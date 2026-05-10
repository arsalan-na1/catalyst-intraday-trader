"""Gemini 2.5 Flash catalyst scorer (two-stage with grounding).

Google Search grounding is incompatible with response_mime_type="application/json"
in a single call, so scoring is split:

  Step 1 (grounded research): Gemini with `tools=[google_search]` and no
    schema — free-form text summarizing what Google Search finds about the
    ticker's move and candidate catalysts.

  Step 2 (structured verdict): Gemini with `response_schema=CatalystVerdict`
    and no tools — takes the step-1 research plus the original trigger data
    and news items, returns a pydantic-validated JSON verdict.

Both calls count toward the 15 RPM free-tier ceiling.

Verdict cache: results are reused for VERDICT_CACHE_MINUTES minutes per ticker
to avoid re-scoring the same event and to prevent re-entering a position that
was just stopped out.

Concurrency: a semaphore (GEMINI_CONCURRENCY) limits simultaneous in-flight
Gemini requests so a burst of triggers doesn't flood the API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from alpaca.data.historical.stock import StockHistoricalDataClient
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

import config
from estimate_revisions import get_estimate_revisions
from self_improvement import format_insights_block, load_insights_if_fresh
from finnhub_data import (
    get_congressional_trades,
    get_earnings_calendar,
    get_earnings_surprise,
    get_insider_score,
    get_insider_sentiment,
)
from fundamentals import get_fundamentals
from news import NewsItem
from reddit_sentiment import get_reddit_sentiment
from sector_momentum import get_sector_momentum
from short_interest import get_short_interest
from technicals import Technicals, get_technicals

log = logging.getLogger("scorer")

# Research uses gemini-2.5-flash: native Google Search grounding,
# highest-stakes call in the pipeline.
# Verdict/eval uses gemini-2.5-flash-lite: 5x cheaper output tokens,
# full response_schema support confirmed, research context already
# provided so quality risk is minimal.
_MODEL_RESEARCH = config.GEMINI_MODEL_RESEARCH
_MODEL_VERDICT  = config.GEMINI_MODEL_VERDICT


class CatalystVerdict(BaseModel):
    """Strict schema Gemini must match. Pydantic validates the response."""

    catalyst_found: bool = Field(description="Is there a real news catalyst for the move?")
    catalyst_summary: str = Field(description="One-sentence description of the catalyst.")
    catalyst_type: Literal[
        "FDA", "earnings", "M&A", "analyst_upgrade", "short_squeeze",
        "insider_buying", "contract", "buyback", "dividend", "other"
    ] = Field(description="Single best-fit category for the catalyst.")
    magnitude_estimate: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=10)
    already_priced_in: bool = Field(
        default=False,
        description=(
            "True if the catalyst broke pre-market and the gap is already fully reflected. "
            "False if catalyst is breaking intraday or still being discovered."
        ),
    )
    technical_signal: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral",
        description=(
            "Overall technical posture based on the provided indicators. "
            "'bullish' if uptrend (price > SMA50 > SMA200) and RSI between 30–70. "
            "'bearish' if downtrend (price < SMA50 < SMA200), RSI ≥ 75, or near 52w low. "
            "'neutral' if sideways or data is missing."
        ),
    )
    suggested_entry: float = Field(description="Suggested entry price.")
    # --- Gemini-driven trade parameters ---
    # Hard floors/ceilings are enforced in trader.py; Gemini sets values freely within them.
    take_profit_pct: float = Field(
        default=0.15,
        description=(
            "Target gain as a decimal (0.20 = 20%). Aggressive for high-conviction "
            "breakouts (0.20–0.35 for magnitude 9–10), conservative for weaker setups "
            "(0.08–0.12 for magnitude 5–6). Clamped to 0.05–0.50."
        ),
    )
    stop_loss_pct: float = Field(
        default=0.05,
        description=(
            "Maximum loss from entry as a positive decimal (0.05 = 5% stop). "
            "Wider for volatile/low-float stocks and binary FDA events (0.08–0.10); "
            "tighter for large-caps with strong catalysts (0.02–0.04). Clamped to 0.02–0.10."
        ),
    )
    position_size_pct: float = Field(
        default=0.10,
        description=(
            "Fraction of portfolio equity to deploy (0.10 = 10%). Scale up for 9–10 "
            "confidence (0.10–0.12), scale down for uncertain setups (0.02–0.05). "
            "Use 0.02–0.05 for biotech/FDA binary events. Clamped to 0.02–0.12."
        ),
    )
    hold_strategy: Literal["momentum", "catalyst", "swing"] = Field(
        default="momentum",
        description=(
            "'momentum' — fast intraday move, 30–60 min; "
            "'catalyst' — news-driven over hours, 90–240 min; "
            "'swing' — could extend into next session, 240–390 min."
        ),
    )
    max_hold_minutes: int = Field(
        default=90,
        description=(
            "Maximum hold time in minutes. Momentum: 30–60. Catalyst: 90–180. "
            "Swing: 240–390. Clamped to 30–390."
        ),
    )
    should_trade: bool = Field(
        default=True,
        description=(
            "Explicit trade/no-trade decision. Set to false if: no real catalyst found, "
            "stock already up 30%+ before entry (fully priced in), RSI > 85 (severely "
            "overextended), risk/reward is unfavorable, or technicals severely broken "
            "with a weak catalyst. Skipping bad trades is as important as entering good ones."
        ),
    )
    reasoning: str = Field(description="One paragraph explaining the catalyst, setup, and all trade parameter decisions.")
    catalyst_quality: Literal["strong", "moderate", "weak"] = Field(
        default="moderate",
        description=(
            "Catalyst quality rating from analysis. 'strong' = real, well-sourced, "
            "proportional to the move (e.g. beat-and-raise, FDA approval, M&A premium). "
            "'moderate' = real but partial confirmation or some priced-in risk. "
            "'weak' = thin sourcing, routine news, or the gap is extended beyond what the news justifies."
        ),
    )
    reversal_risk: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Reversal risk rating from analysis. 'low' = clear path higher with healthy "
            "momentum and supportive tape. 'medium' = some headwinds (overall market "
            "weakness, sector rotation, modest extension) but thesis still tracks. "
            "'high' = stock already extended, RSI > 80 with weak volume, sell-the-news "
            "setup, thin float, or earnings beat already priced in pre-market."
        ),
    )
    skip_reason: str | None = Field(
        default=None,
        description=(
            "Required when should_trade=false. One-sentence explanation, e.g. "
            "'RSI 84 — stock is extended, high fade risk on any catalyst.' "
            "Set to null when should_trade=true."
        ),
    )


class PositionVerdict(BaseModel):
    """Gemini re-evaluation verdict for an open position."""

    action: Literal["hold", "exit", "raise_target", "tighten_sl", "adjust_tp", "add_time"] = Field(
        description=(
            "'hold' = original thesis intact, no changes; "
            "'exit' = close now regardless of P&L; "
            "'raise_target' = new catalyst justifies higher TP (provide new_take_profit_pct); "
            "'tighten_sl' = lock in gains by raising the stop floor (provide new_stop_loss_pct); "
            "'adjust_tp' = revise TP up or down based on updated momentum (provide new_take_profit_pct); "
            "'add_time' = extend hold window, momentum still strong (provide add_minutes)."
        )
    )
    new_take_profit_pct: float | None = Field(
        default=None,
        description=(
            "Required for raise_target and adjust_tp. "
            "New take-profit as a decimal fraction (0.35 = 35%). "
            "For raise_target must exceed current TP; for adjust_tp can be higher or lower. "
            "Null for all other actions."
        ),
    )
    new_stop_loss_pct: float | None = Field(
        default=None,
        description=(
            "Required for tighten_sl. "
            "New minimum P&L floor as a decimal — can be positive to lock in gains "
            "(e.g., 0.08 means close if unrealized P&L drops below +8%). "
            "Must be higher than the current SL floor. Null for all other actions."
        ),
    )
    add_minutes: int | None = Field(
        default=None,
        description=(
            "Required for add_time. "
            "Additional minutes to grant beyond the current max_hold_minutes. "
            "Total will be capped at 390 (full session). Null for all other actions."
        ),
    )
    reason: str = Field(description="One-sentence explanation of the decision.")
    confidence: int = Field(ge=1, le=10, description="Verdict confidence 1–10.")


@dataclass
class TriggerContext:
    """Snapshot passed to the scorer. Keeps this module decoupled from stream.py."""

    ticker: str
    price: float
    price_move_pct: float
    volume_ratio: float
    technicals: Technicals | None = None  # populated by scorer after tech fetch
    trigger_type: str = "intraday"  # "intraday" | "gap_open"
    # Latest session VWAP threaded through from stream.TriggerEvent (None on
    # synthetic CLI triggers and on any bar that did not carry a VWAP value).
    window_vwap: float | None = None


_RESEARCH_SYSTEM_INSTRUCTION = """You are a research analyst investigating a sudden
price/volume spike in a US equity. Use Google Search to find the most likely cause.

Search specifically for:
- FDA decisions, PDUFA dates, clinical trial readouts, complete response letters
- Earnings releases, revenue/EPS beats or misses, guidance changes
- M&A: mergers, acquisitions, buyouts, asset sales, strategic reviews
- Analyst upgrades/downgrades with large price target changes (>25% move in PT)
- SEC filings: 13D/13G activist disclosures, Form 4 insider buying clusters
- Contract wins (government, DoD, large enterprise)
- Short squeeze conditions: high short float + price breaking above resistance
- Buyback authorizations, special dividends, spin-off announcements

Write a plain-text briefing (under 300 words) covering:
1. Most likely catalyst — what happened and when (pre-market or intraday?)
2. Source credibility (SEC filing > press release > analyst note > social media)
3. Whether the move size is proportionate to the catalyst
4. Whether the catalyst is still being discovered or already fully priced in
5. Any red flags: thin corroboration, conflicting headlines, no news found

Do NOT return JSON. Do NOT invent events. If no catalyst is found, say so explicitly."""

_VERDICT_SYSTEM_INSTRUCTION = """You are an intraday catalyst trading analyst. Your job is to evaluate
whether a stock's current move has the characteristics of a trade worth
taking RIGHT NOW — not whether the company is a good long-term investment.

You think like a desk trader, not a portfolio manager. You care about:
- Whether the catalyst is real and proportional to the move
- Whether momentum is likely to continue for the next 2-4 hours
- Whether the risk of a sharp reversal is low enough to justify entry

You do NOT hedge every sentence. You form a clear view and state it.
You gather evidence first, form a conclusion second — never the reverse.
When price action and narrative conflict, you say which one you trust and why.
You respond only in valid JSON. No markdown, no preamble, no explanation outside the JSON."""


_POSITION_EVAL_SYSTEM_INSTRUCTION = """\
You are actively managing an open intraday trade. Think like a professional trader
watching this position in real time — not a rules engine. Use all available context.

HOLD — Default action. Choose when:
  • Original thesis is still intact and no new negative developments have emerged
  • Momentum has not broken down (price above entry, volume still elevated)
  • If uncertain between EXIT and HOLD, always choose HOLD

EXIT — Close the position immediately. Choose when any of the following apply:
  • Stock retraced >50% of post-entry move without reaching TP — momentum failed
  • ANY new negative news since entry: analyst downgrade, secondary offering, regulatory
    action, earnings miss, sector selloff, competing product announcement. Weight heavily.
    A fast exit on bad news is always better than riding a broken thesis to the stop loss.
  • Technicals reversed: RSI collapsed below 40, price fell below SMA50 or SMA200
  • Risk/reward no longer favorable — upside remaining to TP is smaller than downside to SL floor
  • Catalyst was a one-time event that has fully played out with no follow-through buying

RAISE_TARGET — Update take-profit upward. Choose only when:
  • A new confirming catalyst emerged after entry (second upgrade, partnership, approval)
  • RSI is in the healthy 55–72 range with price approaching TP on rising volume
  • Momentum is accelerating, not decelerating
  • Provide new_take_profit_pct strictly higher than current TP

TIGHTEN_SL — Raise the stop floor to lock in gains. Choose when:
  • Stock is significantly profitable (up 10%+ from entry) and you want to protect gains
  • Provide new_stop_loss_pct as the new minimum P&L floor (e.g., 0.08 = close if P&L
    drops below +8% from entry). Must exceed the current SL floor.

ADJUST_TP — Revise take-profit up or down. Choose when:
  • Momentum is decelerating and the current TP is unlikely to be hit: lower TP to bank
    gains early rather than giving them back
  • Strong momentum suggests more upside than the current TP: raise it
  • Provide new_take_profit_pct (can be higher or lower than current)

ADD_TIME — Extend the hold window. Choose when:
  • Momentum is still strong and the time limit is approaching before TP/SL is reached
  • The catalyst is still unfolding and more news is expected within the session
  • Provide add_minutes (additional minutes; total will be capped at session end)

Return valid JSON only. Do not invent news or catalysts not present in the input data."""


class RateLimiter:
    """Sliding-window limiter: at most `limit` calls in `window_s` seconds."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window = window_s
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self._window:
                    self._calls.popleft()
                if len(self._calls) < self._limit:
                    self._calls.append(now)
                    return
                wait = self._window - (now - self._calls[0]) + 0.05
            log.debug("rate limit reached; sleeping %.2fs", wait)
            await asyncio.sleep(wait)


async def _coro_none() -> None:
    return None


class Scorer:
    def __init__(self, hist_client: StockHistoricalDataClient | None = None) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._limiter = RateLimiter(config.GEMINI_REQUESTS_PER_MINUTE, 60.0)
        self._sem = asyncio.Semaphore(config.GEMINI_CONCURRENCY)
        # Separate semaphore for position re-evaluation so open-position scoring
        # never queues behind new-entry scoring (and vice versa).
        self._position_sem = asyncio.Semaphore(config.POSITION_EVAL_SEMAPHORE_LIMIT)
        # verdict cache: ticker -> (cached_at, verdict)
        self._cache: dict[str, tuple[datetime, CatalystVerdict]] = {}
        # call counters (reset by caller at session boundary if desired)
        self.calls_today: int = 0
        self.calls_grounded: int = 0    # scoring events that used Google Search grounding
        self.calls_ungrounded: int = 0  # scoring events that skipped grounding
        self.calls_skipped: int = 0     # events skipped by hourly limit or 30-min cooldown
        # Opt 4: sliding window of API call timestamps for hourly cap.
        self._hourly_calls: deque[float] = deque()
        # Opt 1: per-ticker timestamp of last scoring attempt (30-min cooldown).
        self._last_scored: dict[str, float] = {}
        self._hist_client = hist_client
        # Monthly Gemini cost cap. Reset by daily_summary on the 1st of each month.
        self.monthly_cost_estimate: float = 0.0
        self._budget_mode: str = "normal"  # "normal" | "ungrounded_only" | "halted"
        # Set from bot.py after construction; called with the alert message.
        self._alert_callback = None
        # Optional regime detector label (HMM); None until set by bot.
        self._current_regime_label: str = "unknown"
        # Cached self-improvement insights — refreshed when state/insights.json mtime changes.
        self._insights_cache: dict | None = None
        self._insights_mtime: float | None = None
        # Optional ReflectionStore — set by bot.py at startup. When present,
        # _build_verdict_prompt injects same-ticker + cross-ticker prior
        # reflections under a "PRIOR LESSONS" section.
        self._reflection_store: Any = None

    def _get_reflections_block(self, ticker: str) -> str:
        """Render PRIOR LESSONS injection block for the given ticker.

        Returns "" when no store is wired or no usable reflections exist.
        Token budget cap matches config.REFLECTION_INJECTED_TOKEN_BUDGET;
        the helper drops oldest cross-ticker first, then oldest same-ticker.
        Failure modes are silent — a bad store must never block scoring.
        """
        store = self._reflection_store
        if store is None:
            return ""
        try:
            from reflection_store import format_lessons_block

            same = store.get_recent(
                ticker, n=config.REFLECTION_MAX_PER_TICKER_INJECTED,
            )
            cross = store.get_global_lessons(
                n=config.REFLECTION_MAX_GLOBAL_INJECTED, exclude_ticker=ticker,
            )
            return format_lessons_block(
                same, cross, ticker, max_tokens=config.REFLECTION_INJECTED_TOKEN_BUDGET,
            )
        except Exception:
            log.debug("reflections block render failed", exc_info=True)
            return ""

    def _get_insights_block(self) -> str:
        """Return the formatted lessons block, or empty string when no fresh insights file exists.

        Cached by mtime so we re-parse only when self_improvement writes a new file.
        Failure-modes are silent — a bad/missing insights file must never block scoring.
        """
        try:
            from self_improvement import INSIGHTS_PATH
            if not INSIGHTS_PATH.exists():
                self._insights_cache = None
                self._insights_mtime = None
                return ""
            mtime = INSIGHTS_PATH.stat().st_mtime
            if self._insights_mtime != mtime:
                self._insights_cache = load_insights_if_fresh()
                self._insights_mtime = mtime
            if self._insights_cache is None:
                return ""
            return format_insights_block(self._insights_cache)
        except Exception:
            log.debug("insights load failed; continuing without lessons", exc_info=True)
            return ""

    def _record_call_cost(self, is_grounded: bool) -> bool:
        """Decide whether an upcoming Gemini call may proceed and, if so, add
        its cost to the monthly tally. Returns True only when the call is
        actually allowed — blocked calls do NOT accrue cost (the caller must
        skip the request).

        - At 100% of budget: set _budget_mode='halted', log critical, fire alert,
          return False (caller skips, no cost added).
        - At 80% (only if currently 'normal'): set 'ungrounded_only', warn, alert,
          and still allow the current call (cost added).
        - Below 80%: passthrough, cost added, True.
        """
        cost = (
            config.GEMINI_COST_PER_GROUNDED_CALL
            if is_grounded
            else config.GEMINI_COST_PER_UNGROUNDED_CALL
        )
        cap = config.MONTHLY_GEMINI_BUDGET_USD
        projected = self.monthly_cost_estimate + cost

        if projected >= cap:
            if self._budget_mode != "halted":
                self._budget_mode = "halted"
                msg = (
                    f"🚨 Gemini monthly budget exhausted "
                    f"(${self.monthly_cost_estimate:.2f}) — "
                    f"all Gemini calls halted until next month. Bot running "
                    f"on TP/SL/timeout rules only."
                )
                log.critical(msg)
                if self._alert_callback is not None:
                    try:
                        cb = self._alert_callback(msg)
                        if asyncio.iscoroutine(cb):
                            asyncio.create_task(cb)
                    except Exception:
                        log.exception("budget alert callback failed")
            return False

        # Call is allowed — commit the cost.
        self.monthly_cost_estimate = projected
        used = self.monthly_cost_estimate

        if used >= cap * 0.8 and self._budget_mode == "normal":
            self._budget_mode = "ungrounded_only"
            msg = (
                f"⚠️ Gemini budget 80% used (${used:.2f}/${cap:.0f}) — "
                f"switching to ungrounded-only for rest of month"
            )
            log.warning(msg)
            if self._alert_callback is not None:
                try:
                    cb = self._alert_callback(msg)
                    if asyncio.iscoroutine(cb):
                        asyncio.create_task(cb)
                except Exception:
                    log.exception("budget alert callback failed")

        return True

    def _cache_get(self, ticker: str) -> CatalystVerdict | None:
        entry = self._cache.get(ticker)
        if entry is None:
            return None
        cached_at, verdict = entry
        age_min = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60.0
        if age_min < config.VERDICT_CACHE_MINUTES:
            log.info("[CACHE HIT] %s verdict %.0f min old; reusing", ticker, age_min)
            return verdict
        del self._cache[ticker]
        return None

    def _cache_put(self, ticker: str, verdict: CatalystVerdict) -> None:
        self._cache[ticker] = (datetime.now(timezone.utc), verdict)

    def _check_hourly_limit(self) -> bool:
        """Return True if we have headroom under the hourly API call cap.

        This is a non-mutating check — it does NOT reserve a slot. Callers
        must invoke `_consume_hourly_slot()` immediately before firing the
        actual Gemini request, after every other gate (budget/halt) has
        passed, so blocked calls don't burn slots.

        Fails open on any error so a bad clock or deque corruption never
        silently drops a potentially good trade.
        """
        try:
            now = time.monotonic()
            while self._hourly_calls and now - self._hourly_calls[0] > 3600:
                self._hourly_calls.popleft()
            if len(self._hourly_calls) >= config.MAX_GEMINI_CALLS_PER_HOUR:
                log.warning(
                    "[GEMINI] hourly call limit reached (%d calls in last hour), skipping",
                    len(self._hourly_calls),
                )
                return False
            return True
        except Exception:
            log.exception("hourly limit check failed; proceeding with Gemini call")
            return True  # fail open

    def _consume_hourly_slot(self) -> None:
        """Reserve one slot in the hourly window. Call only after all gates pass
        and immediately before firing the actual Gemini request."""
        try:
            self._hourly_calls.append(time.monotonic())
        except Exception:
            log.debug("hourly slot consume failed", exc_info=True)

    @staticmethod
    def _news_block(news: list[NewsItem]) -> str:
        return (
            "\n".join(item.to_prompt_line() for item in news) if news else "(no recent news)"
        )

    @staticmethod
    def _fundamentals_block(f: dict, current_price: float) -> str:
        if not f:
            return "Fundamentals: unavailable."

        parts: list[str] = []
        mcap = f.get("market_cap")
        if mcap:
            mcap_str = (
                f"${mcap / 1e12:.1f}T" if mcap >= 1e12
                else f"${mcap / 1e9:.1f}B" if mcap >= 1e9
                else f"${mcap / 1e6:.0f}M"
            )
            parts.append(f"Market cap: {mcap_str}")
        float_sh = f.get("float_shares")
        if float_sh:
            float_str = (
                f"{float_sh / 1e9:.1f}B shares" if float_sh >= 1e9
                else f"{float_sh / 1e6:.0f}M shares"
            )
            parts.append(f"Float: {float_str}")
        if f.get("sector"):
            parts.append(f"Sector: {f['sector']}")
        pe = f.get("pe_ratio")
        if pe:
            parts.append(f"P/E: {pe:.1f}")

        lines: list[str] = ["FUNDAMENTALS (yfinance):"]
        if parts:
            lines.append(" | ".join(parts))

        high_52 = f.get("52_week_high")
        low_52 = f.get("52_week_low")
        if high_52 and low_52:
            range_line = f"52w range: ${low_52:.2f} – ${high_52:.2f}"
            if current_price > 0 and high_52 > 0:
                pct_from_high = (current_price - high_52) / high_52
                range_line += f" (current is {pct_from_high:+.0%} from 52w high)"
            lines.append(range_line)

        target = f.get("analyst_target_price")
        if target and current_price > 0:
            upside = (target - current_price) / current_price
            lines.append(f"Analyst mean target: ${target:.2f} ({upside:+.0%} vs current)")

        if f.get("earnings_date"):
            lines.append(f"Next earnings: {f['earnings_date']}")

        return "\n".join(lines)

    @staticmethod
    def _alt_data_block(
        insider: dict,
        earnings_cal: dict,
        congress: dict,
        earnings_surp: dict,
    ) -> str:
        lines: list[str] = ["ALTERNATIVE DATA:"]

        if insider:
            mspr = insider.get("mspr", 0.0)
            net  = insider.get("net_change", 0)
            label = "heavy buying" if mspr > 50 else "heavy selling" if mspr < -50 else "neutral"
            lines.append(
                f"- Insider sentiment MSPR: {mspr:+.1f} ({label}); "
                f"net share change: {net:+,}"
            )
        else:
            lines.append("- Insider sentiment: unavailable")

        if earnings_cal:
            next_date = earnings_cal.get("next_earnings_date", "unknown")
            imminent  = earnings_cal.get("earnings_imminent", False)
            days      = earnings_cal.get("days_until_earnings")
            imm_str   = f"imminent — {days}d away" if imminent else "not imminent"
            beat_str  = ""
            if earnings_surp:
                beat_str = f" | Beat rate: {earnings_surp.get('beat_rate', '?')} last quarters"
            lines.append(
                f"- Earnings: {imm_str} — next: {next_date}{beat_str}"
            )
        else:
            beat_str = f" | Beat rate: {earnings_surp['beat_rate']} last quarters" if earnings_surp else ""
            lines.append(f"- Earnings: date unavailable{beat_str}")

        if congress:
            count  = congress.get("count", 0)
            trades = congress.get("trades") or []
            if count == 0:
                lines.append("- Congressional trades (90d): none")
            else:
                summaries = [
                    f"{t['name']}: {t['transaction_type']} {t['amount']} ({t['date']})"
                    for t in trades[:3]
                ]
                lines.append(
                    f"- Congressional trades (90d): {count} trade(s) — "
                    + "; ".join(summaries)
                )
        else:
            lines.append("- Congressional trades (90d): unavailable")

        return "\n".join(lines)

    @staticmethod
    def _quant_signals_block(
        short_int: dict | None,
        est_rev: dict | None,
        ins_score: dict | None,
        sector_mom: dict | None,
        sector_name: str = "",
    ) -> str:
        lines: list[str] = []
        if short_int:
            lines.append(
                f"Short Interest: {short_int['short_float_pct']:.1f}% float short | "
                f"{short_int['days_to_cover']:.1f}d to cover | Squeeze: {short_int['squeeze_score']}"
            )
        if est_rev:
            lines.append(
                f"Estimate Revisions: {est_rev['eps_surprise_avg']:+.1f}% avg beat | "
                f"Trend: {est_rev['eps_trend']} | Analysts: {est_rev['analyst_revision']} | "
                f"Score: {est_rev['revision_score']:+d}"
            )
        if ins_score:
            buyer_str = (
                f" | Largest buyer: {ins_score['largest_buyer']}"
                if ins_score.get("largest_buyer") else ""
            )
            lines.append(
                f"Insider Activity: Net {ins_score['net_shares_3m']:+,} shares (90d) | "
                f"Signal: {ins_score['insider_signal']}{buyer_str}"
            )
        if sector_mom:
            sec_label = f" ({sector_name})" if sector_name else ""
            lines.append(
                f"Sector{sec_label}: {sector_mom['sector_etf']} "
                f"{sector_mom['sector_change_pct']:+.2f}% today ({sector_mom['sector_momentum']})"
            )
        if not lines:
            return ""
        return "📊 QUANTITATIVE SIGNALS:\n" + "\n".join(lines)

    @staticmethod
    def _atr_hint(tech: Technicals | None) -> str:
        if tech is None or tech.atr_14_pct is None:
            return ""
        atr = tech.atr_14_pct  # already in % units (8.0 = 8% daily range)
        if atr > 8.0:
            return (
                f"\n⚠️ HIGH VOLATILITY: ATR={atr:.1f}% daily range. "
                "Consider smaller position size and wider stop loss."
            )
        if atr < 2.0:
            return (
                f"\nℹ️ LOW VOLATILITY: ATR={atr:.1f}% daily range. "
                "Tighter stops are appropriate."
            )
        return ""

    @staticmethod
    def _reddit_block(reddit: dict) -> str:
        if not reddit:
            return "SOCIAL SENTIMENT (Reddit): unavailable"
        mentions = reddit.get("mention_count", 0)
        if mentions == 0:
            return "SOCIAL SENTIMENT (Reddit, last 24h): 0 mentions across WSB/stocks/investing"
        score = reddit.get("sentiment_score", 0.0)
        label = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
        top_title   = reddit.get("top_post_title") or ""
        top_upvotes = reddit.get("top_post_upvotes", 0)
        lines = [
            "SOCIAL SENTIMENT (Reddit, last 24h):",
            f"- Mentions: {mentions} across WSB/stocks/investing",
            f"- Sentiment: {score:+.2f} ({label})",
        ]
        if top_title:
            truncated = top_title[:100] + "…" if len(top_title) > 100 else top_title
            lines.append(f'- Top post: "{truncated}" ({top_upvotes:,} upvotes)')
        return "\n".join(lines)

    def _build_research_prompt(self, ctx: TriggerContext, news: list[NewsItem]) -> str:
        now_et = datetime.now(tz=config.MARKET_TZ).strftime("%H:%M ET")
        if ctx.trigger_type == "gap_open":
            move_line = f"GAPPED OPEN +{ctx.price_move_pct:.1%} vs previous session close"
        else:
            direction = "UP" if ctx.price_move_pct >= 0 else "DOWN"
            move_line = f"{direction} {abs(ctx.price_move_pct):.1%} in last 2 min"
        return (
            f"Ticker: {ctx.ticker}\n"
            f"Current price: ${ctx.price:.2f}  ({move_line})\n"
            f"Current time: {now_et}\n"
            f"Volume vs expected pace: {ctx.volume_ratio:.1f}× average\n\n"
            f"News already fetched (real-time, may be incomplete):\n{self._news_block(news)}\n\n"
            "Search Google for the most likely catalyst. Write a concise briefing."
        )

    def _build_verdict_prompt(
        self, ctx: TriggerContext, news: list[NewsItem], research: str,
        tech: Technicals | None = None,
        fundamentals: dict | None = None,
        insider: dict | None = None,
        earnings_cal: dict | None = None,
        congress: dict | None = None,
        earnings_surp: dict | None = None,
        reddit: dict | None = None,
        quant_signals: dict | None = None,
        sector_name: str = "",
    ) -> str:
        gap_pct_val = abs(ctx.price_move_pct) * 100
        rsi_val = (
            f"{tech.rsi:.0f}"
            if tech is not None and tech.rsi is not None
            else "N/A"
        )
        news_summary = self._news_block(news)
        beat_rate = (earnings_surp or {}).get("beat_rate")
        earnings_beat = str(beat_rate) if beat_rate else "N/A"
        guidance_change = "N/A"
        market_regime = self._current_regime_label or "unknown"
        insights_block = self._get_insights_block() or "(none)"
        # Per-ticker + cross-ticker post-trade reflections. When the store has
        # no usable entries the block is empty and we omit the section
        # entirely (per spec: do NOT inject "no prior lessons").
        reflections_block = self._get_reflections_block(ctx.ticker)
        reflections_section = (
            f"\n{reflections_block}\n" if reflections_block else ""
        )

        prompt = (
            "Evaluate this intraday catalyst trade opportunity. Work through each\n"
            "section in order before giving your final score. Do not jump to the\n"
            "conclusion first.\n\n"
            f"TICKER: {ctx.ticker}\n"
            f"TRIGGER: {ctx.trigger_type}\n"
            f"GAP: {gap_pct_val:.1f}%\n"
            f"VOLUME RATIO: {ctx.volume_ratio:.1f}x average\n"
            f"PRICE: ${ctx.price:.2f}\n"
            f"RSI: {rsi_val}\n"
            f"NEWS SUMMARY: {news_summary}\n"
            f"EARNINGS BEAT: {earnings_beat}  (if available, else \"N/A\")\n"
            f"GUIDANCE CHANGE: {guidance_change}  (if available, else \"N/A\")\n"
            f"SECTOR REGIME: {market_regime}\n"
            f"RECENT INSIGHTS: {insights_block}\n"
            f"{reflections_section}\n"
            "Work through these four questions in order:\n\n"
            "1. CATALYST QUALITY\n"
            "Is the catalyst real and significant? For earnings: was it a\n"
            "beat-and-raise (best), beat-and-hold (neutral), or beat-and-lower (trap)?\n"
            "Is the gap size proportional to the news, or does the stock look\n"
            "extended beyond what the news justifies? Rate quality: strong / moderate / weak.\n\n"
            "2. MOMENTUM SUSTAINABILITY  \n"
            "Based on gap size, volume ratio, and RSI — is this move likely to\n"
            "continue or fade in the next 2-4 hours? Stocks that gap 20%+ on 5x volume\n"
            "with RSI under 75 tend to grind higher. Stocks already at RSI 85+ on moderate\n"
            "volume often stall. Is there a clear path higher or is this a likely\n"
            "sell-the-news situation?\n\n"
            "3. REVERSAL RISK\n"
            "What is the single most likely reason this trade fails? (e.g. overall\n"
            "market weakness, sector rotation, stock already extended from prior run,\n"
            "thin float, earnings beat already priced in premarket.) How severe would\n"
            "a reversal likely be — sharp and fast, or gradual?\n\n"
            "4. FINAL VERDICT\n"
            "Based only on what you reasoned above — not gut feel — give:\n"
            "- confidence: integer 1-10 (1=strong skip, 10=strong buy)\n"
            "- verdict: \"buy\", \"skip\", or \"avoid\"  \n"
            "- reason: one sentence max, no hedging\n"
            "- catalyst_quality: \"strong\", \"moderate\", or \"weak\"\n"
            "- reversal_risk: \"low\", \"medium\", or \"high\"\n\n"
            "Respond in this exact JSON format:\n"
            "{\n"
            "  \"confidence\": <int>,\n"
            "  \"verdict\": \"<buy|skip|avoid>\",\n"
            "  \"reason\": \"<one sentence>\",\n"
            "  \"catalyst_quality\": \"<strong|moderate|weak>\",\n"
            "  \"reversal_risk\": \"<low|medium|high>\",\n"
            "  \"reasoning\": {\n"
            "    \"catalyst\": \"<2-3 sentences>\",\n"
            "    \"momentum\": \"<2-3 sentences>\",\n"
            "    \"reversal_risk\": \"<2-3 sentences>\"\n"
            "  }\n"
            "}"
        )
        log.debug(
            "[SCORER] verdict prompt for %s: %d chars", ctx.ticker, len(prompt)
        )
        if len(prompt) > 8000:
            log.warning(
                "[SCORER] verdict prompt for %s is %d chars (>8000); consider trimming",
                ctx.ticker, len(prompt),
            )
        return prompt

    async def _research_with_grounding(
        self, ctx: TriggerContext, news: list[NewsItem]
    ) -> str | None:
        if not self._check_hourly_limit():
            self.calls_skipped += 1
            return None
        # Monthly budget gate — must pass before incurring grounded-call cost.
        if not self._record_call_cost(is_grounded=True):
            self.calls_skipped += 1
            return None
        prompt = self._build_research_prompt(ctx, news)
        await self._limiter.acquire()
        self._consume_hourly_slot()
        self.calls_today += 1
        try:
            response = await self._client.aio.models.generate_content(
                model=_MODEL_RESEARCH,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_RESEARCH_SYSTEM_INSTRUCTION,
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                    temperature=0.2,
                ),
            )
        except Exception:
            log.exception("gemini research call failed for %s", ctx.ticker)
            return None

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            log.warning("gemini research returned empty text for %s", ctx.ticker)
            return None
        return text

    async def _verdict_from_research(
        self, ctx: TriggerContext, news: list[NewsItem], research: str,
        tech: Technicals | None = None,
        fundamentals: dict | None = None,
        insider: dict | None = None,
        earnings_cal: dict | None = None,
        congress: dict | None = None,
        earnings_surp: dict | None = None,
        reddit: dict | None = None,
        quant_signals: dict | None = None,
        sector_name: str = "",
    ) -> CatalystVerdict | None:
        if not self._check_hourly_limit():
            self.calls_skipped += 1
            return None
        # Monthly budget gate — verdict calls are always ungrounded.
        if not self._record_call_cost(is_grounded=False):
            self.calls_skipped += 1
            return None
        prompt = self._build_verdict_prompt(
            ctx, news, research, tech, fundamentals,
            insider, earnings_cal, congress, earnings_surp, reddit,
            quant_signals=quant_signals,
            sector_name=sector_name,
        )
        await self._limiter.acquire()
        self._consume_hourly_slot()
        self.calls_today += 1
        # Retry once on transient failure (timeouts, 5xx) — common at market open
        # when Gemini is under heavy load. The retry uses a single 3s backoff and
        # only the second failure is error-logged.
        response = None
        for attempt in (1, 2):
            try:
                response = await self._client.aio.models.generate_content(
                    model=_MODEL_VERDICT,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_VERDICT_SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=CatalystVerdict,
                        temperature=0.2,
                    ),
                )
                break
            except Exception as exc:
                if attempt == 1:
                    log.warning(
                        "gemini verdict call failed for %s (attempt 1, retrying in 3s): %r",
                        ctx.ticker, exc,
                    )
                    await asyncio.sleep(3.0)
                    continue
                log.exception("gemini verdict call failed for %s after retry", ctx.ticker)
                return None

        text = getattr(response, "text", None) or ""
        if not text:
            log.warning("gemini verdict returned empty text for %s", ctx.ticker)
            return None
        try:
            verdict = CatalystVerdict.model_validate_json(text)
        except ValidationError:
            log.exception("gemini verdict failed validation for %s: %s", ctx.ticker, text[:300])
            return None
        log.info(
            "[VERDICT] %s: should_trade=%s conf=%d mag=%d catalyst=%s "
            "catalyst_quality=%s reversal_risk=%s technical=%s",
            ctx.ticker, verdict.should_trade, verdict.confidence,
            verdict.magnitude_estimate, verdict.catalyst_type,
            verdict.catalyst_quality, verdict.reversal_risk,
            verdict.technical_signal,
        )
        return verdict

    async def _score_without_grounding(
        self,
        ctx: TriggerContext,
        news: list[NewsItem],
        tech: Technicals | None = None,
        fundamentals: dict | None = None,
        insider: dict | None = None,
        earnings_cal: dict | None = None,
        congress: dict | None = None,
        earnings_surp: dict | None = None,
        reddit: dict | None = None,
        quant_signals: dict | None = None,
        sector_name: str = "",
    ) -> CatalystVerdict | None:
        """Single Gemini verdict call with no Google Search grounding.

        Roughly 10× cheaper than the full two-call grounded path.  Used for:
          • Weak triggers (vol_ratio < GROUNDING_VOL_THRESHOLD AND move < GROUNDING_PRICE_THRESHOLD)
          • Microcap screener Stage-2 quick check before committing to full grounding.
        Accepts optional pre-fetched data; any missing field defaults to None (neutral).
        Increments calls_ungrounded so the daily summary can report the breakdown.
        """
        self.calls_ungrounded += 1
        research = (
            "[No web search grounding — verdict based on available news only]\n\n"
            f"Available news for {ctx.ticker}:\n{self._news_block(news)}"
        )
        return await self._verdict_from_research(
            ctx, news, research, tech, fundamentals,
            insider, earnings_cal, congress, earnings_surp, reddit,
            quant_signals=quant_signals,
            sector_name=sector_name,
        )

    async def score(
        self, ctx: TriggerContext, news: list[NewsItem]
    ) -> tuple[CatalystVerdict | None, Technicals | None, dict]:
        """Score a trigger event. Returns (verdict, technicals, quant_signals).

        quant_signals is a dict with keys: short_interest, estimate_revisions,
        insider_score, sector_momentum — each value is a dict or None.
        Returns an empty dict on cache/cooldown hits.
        """
        cached = self._cache_get(ctx.ticker)
        if cached is not None:
            return cached, None, {}

        # 30-minute per-ticker scoring cooldown (Opt 1): prevents spending Gemini
        # calls on rapid re-triggers of the same ticker.
        now_mono = time.monotonic()
        last = self._last_scored.get(ctx.ticker)
        if last is not None and now_mono - last < 1800:
            log.info(
                "[SCORER] %s scored %.0fmin ago; skipping re-score (30-min cooldown)",
                ctx.ticker, (now_mono - last) / 60,
            )
            self.calls_skipped += 1
            return None, None, {}

        # Cooldown is set only on success (see end of this function). A failed
        # Gemini call must NOT poison the per-ticker 30-min cooldown — the next
        # trigger for this ticker should be allowed to retry immediately.

        # Opt 1: skip expensive Google Search grounding for weak signals.
        use_grounding = not (
            ctx.volume_ratio < config.GROUNDING_VOL_THRESHOLD
            and abs(ctx.price_move_pct) < config.GROUNDING_PRICE_THRESHOLD
        )
        # Monthly budget overrides — once 80% used, no more grounded calls.
        if self._budget_mode == "ungrounded_only":
            use_grounding = False

        async with self._sem:
            # Each helper already catches its own internal errors and returns
            # None / {}. return_exceptions=True is a defense-in-depth: a future
            # library upgrade that raises a new exception type past the helper's
            # try/except shouldn't kill the whole scoring path. Failed sources
            # are logged and substituted with None; downstream consumers all
            # treat None / {} as "signal unavailable".
            _enrichment_sources = (
                ("technicals",
                 get_technicals(ctx.ticker, self._hist_client, ctx.price)
                 if self._hist_client else _coro_none()),
                ("fundamentals",       get_fundamentals(ctx.ticker)),
                ("insider_sentiment",  get_insider_sentiment(ctx.ticker)),
                ("earnings_calendar",  get_earnings_calendar(ctx.ticker)),
                ("congressional",      get_congressional_trades(ctx.ticker)),
                ("earnings_surprise",  get_earnings_surprise(ctx.ticker)),
                ("reddit_sentiment",   get_reddit_sentiment(ctx.ticker)),
                ("short_interest",     get_short_interest(ctx.ticker)),
                ("estimate_revisions", get_estimate_revisions(ctx.ticker)),
                ("insider_score",      get_insider_score(ctx.ticker)),
            )
            _raw = await asyncio.gather(
                *(coro for _, coro in _enrichment_sources),
                return_exceptions=True,
            )
            _safe: list = []
            for (name, _), result in zip(_enrichment_sources, _raw):
                if isinstance(result, BaseException):
                    log.warning(
                        "[SCORER] enrichment fetch '%s' raised for %s: %r",
                        name, ctx.ticker, result,
                    )
                    _safe.append(None)
                else:
                    _safe.append(result)
            (tech, fundamentals,
             insider, earnings_cal, congress, earnings_surp,
             reddit,
             short_int, est_rev, ins_score) = _safe

            # Sector momentum needs the sector from fundamentals; fetch after gather.
            sector_name = (fundamentals or {}).get("sector", "") or ""
            sector_mom = await get_sector_momentum(sector_name, self._hist_client)

            quant_signals: dict = {
                "short_interest": short_int,
                "estimate_revisions": est_rev,
                "insider_score": ins_score,
                "sector_momentum": sector_mom,
            }

            # Halted mode: short-circuit entirely — no Gemini calls allowed.
            if self._budget_mode == "halted":
                log.info(
                    "[SCORER] %s — Gemini halted (monthly budget exhausted); skipping",
                    ctx.ticker,
                )
                self.calls_skipped += 1
                return None, tech, quant_signals

            if use_grounding:
                research = await self._research_with_grounding(ctx, news)
                if research is None:
                    return None, tech, quant_signals
                verdict = await self._verdict_from_research(
                    ctx, news, research, tech, fundamentals,
                    insider, earnings_cal, congress, earnings_surp, reddit,
                    quant_signals=quant_signals,
                    sector_name=sector_name,
                )
                self.calls_grounded += 1
            else:
                if self._budget_mode == "ungrounded_only":
                    log.info(
                        "[SCORER] %s budget mode=ungrounded_only; skipping grounding",
                        ctx.ticker,
                    )
                else:
                    log.info(
                        "[SCORER] %s weak signal (vol=%.1fx, move=%.1f%%); skipping grounding",
                        ctx.ticker, ctx.volume_ratio, ctx.price_move_pct * 100,
                    )
                verdict = await self._score_without_grounding(
                    ctx, news, tech, fundamentals, insider,
                    earnings_cal, congress, earnings_surp, reddit,
                    quant_signals=quant_signals,
                    sector_name=sector_name,
                )
                # calls_ungrounded already incremented inside _score_without_grounding

        if verdict is not None:
            self._cache_put(ctx.ticker, verdict)
            self._last_scored[ctx.ticker] = now_mono
        return verdict, tech, quant_signals

    # --- open-position re-evaluation ---

    def _build_position_eval_prompt(
        self,
        ctx: TriggerContext,
        entry_price: float,
        hold_minutes: float,
        original_take_profit_pct: float,
        news: list[NewsItem],
        tech: Technicals | None,
        sl_floor: float = 0.0,
        max_hold_minutes: int = 90,
        quant_signals: dict | None = None,
        sector_name: str = "",
    ) -> str:
        # NOTE: Reflections are injected into _build_verdict_prompt only.
        # Position-eval injection deferred to v2 — needs filtering design
        # (loss-only? same-side? recency-weighted?). See HANDOFF.md.
        current_price = ctx.price
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        remaining_to_tp = original_take_profit_pct - pnl_pct
        dist_above_sl_floor = pnl_pct - sl_floor
        time_remaining = max_hold_minutes - hold_minutes

        tech_section = (
            f"\nCurrent technical indicators:\n{tech.to_prompt_text()}"
            if tech is not None
            else "\nTechnical indicators: unavailable."
        )

        sl_desc = (
            f"trailing stop (locks in {sl_floor:.0%} gain)"
            if sl_floor > 0
            else f"loss stop at {sl_floor:.0%} from entry"
        )

        qs = quant_signals or {}
        quant_block = self._quant_signals_block(
            qs.get("short_interest"),
            qs.get("estimate_revisions"),
            qs.get("insider_score"),
            qs.get("sector_momentum"),
            sector_name=sector_name,
        )
        quant_section = f"\n{quant_block}\n" if quant_block else ""

        return (
            f"Ticker: {ctx.ticker}\n"
            f"Entry price: ${entry_price:.2f}\n"
            f"Current price: ${current_price:.2f} ({pnl_pct:+.1%} from entry)\n"
            f"Hold time: {hold_minutes:.0f} min | Time remaining before timeout: {time_remaining:.0f} min\n"
            f"Current TP target: +{original_take_profit_pct:.0%} "
            f"(remaining upside to TP: {remaining_to_tp:+.1%})\n"
            f"Current SL floor: {sl_floor:+.0%} ({sl_desc}; "
            f"P&L distance above floor: {dist_above_sl_floor:+.1%})\n"
            f"{quant_section}\n"
            f"Recent news:\n{self._news_block(news)}"
            f"{tech_section}\n\n"
            "You have full control over this position. Choose hold, exit, raise_target, "
            "tighten_sl, adjust_tp, or add_time. Weigh any new negative news heavily — "
            "a fast exit on bad news beats riding a broken thesis to the stop loss."
        )

    async def score_open_position(
        self,
        ctx: TriggerContext,
        entry_price: float,
        hold_minutes: float,
        original_take_profit_pct: float,
        fresh_news: list[NewsItem],
        fresh_technicals: Technicals | None = None,
        sl_floor: float = 0.0,
        max_hold_minutes: int = 90,
        sector: str = "unknown",
    ) -> PositionVerdict | None:
        """Single Gemini call to re-evaluate an open position (no Google Search grounding).

        Uses a dedicated semaphore so position re-evals never queue behind new-entry
        scoring. Both share the same rate limiter (same Gemini API quota).
        Fetches fresh quant signals (uses TTL caches so mostly instant).
        """
        # Fetch fresh quant signals outside the semaphore (data I/O, not Gemini quota).
        # return_exceptions=True so one source raising can't abort re-evaluation.
        _quant_sources = (
            ("short_interest",     get_short_interest(ctx.ticker)),
            ("estimate_revisions", get_estimate_revisions(ctx.ticker)),
            ("insider_score",      get_insider_score(ctx.ticker)),
        )
        _raw = await asyncio.gather(
            *(coro for _, coro in _quant_sources),
            return_exceptions=True,
        )
        _safe: list = []
        for (name, _), result in zip(_quant_sources, _raw):
            if isinstance(result, BaseException):
                log.warning(
                    "[SCORER] position-eval fetch '%s' raised for %s: %r",
                    name, ctx.ticker, result,
                )
                _safe.append(None)
            else:
                _safe.append(result)
        short_int, est_rev, ins_score = _safe
        sector_mom = await get_sector_momentum(sector, self._hist_client)
        quant_signals: dict = {
            "short_interest": short_int,
            "estimate_revisions": est_rev,
            "insider_score": ins_score,
            "sector_momentum": sector_mom,
        }

        prompt = self._build_position_eval_prompt(
            ctx, entry_price, hold_minutes, original_take_profit_pct,
            fresh_news, fresh_technicals,
            sl_floor=sl_floor,
            max_hold_minutes=max_hold_minutes,
            quant_signals=quant_signals,
            sector_name=sector,
        )
        async with self._position_sem:
            if not self._check_hourly_limit():
                self.calls_skipped += 1
                return None
            # Monthly budget gate — position re-evals are always ungrounded.
            # Halted mode hard-skips; ungrounded_only allows the call.
            if self._budget_mode == "halted":
                self.calls_skipped += 1
                return None
            if not self._record_call_cost(is_grounded=False):
                self.calls_skipped += 1
                return None
            await self._limiter.acquire()
            self._consume_hourly_slot()
            self.calls_today += 1
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=_MODEL_VERDICT,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=_POSITION_EVAL_SYSTEM_INSTRUCTION,
                            response_mime_type="application/json",
                            response_schema=PositionVerdict,
                            temperature=0.1,
                        ),
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "gemini position eval timed out for %s; defaulting to hold",
                    ctx.ticker,
                )
                return None
            except Exception:
                log.exception("gemini position eval failed for %s", ctx.ticker)
                return None

        text = getattr(response, "text", None) or ""
        if not text:
            log.warning("gemini position eval returned empty text for %s", ctx.ticker)
            return None
        try:
            return PositionVerdict.model_validate_json(text)
        except ValidationError:
            log.exception(
                "position verdict validation failed for %s: %s",
                ctx.ticker, text[:200],
            )
            return None

    def passes_thresholds(self, verdict: CatalystVerdict) -> bool:
        return (
            verdict.catalyst_found
            and verdict.confidence >= config.MIN_GEMINI_CONFIDENCE
            and verdict.magnitude_estimate >= config.MIN_GEMINI_MAGNITUDE
        )


async def _smoke() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    scorer = Scorer()
    fake_ctx = TriggerContext(
        ticker="ABCD",
        price=42.10,
        price_move_pct=0.12,
        volume_ratio=7.4,
    )
    fake_news = [
        NewsItem(
            headline="ABCD receives FDA approval for lead drug candidate",
            source="Reuters",
            url="https://example.com",
            published_at=None,
            summary="The approval exceeded expectations and expands the addressable market.",
        )
    ]
    verdict, tech, quant = await scorer.score(fake_ctx, fake_news)
    print(verdict)
    print(tech)
    print(quant)


if __name__ == "__main__":
    asyncio.run(_smoke())
