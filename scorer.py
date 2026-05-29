"""Gemini catalyst scorer (single structured-verdict call).

Scoring is a single Gemini call: `response_schema=CatalystVerdict` with no
tools, taking the trigger data, news items, technicals and quant signals and
returning a pydantic-validated JSON verdict. The call counts toward the
15 RPM free-tier ceiling.

(Historically this was a two-stage pipeline with a leading Google Search
grounding call. That grounded research text was never inserted into the
verdict prompt — it was the priciest call in the pipeline yet had zero effect
on the verdict — so it was removed. See git history for the old behavior.)

Verdict cache: results are reused for VERDICT_CACHE_MINUTES minutes per ticker
to avoid re-scoring the same event and to prevent re-entering a position that
was just stopped out.

Concurrency: a semaphore (GEMINI_CONCURRENCY) limits simultaneous in-flight
Gemini requests so a burst of triggers doesn't flood the API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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

# Verdict/eval uses gemini-2.5-flash-lite: cheap output tokens, full
# response_schema support. This is the only scoring model — the former grounded
# research stage (gemini-2.5-flash + Google Search) was removed because its
# output was never inserted into the verdict prompt.
_MODEL_VERDICT  = config.GEMINI_MODEL_VERDICT


def compute_gemini_cost(
    model: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
    *,
    grounded_queries: int = 0,
) -> float:
    """Real USD cost for one Gemini call from token counts.

    Token cost uses config.GEMINI_TOKEN_PRICING ($/1M tokens). An unknown model
    contributes 0 token cost (the caller logs a warning). Grounding is billed
    only when config.GEMINI_GROUNDING_COST_PER_1K is non-zero (default 0 — the
    free quota; the grounded call was removed so grounded_queries is normally 0).
    Pure function — no I/O, safe to unit-test in isolation.
    """
    rates = config.GEMINI_TOKEN_PRICING.get(model)
    token_cost = 0.0
    if rates:
        token_cost = (
            (prompt_tokens or 0) / 1_000_000.0 * rates["input"]
            + (output_tokens or 0) / 1_000_000.0 * rates["output"]
        )
    grounding_cost = (grounded_queries or 0) / 1000.0 * config.GEMINI_GROUNDING_COST_PER_1K
    return token_cost + grounding_cost


def persist_cost_state(scorer: "Scorer", path) -> None:
    """Atomically write the running Gemini cost tally to `path` (Fix 4).

    A tiny sidecar, flushed on a short timer, so a mid-day restart resumes from
    the true accumulated monthly cost instead of the last daily-summary
    snapshot. Atomic (tmp→rename). Never raises — persistence must not break
    the bot.
    """
    try:
        path = Path(path)
        state = {
            "month": datetime.now(tz=config.MARKET_TZ).strftime("%Y-%m"),
            "monthly_cost_estimate": round(scorer.monthly_cost_estimate, 6),
            "usage_day": scorer.usage_day,
            "usage_by_model": scorer.usage_by_model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)  # atomic rename — no partial read on crash
    except Exception:
        log.warning("failed to persist Gemini cost state", exc_info=True)


def restore_cost_state(scorer: "Scorer", path) -> None:
    """Restore the running cost tally from `path` when it is for the current
    calendar month (Fix 4). Recomputes _budget_mode so a restored high tally
    re-enters ungrounded_only / halted. Stale-month or missing file → no-op.
    Never raises.
    """
    try:
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        current_month = datetime.now(tz=config.MARKET_TZ).strftime("%Y-%m")
        if data.get("month") != current_month:
            log.info(
                "Gemini cost state is for %s, not %s; ignoring",
                data.get("month"), current_month,
            )
            return
        cost = data.get("monthly_cost_estimate")
        if isinstance(cost, (int, float)):
            scorer.monthly_cost_estimate = float(cost)
        ubm = data.get("usage_by_model")
        if isinstance(ubm, dict):
            scorer.usage_by_model = ubm
        ud = data.get("usage_day")
        if isinstance(ud, str):
            scorer.usage_day = ud
        cap = config.MONTHLY_GEMINI_BUDGET_USD
        if scorer.monthly_cost_estimate >= cap:
            scorer._budget_mode = "halted"
        elif scorer.monthly_cost_estimate >= cap * 0.8:
            scorer._budget_mode = "ungrounded_only"
        log.info(
            "restored Gemini cost state: $%.4f for %s (mode=%s)",
            scorer.monthly_cost_estimate, current_month, scorer._budget_mode,
        )
    except Exception:
        log.warning("failed to restore Gemini cost state", exc_info=True)


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
        # Post-call this is reconciled to the REAL token cost (see _account_usage);
        # the flat per-call constant added by _record_call_cost is only a
        # conservative pre-call reservation that prevents racing past the cap.
        self.monthly_cost_estimate: float = 0.0
        # Real per-day, per-model usage breakdown (Fix 2). usage_day is the ET
        # date the breakdown covers; usage_by_model maps model -> {calls,
        # prompt_tokens, output_tokens, cost_usd}. Persisted to performance.json.
        self.usage_day: str = ""
        self.usage_by_model: dict[str, dict] = {}
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

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int]:
        """(prompt_tokens, output_tokens) from a Gemini response.

        Returns (0, 0) when usage_metadata is absent or its fields are None —
        accounting must never raise on a malformed/partial response.
        """
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return 0, 0
        return (
            int(getattr(um, "prompt_token_count", 0) or 0),
            int(getattr(um, "candidates_token_count", 0) or 0),
        )

    def _roll_usage_day(self) -> None:
        """Reset the per-model breakdown when the ET calendar day changes."""
        today = datetime.now(tz=config.MARKET_TZ).strftime("%Y-%m-%d")
        if self.usage_day != today:
            self.usage_day = today
            self.usage_by_model = {}

    def _account_usage(
        self,
        model: str,
        response: Any,
        *,
        is_grounded: bool = False,
        grounded_queries: int = 0,
    ) -> float:
        """Post-call real-cost accounting (Fix 2).

        Reconciles the flat pre-charge added by _record_call_cost to the REAL
        token cost from usage_metadata, and records a per-day, per-model
        token + dollar breakdown. Returns the real cost. Never raises — cost
        bookkeeping must not break the scoring/trade path.
        """
        try:
            prompt_tok, out_tok = self._extract_usage(response)
            real = compute_gemini_cost(
                model, prompt_tok, out_tok, grounded_queries=grounded_queries
            )
            reserved = (
                config.GEMINI_COST_PER_GROUNDED_CALL
                if is_grounded
                else config.GEMINI_COST_PER_UNGROUNDED_CALL
            )
            # Replace the conservative flat reservation with the real cost so the
            # monthly tally reflects actual token spend.
            self.monthly_cost_estimate = max(
                0.0, self.monthly_cost_estimate - reserved + real
            )
            self._roll_usage_day()
            bucket = self.usage_by_model.setdefault(
                model,
                {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt_tok
            bucket["output_tokens"] += out_tok
            bucket["cost_usd"] = round(bucket["cost_usd"] + real, 6)
            if model not in config.GEMINI_TOKEN_PRICING:
                log.warning(
                    "[COST] unknown model %r — token cost not computed; add it "
                    "to config.GEMINI_TOKEN_PRICING", model,
                )
            return real
        except Exception:
            log.debug("[COST] usage accounting failed", exc_info=True)
            return 0.0

    # TODO(ruflo-cost-tracker): ruflo-cost-tracker is a documentation/plugin
    # reference (CLAUDE.md), not a runtime dependency — it exposes no in-process
    # ingestion API importable from this codebase. If/when one ships, forward
    # (model, prompt_tok, out_tok, real_cost) from _account_usage to it here.

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

    def _build_verdict_prompt(
        self, ctx: TriggerContext, news: list[NewsItem],
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
            f"{tech.rsi_14:.0f}"
            if tech is not None and tech.rsi_14 is not None
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

    async def _verdict_from_research(
        self, ctx: TriggerContext, news: list[NewsItem],
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
        # Build prompt BEFORE charging budget/slots: if our own code raises
        # during prompt construction, we haven't actually called Gemini and
        # must not consume any of the 60/hr cap, monthly budget, or RPM slot.
        try:
            prompt = self._build_verdict_prompt(
                ctx, news, tech, fundamentals,
                insider, earnings_cal, congress, earnings_surp, reddit,
                quant_signals=quant_signals,
                sector_name=sector_name,
            )
        except Exception:
            log.exception(
                "verdict prompt build failed for %s; budget not charged",
                ctx.ticker,
            )
            return None
        # Monthly budget gate — verdict calls are always ungrounded.
        if not self._record_call_cost(is_grounded=False):
            self.calls_skipped += 1
            return None
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

        # Real token accounting (Fix 2). Account as soon as we have a response —
        # tokens are billed even if the verdict text is empty or fails validation.
        if response is not None:
            self._account_usage(_MODEL_VERDICT, response, is_grounded=False)

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
        """Single Gemini verdict call (the only scoring call there is).

        Used for:
          • Weak triggers (vol_ratio < GROUNDING_VOL_THRESHOLD AND move < GROUNDING_PRICE_THRESHOLD)
          • Microcap screener Stage-2 quick check.
        Accepts optional pre-fetched data; any missing field defaults to None (neutral).
        Increments calls_ungrounded so the daily summary can report the breakdown.
        """
        self.calls_ungrounded += 1
        return await self._verdict_from_research(
            ctx, news, tech, fundamentals,
            insider, earnings_cal, congress, earnings_surp, reddit,
            quant_signals=quant_signals,
            sector_name=sector_name,
        )

    async def fetch_technicals(self, ctx: TriggerContext) -> Technicals | None:
        """Fetch technicals for the deterministic pre-Gemini reject gates (Fix 3).

        Mirrors the fetch score() does internally so the caller can gate on RSI /
        trend / 52w-distance BEFORE spending a Gemini call, then hand the result
        back to score(tech=...) to avoid a second fetch. Fail-soft: returns None
        on any error (the gates fail-open/closed on None exactly as before).
        """
        if self._hist_client is None:
            return None
        try:
            return await get_technicals(ctx.ticker, self._hist_client, ctx.price)
        except Exception:
            log.debug("technicals prefetch failed for %s", ctx.ticker, exc_info=True)
            return None

    def will_score_call_model(self, ctx: TriggerContext) -> bool:
        """True iff score(ctx) would actually invoke Gemini right now (Fix 3).

        Non-mutating mirror of score()'s early-return guards — verdict cache,
        the per-ticker 30-min scoring cooldown, and a halted budget. The caller
        uses this to decide whether the pre-Gemini reject gate may short-circuit:
        it only does so when a real (billable) call is imminent, so cache- /
        cooldown-served candidates keep flowing through score() and the
        authoritative post-score gates exactly as before. Keep in lockstep with
        the guards at the top of score().
        """
        if self._budget_mode == "halted":
            return False
        entry = self._cache.get(ctx.ticker)
        if entry is not None:
            cached_at, _ = entry
            age_min = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60.0
            if age_min < config.VERDICT_CACHE_MINUTES:
                return False
        last = self._last_scored.get(ctx.ticker)
        if last is not None and time.monotonic() - last < 1800:
            return False
        return True

    def note_scored(self, ticker: str) -> None:
        """Start the 30-min per-ticker scoring cooldown without calling Gemini.

        Used when a pre-Gemini reject gate (bot._process_trigger, Fix 3) skips
        the scoring call: the old flow ran score() — which set this cooldown on
        success — before the gate rejected, so re-triggers were suppressed for
        30 min. Marking it here preserves that suppression, keeping the
        accept/reject set identical to the pre-reorder behavior.
        """
        self._last_scored[ticker] = time.monotonic()

    async def score(
        self, ctx: TriggerContext, news: list[NewsItem],
        *, tech: Technicals | None = None,
    ) -> tuple[CatalystVerdict | None, Technicals | None, dict]:
        """Score a trigger event. Returns (verdict, technicals, quant_signals).

        quant_signals is a dict with keys: short_interest, estimate_revisions,
        insider_score, sector_momentum — each value is a dict or None.
        Returns an empty dict on cache/cooldown hits.

        When `tech` is supplied (pre-fetched by the caller for the pre-Gemini
        reject gates — see bot._process_trigger, Fix 3), it is reused as-is and
        the internal technicals fetch is skipped so we never double-fetch.
        """
        precomputed_tech = tech
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

        # Strong vs weak signal. Historically this gated an extra Google-Search
        # grounding call; that call was removed, so both buckets now make the
        # same single verdict call and this only selects the telemetry counter
        # (calls_grounded vs calls_ungrounded).
        use_grounding = not (
            ctx.volume_ratio < config.GROUNDING_VOL_THRESHOLD
            and abs(ctx.price_move_pct) < config.GROUNDING_PRICE_THRESHOLD
        )
        # Monthly budget overrides — once 80% used, force the weak/ungrounded bucket.
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
                 _coro_none() if precomputed_tech is not None
                 else (get_technicals(ctx.ticker, self._hist_client, ctx.price)
                       if self._hist_client else _coro_none())),
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

            # Reuse caller-supplied technicals (Fix 3) — the gather slot above
            # was a no-op when precomputed_tech was provided.
            if precomputed_tech is not None:
                tech = precomputed_tech

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
                verdict = await self._verdict_from_research(
                    ctx, news, tech, fundamentals,
                    insider, earnings_cal, congress, earnings_surp, reddit,
                    quant_signals=quant_signals,
                    sector_name=sector_name,
                )
                self.calls_grounded += 1
            else:
                if self._budget_mode == "ungrounded_only":
                    log.info(
                        "[SCORER] %s budget mode=ungrounded_only (weak bucket)",
                        ctx.ticker,
                    )
                else:
                    log.info(
                        "[SCORER] %s weak signal (vol=%.1fx, move=%.1f%%)",
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

        # Real token accounting (Fix 2).
        if response is not None:
            self._account_usage(_MODEL_VERDICT, response, is_grounded=False)

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
