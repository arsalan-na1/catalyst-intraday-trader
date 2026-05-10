"""Post-trade reflection generator.

Called from trader.py after a SELL fill is finalized. Computes alpha vs SPY
over the holding window, asks Gemini for a one-paragraph reflection, and
persists the record to state/reflections.jsonl.

The close path NEVER awaits reflection generation — the trader fires this
as a fire-and-forget asyncio task. Failure modes (Gemini error, SPY data
unavailable, store write failure) are logged but cannot affect the close
flow. When Gemini fails, we still persist a record with reflection_text=None
so there's an audit trail of every trade.

Schema of an entry written to reflections.jsonl:
    ticker            : str
    trade_id          : str   (f"{ticker}-{opened_at_iso}")
    entry_ts          : str   (ISO-8601)
    exit_ts           : str   (ISO-8601)
    entry_px          : float
    exit_px           : float
    raw_return_pct    : float | None  (decimal, 0.05 = +5%)
    alpha_vs_spy_pct  : float | None  (decimal)
    side              : str   ("long")
    exit_reason       : str
    reflection_text   : str | None    (None on Gemini failure)
    reflection_model  : str | None
    reflection_ts     : str   (ISO-8601 — always set)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from google.genai import types as genai_types

import config
from reflection_store import ReflectionStore

if TYPE_CHECKING:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from scorer import Scorer

log = logging.getLogger("reflection_generator")


_SYSTEM_INSTRUCTION = """You are an intraday catalyst trader reviewing one of
your own closed trades. Your job is to write ONE short paragraph (≤80 words)
that captures the lesson — what worked, what didn't, and what to do differently
next time on this name. Be specific and grounded in the trade data. No fluff,
no generic trading advice, no hedging. If the data is too thin to draw a
useful lesson, say so plainly in one sentence.

Return PLAIN TEXT only — no JSON, no markdown headers, no preamble."""


class ReflectionGenerator:
    """Builds and persists post-trade reflections."""

    def __init__(
        self,
        store: ReflectionStore,
        scorer: "Scorer",
        hist_client: "StockHistoricalDataClient | None" = None,
    ) -> None:
        self._store = store
        # Routes the Gemini call through the scorer's monthly budget gate,
        # RPM limiter, hourly cap, and call counters — same shape as
        # self_improvement._call_gemini.
        self._scorer = scorer
        self._hist_client = hist_client

    async def on_trade_closed(
        self,
        *,
        ticker: str,
        side: str,
        entry_price: float,
        exit_price: float,
        opened_at: datetime,
        closed_at: datetime,
        raw_return_pct: float,
        exit_reason: str,
        entry_rationale_text: str | None = None,
        entry_top_signals: list[str] | None = None,
    ) -> None:
        """Compute alpha, generate reflection via Gemini, persist record.

        Designed to run as a background task — never raises. Always writes
        a record (with reflection_text=None on Gemini failure) so the audit
        trail captures every closed trade.
        """
        if not config.REFLECTION_ENABLED:
            return
        try:
            await self._do(
                ticker=ticker,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                opened_at=opened_at,
                closed_at=closed_at,
                raw_return_pct=raw_return_pct,
                exit_reason=exit_reason,
                entry_rationale_text=entry_rationale_text,
                entry_top_signals=entry_top_signals or [],
            )
        except Exception:
            log.exception("[REFLECTION] on_trade_closed crashed for %s", ticker)

    async def _do(
        self,
        *,
        ticker: str,
        side: str,
        entry_price: float,
        exit_price: float,
        opened_at: datetime,
        closed_at: datetime,
        raw_return_pct: float,
        exit_reason: str,
        entry_rationale_text: str | None,
        entry_top_signals: list[str],
    ) -> None:
        opened_at = _ensure_utc(opened_at)
        closed_at = _ensure_utc(closed_at)
        hold_minutes = max(0, int((closed_at - opened_at).total_seconds() / 60))

        alpha_vs_spy_pct = await self._compute_alpha(
            opened_at, closed_at, raw_return_pct,
        )

        prompt = _build_prompt(
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            raw_return_pct=raw_return_pct,
            alpha_vs_spy_pct=alpha_vs_spy_pct,
            hold_minutes=hold_minutes,
            exit_reason=exit_reason,
            entry_rationale_text=entry_rationale_text,
            entry_top_signals=entry_top_signals,
        )

        reflection_text, model_used = await self._call_gemini(prompt)

        entry: dict[str, Any] = {
            "ticker": ticker,
            "trade_id": f"{ticker}-{opened_at.isoformat()}",
            "entry_ts": opened_at.isoformat(),
            "exit_ts": closed_at.isoformat(),
            "entry_px": float(entry_price),
            "exit_px": float(exit_price),
            "raw_return_pct": float(raw_return_pct),
            "alpha_vs_spy_pct": alpha_vs_spy_pct,
            "side": side,
            "exit_reason": exit_reason,
            "reflection_text": reflection_text,
            "reflection_model": model_used,
            "reflection_ts": datetime.now(timezone.utc).isoformat(),
        }
        await self._store.append(entry)
        if reflection_text is None:
            log.warning(
                "[REFLECTION] %s persisted with null reflection_text "
                "(Gemini unavailable or budget halted)", ticker,
            )
        else:
            log.info(
                "[REFLECTION] %s saved (return=%+.2f%% alpha=%s reason=%s)",
                ticker, raw_return_pct * 100,
                f"{alpha_vs_spy_pct * 100:+.2f}%" if alpha_vs_spy_pct is not None else "n/a",
                exit_reason,
            )

    async def _compute_alpha(
        self,
        opened_at: datetime,
        closed_at: datetime,
        raw_return_pct: float,
    ) -> float | None:
        """Return raw_return_pct - SPY's return over the same window.

        Uses Alpaca minute bars (IEX feed) — same data source the trader
        already pulls for Kronos minute-bar fetches and the regime detector.
        Fail-open: any error returns None (alpha unavailable but record still
        written).
        """
        if self._hist_client is None:
            return None
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed

            # Pad the window slightly so we always catch a bar at each end.
            start = opened_at - timedelta(minutes=5)
            end = closed_at + timedelta(minutes=5)
            req = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            raw = await asyncio.to_thread(self._hist_client.get_stock_bars, req)
            df = getattr(raw, "df", None)
            if df is None or df.empty:
                return None
            if hasattr(df.index, "names") and "symbol" in (df.index.names or []):
                try:
                    df = df.xs("SPY", level="symbol")
                except KeyError:
                    return None
            if df.empty or "open" not in df.columns or "close" not in df.columns:
                return None

            # First bar at/after open, last bar at/before close. Index is UTC
            # timestamps; pad-window guarantees both sides exist.
            entry_bar = df.iloc[0]
            exit_bar = df.iloc[-1]
            spy_open = float(entry_bar["open"])
            spy_close = float(exit_bar["close"])
            if spy_open <= 0:
                return None
            spy_return = (spy_close - spy_open) / spy_open
            return raw_return_pct - spy_return
        except Exception:
            log.debug(
                "[REFLECTION] SPY alpha computation failed; alpha=null",
                exc_info=True,
            )
            return None

    async def _call_gemini(self, prompt: str) -> tuple[str | None, str | None]:
        """Single ungrounded Gemini call gated by the scorer's budget.

        Returns (reflection_text, model_used). Both are None on any failure.
        """
        scorer = self._scorer
        model = config.REFLECTION_MODEL

        if getattr(scorer, "_budget_mode", "normal") == "halted":
            log.info("[REFLECTION] Gemini halted (monthly budget); skipping call")
            scorer.calls_skipped += 1
            return None, None
        if not scorer._check_hourly_limit():
            scorer.calls_skipped += 1
            return None, None
        if not scorer._record_call_cost(is_grounded=False):
            scorer.calls_skipped += 1
            return None, None
        await scorer._limiter.acquire()
        scorer._consume_hourly_slot()
        scorer.calls_today += 1
        scorer.calls_ungrounded += 1

        try:
            response = await asyncio.wait_for(
                scorer._client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=200,
                    ),
                ),
                timeout=30.0,
            )
        except Exception:
            log.exception("[REFLECTION] Gemini call failed")
            return None, None

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            log.warning("[REFLECTION] Gemini returned empty text")
            return None, None
        # Cap the saved reflection so a runaway response can't bloat future
        # injected prompts (~80 words = ~500 chars; cap at 600).
        return text[:600], model


def _build_prompt(
    *,
    ticker: str,
    side: str,
    entry_price: float,
    exit_price: float,
    raw_return_pct: float,
    alpha_vs_spy_pct: float | None,
    hold_minutes: int,
    exit_reason: str,
    entry_rationale_text: str | None,
    entry_top_signals: list[str],
) -> str:
    alpha_str = (
        f"{alpha_vs_spy_pct * 100:+.2f}%"
        if alpha_vs_spy_pct is not None
        else "n/a"
    )
    rationale_line = (
        f"Entry rationale: {entry_rationale_text}"
        if entry_rationale_text
        else "Entry rationale: (not captured)"
    )
    signals_line = (
        f"Top entry signals: {', '.join(entry_top_signals)}"
        if entry_top_signals
        else "Top entry signals: (none recorded)"
    )
    return (
        f"Trade closed: {ticker} {side}, entry ${entry_price:.2f} → "
        f"exit ${exit_price:.2f}, raw return {raw_return_pct * 100:+.2f}%, "
        f"alpha vs SPY {alpha_str}, held {hold_minutes} min. "
        f"Exit reason: {exit_reason}.\n"
        f"{rationale_line}\n"
        f"{signals_line}\n\n"
        "In ONE paragraph (≤80 words), reflect on what worked or didn't. "
        "Be specific and actionable. No fluff."
    )


def _ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
