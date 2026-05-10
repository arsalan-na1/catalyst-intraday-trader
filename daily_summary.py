"""Daily end-of-session summary and performance tracking.

Defines SessionStats (shared mutable state across all modules) and
TradeRecord (per-trade log populated by Trader on every close).

Scheduling: sleep until next 16:05 ET, post rich recap, append to
performance.json, reset counters, loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
import config
from telegram_handler import TelegramHandler

log = logging.getLogger("daily_summary")


@dataclass
class TradeRecord:
    ticker: str
    qty: int
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_pct: float
    hold_secs: float
    reason: str
    opened_at: datetime
    # HMM SPY regime label captured at entry time. Defaults to "unknown" so
    # earlier records that never set the field still deserialise cleanly.
    regime: str = "unknown"


@dataclass
class SessionStats:
    spikes_detected: int = 0
    alerts_sent: int = 0
    trades_taken: int = 0
    trades_skipped: int = 0
    realized_pnl: float = 0.0
    technicals_fetched: int = 0
    technicals_failed: int = 0
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trade_records: list[TradeRecord] = field(default_factory=list)
    # Circuit breaker state: set once at startup and refreshed after daily reset.
    opening_equity: float = 0.0
    halt_new_entries: bool = False
    # Tiered risk controls.
    # size_reduction is applied as a multiplier to position_size_pct in execute_auto_buy.
    # weekly_pnl and week_start_equity persist across daily resets within the same week.
    size_reduction: float = 1.0
    weekly_pnl: float = 0.0
    week_start_equity: float = 0.0
    weekly_halt_active: bool = False
    # Quantitative factor signal tracking (reset daily).
    squeeze_setups_seen: int = 0
    insider_bullish_seen: int = 0
    revision_strong_seen: int = 0
    factor_signals_traded: int = 0
    factor_signals_correct: int = 0

    def reset(self) -> None:
        self.spikes_detected = 0
        self.alerts_sent = 0
        self.trades_taken = 0
        self.trades_skipped = 0
        self.realized_pnl = 0.0
        self.technicals_fetched = 0
        self.technicals_failed = 0
        self.session_start = datetime.now(timezone.utc)
        self.trade_records = []
        self.opening_equity = 0.0  # re-fetched by run_daily_summary after reset
        self.squeeze_setups_seen = 0
        self.insider_bullish_seen = 0
        self.revision_strong_seen = 0
        self.factor_signals_traded = 0
        self.factor_signals_correct = 0
        # Re-apply weekly restrictions if active; otherwise clear daily state.
        if self.weekly_halt_active:
            self.halt_new_entries = True
            self.size_reduction = 0.5
        else:
            self.halt_new_entries = False
            self.size_reduction = 1.0

    def format(self) -> str:
        return (
            "📋 <b>Daily Summary</b>\n"
            f"Spikes detected: {self.spikes_detected}\n"
            f"Alerts sent: {self.alerts_sent}\n"
            f"Trades taken: {self.trades_taken}\n"
            f"Trades skipped: {self.trades_skipped}\n"
            f"Realized P&amp;L: <b>${self.realized_pnl:+.2f}</b>"
        )


def _hold_str(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h = secs // 3600
    m = (secs % 3600) // 60
    return f"{h}h {m}m"


def _build_rich_summary(
    stats: SessionStats,
    gemini_calls: int,
    balance: float | None,
    today: datetime,
    gemini_calls_grounded: int = 0,
    gemini_calls_ungrounded: int = 0,
    gemini_calls_skipped: int = 0,
    monthly_cost_estimate: float | None = None,
    monthly_cap: float | None = None,
) -> str:
    date_str = today.astimezone(config.MARKET_TZ).strftime("%B %-d, %Y")
    lines: list[str] = [f"📊 <b>DAILY RECAP — {date_str}</b>\n"]

    if stats.trade_records:
        lines.append(f"Trades today: {len(stats.trade_records)}")
        for rec in stats.trade_records:
            emoji = "✅" if rec.pnl_dollars >= 0 else "❌"
            sign = "+" if rec.pnl_dollars >= 0 else ""
            reason_str = "stopped out" if rec.reason.startswith("SL") else "held"
            lines.append(
                f"{emoji} {rec.ticker}: {sign}${rec.pnl_dollars:,.0f} "
                f"({sign}{rec.pnl_pct:.1%}) — {reason_str} {_hold_str(rec.hold_secs)}"
            )
    else:
        lines.append("Trades today: 0")

    lines.append("")
    pnl_sign = "+" if stats.realized_pnl >= 0 else ""
    if balance and balance > 0:
        basis = balance - stats.realized_pnl
        pnl_on_basis = stats.realized_pnl / basis if basis > 0 else 0.0
        lines.append(
            f"Net P&amp;L: <b>{pnl_sign}${stats.realized_pnl:,.2f}</b> "
            f"({pnl_sign}{pnl_on_basis:.2%} on ${basis:,.0f})"
        )
    else:
        lines.append(f"Net P&amp;L: <b>{pnl_sign}${stats.realized_pnl:,.2f}</b>")

    lines.append(f"Triggers seen: {stats.spikes_detected}")
    lines.append(
        f"Gemini calls: {gemini_calls} total "
        f"({gemini_calls_grounded} grounded / "
        f"{gemini_calls_ungrounded} ungrounded / "
        f"{gemini_calls_skipped} skipped)"
    )
    if monthly_cost_estimate is not None and monthly_cap is not None:
        lines.append(
            f"Gemini est. monthly cost: ${monthly_cost_estimate:.2f} / ${monthly_cap:.0f}"
        )
    total_tech = stats.technicals_fetched + stats.technicals_failed
    if total_tech > 0:
        lines.append(
            f"📊 Technicals: {stats.technicals_fetched}/{total_tech} fetched"
            f" ({stats.technicals_failed} failed)"
        )

    if stats.spikes_detected > 0:
        pct = stats.trades_taken / stats.spikes_detected * 100
        lines.append(f"Trades taken: {stats.trades_taken} ({pct:.1f}% of triggers)")

    wins = sum(1 for r in stats.trade_records if r.pnl_dollars > 0)
    total = len(stats.trade_records)
    if total > 0:
        lines.append(f"Win rate: {wins}/{total} ({wins/total:.1%})")

    if balance:
        label = "Paper" if config.ALPACA_PAPER else "Live"
        lines.append(f"\n{label} account balance: <b>${balance:,.2f}</b>")

    # Factor signal summary (only show if any signals were seen today).
    if (stats.squeeze_setups_seen or stats.insider_bullish_seen
            or stats.revision_strong_seen or stats.factor_signals_traded):
        lines.append("")
        lines.append("📊 <b>Factor Signal Summary:</b>")
        lines.append(f"- Squeeze setups seen today: {stats.squeeze_setups_seen} tickers")
        lines.append(f"- Insider bullish signals: {stats.insider_bullish_seen} tickers")
        lines.append(f"- Strong estimate revision scores: {stats.revision_strong_seen} tickers")
        lines.append(f"- Signals that led to trades: {stats.factor_signals_traded}")
        if stats.factor_signals_traded > 0:
            lines.append(
                f"- Signals that were correct (TP hit): {stats.factor_signals_correct}"
            )

    # Regime breakdown — group today's trades by HMM regime label at entry.
    if stats.trade_records:
        by_regime: dict[str, list[TradeRecord]] = {}
        for rec in stats.trade_records:
            label = getattr(rec, "regime", "unknown") or "unknown"
            by_regime.setdefault(label, []).append(rec)
        if by_regime:
            lines.append("")
            lines.append("🌐 <b>By Regime:</b>")
            for label in sorted(by_regime):
                bucket = by_regime[label]
                wins_b = sum(1 for r in bucket if r.pnl_dollars > 0)
                avg_pnl_pct = sum(r.pnl_pct for r in bucket) / len(bucket)
                lines.append(
                    f"- {label}: {len(bucket)} trades, "
                    f"{wins_b}/{len(bucket)} win "
                    f"({wins_b / len(bucket):.0%}), "
                    f"avg P&amp;L {avg_pnl_pct:+.1%}"
                )

    return "\n".join(lines)


def _append_performance(
    stats: SessionStats,
    gemini_calls: int,
    balance: float | None,
    today: datetime,
    monthly_cost_estimate: float | None = None,
    monthly_key: str | None = None,
) -> None:
    path = config.PERFORMANCE_FILE
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("performance.json unreadable; starting fresh")

    date_key = today.astimezone(config.MARKET_TZ).strftime("%Y-%m-%d")
    wins = sum(1 for r in stats.trade_records if r.pnl_dollars > 0)
    data[date_key] = {
        "pnl": round(stats.realized_pnl, 2),
        "trades": stats.trades_taken,
        "wins": wins,
        "triggers": stats.spikes_detected,
        "gemini_calls": gemini_calls,
        "balance": round(balance, 2) if balance else None,
    }
    # Top-level monthly cost tracking (not nested under a date entry).
    if monthly_cost_estimate is not None:
        data["monthly_cost_estimate"] = round(monthly_cost_estimate, 4)
    if monthly_key is not None:
        data["monthly_cost_month"] = monthly_key
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic rename — avoids partial-read on crash
        log.info("performance.json updated for %s", date_key)
    except Exception:
        log.exception("failed to write performance.json")


def _seconds_until_next_summary(now: datetime | None = None) -> float:
    now = now or datetime.now(tz=config.MARKET_TZ)
    target_time = time(config.DAILY_SUMMARY_HOUR, config.DAILY_SUMMARY_MINUTE)
    target = datetime.combine(now.date(), target_time, tzinfo=config.MARKET_TZ)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def run_daily_summary(
    stats: SessionStats,
    telegram: TelegramHandler,
    trading_client,
    scorer,
    stop_event: asyncio.Event,
    trader=None,
) -> None:
    while not stop_event.is_set():
        wait = _seconds_until_next_summary()
        log.info("next daily summary in %.0f seconds", wait)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait)
            return
        except asyncio.TimeoutError:
            pass

        now_utc = datetime.now(timezone.utc)
        balance: float | None = None
        try:
            account = await asyncio.to_thread(trading_client.get_account)
            balance = float(account.equity)
        except Exception:
            log.warning("failed to fetch account balance for daily summary")

        gemini_calls = getattr(scorer, "calls_today", 0)
        gemini_calls_grounded = getattr(scorer, "calls_grounded", 0)
        gemini_calls_ungrounded = getattr(scorer, "calls_ungrounded", 0)
        gemini_calls_skipped = getattr(scorer, "calls_skipped", 0)
        log.info(
            "Gemini usage: %d total (%d grounded, %d ungrounded, %d skipped)",
            gemini_calls, gemini_calls_grounded, gemini_calls_ungrounded, gemini_calls_skipped,
        )

        # Calendar-month rollover: reset the cost meter on the 1st each month.
        now_et_for_month = datetime.now(tz=config.MARKET_TZ)
        current_month = now_et_for_month.strftime("%Y-%m")
        if hasattr(scorer, "monthly_cost_estimate") and now_et_for_month.day == 1:
            log.info("Month rollover (%s): resetting Gemini cost meter", current_month)
            scorer.monthly_cost_estimate = 0.0
            scorer._budget_mode = "normal"

        monthly_cost = getattr(scorer, "monthly_cost_estimate", None)
        monthly_cap = getattr(config, "MONTHLY_GEMINI_BUDGET_USD", None)

        # Log regime breakdown for ops visibility before sending.
        if stats.trade_records:
            by_regime: dict[str, list[TradeRecord]] = {}
            for rec in stats.trade_records:
                by_regime.setdefault(
                    getattr(rec, "regime", "unknown") or "unknown", []
                ).append(rec)
            for label, bucket in sorted(by_regime.items()):
                wins_b = sum(1 for r in bucket if r.pnl_dollars > 0)
                avg_pnl_pct = sum(r.pnl_pct for r in bucket) / len(bucket)
                log.info(
                    "[REGIME RECAP] %s: %d trades, %d/%d win (%.0f%%), avg P&L %.1f%%",
                    label, len(bucket), wins_b, len(bucket),
                    wins_b / len(bucket) * 100, avg_pnl_pct * 100,
                )

        try:
            msg = _build_rich_summary(
                stats, gemini_calls, balance, now_utc,
                gemini_calls_grounded, gemini_calls_ungrounded, gemini_calls_skipped,
                monthly_cost_estimate=monthly_cost,
                monthly_cap=monthly_cap,
            )
            await telegram.send_message(msg)
            log.info("daily summary sent")
        except Exception:
            log.exception("failed to send daily summary")
            try:
                await telegram.send_message(stats.format())
            except Exception:
                log.exception("fallback daily summary also failed")

        try:
            await asyncio.to_thread(
                _append_performance, stats, gemini_calls, balance, now_utc,
                monthly_cost, current_month,
            )
        except Exception:
            log.exception("failed to append performance record")

        # Monday (weekday 0) starts a new week — reset weekly accumulators.
        now_et = datetime.now(tz=config.MARKET_TZ)
        if now_et.weekday() == 0:
            stats.weekly_pnl = 0.0
            stats.weekly_halt_active = False
            stats.week_start_equity = balance or 0.0
            log.info("Monday reset: weekly_pnl cleared, week_start_equity=$%.2f", stats.week_start_equity)

        stats.reset()
        if trader is not None:
            trader.session_blocked_tickers.clear()
            trader._sector_last_entry.clear()
            log.info(
                "daily reset: session_blocked_tickers and _sector_last_entry cleared"
            )
        if balance:
            stats.opening_equity = balance
            log.info("opening equity refreshed to $%.2f for new session", balance)
