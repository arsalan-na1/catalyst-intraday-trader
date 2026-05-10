"""Telegram bot front-end.

Outbound: status messages for buys, closes, P&L updates, daily summary.
Inbound commands (chat_id-gated):
  /status   — current bot state, today's stats, open positions
  /positions — detailed open-position list with entry price and P&L
  /close TICKER — manually close a specific position at market price

Lifecycle: PTB v20 manual pattern (initialize → start → polling; reverse on
shutdown) so it cooperates cleanly with asyncio.gather.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

import config
from scorer import CatalystVerdict, TriggerContext

log = logging.getLogger("telegram")

_MARKET_TZ = ZoneInfo("America/New_York")


def _redact(text: str) -> str:
    """Strip the Telegram bot token from arbitrary text.

    The python-telegram-bot client embeds the token in every Bot API URL,
    so any HTTP-related exception (httpx.HTTPError, request URLs in
    error chains, etc.) can leak it through __str__/__repr__. Call this
    on any string before logging when the source might include URL text.
    """
    if not config.TELEGRAM_BOT_TOKEN:
        return text
    return text.replace(config.TELEGRAM_BOT_TOKEN, "<TELEGRAM_BOT_TOKEN>")


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


class TelegramHandler:
    def __init__(self, bot_token: str, chat_id: int) -> None:
        self._chat_id = chat_id
        self._app: Application = ApplicationBuilder().token(bot_token).build()
        self._app.add_handler(CommandHandler("status", self._on_status))
        self._app.add_handler(CommandHandler("positions", self._on_positions))
        self._app.add_handler(CommandHandler("close", self._on_close))
        self._app.add_handler(CommandHandler("watchlist", self._on_watchlist))
        self._app.add_handler(CommandHandler("trades", self._on_trades))
        self._app.add_handler(CommandHandler("regime", self._on_regime))
        self._app.add_handler(CommandHandler("perf", self._on_perf))
        # injected after construction via set_context()
        self._trader: Any = None
        self._stats: Any = None
        self._bot_start_utc: datetime | None = None
        self._watchlist_info: dict | None = None
        self._regime_detector: Any = None

    def set_context(
        self,
        *,
        trader: Any,
        stats: Any,
        watchlist_info: dict | None = None,
        regime_detector: Any = None,
    ) -> None:
        self._trader = trader
        self._stats = stats
        self._watchlist_info = watchlist_info
        self._regime_detector = regime_detector
        self._bot_start_utc = datetime.now(timezone.utc)

    # --- lifecycle ---

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        log.info("Telegram polling started")

    async def stop(self) -> None:
        log.info("Stopping Telegram handler...")
        try:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            # Redact: PTB exceptions chain httpx errors whose __str__ includes
            # the request URL — and the token sits in that URL.
            log.error("error during Telegram shutdown: %s", _redact(repr(e)))

    # --- generic outbound ---

    async def send_message(self, text: str, *, parse_mode: str = ParseMode.HTML) -> None:
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id, text=text, parse_mode=parse_mode
            )
        except Exception as e:
            # log.exception would dump a traceback containing the Bot API URL
            # (which embeds the token). Redact and skip exc_info here.
            log.error("send_message failed: %s", _redact(repr(e)))

    # --- trade status messages ---

    async def send_buy_status(
        self,
        *,
        ctx: TriggerContext,
        verdict: CatalystVerdict,
        qty: int,
        take_profit_pct: float,
        stop_loss_pct: float = 0.05,
        hold_strategy: str = "momentum",
    ) -> None:
        v = verdict
        tech_line = (
            f"\n📊 {ctx.technicals.to_compact_str()} | Signal: {v.technical_signal}"
            if ctx.technicals is not None else f"\nSignal: {v.technical_signal}"
        )
        gap_tag = " [GAP OPEN]" if getattr(ctx, "trigger_type", "") == "gap_open" else ""
        text = (
            f"📈 <b>AUTO-BUY: {ctx.ticker}</b>{gap_tag}\n"
            f"Price: <b>${ctx.price:.2f}</b> × {qty} shares\n"
            f"Catalyst: {v.catalyst_summary}\n"
            f"Confidence: {v.confidence}/10 | Magnitude: {v.magnitude_estimate}/10"
            f"{tech_line}\n"
            f"🎯 TP: +{take_profit_pct:.0%} | 🛑 SL: -{stop_loss_pct:.0%} | "
            f"⏱ {hold_strategy} ({v.max_hold_minutes}min)"
        )
        await self.send_message(text)

    async def send_microcap_buy_status(
        self,
        *,
        ctx: TriggerContext,
        verdict: CatalystVerdict,
        qty: int,
        market_cap: float | None,
        today_volume: int | None,
        vol_ratio: float | None,
        has_news: bool,
        biotech_fast_track: bool = False,
    ) -> None:
        v = verdict
        catalyst_line = (
            v.catalyst_summary if has_news and v.catalyst_summary
            else "Volume-only signal"
        )
        biotech_tag = "🧬 BIOTECH | " if biotech_fast_track else ""
        mcap_str = f"~${market_cap / 1e6:.0f}M" if market_cap else "unknown"
        vol_str = f"{vol_ratio:.1f}x ADV" if vol_ratio else ""
        tech_line = (
            f"\n📊 {ctx.technicals.to_compact_str()} | Signal: {v.technical_signal}"
            if ctx.technicals is not None else f"\nSignal: {v.technical_signal}"
        )
        text = (
            f"⚠️ <b>AUTO-BUY [MICROCAP]: {ctx.ticker}</b>\n"
            f"{biotech_tag}Price: <b>${ctx.price:.2f}</b> × {qty} shares\n"
            f"MCap: {mcap_str} | Vol: {vol_str}\n"
            f"Catalyst: {catalyst_line}\n"
            f"Confidence: {v.confidence}/10 | Magnitude: {v.magnitude_estimate}/10"
            f"{tech_line}"
        )
        await self.send_message(text)

    async def send_close_status(
        self,
        *,
        ticker: str,
        pnl_pct: float,
        pnl_dollars: float,
        hold_str: str,
        reason: str,
        exit_price: float | None = None,
    ) -> None:
        emoji = "💰" if pnl_pct >= 0 else "🛟"
        price_line = f" @ <b>${exit_price:.2f}</b>" if exit_price is not None else ""
        text = (
            f"{emoji} <b>CLOSED: {ticker}</b>{price_line}\n"
            f"P&amp;L: {pnl_pct:+.2%} (${pnl_dollars:+.2f}) | Held: {hold_str}\n"
            f"Reason: {reason}"
        )
        await self.send_message(text)

    async def send_premarket_watch(
        self,
        *,
        ctx: TriggerContext,
        verdict: CatalystVerdict,
    ) -> None:
        direction = "+" if ctx.price_move_pct >= 0 else ""
        text = (
            f"👀 <b>PRE-MARKET: {ctx.ticker}</b>\n"
            f"${ctx.price:.2f} ({direction}{ctx.price_move_pct:.1%}) | "
            f"Vol: {ctx.volume_ratio:.1f}x\n"
            f"Catalyst: {verdict.catalyst_summary}\n"
            f"<i>Will auto-buy at market open if thresholds still pass.</i>"
        )
        await self.send_message(text)

    async def send_position_update(
        self,
        *,
        ticker: str,
        action: str,
        reason: str,
        current_pnl_pct: float,
        new_tp_pct: float | None = None,
        new_sl_floor: float | None = None,
        add_minutes: int | None = None,
    ) -> None:
        if action == "exit":
            text = (
                f"🔄 <b>GEMINI EXIT: {ticker}</b>\n"
                f"P&L: {current_pnl_pct:+.1%} — {reason}"
            )
            await self.send_message(text)
        elif action == "raise_target" and new_tp_pct is not None:
            text = (
                f"📈 <b>TARGET RAISED: {ticker}</b>\n"
                f"TP raised to +{new_tp_pct:.0%} | P&L: {current_pnl_pct:+.1%} — {reason}"
            )
            await self.send_message(text)
        elif action == "tighten_sl" and new_sl_floor is not None:
            sl_desc = (
                f"trailing floor +{new_sl_floor:.0%}"
                if new_sl_floor > 0
                else f"SL floor {new_sl_floor:.0%}"
            )
            text = (
                f"🔒 <b>SL TIGHTENED: {ticker}</b>\n"
                f"New {sl_desc} | P&L: {current_pnl_pct:+.1%} — {reason}"
            )
            await self.send_message(text)
        elif action == "adjust_tp" and new_tp_pct is not None:
            text = (
                f"⚙️ <b>TP ADJUSTED: {ticker}</b>\n"
                f"New TP: +{new_tp_pct:.0%} | P&L: {current_pnl_pct:+.1%} — {reason}"
            )
            await self.send_message(text)
        elif action == "add_time" and add_minutes is not None:
            text = (
                f"⏳ <b>TIME EXTENDED: {ticker}</b>\n"
                f"+{add_minutes}min granted | P&L: {current_pnl_pct:+.1%} — {reason}"
            )
            await self.send_message(text)
        # "hold" → no message (would be too noisy)

    # --- helpers for inbound ---

    def _authorized(self, update: Update) -> bool:
        return (
            update.effective_chat is not None
            and update.effective_chat.id == self._chat_id
        )

    async def _reply(self, update: Update, text: str) -> None:
        if update.effective_chat:
            try:
                await update.effective_chat.send_message(text, parse_mode=ParseMode.HTML)
            except Exception as e:
                log.error("reply failed: %s", _redact(repr(e)))

    # --- inbound command handlers ---

    async def _on_status(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return

        now_et = datetime.now(tz=_MARKET_TZ)
        lines: list[str] = [
            f"🤖 <b>Bot Status</b> — {now_et.strftime('%I:%M %p ET')}"
        ]

        if self._bot_start_utc:
            elapsed = datetime.now(timezone.utc) - self._bot_start_utc
            total_secs = int(elapsed.total_seconds())
            h, rem = divmod(total_secs, 3600)
            m = rem // 60
            lines.append(f"✅ Running (uptime: {h}h {m}m)")
        else:
            lines.append("✅ Running")

        stats = self._stats
        if stats:
            pnl = getattr(stats, "realized_pnl", 0.0)
            trades = getattr(stats, "trades_taken", 0)
            triggers = getattr(stats, "spikes_detected", 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"📊 Today: {triggers} triggers, {trades} trades, {sign}${pnl:,.2f}"
            )

        trader = self._trader
        if trader:
            positions = getattr(trader, "_positions", {})
            if positions:
                pos_parts: list[str] = []
                for ticker, pos in positions.items():
                    if getattr(pos, "entry_price", None):
                        pos_parts.append(f"{ticker} (entry ${pos.entry_price:.2f})")
                    else:
                        pos_parts.append(ticker)
                lines.append(f"💼 Open: {', '.join(pos_parts)}")
            else:
                lines.append("💼 No open positions")

        await self._reply(update, "\n".join(lines))

    async def _on_positions(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return

        trader = self._trader
        if not trader:
            await self._reply(update, "Trader not available.")
            return

        positions = getattr(trader, "_positions", {})
        if not positions:
            await self._reply(update, "No open positions.")
            return

        now_utc = datetime.now(timezone.utc)
        lines = ["💼 <b>Open Positions</b>"]
        for ticker, pos in positions.items():
            entry_str = f"${pos.entry_price:.2f}" if pos.entry_price else "pending fill"
            opened_at = getattr(pos, "opened_at", None)
            if opened_at:
                elapsed_secs = (now_utc - opened_at.astimezone(timezone.utc) if opened_at.tzinfo else now_utc - opened_at.replace(tzinfo=timezone.utc)).total_seconds()
                hold = _hold_str(elapsed_secs)
            else:
                hold = "?"
            lines.append(f"• <b>{ticker}</b> — entry {entry_str} | held {hold}")

        await self._reply(update, "\n".join(lines))

    async def _on_watchlist(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        info = self._watchlist_info
        if not info:
            await self._reply(update, "Watchlist info not available.")
            return
        lines = [
            "📋 <b>Watchlist Status</b>",
            f"Main: {info.get('base', 0)} tickers",
            f"Catalyst: {info.get('catalyst', 0)} tickers",
            f"ADV prefetched: {info.get('adv_prefetched', 0)}",
            f"WebSocket subscribed: {info.get('ws_subscribed', 0)}",
        ]
        await self._reply(update, "\n".join(lines))

    async def _on_trades(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        path = config.STATE_DIR / "trade_log.jsonl"
        try:
            if not path.exists():
                await self._reply(update, "No trades logged yet.")
                return
            raw = await asyncio.to_thread(lambda: path.read_text(encoding="utf-8"))
            lines = raw.strip().splitlines()
            if not lines:
                await self._reply(update, "No trades logged yet.")
                return
            parts = ["📜 <b>Last 5 Trades</b>"]
            for line in reversed(lines[-5:]):
                try:
                    r = json.loads(line)
                    pnl_d = r.get("pnl_dollar", 0.0)
                    pnl_p = r.get("pnl_pct", 0.0) * 100
                    emoji = "✅" if pnl_d >= 0 else "❌"
                    sig = r.get("technical_signal") or "?"
                    conf = r.get("gemini_confidence") or "?"
                    parts.append(
                        f"{emoji} <b>{r['ticker']}</b> "
                        f"${pnl_d:+.2f} ({pnl_p:+.1f}%) | "
                        f"{r.get('hold_minutes', 0):.0f}m | "
                        f"{r.get('exit_reason', '?')} | "
                        f"conf={conf} tech={sig}"
                    )
                except Exception:
                    log.debug("trade log line parse failed", exc_info=True)
                    continue
            await self._reply(update, "\n".join(parts))
        except Exception as e:
            # _reply makes an HTTP call; the caught exception may have come
            # from there, so the URL (and token) might be in repr(e).
            log.error("/trades command failed: %s", _redact(repr(e)))
            await self._reply(update, "Failed to read trade log.")

    async def _on_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return

        args = ctx.args or []
        if not args:
            await self._reply(update, "Usage: /close TICKER")
            return

        ticker = args[0].upper()
        trader = self._trader
        if not trader:
            await self._reply(update, "Trader not available.")
            return

        positions = getattr(trader, "_positions", {})
        if ticker not in positions:
            await self._reply(update, f"{ticker} not in open positions.")
            return

        await self._reply(update, f"Closing {ticker}…")
        try:
            await trader._close_position(ticker, reason="manual /close")
        except Exception as e:
            log.error("/close failed for %s: %s", ticker, _redact(repr(e)))
            await self._reply(update, f"❌ Close failed for {ticker}. Check logs.")

    async def _on_regime(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        det = self._regime_detector
        if det is None:
            await self._reply(update, "Regime detector not available.")
            return
        try:
            adj = det.get_regime_adjustments()
        except Exception:
            log.exception("/regime failed")
            await self._reply(update, "Failed to read regime state.")
            return
        last_updated = getattr(det, "_last_updated", None)
        if last_updated is not None:
            age_min = (datetime.now(timezone.utc) - last_updated).total_seconds() / 60.0
            age_str = f"{age_min:.0f} min ago"
        else:
            age_str = "never (no successful refresh yet)"
        text = (
            "📊 <b>Market Regime (HMM/SPY)</b>\n"
            f"Label: <b>{adj.get('label', 'unknown')}</b>\n"
            f"Size multiplier: {adj.get('size_multiplier', 1.0):.2f}×\n"
            f"Hold multiplier: {adj.get('hold_multiplier', 1.0):.2f}×\n"
            f"Last refresh: {age_str}"
        )
        await self._reply(update, text)

    async def _on_perf(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        stats = self._stats
        if stats is None:
            await self._reply(update, "Stats not available.")
            return

        records = list(getattr(stats, "trade_records", []) or [])
        total = len(records)
        wins = sum(1 for r in records if r.pnl_dollars > 0)
        losses = total - wins
        win_rate = (wins / total) if total else 0.0
        realized = sum(r.pnl_dollars for r in records)

        lines: list[str] = ["📈 <b>Session Performance</b>"]
        if total == 0:
            lines.append("No trades closed today.")
        else:
            lines.append(
                f"Trades: {total} ({wins}W / {losses}L) — "
                f"Win rate: {win_rate:.0%}"
            )
            sign = "+" if realized >= 0 else ""
            lines.append(f"Realized P&L: {sign}${realized:,.2f}")

            # By catalyst type — pulled from the close `reason` is unreliable;
            # the trader doesn't tag catalyst type on TradeRecord. Use the
            # Gemini verdict's catalyst_type via the trade_log.jsonl file when
            # available; fall back to grouping by close reason prefix otherwise.
            by_reason: dict[str, list] = defaultdict(list)
            for r in records:
                bucket_key = r.reason.split(" ", 1)[0] if r.reason else "unknown"
                by_reason[bucket_key].append(r)
            if by_reason:
                lines.append("\nBy exit reason:")
                for label in sorted(by_reason):
                    bucket = by_reason[label]
                    bw = sum(1 for r in bucket if r.pnl_dollars > 0)
                    avg_pct = sum(r.pnl_pct for r in bucket) / len(bucket)
                    lines.append(
                        f"  {label}: {len(bucket)} trades, "
                        f"{(bw / len(bucket)):.0%} win, "
                        f"avg {avg_pct:+.1%}"
                    )

        # Factor signals
        traded = getattr(stats, "factor_signals_traded", 0)
        correct = getattr(stats, "factor_signals_correct", 0)
        squeeze = getattr(stats, "squeeze_setups_seen", 0)
        ins_seen = getattr(stats, "insider_bullish_seen", 0)
        rev_seen = getattr(stats, "revision_strong_seen", 0)
        if traded or squeeze or ins_seen or rev_seen:
            lines.append("")
            if traded:
                rate = (correct / traded) if traded else 0.0
                lines.append(
                    f"Factor signals: {traded} traded, "
                    f"{correct} correct ({rate:.0%})"
                )
            lines.append(
                f"Squeeze setups seen: {squeeze} | "
                f"Insider bullish seen: {ins_seen} | "
                f"Revision strong seen: {rev_seen}"
            )

        # By regime
        if records:
            by_regime: dict[str, list] = defaultdict(list)
            for r in records:
                by_regime[getattr(r, "regime", "unknown") or "unknown"].append(r)
            if by_regime:
                lines.append("\nBy regime:")
                for label in sorted(by_regime):
                    bucket = by_regime[label]
                    bw = sum(1 for r in bucket if r.pnl_dollars > 0)
                    lines.append(
                        f"  {label}: {len(bucket)} trades, "
                        f"{(bw / len(bucket)):.0%} win"
                    )

        await self._reply(update, "\n".join(lines))
