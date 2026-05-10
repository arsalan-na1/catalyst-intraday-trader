"""Autonomous order execution, position monitoring, auto-exit.

Responsibilities:
  - execute_auto_buy: size and place a market buy when thresholds pass.
  - monitor_positions: poll every 15s; auto-close at per-position take_profit_pct
    (dynamic, magnitude-derived) or fixed STOP_LOSS_PCT (using unrealized_plpc from Alpaca).
  - run_eod_close: close all open positions at EOD_CLOSE_HOUR:EOD_CLOSE_MINUTE ET.
  - run_trading_stream: subscribe to fill events so entry price is recorded
    authoritatively rather than estimated.
  - Block new buys after NO_NEW_POSITIONS_AFTER_HOUR:MINUTE ET.
  - Post milestone P&L notifications (5% increments, went negative, near SL/TP).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream

import config
from cooldown_store import CooldownStore
from daily_summary import SessionStats
from scorer import CatalystVerdict, TriggerContext
from telegram_handler import TelegramHandler

log = logging.getLogger("trader")

_ACQUISITION_BLOCK_KEYWORDS = frozenset(
    {"acquisition", "merger", "buyout", "offer price", "capped", "takeover"}
)


def _format_hold_time(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    h = secs // 3600
    m = (secs % 3600) // 60
    return f"{h}h {m}m"


def _derive_top_signals(verdict: CatalystVerdict, quant: dict | None) -> list[str]:
    """Pick up to 3 short tags describing the strongest signals at entry.

    Used by the post-trade reflection generator to give Gemini a compact view
    of what the bot found compelling at entry, alongside the rationale text.
    """
    qs = quant or {}
    short_int = qs.get("short_interest") or {}
    ins_score = qs.get("insider_score") or {}
    est_rev = qs.get("estimate_revisions") or {}
    sector_mom = qs.get("sector_momentum") or {}

    tags: list[str] = []
    if verdict.catalyst_quality in ("strong", "moderate"):
        tags.append(f"catalyst:{verdict.catalyst_type}-{verdict.catalyst_quality}")
    if short_int.get("squeeze_score") in ("high", "medium"):
        tags.append(f"squeeze:{short_int['squeeze_score']}")
    if ins_score.get("insider_signal") == "bullish":
        tags.append("insider:bullish")
    rev_score = est_rev.get("revision_score") or 0
    if isinstance(rev_score, (int, float)) and rev_score >= 1:
        tags.append(f"revisions:+{int(rev_score)}")
    if sector_mom.get("sector_momentum") == "bullish":
        tags.append("sector:bullish")
    if verdict.technical_signal == "bullish":
        tags.append("tech:bullish")
    return tags[:3]


def _calc_take_profit(magnitude: int) -> float | None:
    """Return the take-profit percentage for a given Gemini magnitude score.

    Returns None when magnitude is below the minimum threshold — callers must
    skip the trade in that case (the scorer should already filter these out,
    but this acts as a final guard).
    """
    if magnitude >= config.TP_MAGNITUDE_HIGH:
        return config.TP_HIGH_PCT
    if magnitude >= config.TP_MAGNITUDE_MID:
        return config.TP_MID_PCT
    if magnitude >= config.TP_MAGNITUDE_LOW:
        return config.TP_LOW_PCT
    return None


@dataclass
class ActivePosition:
    ticker: str
    qty: int
    entry_price: float | None  # None until fill confirmed
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_order_id: str | None = None
    gemini_confidence: int | None = None
    gemini_magnitude: int | None = None
    technical_signal: str | None = None
    sector: str = "unknown"  # for sector concentration check
    take_profit_pct: float = 0.15        # Gemini-set TP; clamped to GEMINI_TP_MIN/MAX
    stop_loss_pct: float = 0.05          # Gemini-set SL at entry; positive decimal
    # sl_floor: actual P&L floor for the monitor check. Starts at -stop_loss_pct.
    # Raised by position_monitor on "tighten_sl" verdicts to lock in gains.
    sl_floor: float = -0.05
    max_hold_minutes: int = 90           # Gemini-set hold time; clamped to GEMINI_HOLD_MIN/MAX
    hold_strategy: str = "momentum"      # "momentum" | "catalyst" | "swing"
    # Close-in-flight state — set when a close order is submitted, cleared by fill
    close_submitted: bool = False
    close_submitted_at: datetime | None = None
    close_reason: str | None = None
    last_known_pnl_pct: float = 0.0      # captured from Alpaca just before close
    last_known_pnl_dollars: float = 0.0
    notified_milestones: set[str] = field(default_factory=set)
    had_factor_signal: bool = False       # True if any non-neutral quant signal at entry
    regime: str = "unknown"               # HMM SPY regime label at entry time
    # Captured at entry for the post-trade reflection prompt.
    entry_rationale_text: str | None = None      # ≤200 chars from verdict.reasoning
    entry_top_signals: list[str] = field(default_factory=list)  # up to 3 tags


class Trader:
    def __init__(
        self,
        trading_client: TradingClient,
        telegram: TelegramHandler,
        stats: SessionStats,
        cooldown_store: CooldownStore | None = None,
        hist_client: StockHistoricalDataClient | None = None,
        regime_detector=None,
    ) -> None:
        self._trading = trading_client
        self._telegram = telegram
        self._stats = stats
        self._cooldown_store = cooldown_store
        self._hist_client = hist_client
        # Optional MarketRegimeDetector — when present, applies regime
        # adjustments to position size and hold time. Failures fall back
        # to defaults silently (regime detector itself never raises).
        self._regime_detector = regime_detector
        # Keyed by ticker because Alpaca positions are keyed by symbol.
        self._positions: dict[str, ActivePosition] = {}
        # Tickers where a close order is in-flight on Alpaca.  _monitor_tick skips
        # these so we don't spam close orders while one is already pending.
        self._closing_in_progress: set[str] = set()
        # Per-ticker 60-second fallback tasks — cancelled when a fill event arrives.
        self._close_fallback_tasks: dict[str, asyncio.Task] = {}
        # Correlation cache: (ticker_a, ticker_b) → (corr_value, computed_at)
        # 24-hour TTL — correlations don't change meaningfully intraday.
        self._corr_cache: dict[tuple[str, str], tuple[float, datetime]] = {}
        # Guard against concurrent pipeline tasks submitting duplicate buys for the same ticker.
        self._buy_in_flight: set[str] = set()
        # Session-level block for acquisition/merger tickers — cleared at EOD / daily reset.
        self.session_blocked_tickers: set[str] = set()
        # Per-sector last entry time for duplicate catalyst guard (5-min window).
        self._sector_last_entry: dict[str, datetime] = {}
        # References injected by run_position_monitor at startup so _monitor_tick
        # can fire a timeout-driven Gemini re-eval instead of an unconditional
        # close. None until run_position_monitor runs; _monitor_tick falls back
        # to a hard close on timeout if any of these is still None.
        self._monitor_scorer: Any = None
        self._monitor_telegram: Any = None
        self._monitor_http_session: Any = None
        self._monitor_news_fingerprints: dict[str, frozenset] | None = None
        # Optional ReflectionGenerator — set by bot.py at startup. Fired as a
        # background task at the END of _handle_sell_fill / _close_fallback so
        # the close path itself never awaits Gemini.
        self._reflection_generator: Any = None
        # Strong references for fire-and-forget background tasks (telegram
        # sends, circuit-breaker checks, snapshot saves). The asyncio loop
        # only holds weak refs, so without this set a task could be GC'd
        # before it runs. Tasks remove themselves on completion.
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn_background(self, coro) -> asyncio.Task:
        """Schedule a fire-and-forget coroutine with a strong ref kept until done."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _dispatch_reflection(
        self,
        pos: ActivePosition,
        *,
        exit_price: float,
        raw_return_pct: float,
        exit_time: datetime,
        reason: str,
    ) -> None:
        """Fire-and-forget reflection generation. Never awaits, never raises.

        Skipped when no generator is wired (e.g. bot.py disabled the feature)
        or when entry_price is unknown (ghost-position branch — no useful
        reflection input). Pre-flight skip avoids creating a no-op task.
        """
        gen = self._reflection_generator
        if gen is None or pos.entry_price is None:
            return
        self._spawn_background(
            gen.on_trade_closed(
                ticker=pos.ticker,
                side="long",
                entry_price=pos.entry_price,
                exit_price=exit_price,
                opened_at=pos.opened_at,
                closed_at=exit_time,
                raw_return_pct=raw_return_pct,
                exit_reason=reason,
                entry_rationale_text=pos.entry_rationale_text,
                entry_top_signals=list(pos.entry_top_signals),
            )
        )

    # --- startup position re-sync ---

    async def sync_open_positions(self) -> None:
        """Recover in-memory positions after a restart.

        Calls get_all_positions from Alpaca, then overlays metadata (entry price,
        Gemini scores, opened_at) from state/open_positions.json if it exists.
        Falls back to Alpaca avg_entry_price and None metadata when the snapshot
        has no entry for a position.

        Note: trade_log.jsonl is not used here because it only records *closed*
        trades; open positions have no entry there until they close.
        """
        try:
            alpaca_positions = await asyncio.to_thread(self._trading.get_all_positions)
        except Exception:
            log.exception("sync_open_positions: get_all_positions failed; skipping re-sync")
            return

        if not alpaca_positions:
            log.info("sync_open_positions: no open positions on Alpaca")
            return

        snapshot: dict = {}
        path = config.STATE_DIR / "open_positions.json"
        if path.exists():
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("sync_open_positions: snapshot unreadable; using Alpaca data only")

        recovered = 0
        for ap in alpaca_positions:
            ticker = ap.symbol
            try:
                qty = int(float(ap.qty))
                avg_entry = float(ap.avg_entry_price)
            except (TypeError, ValueError):
                log.warning(
                    "sync_open_positions: %s has non-numeric qty/avg_entry_price; skipping",
                    ticker,
                )
                continue
            if qty <= 0 or avg_entry <= 0:
                log.warning(
                    "sync_open_positions: %s has zero/negative qty=%s entry=%s; skipping",
                    ticker, qty, avg_entry,
                )
                continue

            saved = snapshot.get(ticker, {})
            entry_price = saved.get("entry_price") or avg_entry

            opened_at_raw = saved.get("opened_at")
            try:
                opened_at = datetime.fromisoformat(opened_at_raw) if opened_at_raw else datetime.now(timezone.utc)
            except Exception:
                opened_at = datetime.now(timezone.utc)

            # If the position was opened before today's 9:30 ET market open it is
            # an overnight carry.  Reset opened_at to now so the hold timer starts
            # fresh rather than immediately expiring on restart.
            today_open_et = datetime.now(tz=config.MARKET_TZ).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            today_open_utc = today_open_et.astimezone(timezone.utc)
            opened_at_utc = (
                opened_at.astimezone(timezone.utc)
                if opened_at.tzinfo
                else opened_at.replace(tzinfo=timezone.utc)
            )
            if opened_at_utc < today_open_utc:
                log.info(
                    "sync_open_positions: %s opened_at %s is prior session — "
                    "resetting hold timer to now",
                    ticker, opened_at.isoformat(),
                )
                opened_at = datetime.now(timezone.utc)

            # Recover take_profit_pct: snapshot → magnitude-derived fallback → default.
            saved_tp = saved.get("take_profit_pct")
            if saved_tp is not None:
                take_profit_pct = float(saved_tp)
            else:
                saved_mag = saved.get("gemini_magnitude")
                if saved_mag is not None:
                    computed_tp = _calc_take_profit(int(saved_mag))
                    take_profit_pct = computed_tp if computed_tp is not None else 0.15
                else:
                    take_profit_pct = 0.15

            saved_sl = saved.get("stop_loss_pct")
            stop_loss_pct = float(saved_sl) if saved_sl is not None else config.STOP_LOSS_PCT
            saved_sl_floor = saved.get("sl_floor")
            sl_floor = float(saved_sl_floor) if saved_sl_floor is not None else -stop_loss_pct
            max_hold_minutes = int(saved.get("max_hold_minutes") or config.MAX_HOLD_MINUTES)
            hold_strategy = saved.get("hold_strategy", "momentum")

            saved_rationale = saved.get("entry_rationale_text")
            entry_rationale_text = (
                str(saved_rationale)[:200] if isinstance(saved_rationale, str) else None
            )
            saved_top = saved.get("entry_top_signals") or []
            entry_top_signals = (
                [str(t) for t in saved_top][:3] if isinstance(saved_top, list) else []
            )

            pos = ActivePosition(
                ticker=ticker,
                qty=qty,
                entry_price=entry_price,
                opened_at=opened_at,
                gemini_confidence=saved.get("gemini_confidence"),
                gemini_magnitude=saved.get("gemini_magnitude"),
                technical_signal=saved.get("technical_signal"),
                sector=saved.get("sector", "unknown"),
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                sl_floor=sl_floor,
                max_hold_minutes=max_hold_minutes,
                hold_strategy=hold_strategy,
                regime=saved.get("regime", "unknown"),
                had_factor_signal=bool(saved.get("had_factor_signal", False)),
                entry_rationale_text=entry_rationale_text,
                entry_top_signals=entry_top_signals,
            )
            self._positions[ticker] = pos
            source = "snapshot" if saved else "Alpaca fallback (no snapshot entry)"
            log.info(
                "sync_open_positions: recovered %s qty=%d entry=$%.2f "
                "tp=%.0f%% sl=%.0f%% hold=%dmin strategy=%s [%s] "
                "conf=%s mag=%s signal=%s",
                ticker, qty, entry_price,
                take_profit_pct * 100, stop_loss_pct * 100, max_hold_minutes, hold_strategy,
                source,
                pos.gemini_confidence, pos.gemini_magnitude, pos.technical_signal,
            )
            recovered += 1

        log.info("sync_open_positions: %d position(s) recovered", recovered)
        await self._save_positions_snapshot()

    async def _save_positions_snapshot(self) -> None:
        snapshot = {
            ticker: {
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "opened_at": pos.opened_at.isoformat(),
                "gemini_confidence": pos.gemini_confidence,
                "gemini_magnitude": pos.gemini_magnitude,
                "technical_signal": pos.technical_signal,
                "sector": pos.sector,
                "take_profit_pct": pos.take_profit_pct,
                "stop_loss_pct": pos.stop_loss_pct,
                "sl_floor": pos.sl_floor,
                "max_hold_minutes": pos.max_hold_minutes,
                "hold_strategy": pos.hold_strategy,
                # Persisted so a restart preserves the regime tag (used in EOD
                # recap by-regime breakdown) and the factor-signal flag
                # (used in _handle_sell_fill to count factor_signals_correct).
                "regime": pos.regime,
                "had_factor_signal": pos.had_factor_signal,
                # Persisted so the post-trade reflection prompt has access to
                # the original entry rationale even after a bot restart.
                "entry_rationale_text": pos.entry_rationale_text,
                "entry_top_signals": pos.entry_top_signals,
            }
            for ticker, pos in self._positions.items()
        }
        data = json.dumps(snapshot, indent=2)
        path = config.STATE_DIR / "open_positions.json"

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)  # atomic rename — avoids partial-read on crash

        try:
            await asyncio.to_thread(_write)
        except Exception:
            log.warning("open_positions snapshot write failed", exc_info=True)

    async def _fetch_snapshot_price(
        self, ticker: str, session: aiohttp.ClientSession
    ) -> float | None:
        """Return the latest trade price for ticker via Alpaca IEX snapshot."""
        url = "https://data.alpaca.markets/v2/stocks/snapshots"
        try:
            async with session.get(
                url,
                params={"symbols": ticker, "feed": "iex"},
                headers={
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                snap = data.get(ticker) or {}
                raw = (snap.get("latestTrade") or {}).get("p") or (snap.get("dailyBar") or {}).get("c")
                return float(raw) if raw else None
        except Exception:
            log.debug("_fetch_snapshot_price(%s) failed", ticker, exc_info=True)
            return None

    async def _fetch_bid_ask(
        self, ticker: str, session: aiohttp.ClientSession
    ) -> tuple[float, float] | None:
        """Return (bid, ask) for ticker via Alpaca IEX snapshot. Returns None on failure."""
        url = "https://data.alpaca.markets/v2/stocks/snapshots"
        try:
            async with session.get(
                url,
                params={"symbols": ticker, "feed": "iex"},
                headers={
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                snap = data.get(ticker) or {}
                quote = snap.get("latestQuote") or {}
                bid = quote.get("bp")
                ask = quote.get("ap")
                if bid and ask and float(ask) > 0:
                    return float(bid), float(ask)
                return None
        except Exception:
            log.debug("_fetch_bid_ask(%s) failed", ticker, exc_info=True)
            return None

    async def _fetch_daily_returns(self, ticker: str) -> list[float] | None:
        """Fetch 60 calendar days of daily close prices and return daily return series."""
        if self._hist_client is None:
            return None
        try:
            now_et = datetime.now(tz=config.MARKET_TZ)
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=now_et - timedelta(days=90),  # extra buffer for weekends/holidays
                feed=DataFeed.IEX,
            )
            raw = await asyncio.to_thread(self._hist_client.get_stock_bars, req)
            df = getattr(raw, "df", None)
            if df is None or df.empty:
                return None
            if hasattr(df.index, "names") and "symbol" in (df.index.names or []):
                try:
                    df = df.xs(ticker, level="symbol")
                except KeyError:
                    return None
            closes = [float(v) for v in df["close"]]
            if len(closes) < 10:
                return None
            closes = closes[-61:]  # last 61 bars → 60 returns
            return [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]
        except Exception:
            log.debug("_fetch_daily_returns(%s) failed", ticker, exc_info=True)
            return None

    def _pearson_corr(self, xs: list[float], ys: list[float]) -> float | None:
        """Pearson correlation for two equal-length series. Returns None if degenerate."""
        n = min(len(xs), len(ys))
        if n < 10:
            return None
        xs, ys = xs[-n:], ys[-n:]
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dxs = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
        dys = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
        if dxs == 0 or dys == 0:
            return None
        return num / (dxs * dys)

    async def _close_all_positions(self, reason: str) -> None:
        """Close every tracked open position with the given reason."""
        tickers = [t for t in self._positions if t not in self._closing_in_progress]
        if not tickers:
            return
        log.warning("closing all %d position(s): reason=%s", len(tickers), reason)
        for ticker in tickers:
            try:
                await self._close_position(ticker, reason=reason)
            except Exception:
                log.exception("_close_all_positions: close failed for %s", ticker)

    async def _check_circuit_breakers(self, pnl_dollars: float) -> None:
        """Evaluate tiered circuit breakers after every P&L update.

        Called from _handle_sell_fill and _close_fallback after updating
        realized_pnl / weekly_pnl.  Updates stats flags and closes all
        positions when a hard halt tier is triggered.
        """
        equity = self._stats.opening_equity
        week_eq = self._stats.week_start_equity or equity

        daily_dd = -self._stats.realized_pnl / equity if equity > 0 else 0.0
        weekly_dd = -self._stats.weekly_pnl / week_eq if week_eq > 0 else 0.0

        # --- weekly tiers (check first — more severe) ---
        if not self._stats.weekly_halt_active and weekly_dd >= config.WEEKLY_HALT_PCT:
            self._stats.weekly_halt_active = True
            self._stats.halt_new_entries = True
            self._stats.size_reduction = 0.5
            msg = (
                f"🚨 Weekly DD >{config.WEEKLY_HALT_PCT:.0%} — "
                "all positions closed, halted for week"
            )
            log.warning(msg)
            self._spawn_background(self._telegram.send_message(msg))
            self._spawn_background(self._close_all_positions("weekly_halt"))
            return

        if (
            not self._stats.weekly_halt_active
            and weekly_dd >= config.WEEKLY_WARN_PCT
            and self._stats.size_reduction == 1.0
        ):
            self._stats.size_reduction = 0.5
            msg = (
                f"⚠️ Weekly DD >{config.WEEKLY_WARN_PCT:.0%} — "
                "position sizes halved for rest of week"
            )
            log.warning(msg)
            self._spawn_background(self._telegram.send_message(msg))

        # --- daily tiers ---
        if not self._stats.halt_new_entries and daily_dd >= config.DAILY_HALT_PCT:
            self._stats.halt_new_entries = True
            self._stats.size_reduction = 0.5
            msg = (
                f"🚨 Daily DD >{config.DAILY_HALT_PCT:.0%} — "
                "all positions closed, trading halted"
            )
            log.warning(msg)
            self._spawn_background(self._telegram.send_message(msg))
            self._spawn_background(self._close_all_positions("daily_halt"))
            return

        if (
            not self._stats.halt_new_entries
            and daily_dd >= config.DAILY_LOSS_LIMIT_PCT
            and self._stats.size_reduction == 1.0
            and not self._stats.weekly_halt_active  # weekly tier already set it
        ):
            self._stats.size_reduction = 0.5
            log.warning(
                "⚠️ Daily DD >%.0f%% — position sizes halved for rest of session",
                config.DAILY_LOSS_LIMIT_PCT * 100,
            )

    # --- autonomous execution ---

    async def execute_auto_buy(
        self,
        ctx: TriggerContext,
        verdict: CatalystVerdict,
        *,
        http_session: aiohttp.ClientSession | None = None,
        is_microcap: bool = False,
        market_cap: float | None = None,
        today_volume: int | None = None,
        vol_ratio: float | None = None,
        has_news: bool = True,
        biotech_fast_track: bool = False,
        is_biotech: bool = False,
        sector: str = "unknown",
        quant_signals: dict | None = None,
        regime: str = "unknown",
    ) -> bool:
        """Return True if an order was submitted; False for any capacity/quality rejection.

        Callers should only set the persistent cooldown when True is returned —
        capacity rejections (MAX_POSITIONS, sector limit) should allow retry once
        positions free up.
        """
        ticker = ctx.ticker
        if ticker in self._buy_in_flight:
            log.info("concurrent buy already in flight for %s; skipping", ticker)
            return False
        self._buy_in_flight.add(ticker)
        try:
            return await self._do_execute_buy(
                ctx, verdict,
                http_session=http_session,
                is_microcap=is_microcap,
                market_cap=market_cap,
                today_volume=today_volume,
                vol_ratio=vol_ratio,
                has_news=has_news,
                biotech_fast_track=biotech_fast_track,
                is_biotech=is_biotech,
                sector=sector,
                quant_signals=quant_signals,
                regime=regime,
            )
        finally:
            self._buy_in_flight.discard(ticker)

    async def _do_execute_buy(
        self,
        ctx: TriggerContext,
        verdict: CatalystVerdict,
        *,
        http_session: aiohttp.ClientSession | None = None,
        is_microcap: bool = False,
        market_cap: float | None = None,
        today_volume: int | None = None,
        vol_ratio: float | None = None,
        has_news: bool = True,
        biotech_fast_track: bool = False,
        is_biotech: bool = False,
        sector: str = "unknown",
        quant_signals: dict | None = None,
        regime: str = "unknown",
    ) -> bool:
        ticker = ctx.ticker

        # Hard cutoff: no new positions after NO_NEW_POSITIONS_AFTER time ET.
        now_et = datetime.now(tz=config.MARKET_TZ)
        cutoff = now_et.replace(
            hour=config.NO_NEW_POSITIONS_AFTER_HOUR,
            minute=config.NO_NEW_POSITIONS_AFTER_MINUTE,
            second=0,
            microsecond=0,
        )
        if now_et >= cutoff:
            log.info(
                "past %02d:%02d ET cutoff; skipping buy for %s",
                config.NO_NEW_POSITIONS_AFTER_HOUR,
                config.NO_NEW_POSITIONS_AFTER_MINUTE,
                ticker,
            )
            return False

        if ticker in self._positions:
            log.info("already have open position in %s; skip", ticker)
            return False

        if len(self._positions) >= config.MAX_POSITIONS:
            log.info(
                "[CAPACITY] max positions (%d) reached; skipping buy for %s "
                "(cooldown NOT set — will retry when a position closes)",
                config.MAX_POSITIONS, ticker,
            )
            return False

        # Sector concentration: allow at most 2 positions in the same sector.
        if sector != "unknown":
            same_sector = sum(1 for p in self._positions.values() if p.sector == sector)
            if same_sector >= 2:
                log.info(
                    "[SECTOR LIMIT] %d open %s position(s) already; skipping %s "
                    "(cooldown NOT set — will retry when sector capacity frees up)",
                    same_sector, sector, ticker,
                )
                return False

        # Duplicate catalyst guard: skip a second entry in the same sector within 5 minutes.
        if sector != "unknown":
            last = self._sector_last_entry.get(sector)
            if last and (datetime.now(timezone.utc) - last).total_seconds() < 300:
                log.info(
                    "[DUPLICATE CATALYST] %s skipped — %s sector already entered within 5min",
                    ticker, sector,
                )
                return False

        # Entry retrace guard: skip if ≥40% of the original spike has faded.
        if http_session is not None and ctx.price_move_pct > 0:
            current_price = await self._fetch_snapshot_price(ticker, http_session)
            if current_price is not None:
                pre_spike = ctx.price / (1 + ctx.price_move_pct)
                move_size = ctx.price - pre_spike
                if move_size > 0:
                    remaining = (current_price - pre_spike) / move_size
                    if remaining < config.ENTRY_RETRACE_THRESHOLD:
                        log.warning(
                            "[ENTRY CHASING] %s: %.0f%% of move retraced "
                            "(remaining=%.2f < threshold=%.2f); skipping",
                            ticker,
                            (1.0 - remaining) * 100,
                            remaining,
                            config.ENTRY_RETRACE_THRESHOLD,
                        )
                        return False

        # Gemini sets TP, SL, position size, and hold time per trade.
        # Hard floors/ceilings are enforced here regardless of Gemini's values.
        # Reject NaN/inf upfront — pydantic accepts them but they break clamping.
        def _safe_float(v: float, default: float) -> float:
            return default if (v is None or not math.isfinite(v)) else float(v)

        take_profit_pct = max(
            config.GEMINI_TP_MIN,
            min(config.GEMINI_TP_MAX, _safe_float(verdict.take_profit_pct, 0.15)),
        )
        stop_loss_pct = max(
            config.GEMINI_SL_MIN,
            min(config.GEMINI_SL_MAX, _safe_float(verdict.stop_loss_pct, 0.05)),
        )

        # ATR floor: SL must cover at least ATR_SL_MULTIPLIER × daily ATR
        # so we don't get stopped out by normal intraday noise.
        if ctx.technicals is not None and ctx.technicals.atr_14_pct is not None:
            atr_floor = ctx.technicals.atr_14_pct / 100.0 * config.ATR_SL_MULTIPLIER
            if atr_floor > stop_loss_pct:
                log.info(
                    "[ATR FLOOR] %s SL raised from %.1f%% to %.1f%% (ATR=%.1f%% × %.1f)",
                    ticker, stop_loss_pct * 100, atr_floor * 100,
                    ctx.technicals.atr_14_pct, config.ATR_SL_MULTIPLIER,
                )
                stop_loss_pct = min(atr_floor, config.GEMINI_SL_MAX)

        position_size_pct = max(
            config.GEMINI_SIZE_MIN,
            min(config.GEMINI_SIZE_MAX, _safe_float(verdict.position_size_pct, 0.10)),
        )
        max_hold_minutes  = max(config.GEMINI_HOLD_MIN, min(config.GEMINI_HOLD_MAX, verdict.max_hold_minutes))
        hold_strategy     = verdict.hold_strategy

        # Apply tiered size_reduction (1.0 normally, 0.5 when a DD tier is active).
        position_size_pct = max(config.GEMINI_SIZE_MIN, position_size_pct * self._stats.size_reduction)

        # Late session: halve size and cap hold time when within 60 min of EOD.
        eod_et = now_et.replace(
            hour=config.EOD_CLOSE_HOUR, minute=config.EOD_CLOSE_MINUTE, second=0, microsecond=0,
        )
        minutes_to_eod = max(0.0, (eod_et - now_et).total_seconds() / 60.0)
        if minutes_to_eod < 60.0:
            position_size_pct = max(config.GEMINI_SIZE_MIN, position_size_pct * 0.5)
            max_hold_minutes = min(max_hold_minutes, max(config.GEMINI_HOLD_MIN, int(minutes_to_eod)))
            log.info(
                "late session (%.0fmin to EOD): %s size→%.0f%% hold→%dmin",
                minutes_to_eod, ticker, position_size_pct * 100, max_hold_minutes,
            )

        # Market regime adjustments — apply size_multiplier and hold_multiplier
        # based on SPY HMM regime. Always fail open (regime detector returns
        # 1.0/1.0 for "unknown" so failure never blocks a trade).
        if self._regime_detector is not None:
            try:
                adj = self._regime_detector.get_regime_adjustments()
                size_mult = float(adj.get("size_multiplier", 1.0))
                hold_mult = float(adj.get("hold_multiplier", 1.0))
                label = adj.get("label", "unknown")
                if size_mult != 1.0 or hold_mult != 1.0:
                    new_size = max(config.GEMINI_SIZE_MIN, position_size_pct * size_mult)
                    new_hold = max(config.GEMINI_HOLD_MIN, int(max_hold_minutes * hold_mult))
                    log.info(
                        "[REGIME] %s regime=%s size %.0f%%→%.0f%% hold %dmin→%dmin",
                        ticker, label,
                        position_size_pct * 100, new_size * 100,
                        max_hold_minutes, new_hold,
                    )
                    position_size_pct = new_size
                    max_hold_minutes = new_hold
            except Exception:
                log.debug("[REGIME] adjustment failed; using defaults", exc_info=True)

        log.info(
            "Gemini params: tp=%.0f%% sl=%.0f%% size=%.0f%% hold=%dmin strategy=%s "
            "(conf=%d mag=%d size_reduction=%.1f)",
            take_profit_pct * 100, stop_loss_pct * 100, position_size_pct * 100,
            max_hold_minutes, hold_strategy,
            verdict.confidence, verdict.magnitude_estimate,
            self._stats.size_reduction,
        )

        # --- Bid-ask spread check ---
        if http_session is not None:
            ba = await self._fetch_bid_ask(ticker, http_session)
            if ba is not None:
                bid, ask = ba
                spread_pct = (ask - bid) / ask
                if spread_pct > config.MAX_SPREAD_PCT:
                    log.warning(
                        "[SPREAD] %s skipped — spread %.2f%% exceeds %.1f%% limit",
                        ticker, spread_pct * 100, config.MAX_SPREAD_PCT * 100,
                    )
                    return False

        # --- Correlation check against open positions ---
        if self._hist_client is not None and self._positions:
            new_returns = await self._fetch_daily_returns(ticker)
            if new_returns is not None:
                max_corr = 0.0
                max_corr_ticker = ""
                for open_ticker in list(self._positions):
                    cache_key = tuple(sorted([ticker, open_ticker]))
                    cached = self._corr_cache.get(cache_key)
                    if cached and (datetime.now(timezone.utc) - cached[1]).total_seconds() < 86400:
                        corr = cached[0]
                    else:
                        open_returns = await self._fetch_daily_returns(open_ticker)
                        if open_returns is None:
                            continue
                        corr = self._pearson_corr(new_returns, open_returns)
                        if corr is None:
                            continue
                        self._corr_cache[cache_key] = (corr, datetime.now(timezone.utc))
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        max_corr_ticker = open_ticker

                if max_corr >= config.CORRELATION_REJECT:
                    log.warning(
                        "[CORRELATION] %s skipped — correlation with %s >%.0f%%",
                        ticker, max_corr_ticker, config.CORRELATION_REJECT * 100,
                    )
                    return False
                if max_corr >= config.CORRELATION_REDUCE:
                    log.info(
                        "[CORRELATION] %s size halved — correlation with %s >%.0f%%",
                        ticker, max_corr_ticker, config.CORRELATION_REDUCE * 100,
                    )
                    position_size_pct = max(config.GEMINI_SIZE_MIN, position_size_pct * 0.5)

        try:
            qty, order_id = await self._submit_market_buy(ticker, ctx.price, position_size_pct)
        except Exception as e:
            log.exception("order placement failed for %s", ticker)
            await self._telegram.send_message(
                f"❌ Order failed for <b>{ticker}</b>: {e}"
            )
            return False

        qs = quant_signals or {}
        short_int = qs.get("short_interest") or {}
        ins_score = qs.get("insider_score") or {}
        est_rev = qs.get("estimate_revisions") or {}
        had_factor_signal = bool(
            short_int.get("squeeze_score") in ("high", "medium")
            or ins_score.get("insider_signal") == "bullish"
            or (est_rev.get("revision_score") or 0) >= 1
        )

        rationale_raw = (verdict.reasoning or "").strip()
        entry_rationale_text = rationale_raw[:200] if rationale_raw else None
        entry_top_signals = _derive_top_signals(verdict, quant_signals)

        active = ActivePosition(
            ticker=ticker,
            qty=qty,
            entry_price=None,
            submitted_order_id=order_id,
            gemini_confidence=verdict.confidence,
            gemini_magnitude=verdict.magnitude_estimate,
            technical_signal=verdict.technical_signal,
            sector=sector,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            sl_floor=-stop_loss_pct,
            max_hold_minutes=max_hold_minutes,
            hold_strategy=hold_strategy,
            had_factor_signal=had_factor_signal,
            regime=regime,
            entry_rationale_text=entry_rationale_text,
            entry_top_signals=entry_top_signals,
        )
        self._positions[ticker] = active
        if sector != "unknown":
            self._sector_last_entry[sector] = datetime.now(timezone.utc)
        self._stats.trades_taken += 1
        if had_factor_signal:
            self._stats.factor_signals_traded += 1
        await self._save_positions_snapshot()

        if is_microcap:
            await self._telegram.send_microcap_buy_status(
                ctx=ctx,
                verdict=verdict,
                qty=qty,
                market_cap=market_cap,
                today_volume=today_volume,
                vol_ratio=vol_ratio,
                has_news=has_news,
                biotech_fast_track=biotech_fast_track,
            )
        else:
            await self._telegram.send_buy_status(
                ctx=ctx,
                verdict=verdict,
                qty=qty,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                hold_strategy=hold_strategy,
            )
        return True

    # --- order submission ---

    async def _submit_market_buy(
        self, ticker: str, ref_price: float, position_size_pct: float | None = None
    ) -> tuple[int, str]:
        account = await asyncio.to_thread(self._trading.get_account)
        equity = float(account.non_marginable_buying_power)
        pct = position_size_pct if position_size_pct is not None else config.POSITION_SIZE_PCT
        allocation = equity * pct
        if allocation < 100.0:
            raise ValueError(
                f"allocation ${allocation:.2f} below $100 minimum "
                f"(equity=${equity:.0f}, size={pct:.1%})"
            )
        qty = max(1, math.floor(allocation / max(ref_price, 0.01)))

        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = await asyncio.to_thread(self._trading.submit_order, req)
        log.info(
            "order submitted: %s qty=%d cash=$%.0f alloc=$%.0f (%.0f%%) id=%s",
            ticker, qty, equity, allocation, pct * 100, order.id,
        )
        return qty, str(order.id)

    # --- position close ---

    async def _close_position(self, ticker: str, *, reason: str) -> None:
        """Submit a market-sell order for `ticker`.

        This method only submits the order and marks the position as in-flight.
        All P&L accounting, trade-log writing, circuit-breaker checks, and
        Telegram notifications are handled by _handle_sell_fill() once the
        fill event arrives on the TradingStream.  A 60-second fallback task
        (_close_fallback) fires if no fill event is received in time.
        """
        if ticker not in self._positions:
            log.info("close requested for %s but no position tracked", ticker)
            return

        # Guard: if a close order is already in flight (e.g. from a previous
        # tick that got a 40310000 "close already pending" error), skip so we
        # don't pile up duplicate orders.
        if ticker in self._closing_in_progress:
            log.warning(
                "close already in progress for %s; skipping duplicate call", ticker
            )
            return
        self._closing_in_progress.add(ticker)

        pos = self._positions[ticker]

        # Fetch current P&L estimate now; used by the 60s fallback if the fill
        # event never arrives.  The actual fill price from the stream is preferred.
        pnl_pct = 0.0
        pnl_dollars = 0.0
        try:
            live = await asyncio.to_thread(self._trading.get_open_position, ticker)
            pnl_pct = float(live.unrealized_plpc)
            pnl_dollars = float(live.unrealized_pl)
        except Exception:
            log.debug("could not fetch live P&L for %s before close", ticker, exc_info=True)

        try:
            await asyncio.to_thread(self._trading.close_position, ticker)
        except Exception as e:
            err_str = str(e)
            # 40310000 = "insufficient qty available for order" — Alpaca already has
            # a pending close order for this position.  Stay in _closing_in_progress
            # so monitor_tick keeps skipping; the fill event will clear things up.
            if "40310000" in err_str or "insufficient qty available" in err_str.lower():
                log.warning(
                    "close already pending for %s — a close order is already in "
                    "flight on Alpaca; waiting for fill event to confirm",
                    ticker,
                )
                # Mark close_submitted so the fallback knows to run.
                pos.close_submitted = True
                pos.close_submitted_at = datetime.now(timezone.utc)
                pos.close_reason = reason
                pos.last_known_pnl_pct = pnl_pct
                pos.last_known_pnl_dollars = pnl_dollars
                if ticker not in self._close_fallback_tasks:
                    task = asyncio.create_task(self._close_fallback(ticker))
                    self._close_fallback_tasks[ticker] = task
                return
            # Genuine failure: discard from _closing_in_progress to allow retry.
            self._closing_in_progress.discard(ticker)
            log.exception("close_position failed for %s", ticker)
            await self._telegram.send_message(
                f"❌ Close failed for <b>{ticker}</b>: {e}"
            )
            return

        # Order submitted successfully.  Store context so the fill handler /
        # fallback can complete accounting without re-fetching everything.
        pos.close_submitted = True
        pos.close_submitted_at = datetime.now(timezone.utc)
        pos.close_reason = reason
        pos.last_known_pnl_pct = pnl_pct
        pos.last_known_pnl_dollars = pnl_dollars

        log.info(
            "close order submitted for %s reason=%s; waiting for SELL fill event",
            ticker, reason,
        )

        # Schedule 60-second fallback in case the fill event never arrives
        # (e.g. stream disconnect).
        task = asyncio.create_task(self._close_fallback(ticker))
        self._close_fallback_tasks[ticker] = task

    # --- fill-stream sell handler (single source of truth for P&L accounting) ---

    async def _handle_sell_fill(
        self,
        ticker: str,
        fill_price: float,
        fill_qty: float,
        is_manual: bool = False,
    ) -> None:
        """Called from the TradingStream fill handler when a SELL fill is confirmed.

        This is the authoritative path for P&L accounting, trade-log writing,
        circuit-breaker checks, and the close Telegram notification.
        Works for both bot-initiated closes and manual closes from the Alpaca
        dashboard or /close command.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            log.debug(
                "_handle_sell_fill: %s not in tracked positions; already cleaned up",
                ticker,
            )
            return

        # Cancel the 60-second fallback — we have the real fill.
        fallback = self._close_fallback_tasks.pop(ticker, None)
        if fallback and not fallback.done():
            fallback.cancel()

        entry_price = pos.entry_price if pos.entry_price is not None else fill_price
        pnl_dollars = (fill_price - entry_price) * fill_qty
        pnl_pct = (fill_price - entry_price) / entry_price if entry_price > 0 else 0.0

        exit_time = datetime.now(timezone.utc)
        opened_utc = (
            pos.opened_at.astimezone(timezone.utc)
            if pos.opened_at.tzinfo
            else pos.opened_at.replace(tzinfo=timezone.utc)
        )
        hold_secs = (exit_time - opened_utc).total_seconds()

        reason = pos.close_reason or ("manual close" if is_manual else "fill_stream")
        source = "[manual]" if is_manual else "[pending order]" if ticker in self._closing_in_progress else "[bot-initiated]"
        log.info(
            "SELL fill confirmed: %s %s exit=$%.2f qty=%g pnl=%+.2f%% ($%+.2f) held=%s reason=%s",
            ticker, source, fill_price, fill_qty,
            pnl_pct, pnl_dollars, _format_hold_time(hold_secs), reason,
        )

        self._stats.realized_pnl += pnl_dollars
        self._stats.weekly_pnl += pnl_dollars

        if pos.had_factor_signal and reason.startswith("TP"):
            self._stats.factor_signals_correct += 1

        if self._stats.opening_equity > 0:
            self._spawn_background(self._check_circuit_breakers(pnl_dollars))

        if pos.entry_price is not None:
            try:
                from daily_summary import TradeRecord
                self._stats.trade_records.append(TradeRecord(
                    ticker=ticker,
                    qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=round(fill_price, 4),
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_pct,
                    hold_secs=hold_secs,
                    reason=reason,
                    opened_at=pos.opened_at,
                    regime=pos.regime,
                ))
            except Exception:
                log.debug("TradeRecord append failed", exc_info=True)
            await self._write_trade_log(
                pos, round(fill_price, 4), pnl_pct, pnl_dollars, hold_secs, reason, exit_time
            )
            # Background reflection — close path never awaits this.
            self._dispatch_reflection(
                pos,
                exit_price=float(fill_price),
                raw_return_pct=pnl_pct,
                exit_time=exit_time,
                reason=reason,
            )

        self._positions.pop(ticker, None)
        self._closing_in_progress.discard(ticker)
        await self._save_positions_snapshot()

        if self._cooldown_store is not None:
            await self._cooldown_store.set_cooldown(ticker)
            log.info(
                "exit cooldown set for %s (%dmin) after %s",
                ticker, config.EXIT_COOLDOWN_MINUTES, reason,
            )

        if any(kw in reason.lower() for kw in _ACQUISITION_BLOCK_KEYWORDS):
            self.session_blocked_tickers.add(ticker)
            log.warning("[SESSION BLOCK] %s blocked for session — %s", ticker, reason)

        await self._telegram.send_close_status(
            ticker=ticker,
            exit_price=fill_price,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            hold_str=_format_hold_time(hold_secs),
            reason=reason,
        )

    async def _close_fallback(self, ticker: str) -> None:
        """60-second fallback when no SELL fill event arrives for a submitted close.

        Uses the pre-close P&L estimates stored on the position.  Fires for
        legitimate stream gaps (TradingStream reconnect, brief disconnect) and
        any other case where the fill event is never delivered.
        """
        await asyncio.sleep(60.0)
        self._close_fallback_tasks.pop(ticker, None)

        pos = self._positions.get(ticker)
        if pos is None or not pos.close_submitted:
            # Fill handler already cleaned this up — nothing to do.
            return

        log.warning(
            "SELL fill event not received for %s within 60s; "
            "using pre-close P&L estimate for accounting",
            ticker,
        )

        pnl_pct = pos.last_known_pnl_pct
        pnl_dollars = pos.last_known_pnl_dollars
        entry_price = pos.entry_price or 0.0
        exit_price = entry_price * (1.0 + pnl_pct) if entry_price > 0 else 0.0

        exit_time = datetime.now(timezone.utc)
        opened_utc = (
            pos.opened_at.astimezone(timezone.utc)
            if pos.opened_at.tzinfo
            else pos.opened_at.replace(tzinfo=timezone.utc)
        )
        hold_secs = (exit_time - opened_utc).total_seconds()
        reason = pos.close_reason or "fallback"

        self._stats.realized_pnl += pnl_dollars
        self._stats.weekly_pnl += pnl_dollars

        if self._stats.opening_equity > 0:
            self._spawn_background(self._check_circuit_breakers(pnl_dollars))

        if pos.entry_price is not None and exit_price > 0:
            try:
                from daily_summary import TradeRecord
                self._stats.trade_records.append(TradeRecord(
                    ticker=ticker,
                    qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=round(exit_price, 4),
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_pct,
                    hold_secs=hold_secs,
                    reason=reason,
                    opened_at=pos.opened_at,
                    regime=pos.regime,
                ))
            except Exception:
                log.debug("TradeRecord append failed (fallback path)", exc_info=True)
            await self._write_trade_log(
                pos, round(exit_price, 4), pnl_pct, pnl_dollars, hold_secs, reason, exit_time
            )
            # Background reflection — close path never awaits this.
            self._dispatch_reflection(
                pos,
                exit_price=float(exit_price),
                raw_return_pct=pnl_pct,
                exit_time=exit_time,
                reason=reason,
            )

        self._positions.pop(ticker, None)
        self._closing_in_progress.discard(ticker)
        await self._save_positions_snapshot()

        if self._cooldown_store is not None:
            await self._cooldown_store.set_cooldown(ticker)
            log.info(
                "exit cooldown set for %s (%dmin) after %s [fallback path]",
                ticker, config.EXIT_COOLDOWN_MINUTES, reason,
            )

        if any(kw in reason.lower() for kw in _ACQUISITION_BLOCK_KEYWORDS):
            self.session_blocked_tickers.add(ticker)
            log.warning("[SESSION BLOCK] %s blocked for session (fallback path) — %s", ticker, reason)

        await self._telegram.send_close_status(
            ticker=ticker,
            exit_price=exit_price if exit_price > 0 else None,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            hold_str=_format_hold_time(hold_secs),
            reason=f"{reason} [estimated — fill event not received]",
        )

    async def _write_trade_log(
        self,
        pos: ActivePosition,
        exit_price: float,
        pnl_pct: float,
        pnl_dollars: float,
        hold_secs: float,
        reason: str,
        exit_time: datetime,
    ) -> None:
        record = {
            "ticker": pos.ticker,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "shares": pos.qty,
            "pnl_pct": round(pnl_pct, 6),
            "pnl_dollar": round(pnl_dollars, 2),
            "hold_minutes": round(hold_secs / 60, 1),
            "exit_reason": reason,
            "gemini_confidence": pos.gemini_confidence,
            "gemini_magnitude": pos.gemini_magnitude,
            "technical_signal": pos.technical_signal,
            "entry_time": pos.opened_at.isoformat(),
            "exit_time": exit_time.isoformat(),
        }
        line = json.dumps(record) + "\n"
        path = config.STATE_DIR / "trade_log.jsonl"

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

        try:
            await asyncio.to_thread(_append)
        except Exception:
            log.warning("trade log write failed for %s", pos.ticker, exc_info=True)

    # --- EOD close ---

    async def run_eod_close(self, stop_event: asyncio.Event) -> None:
        """Wait until EOD_CLOSE time ET then close all open positions."""
        while not stop_event.is_set():
            now_et = datetime.now(tz=config.MARKET_TZ)
            eod = now_et.replace(
                hour=config.EOD_CLOSE_HOUR,
                minute=config.EOD_CLOSE_MINUTE,
                second=0,
                microsecond=0,
            )
            if now_et >= eod:
                eod += timedelta(days=1)

            sleep_secs = max(1.0, (eod - datetime.now(tz=config.MARKET_TZ)).total_seconds())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_secs)
                return
            except asyncio.TimeoutError:
                pass

            if self._positions:
                tickers = list(self._positions.keys())
                log.info("EOD close: closing %d position(s): %s", len(tickers), tickers)
                for ticker in tickers:
                    try:
                        await self._close_position(ticker, reason="EOD")
                    except Exception:
                        log.exception("EOD close failed for %s", ticker)
                await self._telegram.send_message(
                    f"🔔 EOD close complete — {len(tickers)} position(s) closed."
                )
            else:
                log.info("EOD close: no open positions")

            self.session_blocked_tickers.clear()
            self._sector_last_entry.clear()
            log.info(
                "EOD: session_blocked_tickers and _sector_last_entry cleared for next session"
            )

    # --- monitoring loop ---

    async def monitor_positions(self, stop_event: asyncio.Event) -> None:
        """Poll open positions; auto-close at thresholds; post milestone P&L notifications."""
        while not stop_event.is_set():
            try:
                await self._monitor_tick()
            except Exception:
                log.exception("monitor tick failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=config.POSITION_POLL_SECONDS)
                return
            except asyncio.TimeoutError:
                continue

    async def _monitor_tick(self) -> None:
        if not self._positions:
            return
        try:
            open_positions = await asyncio.wait_for(
                asyncio.to_thread(self._trading.get_all_positions),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            log.warning("get_all_positions timed out; skipping monitor tick")
            return
        except Exception:
            log.exception("get_all_positions failed")
            return
        by_symbol = {p.symbol: p for p in open_positions}

        for ticker in list(self._positions.keys()):
            # Skip tickers where a close order is already in-flight; no need to
            # re-evaluate TP/SL or re-fire _close_position.
            if ticker in self._closing_in_progress:
                log.debug("monitor_tick: %s close already in progress; skipping", ticker)
                continue

            live = by_symbol.get(ticker)
            if live is None:
                active = self._positions[ticker]
                opened_utc = active.opened_at.astimezone(timezone.utc) if active.opened_at.tzinfo else active.opened_at.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - opened_utc) > timedelta(seconds=30):
                    if active.close_submitted:
                        # Close order filled — the fill event or 60s fallback will
                        # finalize accounting.  Nothing more to do here.
                        log.debug(
                            "monitor_tick: %s gone from Alpaca (close_submitted); "
                            "fill event or fallback will finalize accounting",
                            ticker,
                        )
                    else:
                        # Ghost position: the BUY was submitted (trades_taken
                        # incremented, position tracked) but the TradingStream
                        # fill event never arrived AND the position is now gone
                        # from Alpaca. Without recording it here, trades_taken
                        # diverges from len(stats.trade_records) and the trade
                        # leaves no audit trail.
                        # Synthesize a zero-P&L close record so accounting stays
                        # in parity, and notify operator to investigate.
                        reason = "disappeared_no_fill_confirmation"
                        active.close_reason = reason
                        active.last_known_pnl_pct = 0.0
                        active.last_known_pnl_dollars = 0.0

                        exit_time = datetime.now(timezone.utc)
                        hold_secs = (exit_time - opened_utc).total_seconds()

                        if active.entry_price is not None:
                            log.warning(
                                "monitor_tick: %s gone from Alpaca without "
                                "close_submitted (entry_price known) — "
                                "recording zero-P&L trade for accounting parity",
                                ticker,
                            )
                            try:
                                from daily_summary import TradeRecord
                                self._stats.trade_records.append(TradeRecord(
                                    ticker=ticker,
                                    qty=active.qty,
                                    entry_price=active.entry_price,
                                    exit_price=round(active.entry_price, 4),
                                    pnl_dollars=0.0,
                                    pnl_pct=0.0,
                                    hold_secs=hold_secs,
                                    reason=reason,
                                    opened_at=active.opened_at,
                                    regime=active.regime,
                                ))
                            except Exception:
                                log.debug(
                                    "TradeRecord append failed (ghost path)",
                                    exc_info=True,
                                )
                            try:
                                await self._write_trade_log(
                                    active,
                                    round(active.entry_price, 4),
                                    0.0, 0.0, hold_secs, reason, exit_time,
                                )
                            except Exception:
                                log.debug(
                                    "trade log write failed (ghost path)",
                                    exc_info=True,
                                )
                        else:
                            # entry_price never confirmed — order may have
                            # been rejected after submission. Skip the trade
                            # log entry rather than write nonsense numbers.
                            log.warning(
                                "monitor_tick: %s gone from Alpaca without "
                                "close_submitted AND entry_price unset — "
                                "buy fill never confirmed; cannot synthesize "
                                "trade record",
                                ticker,
                            )

                        try:
                            await self._telegram.send_message(
                                f"⚠️ <b>{ticker}</b> disappeared from Alpaca "
                                f"without fill confirmation — removed from "
                                f"tracking, recorded P&L $0. Check Alpaca "
                                f"dashboard."
                            )
                        except Exception:
                            log.debug(
                                "ghost-position telegram failed",
                                exc_info=True,
                            )

                        self._positions.pop(ticker, None)
                        self._closing_in_progress.discard(ticker)
                        if self._cooldown_store is not None:
                            await self._cooldown_store.set_cooldown(ticker)
                            log.info(
                                "exit cooldown set for %s (%dmin) after %s",
                                ticker, config.EXIT_COOLDOWN_MINUTES, reason,
                            )
                        self._spawn_background(self._save_positions_snapshot())
                continue

            active = self._positions[ticker]
            if active.entry_price is None:
                active.entry_price = float(live.avg_entry_price)

            now_utc = datetime.now(timezone.utc)
            opened_utc = (
                active.opened_at.astimezone(timezone.utc)
                if active.opened_at.tzinfo
                else active.opened_at.replace(tzinfo=timezone.utc)
            )
            hold_mins = (now_utc - opened_utc).total_seconds() / 60.0

            plpc = float(live.unrealized_plpc)

            # Gemini-set hold time: only timeout profitable positions so a
            # losing trade is never forced out by the clock (SL or EOD handles
            # that). Use 0.001 (~10 bps) as the threshold instead of >=0 to
            # avoid float-precision races where Alpaca returns 0.0 at the tick
            # moment and the position is slightly negative by the time the
            # close fills (MRP closed at -0.07% on 2026-05-07 via this race).
            if hold_mins >= active.max_hold_minutes and plpc >= 0.001:
                if (self._monitor_scorer is not None
                        and self._monitor_telegram is not None
                        and self._monitor_http_session is not None):
                    log.info(
                        "[TIMEOUT] %s hold limit %dmin reached (P&L %.2f%%) — "
                        "triggering immediate Gemini re-look",
                        ticker, active.max_hold_minutes, plpc * 100,
                    )
                    # Push the deadline 30 minutes out so the next monitor
                    # tick doesn't re-fire while the task is in flight.
                    active.max_hold_minutes = int(hold_mins) + 30
                    # Lazy import to avoid circular dependency at module load
                    # (position_monitor imports Trader at module level).
                    from position_monitor import _timeout_reeval
                    self._spawn_background(_timeout_reeval(
                        ticker,
                        self,
                        self._monitor_scorer,
                        self._monitor_telegram,
                        self._monitor_http_session,
                        self._monitor_news_fingerprints,
                    ))
                else:
                    # Position monitor never started or refs got cleared —
                    # fall back to the original hard-close behaviour rather
                    # than letting the position drift forever.
                    log.warning(
                        "[TIMEOUT] %s hold limit reached but monitor refs "
                        "unavailable — falling back to hard close",
                        ticker,
                    )
                    await self._close_position(
                        ticker,
                        reason=f"timeout — Gemini hold limit {active.max_hold_minutes}min reached",
                    )
                continue

            if plpc >= active.take_profit_pct:
                await self._close_position(ticker, reason=f"TP {plpc:+.1%}")
                continue
            # SL floor: starts at -stop_loss_pct; can be raised by tighten_sl verdicts.
            if plpc <= active.sl_floor:
                await self._close_position(ticker, reason=f"SL {plpc:+.1%}")
                continue

            # Milestone notifications — fire once per milestone, never on a timer.
            unrealized_pl = float(live.unrealized_pl)

            # +5% increments: fire each band crossed since last tick.
            if plpc >= 0.05:
                for threshold in range(5, 105, 5):
                    if plpc >= threshold / 100:
                        key = f"up_{threshold}"
                        if key not in active.notified_milestones:
                            active.notified_milestones.add(key)
                            await self._telegram.send_message(
                                f"📈 <b>{ticker}</b>: {plpc:+.2%} "
                                f"(${unrealized_pl:+.2f}) — +{threshold}% milestone"
                            )
                    else:
                        break

            # Went negative for the first time.
            if plpc < 0 and "went_negative" not in active.notified_milestones:
                active.notified_milestones.add("went_negative")
                await self._telegram.send_message(
                    f"🔻 <b>{ticker}</b> turned negative: {plpc:+.2%} "
                    f"(${unrealized_pl:+.2f})"
                )

            # Within 1% of stop floor.
            if (
                plpc <= (active.sl_floor + 0.01)
                and "near_sl" not in active.notified_milestones
            ):
                active.notified_milestones.add("near_sl")
                sl_desc = (
                    f"trailing floor at {active.sl_floor:+.0%}"
                    if active.sl_floor > 0
                    else f"SL at {active.sl_floor:.0%}"
                )
                await self._telegram.send_message(
                    f"⚠️ <b>{ticker}</b> approaching stop: {plpc:+.2%} "
                    f"({sl_desc})"
                )

            # Within 1% of take profit.
            near_tp_threshold = active.take_profit_pct - config.NEAR_TP_EXIT_PCT
            if plpc >= near_tp_threshold:
                if "near_tp" not in active.notified_milestones:
                    active.notified_milestones.add("near_tp")
                    await self._telegram.send_message(
                        f"🎯 <b>{ticker}</b> approaching take profit: {plpc:+.2%} "
                        f"(TP at +{active.take_profit_pct:.0%}) — will close here"
                    )
                # Close if we've held long enough — don't bail in the first 20 minutes
                if hold_mins >= config.NEAR_TP_MIN_HOLD_MINUTES:
                    await self._close_position(ticker, reason=f"near-TP {plpc:+.1%}")
                    continue

    # --- trading stream (fill confirmations) ---

    async def run_trading_stream(self, stop_event: asyncio.Event) -> None:
        """Subscribe to trade updates so fills are recorded authoritatively.

        TradingStream.run() creates its own event loop in a worker thread;
        we hop back to the main loop via run_coroutine_threadsafe for anything
        that touches shared state.
        """
        main_loop = asyncio.get_running_loop()

        async def _handle_on_main(data) -> None:
            try:
                order = data.order
                symbol = order.symbol
                event = data.event

                if event in ("fill", "partial_fill"):
                    side = order.side

                    # --- BUY fill: record authoritative entry price ---
                    if side == OrderSide.BUY and symbol in self._positions:
                        pos = self._positions[symbol]
                        if order.filled_avg_price:
                            pos.entry_price = float(order.filled_avg_price)
                        # Send Telegram notification only on the FINAL fill —
                        # a single buy can deliver many partial_fill events
                        # (CORZ produced 7 on 2026-05-07) which would spam chat.
                        if event == "fill":
                            await self._telegram.send_message(
                                f"✅ Filled {order.filled_qty} × <b>{symbol}</b> @ "
                                f"${float(order.filled_avg_price or 0):.2f}"
                            )
                        # partial_fill: silently update entry_price only.

                    # --- SELL fill: single source of truth for P&L accounting ---
                    elif side == OrderSide.SELL and event == "fill":
                        fill_price = float(order.filled_avg_price or 0)
                        fill_qty = float(order.filled_qty or 0)
                        if fill_price <= 0 or fill_qty <= 0:
                            log.warning(
                                "SELL fill for %s has zero price/qty; ignoring "
                                "(fill_price=%.2f fill_qty=%g)",
                                symbol, fill_price, fill_qty,
                            )
                        else:
                            is_manual = symbol not in self._closing_in_progress
                            if is_manual:
                                log.info(
                                    "SELL fill [manual close]: %s fill_price=$%.2f qty=%g",
                                    symbol, fill_price, fill_qty,
                                )
                            await self._handle_sell_fill(symbol, fill_price, fill_qty, is_manual)

                    elif side == OrderSide.SELL and event == "partial_fill":
                        partial_qty = float(data.qty or 0) if data.qty else 0
                        log.info(
                            "SELL partial_fill: %s qty=%g fill_price=$%.2f "
                            "(waiting for final fill event)",
                            symbol, partial_qty,
                            float(data.price or 0) if data.price else 0,
                        )

            except Exception:
                log.exception("trade update main-loop handler failed")

        def _log_future_exc(fut: asyncio.Future) -> None:
            exc = fut.exception()
            if exc:
                log.error("trade update handler raised: %r", exc)

        async def on_trade_update(data) -> None:
            fut = asyncio.run_coroutine_threadsafe(_handle_on_main(data), main_loop)
            fut.add_done_callback(_log_future_exc)

        backoff = 1.0
        while not stop_event.is_set():
            stream = TradingStream(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                paper=config.ALPACA_PAPER,
            )
            stream.subscribe_trade_updates(on_trade_update)
            connected_at = asyncio.get_running_loop().time()
            stream_task = asyncio.create_task(asyncio.to_thread(stream.run))
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                {stream_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if stop_task in done:
                try:
                    await asyncio.to_thread(stream.stop)
                except Exception:
                    log.exception("error stopping TradingStream")
                stream_task.cancel()
                return
            # Reset backoff if the stream ran cleanly for ≥60s — a single late
            # disconnect should not push the next reconnect to a 60s wait.
            ran_secs = asyncio.get_running_loop().time() - connected_at
            if ran_secs >= 60.0:
                backoff = 1.0
            log.warning(
                "TradingStream exited after %.0fs; reconnecting in %.1fs",
                ran_secs, backoff,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(60.0, backoff * 2.0)
