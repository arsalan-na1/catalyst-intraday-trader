"""Entrypoint. Wires every component onto a single asyncio event loop.

Signals handled:
  - SIGINT / SIGTERM: set stop_event; all workers drain and exit cleanly.

CLI flags (for ops/testing):
  --inject-trigger TICKER PRICE MOVE_PCT VOL_RATIO
        Puts a synthetic TriggerEvent on the queue so the downstream pipeline
        (news → scorer → Telegram → trader) can be exercised without waiting
        for real market conditions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, time as _dt_time, timezone

import aiohttp
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

import config
from cooldown_store import CooldownStore
from daily_summary import SessionStats, run_daily_summary
from earnings_calendar import EarningsCalendar
from logger import setup_logging
from market_calendar import MarketCalendar
from market_regime import MarketRegimeDetector
from microcap_screener import supervised_run as run_microcap_screener
from premarket_scanner import run_premarket_scanner
from position_monitor import run_position_monitor
from news import determine_sector, fetch_news, is_analyst_action, is_biotech_catalyst, is_insider_buying
from scorer import Scorer, TriggerContext
from stream import TriggerEvent, load_watchlist, prefetch_adv, prefetch_prev_close, run_stream
from telegram_handler import TelegramHandler
from trader import Trader

log = logging.getLogger("bot")

_MARKET_OPEN = _dt_time(9, 30, 0)


def _to_utc(dt: datetime) -> datetime:
    """Normalise any datetime to UTC-aware, treating naive as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _fetch_current_price(
    ticker: str, http_session: aiohttp.ClientSession
) -> float | None:
    """Snapshot current price for stale-trigger reversal check."""
    url = "https://data.alpaca.markets/v2/stocks/snapshots"
    params = {"symbols": ticker, "feed": "iex"}
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }
    try:
        async with http_session.get(
            url,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            snap = data.get(ticker) or {}
            trade = snap.get("latestTrade") or {}
            bar = snap.get("dailyBar") or {}
            raw = trade.get("p") or bar.get("c")
            return float(raw) if raw else None
    except Exception:
        log.debug("_fetch_current_price(%s) failed", ticker, exc_info=True)
        return None


async def _process_trigger(
    event: TriggerEvent,
    scorer: Scorer,
    telegram: TelegramHandler,
    trader: Trader,
    stats: SessionStats,
    http_session: aiohttp.ClientSession,
    cooldown_store: CooldownStore,
) -> None:
    """Full pipeline for a single trigger: stale-check → cooldown → news → score → trade."""
    try:
        # --- stale trigger guard ---
        # Use _to_utc on both sides — bar.timestamp is UTC-aware; synthetic
        # triggers use datetime.now(timezone.utc) so they're also tz-aware.
        age_secs = (datetime.now(timezone.utc) - _to_utc(event.triggered_at)).total_seconds()
        if age_secs > config.STALE_TRIGGER_MINUTES * 60:
            try:
                current_price = await _fetch_current_price(event.ticker, http_session)
                if current_price is not None:
                    delta = (current_price - event.price) / event.price
                    reversed_ = (
                        (event.price_move_pct > 0 and delta <= -config.STALE_TRIGGER_REVERSAL_PCT)
                        or (event.price_move_pct < 0 and delta >= config.STALE_TRIGGER_REVERSAL_PCT)
                    )
                    if reversed_:
                        log.info(
                            "[STALE TRIGGER] %s age=%.0fs trigger=$%.2f current=$%.2f delta=%.1f%%",
                            event.ticker, age_secs, event.price, current_price, delta * 100,
                        )
                        return
            except Exception:
                log.warning(
                    "stale price check failed for %s; proceeding without check", event.ticker
                )

        # --- session block check (acquisition/merger tickers) ---
        if event.ticker in trader.session_blocked_tickers:
            log.info("[SESSION BLOCK] %s blocked for session — acquisition/merger cap", event.ticker)
            return

        # --- persistent cooldown check ---
        if cooldown_store.is_in_cooldown(event.ticker, window_minutes=config.EXIT_COOLDOWN_MINUTES):
            log.info("[COOLDOWN] %s still in cooldown (%dmin window); skipping",
                     event.ticker, config.EXIT_COOLDOWN_MINUTES)
            return

        ctx = TriggerContext(
            ticker=event.ticker,
            price=event.price,
            price_move_pct=event.price_move_pct,
            volume_ratio=event.volume_ratio,
            trigger_type=event.trigger_type,
        )

        # --- pre-market check ---
        triggered_et = _to_utc(event.triggered_at).astimezone(config.MARKET_TZ)
        is_premarket = triggered_et.time() < _MARKET_OPEN

        try:
            news = await fetch_news(event.ticker, session=http_session)
        except Exception:
            log.exception("news fetch failed for %s", event.ticker)
            news = []

        verdict, tech, quant = await scorer.score(ctx, news)
        if tech is not None:
            stats.technicals_fetched += 1
        else:
            stats.technicals_failed += 1

        if quant:
            if (quant.get("short_interest") or {}).get("squeeze_score") == "high":
                stats.squeeze_setups_seen += 1
            if (quant.get("insider_score") or {}).get("insider_signal") == "bullish":
                stats.insider_bullish_seen += 1
            if ((quant.get("estimate_revisions") or {}).get("revision_score") or 0) >= 1:
                stats.revision_strong_seen += 1

        if verdict is None:
            log.info("no verdict for %s; skipping", event.ticker)
            return
        ctx.technicals = tech

        # Log catalyst context for ops visibility (no longer used for threshold decisions).
        biotech = is_biotech_catalyst(news)
        if biotech:
            log.info("[BIOTECH CATALYST] detected for %s — Gemini will size conservatively", event.ticker)
        elif is_analyst_action(news):
            log.info("[ANALYST CATALYST] detected for %s", event.ticker)
        elif is_insider_buying(news):
            log.info("[INSIDER CATALYST] detected for %s", event.ticker)

        # Gemini makes the trade/no-trade decision via should_trade.
        # Bot only enforces hard safety gates (circuit breaker, cooldown, capacity).
        if not verdict.should_trade:
            skip = verdict.skip_reason or "Gemini declined to trade"
            log.info(
                "[GEMINI SKIP] %s: should_trade=False — %s (conf=%d mag=%d found=%s)",
                event.ticker, skip,
                verdict.confidence, verdict.magnitude_estimate, verdict.catalyst_found,
            )
            return

        if is_premarket:
            log.info("[PRE-MARKET] %s passes thresholds; sending watch alert", event.ticker)
            try:
                await telegram.send_premarket_watch(ctx=ctx, verdict=verdict)
            except Exception:
                log.exception("send_premarket_watch failed for %s", event.ticker)
            # Set cooldown for pre-market alerts so we don't re-alert on restart.
            await cooldown_store.set_cooldown(event.ticker)
            return

        stats.alerts_sent += 1
        sector = determine_sector(event.ticker, news)
        bought = False
        try:
            bought = await trader.execute_auto_buy(
                ctx,
                verdict,
                http_session=http_session,
                is_biotech=biotech,
                sector=sector,
                quant_signals=quant,
            )
        except Exception:
            log.exception("execute_auto_buy failed for %s", event.ticker)

        # Only record the persistent cooldown when an order was actually submitted.
        # Capacity rejections (MAX_POSITIONS, sector limit) must not lock out the
        # ticker so it can retry when capacity frees up.
        if bought:
            await cooldown_store.set_cooldown(event.ticker)
        else:
            log.info(
                "[CAPACITY] %s: no order placed; cooldown NOT set "
                "(trigger eligible to retry when positions free up)",
                event.ticker,
            )

    except Exception:
        log.exception("pipeline failed unexpectedly for %s", event.ticker)


async def pipeline_consumer(
    queue: asyncio.Queue[TriggerEvent],
    scorer: Scorer,
    telegram: TelegramHandler,
    trader: Trader,
    stats: SessionStats,
    http_session: aiohttp.ClientSession,
    stop_event: asyncio.Event,
    cooldown_store: CooldownStore,
) -> None:
    """Drain trigger queue; process up to PIPELINE_CONCURRENCY events concurrently."""
    sem = asyncio.Semaphore(config.PIPELINE_CONCURRENCY)

    async def bounded(event: TriggerEvent) -> None:
        async with sem:
            await _process_trigger(
                event, scorer, telegram, trader, stats, http_session, cooldown_store
            )

    while not stop_event.is_set():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if stats.halt_new_entries:
            log.info("[CIRCUIT BREAKER] halted; dropping trigger for %s", event.ticker)
            continue
        stats.spikes_detected += 1
        asyncio.create_task(bounded(event))


async def run_heartbeat(
    stats: SessionStats,
    trader: Trader,
    scorer: Scorer,
    stop_event: asyncio.Event,
) -> None:
    """Log a heartbeat every HEARTBEAT_INTERVAL_MINUTES minutes."""
    interval = config.HEARTBEAT_INTERVAL_MINUTES * 60.0
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        n_pos = len(getattr(trader, "_positions", {}))
        n_gemini = getattr(scorer, "calls_today", 0)
        monthly_cost = getattr(scorer, "monthly_cost_estimate", 0.0)
        budget_mode = getattr(scorer, "_budget_mode", "normal")
        log.info(
            "[HEARTBEAT] alive | triggers today: %d | trades today: %d | "
            "positions open: %d | Gemini calls: %d | monthly cost: $%.2f/$%d (%s)",
            stats.spikes_detected,
            stats.trades_taken,
            n_pos,
            n_gemini,
            monthly_cost,
            int(config.MONTHLY_GEMINI_BUDGET_USD),
            budget_mode,
        )


async def run_regime_refresher(
    regime_detector: MarketRegimeDetector,
    scorer: Scorer,
    stop_event: asyncio.Event,
    interval_minutes: int = 60,
) -> None:
    """Refresh SPY regime every `interval_minutes`; sync label to scorer."""
    # Prime once at startup.
    try:
        label = await regime_detector.get_regime()
        scorer._current_regime_label = label
    except Exception:
        log.warning("initial regime fetch failed", exc_info=True)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_minutes * 60)
            return
        except asyncio.TimeoutError:
            pass
        try:
            label = await regime_detector.get_regime()
            scorer._current_regime_label = label
        except Exception:
            log.warning("regime refresh failed; keeping previous label", exc_info=True)


async def _supervise(
    name: str,
    coro_factory,
    stop_event: asyncio.Event,
    restart_delay: float = 30.0,
) -> None:
    """Run coro_factory() and restart on unhandled exceptions."""
    while not stop_event.is_set():
        try:
            await coro_factory()
            return  # clean exit (stop_event set or normal finish)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[SUPERVISOR] %s crashed; restarting in %.0fs", name, restart_delay)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=restart_delay)
                return
            except asyncio.TimeoutError:
                pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-driven catalyst trading bot")
    p.add_argument(
        "--inject-trigger",
        nargs=4,
        metavar=("TICKER", "PRICE", "MOVE_PCT", "VOL_RATIO"),
        help="Inject a synthetic trigger on startup for end-to-end testing.",
    )
    return p.parse_args()


async def _startup_checks(
    trading_client,
    http_session: aiohttp.ClientSession,
    scorer,
    watchlist_path,
    n_watchlist: int,
) -> None:
    """Log ✅/❌ for each pre-flight check. Never raises — errors are logged only."""
    log.info("=== STARTUP SELF-CHECK ===")

    # state/ directory
    try:
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("✅ state/ dir ready (%s)", config.STATE_DIR)
    except Exception as e:
        log.error("❌ state/ dir: %s", e)

    # watchlist size
    if not watchlist_path.exists():
        log.error("❌ watchlist not found: %s", watchlist_path)
    elif n_watchlist < 100:
        log.warning("⚠️  watchlist: %d tickers (recommend ≥ 100)", n_watchlist)
    else:
        log.info("✅ watchlist: %d tickers", n_watchlist)

    # Alpaca connectivity
    try:
        account = await asyncio.to_thread(trading_client.get_account)
        label = "paper" if config.ALPACA_PAPER else "LIVE"
        log.info("✅ Alpaca %s account reachable (equity=$%.2f)",
                 label, float(account.equity))
    except Exception as e:
        log.error("❌ Alpaca unreachable: %s", e)

    # Gemini API key — minimal generate call to confirm auth
    try:
        from google.genai import types as _genai_types
        await scorer._client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
            config=_genai_types.GenerateContentConfig(max_output_tokens=1),
        )
        log.info("✅ Gemini API key valid")
    except Exception as e:
        log.error("❌ Gemini API key check failed: %s", e)

    # Telegram bot token — getMe
    # Token is embedded in the URL; never log the raw URL or unredacted exception
    # text, since aiohttp's ClientError messages may include the request URL.
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getMe"
        async with http_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                username = (data.get("result") or {}).get("username", "?")
                log.info("✅ Telegram bot token valid (@%s)", username)
            else:
                log.error("❌ Telegram bot token invalid (HTTP %s)", resp.status)
    except Exception as e:
        redacted = str(e).replace(config.TELEGRAM_BOT_TOKEN, "<TELEGRAM_BOT_TOKEN>")
        log.error("❌ Telegram check failed: %s", redacted)

    log.info("=== STARTUP SELF-CHECK COMPLETE ===")


async def _inject_synthetic(queue: asyncio.Queue[TriggerEvent], spec: list[str]) -> None:
    ticker_raw, price_s, move_s, vol_s = spec
    ticker = ticker_raw.upper().strip()
    if not ticker or not all(c.isalpha() or c == "." for c in ticker) or len(ticker) > 6:
        raise ValueError(f"--inject-trigger: invalid ticker {ticker_raw!r}")
    try:
        price = float(price_s)
        move = float(move_s)
        vol = float(vol_s)
    except ValueError as exc:
        raise ValueError(
            f"--inject-trigger: PRICE/MOVE_PCT/VOL_RATIO must be numeric ({exc})"
        ) from exc
    if price <= 0:
        raise ValueError(f"--inject-trigger: PRICE must be positive (got {price})")
    if not -0.99 < move < 5.0:
        raise ValueError(f"--inject-trigger: MOVE_PCT must be in (-0.99, 5.0) (got {move})")
    if vol < 0:
        raise ValueError(f"--inject-trigger: VOL_RATIO must be non-negative (got {vol})")
    await queue.put(TriggerEvent(
        ticker=ticker,
        price=price,
        price_move_pct=move,
        volume_ratio=vol,
        cumulative_volume=0,
        expected_volume=0.0,
        triggered_at=datetime.now(timezone.utc),
        window_high=price,
        window_low=price * (1 - move),
    ))
    log.info("synthetic trigger injected: %s", ticker)


async def main() -> int:
    args = _parse_args()
    setup_logging(config.LOG_DIR)
    log.info("starting trading bot (paper=%s)", config.ALPACA_PAPER)

    # --- Alpaca clients ---
    trading_client = TradingClient(
        config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER
    )
    hist_client = StockHistoricalDataClient(
        config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
    )

    # --- Watchlist + ADV baseline ---
    base_tickers = load_watchlist(config.WATCHLIST_PATH)
    catalyst_path = config.PROJECT_ROOT / "watchlist_catalyst.txt"
    catalyst_tickers: set[str] = set()
    if catalyst_path.exists():
        catalyst_tickers = set(load_watchlist(catalyst_path))
    else:
        log.warning(
            "catalyst watchlist %s not found; run build_catalyst_watchlist.py to generate",
            catalyst_path,
        )
    watchlist = sorted(set(base_tickers) | catalyst_tickers)
    log.info(
        "watchlist loaded: combined=%d base=%d catalyst=%d overlap=%d",
        len(watchlist),
        len(base_tickers),
        len(catalyst_tickers),
        len(set(base_tickers) & catalyst_tickers),
    )
    adv, prev_close = await asyncio.gather(
        prefetch_adv(hist_client, watchlist),
        prefetch_prev_close(hist_client, watchlist),
    )
    earnings_calendar = EarningsCalendar()

    # --- Peak equity / trading halt lock check ---
    if config.TRADING_HALT_LOCK.exists():
        log.critical(
            "🔒 state/trading_halted.lock exists — peak drawdown limit was breached. "
            "Delete %s to resume trading.",
            config.TRADING_HALT_LOCK,
        )
        return 1

    # --- Shared state ---
    stats = SessionStats()
    try:
        _acct = await asyncio.to_thread(trading_client.get_account)
        stats.opening_equity = float(_acct.equity)
        log.info("opening equity set to $%.2f (daily loss limit: -$%.2f)",
                 stats.opening_equity, stats.opening_equity * config.DAILY_LOSS_LIMIT_PCT)
    except Exception:
        log.warning("could not fetch opening equity; daily loss circuit breaker inactive")

    # --- Peak equity drawdown check ---
    if stats.opening_equity > 0:
        try:
            peak_eq = stats.opening_equity
            peak_path = config.PEAK_EQUITY_FILE

            def _read_peak() -> float:
                if not peak_path.exists():
                    return stats.opening_equity
                stored = json.loads(peak_path.read_text(encoding="utf-8"))
                return float(stored.get("peak_equity", stats.opening_equity))

            peak_eq = await asyncio.to_thread(_read_peak)

            if stats.opening_equity > peak_eq:
                peak_eq = stats.opening_equity

                def _write_peak() -> None:
                    peak_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = peak_path.with_name(peak_path.name + ".tmp")
                    tmp.write_text(
                        json.dumps({"peak_equity": round(peak_eq, 2)}),
                        encoding="utf-8",
                    )
                    tmp.replace(peak_path)  # atomic rename

                await asyncio.to_thread(_write_peak)
                log.info("peak equity updated to $%.2f", peak_eq)

            drawdown = (peak_eq - stats.opening_equity) / peak_eq
            if drawdown >= config.PEAK_DRAWDOWN_PCT:
                config.TRADING_HALT_LOCK.parent.mkdir(parents=True, exist_ok=True)
                config.TRADING_HALT_LOCK.touch()
                msg = (
                    f"🔒 Peak DD >{config.PEAK_DRAWDOWN_PCT:.0%} — trading locked. "
                    f"Delete state/trading_halted.lock to resume."
                )
                log.critical(msg)
                # Attempt to notify via Telegram before exiting.
                try:
                    _tg_early = TelegramHandler(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
                    await _tg_early.start()
                    await _tg_early.send_message(msg)
                    await _tg_early.stop()
                except Exception:
                    pass
                return 1
        except Exception:
            log.warning("peak equity check failed; proceeding without lock check", exc_info=True)

    # Set week_start_equity if not already set (e.g. first run or Monday).
    now_et = datetime.now(tz=config.MARKET_TZ)
    if stats.week_start_equity == 0.0 and stats.opening_equity > 0:
        stats.week_start_equity = stats.opening_equity

    calendar = MarketCalendar(trading_client)
    scorer = Scorer(hist_client=hist_client)

    # Load persisted monthly Gemini cost from performance.json if same month.
    try:
        if config.PERFORMANCE_FILE.exists():
            perf_raw = await asyncio.to_thread(
                config.PERFORMANCE_FILE.read_text, "utf-8",
            )
            perf_data = json.loads(perf_raw)
            stored_month = perf_data.get("monthly_cost_month")
            stored_cost = perf_data.get("monthly_cost_estimate")
            current_month = datetime.now(tz=config.MARKET_TZ).strftime("%Y-%m")
            if stored_month == current_month and isinstance(stored_cost, (int, float)):
                scorer.monthly_cost_estimate = float(stored_cost)
                log.info(
                    "loaded persisted Gemini monthly cost: $%.2f for %s",
                    scorer.monthly_cost_estimate, current_month,
                )
                if scorer.monthly_cost_estimate >= config.MONTHLY_GEMINI_BUDGET_USD:
                    scorer._budget_mode = "halted"
                elif scorer.monthly_cost_estimate >= config.MONTHLY_GEMINI_BUDGET_USD * 0.8:
                    scorer._budget_mode = "ungrounded_only"
    except Exception:
        log.warning("failed to load persisted monthly Gemini cost", exc_info=True)

    telegram = TelegramHandler(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    cooldown_store = CooldownStore()
    await cooldown_store.load()

    # Wire Gemini budget alerts to Telegram. Scorer fires this from
    # _record_call_cost when budget mode transitions (80% / 100%).
    def _budget_alert(msg: str) -> None:
        asyncio.create_task(telegram.send_message(msg))
    scorer._alert_callback = _budget_alert

    regime_detector = MarketRegimeDetector(hist_client=hist_client)
    trader = Trader(
        trading_client, telegram, stats,
        cooldown_store=cooldown_store, hist_client=hist_client,
        regime_detector=regime_detector,
    )
    telegram.set_context(
        trader=trader,
        stats=stats,
        watchlist_info={
            "base": len(base_tickers),
            "catalyst": len(catalyst_tickers),
            "adv_prefetched": len(adv),
            "ws_subscribed": len(watchlist),
        },
    )

    trigger_queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
    stop_event = asyncio.Event()
    intraday_gappers: set[str] = set()
    dynamic_sub_queue: asyncio.Queue[str] = asyncio.Queue()

    # --- Signal handling ---
    loop = asyncio.get_running_loop()

    def _request_stop(signame: str) -> None:
        log.info("%s received; initiating shutdown", signame)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            pass

    await telegram.start()
    await telegram.send_message(
        f"🟢 Bot online at {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z "
        f"({len(watchlist)} tickers: {len(base_tickers)} base + "
        f"{len(catalyst_tickers)} catalyst, {len(adv)} with ADV baseline)"
    )

    if args.inject_trigger:
        await _inject_synthetic(trigger_queue, args.inject_trigger)

    async with aiohttp.ClientSession() as http_session:
        await _startup_checks(
            trading_client, http_session, scorer,
            config.WATCHLIST_PATH, len(watchlist),
        )
        await trader.sync_open_positions()
        await earnings_calendar.load(http_session)
        tasks = [
            run_stream(
                watchlist, adv, calendar, trigger_queue, stop_event,
                catalyst_tickers, earnings_calendar,
                intraday_gappers, dynamic_sub_queue,
                prev_close=prev_close,
            ),
            earnings_calendar.run_refresh(stop_event, http_session),
            _supervise(
                "pipeline_consumer",
                lambda: pipeline_consumer(
                    trigger_queue, scorer, telegram, trader, stats,
                    http_session, stop_event, cooldown_store
                ),
                stop_event,
            ),
            _supervise(
                "monitor_positions",
                lambda: trader.monitor_positions(stop_event),
                stop_event,
            ),
            trader.run_trading_stream(stop_event),
            _supervise(
                "run_eod_close",
                lambda: trader.run_eod_close(stop_event),
                stop_event,
            ),
            _supervise(
                "run_daily_summary",
                lambda: run_daily_summary(
                    stats, telegram, trading_client, scorer, stop_event, trader
                ),
                stop_event,
            ),
            run_heartbeat(stats, trader, scorer, stop_event),
            _supervise(
                "run_position_monitor",
                lambda: run_position_monitor(
                    trader, scorer, telegram, http_session, calendar, stats, stop_event
                ),
                stop_event,
            ),
        ]
        if config.MICROCAP_ENABLED:
            tasks.append(run_microcap_screener(
                trading_client=trading_client,
                hist_client=hist_client,
                http_session=http_session,
                scorer=scorer,
                telegram=telegram,
                trader=trader,
                calendar=calendar,
                watchlist_path=config.WATCHLIST_PATH,
                stop_event=stop_event,
                stats=stats,
                cooldown_store=cooldown_store,
                intraday_gappers=intraday_gappers,
                dynamic_sub_queue=dynamic_sub_queue,
                shared_adv=adv,
            ))
        else:
            log.info("[MICROCAP] disabled via config; not scheduling task")
        tasks.append(_supervise(
            "run_premarket_scanner",
            lambda: run_premarket_scanner(
                hist_client, http_session, adv,
                intraday_gappers, dynamic_sub_queue, stop_event,
            ),
            stop_event,
        ))
        tasks.append(_supervise(
            "run_regime_refresher",
            lambda: run_regime_refresher(regime_detector, scorer, stop_event),
            stop_event,
        ))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await trader._save_positions_snapshot()
                log.info("shutdown: positions snapshot saved")
            except Exception:
                log.exception("shutdown: failed to save positions snapshot")
            await telegram.send_message("🔴 Bot shutting down.")
            await telegram.stop()
    log.info("Bot shutdown complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
