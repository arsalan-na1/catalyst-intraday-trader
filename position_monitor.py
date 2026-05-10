"""Continuous Gemini re-evaluation of open positions.

Wakes every POSITION_EVAL_INTERVAL_MINUTES and re-scores each open position
with fresh news and technicals. Possible outcomes per verdict:

  hold        → nothing logged beyond DEBUG
  exit        → close via trader._close_position("gemini_exit — <reason>")
  raise_target → update pos.take_profit_pct upward; persist snapshot

A position younger than POSITION_EVAL_MIN_HOLD_MINUTES is skipped so the
initial catalyst move has time to play out before Gemini second-guesses it.
One bad evaluation never kills the loop — each position is wrapped in try/except.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

import config
from daily_summary import SessionStats
from market_calendar import MarketCalendar
from news import fetch_news
from scorer import Scorer, TriggerContext
from technicals import get_technicals
from telegram_handler import TelegramHandler
from trader import Trader

log = logging.getLogger("position_monitor")


async def run_position_monitor(
    trader: Trader,
    scorer: Scorer,
    telegram: TelegramHandler,
    http_session: aiohttp.ClientSession,
    calendar: MarketCalendar,
    stats: SessionStats,
    stop_event: asyncio.Event,
) -> None:
    log.info(
        "position monitor starting (eval every %dmin, skip positions < %dmin old)",
        config.POSITION_EVAL_INTERVAL_MINUTES,
        config.POSITION_EVAL_MIN_HOLD_MINUTES,
    )
    # Opt 2: fingerprint of last-seen headlines per ticker; skip Gemini when unchanged.
    _news_fingerprints: dict[str, frozenset] = {}

    # Inject references onto the Trader so _monitor_tick can invoke a
    # timeout-driven Gemini re-look instead of an unconditional close.
    trader._monitor_scorer = scorer
    trader._monitor_telegram = telegram
    trader._monitor_http_session = http_session
    trader._monitor_news_fingerprints = _news_fingerprints
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=config.POSITION_EVAL_INTERVAL_MINUTES * 60,
            )
            return
        except asyncio.TimeoutError:
            pass

        if stats.halt_new_entries:
            log.debug("position monitor: circuit breaker active; skipping cycle")
            continue

        if not await calendar.is_market_open():
            log.debug("position monitor: market closed; skipping cycle")
            continue

        tickers = list(trader._positions.keys())
        if not tickers:
            log.debug("position monitor: no open positions")
            continue

        log.info(
            "position monitor: evaluating %d position(s): %s",
            len(tickers), tickers,
        )
        for ticker in tickers:
            try:
                await _evaluate_position(ticker, trader, scorer, telegram, http_session,
                                         _news_fingerprints)
            except Exception:
                log.exception("position monitor: evaluation failed for %s", ticker)


async def _evaluate_position(
    ticker: str,
    trader: Trader,
    scorer: Scorer,
    telegram: TelegramHandler,
    http_session: aiohttp.ClientSession,
    news_fingerprints: dict[str, frozenset] | None = None,
) -> None:
    pos = trader._positions.get(ticker)
    if pos is None:
        return

    # Skip positions younger than the minimum hold window.
    now_utc = datetime.now(timezone.utc)
    opened_utc = (
        pos.opened_at.astimezone(timezone.utc)
        if pos.opened_at.tzinfo
        else pos.opened_at.replace(tzinfo=timezone.utc)
    )
    hold_minutes = (now_utc - opened_utc).total_seconds() / 60.0
    if hold_minutes < config.POSITION_EVAL_MIN_HOLD_MINUTES:
        log.debug(
            "position monitor: %s only %.1fmin old (< %dmin); skipping",
            ticker, hold_minutes, config.POSITION_EVAL_MIN_HOLD_MINUTES,
        )
        return

    # Current price from the same Alpaca IEX snapshot endpoint used elsewhere.
    current_price = await trader._fetch_snapshot_price(ticker, http_session)
    if current_price is None:
        log.warning("position monitor: cannot fetch price for %s; skipping", ticker)
        return

    entry_price = pos.entry_price or current_price
    current_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # Fresh news (uses the existing 15-second in-process cache from news.py).
    try:
        fresh_news = await fetch_news(ticker, session=http_session)
    except Exception:
        log.warning("position monitor: news fetch failed for %s; proceeding with empty list", ticker)
        fresh_news = []

    # Force a Gemini look whenever the position is bleeding past the
    # FORCED_REEVAL_LOSS_PCT threshold, regardless of whether headlines have
    # changed. A position that is actively losing money should always get a
    # fresh assessment.
    force_reeval = current_pnl_pct < -config.FORCED_REEVAL_LOSS_PCT
    if force_reeval:
        log.info(
            "[MONITOR] %s P&L %.2f%% below -%.1f%% threshold — "
            "forcing Gemini re-look regardless of news fingerprint",
            ticker, current_pnl_pct * 100,
            config.FORCED_REEVAL_LOSS_PCT * 100,
        )

    # Opt 2: skip Gemini call when headlines haven't changed since last check.
    # Always score on the FIRST check (no stored fingerprint yet) and on any
    # forced run.
    if not force_reeval and news_fingerprints is not None:
        new_fp = frozenset(item.headline for item in (fresh_news or [])[:5])
        stored_fp = news_fingerprints.get(ticker)
        if stored_fp is not None and new_fp == stored_fp:
            log.info("[MONITOR] %s — no new news, skipping Gemini re-look", ticker)
            return
        news_fingerprints[ticker] = new_fp
    elif news_fingerprints is not None:
        # Forced run: still update the fingerprint so the next non-forced
        # check has a baseline.
        new_fp = frozenset(item.headline for item in (fresh_news or [])[:5])
        news_fingerprints[ticker] = new_fp

    # Fresh technicals — reuse scorer's hist_client to avoid adding another dependency.
    fresh_tech = None
    if scorer._hist_client is not None:
        try:
            fresh_tech = await get_technicals(ticker, scorer._hist_client, current_price)
        except Exception:
            log.debug(
                "position monitor: technicals fetch failed for %s", ticker, exc_info=True
            )

    ctx = TriggerContext(
        ticker=ticker,
        price=current_price,
        price_move_pct=current_pnl_pct,
        volume_ratio=1.0,  # not meaningful during re-evaluation; field required by dataclass
    )

    verdict = await scorer.score_open_position(
        ctx=ctx,
        entry_price=entry_price,
        hold_minutes=hold_minutes,
        original_take_profit_pct=pos.take_profit_pct,
        fresh_news=fresh_news,
        fresh_technicals=fresh_tech,
        sl_floor=pos.sl_floor,
        max_hold_minutes=pos.max_hold_minutes,
        sector=pos.sector,
    )

    if verdict is None:
        log.warning("position monitor: no verdict returned for %s; skipping", ticker)
        return

    log.debug(
        "position monitor: %s → %s (conf=%d) — %s",
        ticker, verdict.action, verdict.confidence, verdict.reason,
    )

    if verdict.action == "hold":
        return

    if verdict.action == "exit":
        log.info(
            "position monitor: EXIT %s (conf=%d): %s",
            ticker, verdict.confidence, verdict.reason,
        )
        await telegram.send_position_update(
            ticker=ticker,
            action="exit",
            reason=verdict.reason,
            current_pnl_pct=current_pnl_pct,
        )
        # Guard: monitor_tick may have already closed it between score and here.
        if ticker in trader._positions:
            await trader._close_position(ticker, reason=f"gemini_exit — {verdict.reason}")
        return

    if verdict.action == "raise_target":
        new_tp = verdict.new_take_profit_pct
        if new_tp is None or new_tp <= pos.take_profit_pct:
            log.info(
                "position monitor: %s raise_target ignored — new_tp=%s not above current %.0f%%",
                ticker,
                f"{new_tp:.0%}" if new_tp is not None else "None",
                pos.take_profit_pct * 100,
            )
            return
        old_tp = pos.take_profit_pct
        pos.take_profit_pct = new_tp
        log.info(
            "position monitor: RAISE TARGET %s %.0f%% → %.0f%% (conf=%d): %s",
            ticker, old_tp * 100, new_tp * 100, verdict.confidence, verdict.reason,
        )
        await telegram.send_position_update(
            ticker=ticker,
            action="raise_target",
            reason=verdict.reason,
            current_pnl_pct=current_pnl_pct,
            new_tp_pct=new_tp,
        )
        await trader._save_positions_snapshot()
        return

    if verdict.action == "tighten_sl":
        new_floor = verdict.new_stop_loss_pct
        if new_floor is None or new_floor <= pos.sl_floor:
            log.info(
                "position monitor: %s tighten_sl ignored — new_floor=%s not above current %+.0f%%",
                ticker,
                f"{new_floor:+.0%}" if new_floor is not None else "None",
                pos.sl_floor * 100,
            )
            return
        old_floor = pos.sl_floor
        pos.sl_floor = new_floor
        log.info(
            "position monitor: TIGHTEN SL %s floor %+.0f%% → %+.0f%% (conf=%d): %s",
            ticker, old_floor * 100, new_floor * 100, verdict.confidence, verdict.reason,
        )
        await telegram.send_position_update(
            ticker=ticker,
            action="tighten_sl",
            reason=verdict.reason,
            current_pnl_pct=current_pnl_pct,
            new_sl_floor=new_floor,
        )
        await trader._save_positions_snapshot()
        return

    if verdict.action == "adjust_tp":
        new_tp = verdict.new_take_profit_pct
        if new_tp is None:
            log.info("position monitor: %s adjust_tp ignored — no new_take_profit_pct", ticker)
            return
        # Clamp to hard limits
        new_tp = max(config.GEMINI_TP_MIN, min(config.GEMINI_TP_MAX, new_tp))
        old_tp = pos.take_profit_pct
        pos.take_profit_pct = new_tp
        direction = "raised" if new_tp > old_tp else "lowered"
        log.info(
            "position monitor: ADJUST TP %s %s %.0f%% → %.0f%% (conf=%d): %s",
            ticker, direction, old_tp * 100, new_tp * 100, verdict.confidence, verdict.reason,
        )
        await telegram.send_position_update(
            ticker=ticker,
            action="adjust_tp",
            reason=verdict.reason,
            current_pnl_pct=current_pnl_pct,
            new_tp_pct=new_tp,
        )
        await trader._save_positions_snapshot()
        return

    if verdict.action == "add_time":
        add_mins = verdict.add_minutes
        if not add_mins or add_mins <= 0:
            log.info("position monitor: %s add_time ignored — no valid add_minutes", ticker)
            return
        old_max = pos.max_hold_minutes
        pos.max_hold_minutes = min(config.GEMINI_HOLD_MAX, pos.max_hold_minutes + add_mins)
        actual_added = pos.max_hold_minutes - old_max
        log.info(
            "position monitor: ADD TIME %s +%dmin → %dmin total (conf=%d): %s",
            ticker, actual_added, pos.max_hold_minutes, verdict.confidence, verdict.reason,
        )
        await telegram.send_position_update(
            ticker=ticker,
            action="add_time",
            reason=verdict.reason,
            current_pnl_pct=current_pnl_pct,
            add_minutes=actual_added,
        )
        await trader._save_positions_snapshot()


async def _timeout_reeval(
    ticker: str,
    trader: Trader,
    scorer: Scorer,
    telegram: TelegramHandler,
    http_session: aiohttp.ClientSession,
    news_fingerprints: dict[str, frozenset] | None,
) -> None:
    """Hold-limit expiry handler. Routes to _evaluate_position; on failure
    falls back to a profitable-only hard close."""
    try:
        await _evaluate_position(
            ticker, trader, scorer, telegram, http_session, news_fingerprints,
        )
    except Exception:
        log.exception(
            "[TIMEOUT REEVAL] %s position review failed; falling back", ticker,
        )
        pos = trader._positions.get(ticker)
        if pos is None:
            return
        try:
            live = await asyncio.to_thread(
                trader._trading.get_open_position, ticker,
            )
            plpc = float(live.unrealized_plpc)
        except Exception:
            plpc = 0.0
        if plpc >= 0.001:
            await trader._close_position(
                ticker,
                reason="timeout_fallback — review unavailable",
            )
