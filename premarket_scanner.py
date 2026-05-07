# VERIFIED 2026-05-06:
#   1. [SQUEEZE SETUP] log fires when squeeze_score == "high" AND
#      change_pct >= PREMARKET_GAP_PCT (gap check is the gate above the
#      short-interest probe — squeeze setup can never log without the gap).
#   2. Each gapper is added to both intraday_gappers (visible to
#      TriggerDetector via shared set) AND pushed onto dynamic_sub_queue
#      (so run_stream subscribes it mid-session without a reconnect).
#   3. ADV is prefetched via prefetch_adv and merged into the shared adv dict
#      so TriggerDetector has a volume baseline when the bars start arriving.
"""Premarket gap scanner — identifies stocks gapping up before market open.

Polls Alpaca's top movers endpoint every PREMARKET_POLL_SECONDS between
9:00 AM and 9:29 AM ET. For each gainer with a premarket gap ≥
PREMARKET_GAP_PCT, the ticker is:

  1. Added to intraday_gappers (shared set visible to TriggerDetector).
  2. Pushed onto dynamic_sub_queue so the stream subscribes it to the live
     WebSocket mid-session without waiting for the next reconnect.
  3. Added to the shared adv dict with its trailing-20-day average volume
     so the volume trigger has a baseline when bars arrive at market open.

Only runs during 9:00–9:29 AM ET. intraday_gappers is cleared at the start
of each new morning so yesterday's gappers don't carry over.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time as _time

import aiohttp
from alpaca.data.historical.stock import StockHistoricalDataClient

import config
from short_interest import get_short_interest
from stream import prefetch_adv

log = logging.getLogger("premarket_scanner")

_LOG_PREFIX = "[GAPSCANNER]"
_MOVERS_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
_PREMARKET_START = _time(9, 0, 0)
_PREMARKET_END = _time(9, 30, 0)


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }


def _valid_ticker(symbol: str) -> bool:
    if not symbol or len(symbol) > 6:
        return False
    if not all(c.isalpha() or c == "." for c in symbol):
        return False
    return symbol not in {"CASH", "USD", "XTSLA", "MARGIN_USD"}


async def _fetch_gainers(http_session: aiohttp.ClientSession) -> list[dict]:
    try:
        async with http_session.get(
            _MOVERS_URL,
            params={"top": "50"},
            headers=_alpaca_headers(),
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning("%s movers HTTP %s: %s", _LOG_PREFIX, resp.status, body[:200])
                return []
            payload = await resp.json()
    except Exception:
        log.exception("%s movers request failed", _LOG_PREFIX)
        return []
    return list(payload.get("gainers", []) or [])


async def run_premarket_scanner(
    hist_client: StockHistoricalDataClient,
    http_session: aiohttp.ClientSession,
    adv: dict[str, float],
    intraday_gappers: set[str],
    dynamic_sub_queue: "asyncio.Queue[str]",
    stop_event: asyncio.Event,
) -> None:
    """Poll Alpaca movers 9:00–9:29 AM ET and feed gap plays into the stream.

    Designed to be run as a supervised task alongside run_stream and
    microcap_screener. Mutations to adv, intraday_gappers, and
    dynamic_sub_queue are immediately visible to TriggerDetector because all
    are passed by reference from bot.py.
    """
    log.info(
        "%s scanner starting (gap_threshold=%.0f%% poll=%ss)",
        _LOG_PREFIX, config.PREMARKET_GAP_PCT, config.PREMARKET_POLL_SECONDS,
    )

    _cleared_session: date | None = None
    _eod_cleared_session: date | None = None
    _seen_today: set[str] = set()

    while not stop_event.is_set():
        now_et = datetime.now(tz=config.MARKET_TZ)
        today = now_et.date()
        t = now_et.time()

        # EOD reset: clear gappers when the session closes so yesterday's gap
        # plays don't bleed into the next pre-open window with a lower threshold.
        eod_time = _time(config.EOD_CLOSE_HOUR, config.EOD_CLOSE_MINUTE)
        if t >= eod_time and _eod_cleared_session != today:
            if intraday_gappers:
                log.info(
                    "%s EOD reset: clearing %d gapper(s) for session %s",
                    _LOG_PREFIX, len(intraday_gappers), today,
                )
            intraday_gappers.clear()
            _eod_cleared_session = today

        if _PREMARKET_START <= t < _PREMARKET_END:
            # Clear gappers and seen-set at the start of each new session.
            if _cleared_session != today:
                if intraday_gappers:
                    log.info(
                        "%s new session %s; clearing %d gapper(s) from yesterday",
                        _LOG_PREFIX, today, len(intraday_gappers),
                    )
                intraday_gappers.clear()
                _seen_today.clear()
                _cleared_session = today

            gainers = await _fetch_gainers(http_session)
            for g in gainers:
                symbol = (g.get("symbol") or "").upper().strip()
                try:
                    change_pct = float(g.get("change_percent") or 0.0)
                except (TypeError, ValueError):
                    log.debug("%s skipping %s — non-numeric change_percent",
                              _LOG_PREFIX, symbol)
                    continue

                if not symbol or symbol in _seen_today:
                    continue
                if not _valid_ticker(symbol):
                    continue
                if change_pct < config.PREMARKET_GAP_PCT:
                    continue

                _seen_today.add(symbol)
                intraday_gappers.add(symbol)
                await dynamic_sub_queue.put(symbol)

                # Log squeeze setup when short interest is high alongside the gap.
                try:
                    short_data = await get_short_interest(symbol)
                    if (short_data or {}).get("squeeze_score") == "high":
                        log.info(
                            "%s [SQUEEZE SETUP] %s — +%.1f%% premarket gap + high short float "
                            "(%.1f%% short, %.1fd to cover)",
                            _LOG_PREFIX, symbol, change_pct,
                            (short_data or {}).get("short_float_pct", 0.0),
                            (short_data or {}).get("days_to_cover", 0.0),
                        )
                except Exception:
                    log.debug("%s short interest check failed for %s", _LOG_PREFIX, symbol, exc_info=True)

                # Prefetch ADV so TriggerDetector has a volume baseline.
                if symbol not in adv:
                    try:
                        new_adv = await prefetch_adv(hist_client, [symbol])
                        adv_val = new_adv.get(symbol)
                        if adv_val:
                            adv[symbol] = adv_val
                            log.info(
                                "%s Added %s — +%.1f%% premarket gap  ADV=%.0f",
                                _LOG_PREFIX, symbol, change_pct, adv_val,
                            )
                        else:
                            log.info(
                                "%s Added %s — +%.1f%% premarket gap  (no ADV data)",
                                _LOG_PREFIX, symbol, change_pct,
                            )
                    except Exception:
                        log.exception("%s ADV prefetch failed for %s", _LOG_PREFIX, symbol)
                        log.info(
                            "%s Added %s — +%.1f%% premarket gap  (ADV fetch failed)",
                            _LOG_PREFIX, symbol, change_pct,
                        )
                else:
                    log.info(
                        "%s Added %s — +%.1f%% premarket gap  (ADV already known: %.0f)",
                        _LOG_PREFIX, symbol, change_pct, adv[symbol],
                    )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=config.PREMARKET_POLL_SECONDS)
            return
        except asyncio.TimeoutError:
            pass
