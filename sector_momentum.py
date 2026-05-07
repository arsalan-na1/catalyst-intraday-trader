"""Sector ETF momentum context for Gemini scoring.

Maps the stock's sector (from yfinance or the bot's internal sector names)
to the appropriate SPDR sector ETF, fetches today's % change via Alpaca,
and classifies momentum strength.

Cached for 5 minutes (intraday — ETF prices update continuously).
Returns None on error or unrecognised sector so the signal is omitted silently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

log = logging.getLogger("sector_momentum")

_SECTOR_ETF_MAP: dict[str, str] = {
    # yfinance sector names (lowercase)
    "technology": "XLK",
    "health care": "XLV",
    "healthcare": "XLV",
    "financial services": "XLF",
    "energy": "XLE",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "industrials": "XLI",
    "basic materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC",
    # bot's internal sector names
    "tech": "XLK",
    "biotech": "XLV",
    "financials": "XLF",
    "consumer": "XLY",
    "materials": "XLB",
    "communication": "XLC",
}

_CACHE_SECONDS = 5 * 60
_cache: dict[str, tuple[float, dict | None]] = {}


def _classify_momentum(change_pct: float) -> str:
    if change_pct > 1.5:
        return "strong_bull"
    if change_pct > 0.5:
        return "bull"
    if change_pct >= -0.5:
        return "neutral"
    if change_pct >= -1.5:
        return "bear"
    return "strong_bear"


def _fetch_sync(etf: str, hist_client: StockHistoricalDataClient) -> dict | None:
    now_et = datetime.now(tz=config.MARKET_TZ)
    req = StockBarsRequest(
        symbol_or_symbols=etf,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=now_et - timedelta(days=7),  # buffer for weekends/holidays
        feed=DataFeed.IEX,
    )
    raw = hist_client.get_stock_bars(req)
    df = getattr(raw, "df", None)
    if df is None or df.empty:
        return None

    if hasattr(df.index, "names") and "symbol" in (df.index.names or []):
        try:
            df = df.xs(etf, level="symbol")
        except KeyError:
            return None

    if len(df) < 2:
        return None

    prev_close = float(df["close"].iloc[-2])
    today_close = float(df["close"].iloc[-1])
    if prev_close <= 0:
        return None

    change_pct = (today_close - prev_close) / prev_close * 100.0
    return {
        "sector_etf": etf,
        "sector_change_pct": round(change_pct, 3),
        "sector_momentum": _classify_momentum(change_pct),
    }


async def get_sector_momentum(
    sector: str, hist_client: StockHistoricalDataClient | None
) -> dict | None:
    """Today's % change for the sector ETF proxy. Cached 5 minutes."""
    if hist_client is None or not sector:
        return None

    etf = _SECTOR_ETF_MAP.get(sector.lower().strip())
    if not etf:
        return None

    now = time.monotonic()
    cached = _cache.get(etf)
    if cached and now - cached[0] < _CACHE_SECONDS:
        return cached[1]

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync, etf, hist_client),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        log.debug("sector momentum timed out for %s (%s)", sector, etf)
        return None
    except Exception:
        log.debug("sector momentum failed for %s (%s)", sector, etf, exc_info=True)
        return None

    _cache[etf] = (now, data)
    stale = [k for k, (t, _) in _cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _cache[k]

    if data:
        log.info(
            "sector momentum %s (%s): %+.2f%% (%s)",
            sector, etf, data["sector_change_pct"], data["sector_momentum"],
        )
    return data
