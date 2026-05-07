"""yfinance fundamentals cache for Gemini context enrichment.

Fetches static/slow-changing data (52w range, analyst targets, float, P/E,
next earnings date) and caches per ticker for 6 hours — fundamentals don't
change intraday so re-fetching on every trigger would be wasteful.

Do NOT use for real-time price or volume data — Alpaca WebSocket owns that.
Returns {} on timeout or any yfinance error (endpoints are unofficial and flaky).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("fundamentals")

_CACHE_SECONDS = 6 * 3600  # 6 hours
_TIMEOUT_SECONDS = 30.0

# ticker -> (cached_at monotonic seconds, data dict)
_cache: dict[str, tuple[float, dict]] = {}


def _fetch_sync(ticker: str) -> dict:
    """Blocking yfinance fetch — always run via asyncio.to_thread."""
    import yfinance as yf  # lazy import: startup safe if yfinance absent

    info = yf.Ticker(ticker).info
    if not isinstance(info, dict):
        return {}

    # earningsDate format varies across yfinance versions: list of unix
    # timestamps, a single timestamp, or absent entirely.
    earnings_date: str | None = None
    raw_ed = info.get("earningsDate") or info.get("earningsTimestamps")
    if isinstance(raw_ed, (list, tuple)) and raw_ed:
        try:
            ts = raw_ed[0]
            if isinstance(ts, (int, float)) and ts > 0:
                earnings_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    elif isinstance(raw_ed, (int, float)) and raw_ed > 0:
        try:
            earnings_date = datetime.fromtimestamp(raw_ed, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass

    return {
        "market_cap": info.get("marketCap"),
        "float_shares": info.get("floatShares"),
        "sector": info.get("sector"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "earnings_date": earnings_date,
        "analyst_target_price": info.get("targetMeanPrice"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
    }


async def get_fundamentals(ticker: str) -> dict:
    """Return cached fundamentals for ticker, fetching from yfinance if stale.

    Always returns a dict (empty on any failure). Callers should treat every
    field as optional — yfinance can return None for any of them.
    """
    now = time.monotonic()
    cached = _cache.get(ticker)
    if cached is not None:
        cached_at, data = cached
        if now - cached_at < _CACHE_SECONDS:
            log.debug("fundamentals cache hit for %s (age %.0fs)", ticker, now - cached_at)
            return data

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("fundamentals fetch timed out for %s (>%.0fs)", ticker, _TIMEOUT_SECONDS)
        return {}
    except Exception:
        log.warning("fundamentals fetch failed for %s", ticker, exc_info=True)
        return {}

    _cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _cache[k]
    log.info(
        "fundamentals fetched for %s — mcap=%s float=%s sector=%s pe=%s 52wH=%s 52wL=%s target=%s earningsDate=%s",
        ticker,
        data.get("market_cap"), data.get("float_shares"), data.get("sector"),
        data.get("pe_ratio"), data.get("52_week_high"), data.get("52_week_low"),
        data.get("analyst_target_price"), data.get("earnings_date"),
    )
    return data
