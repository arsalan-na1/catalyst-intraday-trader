"""Short interest data via yfinance.

Finnhub's Python SDK has no short-interest method (verified against the Pi's
installed package). yfinance provides the same data via info['sharesShort'],
info['floatShares'], and info['shortRatio'].

Cached for SHORT_INTEREST_CACHE_HOURS (default 4h) — FINRA data publishes
twice monthly so intraday re-fetches are always cache hits after the first.
Returns None on any error or missing data so callers can omit the signal
silently.
"""

from __future__ import annotations

import asyncio
import logging
import time

import config

log = logging.getLogger("short_interest")

_TIMEOUT_SECONDS = 20.0
_cache: dict[str, tuple[float, dict | None]] = {}


def _fetch_sync(ticker: str) -> dict | None:
    import yfinance as yf

    info = yf.Ticker(ticker).info
    if not isinstance(info, dict):
        return None

    float_shares = info.get("floatShares")
    shares_short = info.get("sharesShort")
    short_ratio = info.get("shortRatio")  # days to cover

    if not float_shares or not shares_short:
        return None

    # yfinance occasionally returns strings or odd types; coerce defensively.
    try:
        float_shares_f = float(float_shares)
        shares_short_f = float(shares_short)
        if float_shares_f <= 0 or shares_short_f <= 0:
            return None
        # shortPercentOfFloat is a decimal in yfinance (0.056 = 5.6%).
        raw_pct = info.get("shortPercentOfFloat")
        if raw_pct is not None:
            short_float_pct = float(raw_pct) * 100.0
        else:
            short_float_pct = shares_short_f / float_shares_f * 100.0
        days_to_cover = float(short_ratio) if short_ratio else 0.0
    except (TypeError, ValueError):
        return None

    if short_float_pct > 20.0 and days_to_cover > 5.0:
        squeeze_score = "high"
    elif short_float_pct > 10.0 or days_to_cover > 3.0:
        squeeze_score = "medium"
    else:
        squeeze_score = "low"

    return {
        "short_float_pct": round(short_float_pct, 2),
        "days_to_cover": round(days_to_cover, 2),
        "squeeze_score": squeeze_score,
    }


async def get_short_interest(ticker: str) -> dict | None:
    """Short float %, days to cover, and squeeze score (via yfinance)."""
    now = time.monotonic()
    cache_secs = config.SHORT_INTEREST_CACHE_HOURS * 3600
    cached = _cache.get(ticker)
    if cached and now - cached[0] < cache_secs:
        return cached[1]

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("short interest fetch timed out for %s", ticker)
        return None
    except Exception:
        log.warning("short interest fetch failed for %s", ticker, exc_info=True)
        return None

    _cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _cache.items() if now - t >= cache_secs]
    for k in stale:
        del _cache[k]

    if data:
        log.info(
            "short interest %s: %.1f%% float short | %.1fd to cover | squeeze=%s",
            ticker, data["short_float_pct"], data["days_to_cover"], data["squeeze_score"],
        )
    return data
