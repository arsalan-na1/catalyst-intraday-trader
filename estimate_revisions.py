"""Analyst estimate revision score via Finnhub.

Uses recommendation_trends() for analyst sentiment momentum and
company_earnings() for EPS beat/miss history. Combines into a -2 to +2
composite score that Gemini uses to weight earnings catalyst strength.

Cached for ESTIMATE_REVISION_CACHE_HOURS (default 6h). Returns None on
missing API key or any error so the signal is omitted silently.
"""

from __future__ import annotations

import asyncio
import logging
import time

import config

log = logging.getLogger("estimate_revisions")

_TIMEOUT_SECONDS = 15.0
_cache: dict[str, tuple[float, dict | None]] = {}


def _client():
    import finnhub
    return finnhub.Client(api_key=config.FINNHUB_API_KEY)


def _fetch_sync(ticker: str) -> dict | None:
    c = _client()

    # --- 1. Analyst recommendation trend ---
    analyst_revision = "stable"
    try:
        trends = c.recommendation_trends(ticker)
        if trends and len(trends) >= 2:
            # trends[0] = most recent month, trends[1] = previous month
            recent_bull = trends[0].get("strongBuy", 0) + trends[0].get("buy", 0)
            prev_bull = trends[1].get("strongBuy", 0) + trends[1].get("buy", 0)
            if recent_bull > prev_bull:
                analyst_revision = "upgrading"
            elif recent_bull < prev_bull:
                analyst_revision = "downgrading"
    except Exception:
        log.debug("recommendation_trends failed for %s", ticker, exc_info=True)

    # --- 2. EPS surprise history (last 4 quarters) ---
    eps_surprise_avg = 0.0
    eps_trend = "stable"
    try:
        earnings = c.company_earnings(ticker, limit=4)
        if earnings and isinstance(earnings, list):
            surprises: list[float] = []
            for r in earnings:
                actual = r.get("actual")
                estimate = r.get("estimate")
                if actual is not None and estimate is not None and estimate != 0:
                    surprises.append((actual - estimate) / abs(estimate) * 100.0)

            if surprises:
                eps_surprise_avg = sum(surprises) / len(surprises)
                # Compare most recent vs average of older quarters for trend.
                # Finnhub returns newest-first; surprises[0] = latest quarter.
                if len(surprises) >= 2:
                    older_avg = sum(surprises[1:]) / len(surprises[1:])
                    if surprises[0] > older_avg + 2.0:
                        eps_trend = "accelerating"
                    elif surprises[0] < older_avg - 2.0:
                        eps_trend = "decelerating"
    except Exception:
        log.debug("company_earnings failed for %s", ticker, exc_info=True)

    # --- 3. Composite score (-2 to +2) ---
    score = 0
    if eps_surprise_avg > 5.0:
        score += 1
    elif eps_surprise_avg < -5.0:
        score -= 1
    if eps_trend == "accelerating":
        score += 1
    elif eps_trend == "decelerating":
        score -= 1
    if analyst_revision == "upgrading":
        score += 1
    elif analyst_revision == "downgrading":
        score -= 1

    return {
        "eps_surprise_avg": round(eps_surprise_avg, 2),
        "eps_trend": eps_trend,
        "analyst_revision": analyst_revision,
        "revision_score": max(-2, min(2, score)),
    }


async def get_estimate_revisions(ticker: str) -> dict | None:
    """Analyst revision score and EPS trend for ticker."""
    if not config.FINNHUB_API_KEY:
        return None

    now = time.monotonic()
    cache_secs = config.ESTIMATE_REVISION_CACHE_HOURS * 3600
    cached = _cache.get(ticker)
    if cached and now - cached[0] < cache_secs:
        return cached[1]

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("estimate revisions timed out for %s", ticker)
        return None
    except Exception:
        log.warning("estimate revisions failed for %s", ticker, exc_info=True)
        return None

    _cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _cache.items() if now - t >= cache_secs]
    for k in stale:
        del _cache[k]

    if data:
        log.info(
            "estimate revisions %s: eps_avg=%+.1f%% trend=%s analysts=%s score=%+d",
            ticker, data["eps_surprise_avg"], data["eps_trend"],
            data["analyst_revision"], data["revision_score"],
        )
    return data
