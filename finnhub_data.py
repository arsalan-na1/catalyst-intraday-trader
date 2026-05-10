"""Finnhub alternative data: insider sentiment, earnings calendar, congressional
trades, and earnings surprise history.

All Finnhub calls are synchronous (requests-based), so each runs via
asyncio.to_thread. Results are cached per ticker for 6 hours — this data
changes at most daily. Returns {} on missing API key, timeout, or any error
(small tickers often have no Finnhub coverage at all).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

import config

log = logging.getLogger("finnhub_data")

_CACHE_SECONDS = 6 * 3600
_TIMEOUT_SECONDS = 15.0

_insider_cache:  dict[str, tuple[float, dict]] = {}
_earnings_cal_cache: dict[str, tuple[float, dict]] = {}
_congress_cache: dict[str, tuple[float, dict]] = {}
_earnings_surp_cache: dict[str, tuple[float, dict]] = {}
_insider_score_cache: dict[str, tuple[float, dict | None]] = {}

# Once Finnhub returns 403 for the congressional-trades endpoint we know it is
# not available on the current plan. Latch the flag for the rest of the session
# so we stop burning rate-limit slots and stop spamming warnings.
_congressional_403: bool = False


def _client():
    import finnhub  # lazy import — only used when key is set
    return finnhub.Client(api_key=config.FINNHUB_API_KEY)


# ---------------------------------------------------------------------------
# 1. Insider sentiment (MSPR)
# ---------------------------------------------------------------------------

def _fetch_insider_sentiment_sync(ticker: str) -> dict:
    from_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    to_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = _client().stock_insider_sentiment(ticker, from_date, to_date)
    records = data.get("data") or []
    if not records:
        return {}
    mspr_values = [r["mspr"] for r in records if "mspr" in r]
    net_change  = sum(r.get("change", 0) for r in records)
    avg_mspr    = sum(mspr_values) / len(mspr_values) if mspr_values else 0.0
    return {
        "mspr": round(avg_mspr, 1),
        "net_change": round(net_change),
    }


async def get_insider_sentiment(ticker: str) -> dict:
    """MSPR score (-100–+100, avg over 90 days) and net share change."""
    if not config.FINNHUB_API_KEY:
        return {}
    now = time.monotonic()
    cached = _insider_cache.get(ticker)
    if cached and now - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_insider_sentiment_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("finnhub insider sentiment timed out for %s", ticker)
        return {}
    except Exception:
        log.warning("finnhub insider sentiment failed for %s", ticker, exc_info=True)
        return {}
    _insider_cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _insider_cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _insider_cache[k]
    return data


# ---------------------------------------------------------------------------
# 2. Earnings calendar (next date + imminent flag)
# ---------------------------------------------------------------------------

def _fetch_earnings_calendar_sync(ticker: str) -> dict:
    from_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    to_date   = (datetime.now(timezone.utc) + timedelta(days=90)).strftime("%Y-%m-%d")
    data = _client().earnings_calendar(
        _from=from_date, to=to_date, symbol=ticker, international=False
    )
    calendar = data.get("earningsCalendar") or []
    if not calendar:
        return {}
    earliest = min(calendar, key=lambda e: e.get("date", "9999-99-99"))
    next_date_str = earliest.get("date")
    if not next_date_str:
        return {}
    try:
        next_date = datetime.strptime(next_date_str, "%Y-%m-%d").date()
        days_until = (next_date - date.today()).days
    except ValueError:
        return {}
    return {
        "next_earnings_date": next_date_str,
        "earnings_imminent": days_until <= 7,
        "days_until_earnings": days_until,
    }


async def get_earnings_calendar(ticker: str) -> dict:
    """Next scheduled earnings date and whether it falls within the next 7 days."""
    if not config.FINNHUB_API_KEY:
        return {}
    now = time.monotonic()
    cached = _earnings_cal_cache.get(ticker)
    if cached and now - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_earnings_calendar_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("finnhub earnings calendar timed out for %s", ticker)
        return {}
    except Exception:
        log.warning("finnhub earnings calendar failed for %s", ticker, exc_info=True)
        return {}
    _earnings_cal_cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _earnings_cal_cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _earnings_cal_cache[k]
    return data


# ---------------------------------------------------------------------------
# 3. Congressional trades (last 90 days)
# ---------------------------------------------------------------------------

def _fetch_congressional_trades_sync(ticker: str) -> dict:
    from_date = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    to_date   = date.today().strftime("%Y-%m-%d")
    data = _client().congressional_trading(ticker, from_date, to_date)
    trades = data.get("data") or []
    if not trades:
        return {"trades": [], "count": 0}
    recent: list[dict] = []
    for trade in trades:
        recent.append({
            "name": trade.get("name") or trade.get("representative", ""),
            "transaction_type": trade.get("transactionType", ""),
            "amount": trade.get("amount", ""),
            "date": trade.get("txDate", ""),
        })
    return {"trades": recent[:10], "count": len(recent)}


async def get_congressional_trades(ticker: str) -> dict:
    """Congressional buy/sell trades in the last 90 days."""
    global _congressional_403
    if not config.FINNHUB_API_KEY or _congressional_403:
        return {}
    now = time.monotonic()
    cached = _congress_cache.get(ticker)
    if cached and now - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_congressional_trades_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("finnhub congressional trades timed out for %s", ticker)
        return {}
    except Exception as e:
        # FinnhubAPIException carries .status_code on HTTP errors. Use getattr
        # so the check still works if the library wraps the response differently.
        if getattr(e, "status_code", None) == 403:
            _congressional_403 = True
            log.warning(
                "[FINNHUB] Congressional trades endpoint returned 403 — "
                "not available on this plan. Disabling for session."
            )
            return {}
        log.warning("finnhub congressional trades failed for %s", ticker, exc_info=True)
        return {}
    _congress_cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _congress_cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _congress_cache[k]
    return data


# ---------------------------------------------------------------------------
# 4. Earnings surprise (last 4 quarters)
# ---------------------------------------------------------------------------

def _fetch_earnings_surprise_sync(ticker: str) -> dict:
    records = _client().company_earnings(ticker, limit=4)
    if not records or not isinstance(records, list):
        return {}
    beats = sum(
        1 for r in records
        if r.get("actual") is not None
        and r.get("estimate") is not None
        and r["actual"] > r["estimate"]
    )
    total = len(records)
    return {"beat_rate": f"{beats}/{total}", "beats": beats, "total": total}


async def get_earnings_surprise(ticker: str) -> dict:
    """Beat rate string (e.g. '3/4') for the last 4 reported quarters."""
    if not config.FINNHUB_API_KEY:
        return {}
    now = time.monotonic()
    cached = _earnings_surp_cache.get(ticker)
    if cached and now - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_earnings_surprise_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("finnhub earnings surprise timed out for %s", ticker)
        return {}
    except Exception:
        log.warning("finnhub earnings surprise failed for %s", ticker, exc_info=True)
        return {}
    _earnings_surp_cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _earnings_surp_cache.items() if now - t >= _CACHE_SECONDS]
    for k in stale:
        del _earnings_surp_cache[k]
    return data


# ---------------------------------------------------------------------------
# 5. Insider score (individual buy/sell transactions, last 90 days)
# ---------------------------------------------------------------------------

def _fetch_insider_score_sync(ticker: str) -> dict | None:
    from_date = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    to_date = date.today().strftime("%Y-%m-%d")
    # Note: the finnhub Python SDK uses _from (not from_date) because 'from' is
    # a Python keyword.
    data = _client().stock_insider_transactions(ticker, _from=from_date, to=to_date)
    records = data.get("data") or []
    if not records:
        return None

    # Only include non-derivative open-market purchases and sales.
    transactions = [
        r for r in records
        if not r.get("isDerivative", False)
        and r.get("transactionCode") in ("P", "S")
    ]
    if not transactions:
        return None

    net_shares = sum(r.get("change", 0) for r in transactions)
    transaction_count = len(transactions)

    buyers = [
        (r.get("change", 0), r.get("name", ""))
        for r in transactions
        if r.get("change", 0) > 0
    ]
    largest_buyer: str | None = max(buyers, key=lambda x: x[0])[1] if buyers else None

    if net_shares > 10_000 and transaction_count >= 2:
        insider_signal = "bullish"
    elif net_shares < -50_000:
        insider_signal = "bearish"
    else:
        insider_signal = "neutral"

    return {
        "net_shares_3m": net_shares,
        "transaction_count": transaction_count,
        "insider_signal": insider_signal,
        "largest_buyer": largest_buyer,
    }


async def get_insider_score(ticker: str) -> dict | None:
    """Individual insider buy/sell transactions over the last 90 days."""
    if not config.FINNHUB_API_KEY:
        return None

    now = time.monotonic()
    cache_secs = config.INSIDER_CACHE_HOURS * 3600
    cached = _insider_score_cache.get(ticker)
    if cached and now - cached[0] < cache_secs:
        return cached[1]

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_insider_score_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("finnhub insider score timed out for %s", ticker)
        return None
    except Exception:
        log.warning("finnhub insider score failed for %s", ticker, exc_info=True)
        return None

    _insider_score_cache[ticker] = (now, data)
    stale = [k for k, (t, _) in _insider_score_cache.items() if now - t >= cache_secs]
    for k in stale:
        del _insider_score_cache[k]
    return data
