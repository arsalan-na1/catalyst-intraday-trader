"""Daily earnings calendar.

Primary source: NASDAQ public API.
Fallback: Yahoo Finance earnings calendar page (HTML scrape via bs4).
If both sources fail, returns an empty frozenset — the bot keeps running
without the earnings fast-track threshold adjustment.

Background task refreshes at 8 AM ET each day.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import aiohttp

log = logging.getLogger("earnings_calendar")

_MARKET_TZ = ZoneInfo("America/New_York")
_NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings"
_YAHOO_URL = "https://finance.yahoo.com/calendar/earnings"

_NASDAQ_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
}
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Valid ticker pattern: 1–6 uppercase alpha chars (or with dot for BRK.B style)
_TICKER_RE = re.compile(r'\b([A-Z]{1,6})\b')


class EarningsCalendar:
    def __init__(self) -> None:
        self._date: date | None = None
        self._tickers: frozenset[str] = frozenset()

    def get_tickers(self) -> frozenset[str]:
        return self._tickers

    async def load(self, session: aiohttp.ClientSession) -> None:
        today = datetime.now(tz=_MARKET_TZ).date()
        tickers = await _fetch_for_date(today, session)
        self._date = today
        self._tickers = tickers
        log.info("[EARNINGS] loaded %d tickers reporting today (%s)", len(tickers), today)

    async def run_refresh(
        self, stop_event: asyncio.Event, session: aiohttp.ClientSession
    ) -> None:
        """Background task: refresh at 8 AM ET each day."""
        while not stop_event.is_set():
            now = datetime.now(tz=_MARKET_TZ)
            today = now.date()
            if self._date != today:
                try:
                    await self.load(session)
                except Exception:
                    log.exception("[EARNINGS] calendar load failed; will retry at next 8 AM")
                    self._date = today  # prevent tight retry loop
            next_8am = datetime.combine(today + timedelta(days=1), time(8, 0), tzinfo=_MARKET_TZ)
            sleep_secs = max(1.0, (next_8am - datetime.now(tz=_MARKET_TZ)).total_seconds())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_secs)
                return
            except asyncio.TimeoutError:
                pass


async def _fetch_nasdaq(target: date, session: aiohttp.ClientSession) -> frozenset[str]:
    date_str = target.strftime("%Y-%m-%d")
    try:
        async with session.get(
            _NASDAQ_URL,
            params={"date": date_str},
            headers=_NASDAQ_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning(
                    "NASDAQ earnings HTTP %s for %s: %s", resp.status, date_str, body[:200]
                )
                return frozenset()
            data = await resp.json(content_type=None)
    except Exception:
        log.exception("NASDAQ earnings fetch failed for %s", date_str)
        return frozenset()

    rows: list = []
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, dict):
            rows = inner.get("rows") or []

    tickers: set[str] = set()
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        if symbol:
            tickers.add(symbol)
    return frozenset(tickers)


async def _fetch_yahoo(target: date, session: aiohttp.ClientSession) -> frozenset[str]:
    """Fallback: scrape Yahoo Finance earnings calendar page."""
    date_str = target.strftime("%Y-%m-%d")
    try:
        async with session.get(
            _YAHOO_URL,
            params={"day": date_str},
            headers=_YAHOO_HEADERS,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status != 200:
                log.warning("Yahoo earnings calendar HTTP %s for %s", resp.status, date_str)
                return frozenset()
            html = await resp.text()
    except Exception:
        log.exception("Yahoo earnings calendar fetch failed for %s", date_str)
        return frozenset()

    tickers: set[str] = set()

    # Try JSON embedded in HTML (Yahoo embeds data in <script> tags)
    for match in re.finditer(r'"symbol"\s*:\s*"([A-Z\.]{1,6})"', html):
        sym = match.group(1).strip(".")
        if 1 <= len(sym) <= 6:
            tickers.add(sym)

    # Try data attributes used in table rows
    for match in re.finditer(r'data-symbol="([A-Z\.]{1,6})"', html):
        sym = match.group(1).strip(".")
        if 1 <= len(sym) <= 6:
            tickers.add(sym)

    # Try quote links: /quote/TICKER/
    for match in re.finditer(r'/quote/([A-Z\.]{1,6})/', html):
        sym = match.group(1).strip(".")
        if 1 <= len(sym) <= 6:
            tickers.add(sym)

    # Filter out obvious false positives (single-letter HTML tokens, etc.)
    noise = {"A", "I", "P", "S", "T", "TD", "TR", "TH", "DIV", "NAV", "US", "ET"}
    tickers -= noise

    log.info(
        "[EARNINGS] Yahoo fallback: %d tickers for %s", len(tickers), date_str
    )
    return frozenset(tickers)


async def _fetch_for_date(target: date, session: aiohttp.ClientSession) -> frozenset[str]:
    """Try NASDAQ first; fall back to Yahoo Finance if result is empty."""
    tickers = await _fetch_nasdaq(target, session)
    if tickers:
        return tickers
    log.info("[EARNINGS] NASDAQ returned empty; trying Yahoo Finance fallback")
    return await _fetch_yahoo(target, session)
