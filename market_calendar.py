"""Market calendar wrapper.

Single source of truth for "is the US equity market open right now?". Uses
Alpaca's `/clock` and `/calendar` endpoints so we automatically get holidays
and half-days correct — zoneinfo alone only handles time zones, not holidays.

We cache the clock for a few seconds to avoid hammering the API in hot loops
(the stream's trigger check fires on every bar).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient

import config

log = logging.getLogger("market_calendar")

_CLOCK_CACHE_SECONDS = 5


@dataclass
class _ClockSnapshot:
    is_open: bool
    next_open: datetime
    next_close: datetime
    fetched_at: float


class MarketCalendar:
    """Cached wrapper around Alpaca's /clock.

    Uses a threading.Lock (not asyncio.Lock) so it's safe to call from
    multiple asyncio loops — the Alpaca stream runs its own loop in a
    worker thread, and this object is shared across loops.
    """

    def __init__(self, trading_client: TradingClient) -> None:
        self._client = trading_client
        self._cache: _ClockSnapshot | None = None
        self._lock = threading.Lock()

    async def _refresh(self) -> _ClockSnapshot:
        clock = await asyncio.to_thread(self._client.get_clock)
        snap = _ClockSnapshot(
            is_open=bool(clock.is_open),
            next_open=clock.next_open,
            next_close=clock.next_close,
            fetched_at=time.monotonic(),
        )
        with self._lock:
            self._cache = snap
        return snap

    async def _get(self) -> _ClockSnapshot:
        with self._lock:
            cache = self._cache
            if cache is not None and time.monotonic() - cache.fetched_at < _CLOCK_CACHE_SECONDS:
                return cache
        return await self._refresh()

    async def is_market_open(self) -> bool:
        return (await self._get()).is_open

    async def seconds_until_next_close(self) -> float:
        snap = await self._get()
        if not snap.is_open:
            return 0.0
        now = datetime.now(tz=config.MARKET_TZ)
        return max(0.0, (snap.next_close - now).total_seconds())

    async def seconds_until_next_open(self) -> float:
        snap = await self._get()
        if snap.is_open:
            return 0.0
        now = datetime.now(tz=config.MARKET_TZ)
        return max(0.0, (snap.next_open - now).total_seconds())

    async def minutes_since_open(self) -> float:
        """How many minutes of regular session have elapsed so far today.

        0 pre-open, 390 post-close, linearly scaled while open. Used by the
        volume trigger to compute expected-so-far volume.
        """
        snap = await self._get()
        if not snap.is_open:
            return 0.0
        # next_close is today's close; session_open = next_close - 6.5h.
        session_open = snap.next_close - timedelta(minutes=config.REGULAR_SESSION_MINUTES)
        now = datetime.now(tz=config.MARKET_TZ)
        elapsed_min = (now - session_open).total_seconds() / 60.0
        return max(0.0, min(float(config.REGULAR_SESSION_MINUTES), elapsed_min))


async def _smoke() -> None:
    client = TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=config.ALPACA_PAPER,
    )
    cal = MarketCalendar(client)
    print("is_open:", await cal.is_market_open())
    print("minutes_since_open:", await cal.minutes_since_open())
    print("seconds_until_next_close:", await cal.seconds_until_next_close())
    print("seconds_until_next_open:", await cal.seconds_until_next_open())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke())
