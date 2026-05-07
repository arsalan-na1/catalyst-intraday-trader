"""Persistent cooldown store — survives bot restarts.

Writes per-ticker last-alerted timestamps to a JSON file so the bot
doesn't re-alert for the same ticker after a crash/restart within the
cooldown window.

File: config.STATE_DIR / "cooldowns.json"
Format: {"TICKER": "2024-01-15T14:30:00+00:00", ...}

On load: prunes entries older than _MAX_AGE_HOURS to keep the file small.
On missing/corrupt file: starts fresh with a warning log — never crashes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

log = logging.getLogger("cooldown_store")

_MAX_AGE_HOURS = 1  # entries older than this are discarded on load


class CooldownStore:
    """Persistent per-ticker cooldown tracker backed by a JSON file.

    All mutation is async; file I/O is offloaded via asyncio.to_thread so
    the event loop is never blocked.  The in-memory dict is only accessed
    from the event loop, so no asyncio.Lock is needed.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (config.STATE_DIR / "cooldowns.json")
        self._store: dict[str, datetime] = {}

    async def load(self) -> None:
        """Load and prune the store from disk. Call once at startup."""
        try:
            text = await asyncio.to_thread(self._path.read_text, "utf-8")
            raw: dict[str, str] = json.loads(text)
        except FileNotFoundError:
            log.info("cooldown store: %s not found; starting fresh", self._path)
            return
        except Exception:
            log.warning(
                "cooldown store: %s corrupt or unreadable; starting fresh",
                self._path, exc_info=True,
            )
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=_MAX_AGE_HOURS)
        loaded = 0
        pruned = 0
        for ticker, ts_str in raw.items():
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    self._store[ticker] = ts
                    loaded += 1
                else:
                    pruned += 1
            except (ValueError, TypeError):
                pruned += 1

        log.info(
            "cooldown store loaded: %d active, %d pruned (>%dh old) from %s",
            loaded, pruned, _MAX_AGE_HOURS, self._path,
        )

    def is_in_cooldown(self, ticker: str, window_minutes: int | None = None) -> bool:
        """Return True if ticker was alerted within the cooldown window.

        window_minutes defaults to config.COOLDOWN_MINUTES.
        """
        ts = self._store.get(ticker)
        if ts is None:
            return False
        window = timedelta(minutes=window_minutes if window_minutes is not None
                           else config.COOLDOWN_MINUTES)
        return (datetime.now(timezone.utc) - ts) < window

    async def set_cooldown(self, ticker: str) -> None:
        """Record that ticker was alerted now and persist to disk."""
        self._store[ticker] = datetime.now(timezone.utc)
        await self._persist()

    async def _persist(self) -> None:
        raw = {t: ts.isoformat() for t, ts in self._store.items()}
        text = json.dumps(raw, indent=2)
        path = self._path

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)  # atomic rename — avoids partial-read on crash

        try:
            await asyncio.to_thread(_write)
        except Exception:
            log.warning(
                "cooldown store: failed to persist to %s", self._path, exc_info=True,
            )
