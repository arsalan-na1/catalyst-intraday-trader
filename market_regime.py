"""SPY market regime detector — Hidden Markov Model over daily bars.

Fits a 3-state Gaussian HMM to engineered features (5-day momentum,
10-day annualized volatility, SMA20/SMA50 ratio) over the last 400 calendar
days of SPY daily bars. The three latent states are mapped to human labels
based on each state's volatility mean: lowest=trending, middle=ranging,
highest=volatile.

Refresh cadence: every ~60 minutes (cached). The model is re-fit each
refresh — fast on a daily-bar window of this size.

Fail-open behaviour: every error path returns "unknown", which corresponds
to size_multiplier=1.0 / hold_multiplier=1.0 — i.e. regime detection never
blocks a trade that the rest of the pipeline accepted.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import numpy as np
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

log = logging.getLogger("market_regime")

_REGIME_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "trending": {"size_multiplier": 1.0, "hold_multiplier": 1.2},
    "ranging":  {"size_multiplier": 0.7, "hold_multiplier": 0.8},
    "volatile": {"size_multiplier": 0.5, "hold_multiplier": 0.6},
    "unknown":  {"size_multiplier": 1.0, "hold_multiplier": 1.0},
}


class MarketRegimeDetector:
    def __init__(self, hist_client: StockHistoricalDataClient | None) -> None:
        self._hist_client = hist_client
        self._current_regime: str = "unknown"
        self._last_updated: datetime | None = None
        self._cache_minutes: int = 60
        self._lock = asyncio.Lock()

    async def get_regime(self) -> str:
        """Return the current cached regime label, refreshing if stale.

        Never raises: any internal error leaves the previous regime intact
        and returns it (or "unknown" if the cache is empty).
        """
        if self._last_updated is not None:
            age = (datetime.utcnow() - self._last_updated).total_seconds() / 60.0
            if age < self._cache_minutes:
                return self._current_regime

        async with self._lock:
            # Double-check after acquiring the lock — another waiter may have
            # just refreshed.
            if self._last_updated is not None:
                age = (datetime.utcnow() - self._last_updated).total_seconds() / 60.0
                if age < self._cache_minutes:
                    return self._current_regime
            try:
                await self._update_regime()
            except Exception:
                log.warning("regime update raised; keeping previous label", exc_info=True)
        return self._current_regime

    async def _update_regime(self) -> None:
        """Fetch SPY bars, compute features, fit HMM, set self._current_regime."""
        if self._hist_client is None:
            return

        try:
            now_et = datetime.now(tz=config.MARKET_TZ)
            req = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=now_et - timedelta(days=400),
                feed=DataFeed.IEX,
            )
            raw = await asyncio.to_thread(self._hist_client.get_stock_bars, req)
            df = getattr(raw, "df", None)
            if df is None or df.empty:
                log.warning("[REGIME] SPY bars empty; leaving regime unchanged")
                return

            if hasattr(df.index, "names") and "symbol" in (df.index.names or []):
                try:
                    df = df.xs("SPY", level="symbol")
                except KeyError:
                    log.warning("[REGIME] SPY not in MultiIndex; leaving regime unchanged")
                    return

            closes = np.asarray(df["close"], dtype=float)
            if len(closes) < 60:
                log.warning(
                    "[REGIME] only %d SPY bars; need 50+ history; leaving unchanged",
                    len(closes),
                )
                return

            returns = np.diff(closes) / closes[:-1]

            # Build per-bar feature rows. We need at least 50 bars of history
            # before the first usable feature row (max window = SMA50).
            features: list[list[float]] = []
            min_history = 50
            for i in range(min_history, len(closes)):
                # 5-day momentum: sum of last 5 daily returns ending at index i.
                # returns[k] is the return from closes[k] -> closes[k+1], so the
                # 5 returns ending at bar i are returns[i-5 : i].
                mom_5d = float(np.sum(returns[i - 5 : i]))

                # 10-day annualized volatility: std of last 10 returns × sqrt(252).
                vol_window = returns[i - 10 : i]
                vol_10d = float(np.std(vol_window, ddof=0) * np.sqrt(252))

                # SMA ratio: 20-day SMA / 50-day SMA.
                sma_20 = float(np.mean(closes[i - 20 : i]))
                sma_50 = float(np.mean(closes[i - 50 : i]))
                sma_ratio = sma_20 / sma_50 if sma_50 > 0 else 1.0

                features.append([mom_5d, vol_10d, sma_ratio])

            if len(features) < 60:
                log.warning(
                    "[REGIME] insufficient feature rows (%d); leaving unchanged",
                    len(features),
                )
                return

            X = np.asarray(features, dtype=float)

            # hmmlearn import lazy so missing dep doesn't break import.
            from hmmlearn import hmm

            model = hmm.GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(X)
            states = model.predict(X)

            # Feature index 1 is volatility — sort ascending: low=trending,
            # mid=ranging, high=volatile.
            vol_means = model.means_[:, 1]
            sorted_states = np.argsort(vol_means)
            label_map = {
                int(sorted_states[0]): "trending",
                int(sorted_states[1]): "ranging",
                int(sorted_states[2]): "volatile",
            }
            current_label = label_map[int(states[-1])]

            self._current_regime = current_label
            self._last_updated = datetime.utcnow()
            log.info("[REGIME] SPY regime: %s", current_label)
        except Exception:
            log.warning("[REGIME] update failed; leaving previous label", exc_info=True)

    def get_regime_adjustments(self) -> dict:
        """Return size_multiplier, hold_multiplier, label for the current regime."""
        adj = _REGIME_ADJUSTMENTS.get(self._current_regime, _REGIME_ADJUSTMENTS["unknown"])
        return {
            "size_multiplier": adj["size_multiplier"],
            "hold_multiplier": adj["hold_multiplier"],
            "label": self._current_regime,
        }
