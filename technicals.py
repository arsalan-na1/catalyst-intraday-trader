"""Technical indicator calculator.

Fetches 252 days of daily OHLCV bars and today's 5-minute intraday bars
from Alpaca (IEX feed), computes momentum and trend indicators, and
returns a Technicals dataclass ready to be formatted into Gemini prompts.

Returns None on any error so callers fall back gracefully.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

log = logging.getLogger("technicals")

# 400 calendar days safely covers 252+ trading days for SMA200 / 52-week range.
_DAILY_LOOKBACK_DAYS = 400
_NEAR_52W_THRESHOLD_PCT = 5.0  # within 5% = "near"


@dataclass
class Technicals:
    week52_high: float
    week52_low: float
    dist_from_52w_high_pct: float   # negative means below high
    dist_from_52w_low_pct: float    # positive means above low
    rsi_14: float | None
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    price_vs_sma20_pct: float | None
    price_vs_sma50_pct: float | None
    price_vs_sma200_pct: float | None
    atr_14_pct: float | None        # ATR as % of current price
    intraday_high: float
    intraday_low: float
    intraday_range_pct: float
    trend: str                       # "uptrend" | "downtrend" | "sideways"
    near_52w_high: bool
    near_52w_low: bool

    def to_prompt_text(self) -> str:
        lines = [
            f"52-week range: ${self.week52_low:.2f} – ${self.week52_high:.2f}",
            f"  vs 52w high: {self.dist_from_52w_high_pct:+.1f}% | "
            f"vs 52w low: {self.dist_from_52w_low_pct:+.1f}%",
            f"  Near 52w high: {self.near_52w_high} | Near 52w low: {self.near_52w_low}",
        ]
        if self.rsi_14 is not None:
            lines.append(f"RSI(14): {self.rsi_14:.1f}")
        if self.sma_20 is not None:
            lines.append(
                f"SMA20: ${self.sma_20:.2f} ({self.price_vs_sma20_pct:+.1f}% vs price)"
            )
        if self.sma_50 is not None:
            lines.append(
                f"SMA50: ${self.sma_50:.2f} ({self.price_vs_sma50_pct:+.1f}% vs price)"
            )
        if self.sma_200 is not None:
            lines.append(
                f"SMA200: ${self.sma_200:.2f} ({self.price_vs_sma200_pct:+.1f}% vs price)"
            )
        if self.atr_14_pct is not None:
            lines.append(f"ATR(14) as % of price: {self.atr_14_pct:.2f}%")
        lines.append(f"Overall trend: {self.trend}")
        lines.append(
            f"Today's intraday range: ${self.intraday_low:.2f} – ${self.intraday_high:.2f}"
            f" ({self.intraday_range_pct:.1f}%)"
        )
        return "\n".join(lines)

    def to_compact_str(self) -> str:
        """One-line summary for Telegram alerts."""
        parts: list[str] = []
        if self.rsi_14 is not None:
            parts.append(f"RSI {self.rsi_14:.0f}")
        parts.append(self.trend.capitalize())
        parts.append(f"{self.dist_from_52w_high_pct:+.0f}% from 52w high")
        if self.atr_14_pct is not None:
            parts.append(f"ATR {self.atr_14_pct:.1f}%")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Pure-Python indicator helpers (no numpy dependency)
# ---------------------------------------------------------------------------

def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    if len(closes) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def _determine_trend(
    price: float,
    sma_50: float | None,
    sma_200: float | None,
) -> str:
    if sma_50 is not None and sma_200 is not None:
        if price > sma_50 and sma_50 > sma_200:
            return "uptrend"
        if price < sma_50 and sma_50 < sma_200:
            return "downtrend"
        return "sideways"
    if sma_50 is not None:
        if price > sma_50 * 1.03:
            return "uptrend"
        if price < sma_50 * 0.97:
            return "downtrend"
    return "sideways"


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def get_technicals(
    ticker: str,
    hist_client: StockHistoricalDataClient,
    current_price: float,
) -> Technicals | None:
    """Fetch and compute technical indicators. Returns None on any error."""
    try:
        return await _compute(ticker, hist_client, current_price)
    except Exception:
        log.exception("technicals computation failed for %s", ticker)
        return None


async def _compute(
    ticker: str,
    hist_client: StockHistoricalDataClient,
    current_price: float,
) -> Technicals:
    now_et = datetime.now(tz=config.MARKET_TZ)
    daily_start = now_et - timedelta(days=_DAILY_LOOKBACK_DAYS)
    today_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)

    daily_req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=daily_start,
        feed=DataFeed.IEX,
    )
    intraday_req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=today_start,
        feed=DataFeed.IEX,
    )

    daily_raw, intraday_raw = await asyncio.gather(
        asyncio.to_thread(hist_client.get_stock_bars, daily_req),
        asyncio.to_thread(hist_client.get_stock_bars, intraday_req),
    )

    # --- daily bars ---
    daily_df = getattr(daily_raw, "df", None)
    if daily_df is None or daily_df.empty:
        raise ValueError(f"no daily bars returned for {ticker}")

    # Handle MultiIndex (symbol, timestamp) from the Alpaca SDK
    if hasattr(daily_df.index, "names") and "symbol" in (daily_df.index.names or []):
        try:
            sym_df = daily_df.xs(ticker, level="symbol")
        except KeyError:
            # Fallback: use first symbol's data
            sym_df = daily_df.xs(daily_df.index.get_level_values("symbol")[0], level="symbol")
    else:
        sym_df = daily_df

    closes = [float(v) for v in sym_df["close"]]
    highs = [float(v) for v in sym_df["high"]]
    lows = [float(v) for v in sym_df["low"]]

    if len(closes) < 2:
        raise ValueError(f"insufficient daily bars for {ticker}: {len(closes)}")

    # 52-week range over the last 252 trading-day bars
    tail = min(252, len(closes))
    week52_high = max(highs[-tail:])
    week52_low = min(lows[-tail:])

    dist_from_52w_high_pct = (current_price - week52_high) / week52_high * 100
    dist_from_52w_low_pct = (current_price - week52_low) / week52_low * 100

    near_52w_high = dist_from_52w_high_pct >= -_NEAR_52W_THRESHOLD_PCT
    near_52w_low = dist_from_52w_low_pct <= _NEAR_52W_THRESHOLD_PCT

    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)
    sma_200 = _sma(closes, 200)

    def _vs_pct(sma: float | None) -> float | None:
        if sma is None or sma == 0:
            return None
        return (current_price - sma) / sma * 100

    rsi = _compute_rsi(closes)
    atr = _compute_atr(highs, lows, closes)
    atr_pct = (atr / current_price * 100) if (atr and current_price > 0) else None
    trend = _determine_trend(current_price, sma_50, sma_200)

    # --- intraday bars ---
    intraday_high = current_price
    intraday_low = current_price
    intraday_range_pct = 0.0

    intra_df = getattr(intraday_raw, "df", None)
    if intra_df is not None and not intra_df.empty:
        if hasattr(intra_df.index, "names") and "symbol" in (intra_df.index.names or []):
            try:
                intra_sym = intra_df.xs(ticker, level="symbol")
            except KeyError:
                intra_sym = intra_df.xs(intra_df.index.get_level_values("symbol")[0], level="symbol")
        else:
            intra_sym = intra_df
        intraday_high = float(intra_sym["high"].max())
        intraday_low = float(intra_sym["low"].min())
        if intraday_low > 0:
            intraday_range_pct = (intraday_high - intraday_low) / intraday_low * 100

    return Technicals(
        week52_high=week52_high,
        week52_low=week52_low,
        dist_from_52w_high_pct=dist_from_52w_high_pct,
        dist_from_52w_low_pct=dist_from_52w_low_pct,
        rsi_14=rsi,
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        price_vs_sma20_pct=_vs_pct(sma_20),
        price_vs_sma50_pct=_vs_pct(sma_50),
        price_vs_sma200_pct=_vs_pct(sma_200),
        atr_14_pct=atr_pct,
        intraday_high=intraday_high,
        intraday_low=intraday_low,
        intraday_range_pct=intraday_range_pct,
        trend=trend,
        near_52w_high=near_52w_high,
        near_52w_low=near_52w_low,
    )
