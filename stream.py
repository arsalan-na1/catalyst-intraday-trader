"""Alpaca WebSocket stream + trigger detector.

Subscribes to minute bars for every ticker in watchlist.txt on Alpaca's IEX
feed (free tier). Two triggers produce a `TriggerEvent` on the output queue:

  1. Price move: `(max - min) / min > PRICE_MOVE_THRESHOLD_PCT` over the last
     PRICE_MOVE_WINDOW_SECONDS across bar high/low/close samples.

  2. Volume spike: cumulative-volume-today > VOLUME_RATIO_THRESHOLD × expected-
     so-far, where expected = ADV × (minutes_since_open / 390).
     Chosen so early-session moves don't require a full day's worth of volume.

Per-ticker cooldown of COOLDOWN_MINUTES suppresses repeats. Triggers only
fire during regular market hours (handled via MarketCalendar).

Reliability:
  - alpaca-py has its own WS reconnect logic; we add an outer loop with
    exponential backoff (1..60s) so the entire client gets rebuilt if the
    stream exits for any reason.
  - Cumulative volumes reset on session date change (detected by comparing
    bar.timestamp.date() against the stored session_date).

IEX caveat: on the free feed, bar volumes reflect IEX-only trades, so mid-caps
will undercount. Thresholds may need tuning after real observation. Logged
raw numbers make this tractable.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from earnings_calendar import EarningsCalendar
from market_calendar import MarketCalendar

log = logging.getLogger("stream")

# Max symbols per historical-bars request. Alpaca accepts more but responses
# grow large; 100 is a safe chunk size that stays well under rate limits.
_HISTORICAL_BATCH_SIZE = 100

# Catalyst-watchlist tickers (build_catalyst_watchlist.py output) use a lower
# price-move trigger because FDA/squeeze names move harder and faster on news.
# Threshold is config.PRICE_TRIGGER_CATALYST_PCT (default 5% vs 8% baseline).


@dataclass
class TriggerEvent:
    ticker: str
    price: float
    price_move_pct: float
    volume_ratio: float
    cumulative_volume: int
    expected_volume: float
    triggered_at: datetime
    window_high: float
    window_low: float
    trigger_type: str = field(default="intraday")  # "intraday" | "gap_open"
    # Latest session VWAP from the bar that fired the trigger. None when the
    # bar payload omitted vwap (e.g. some IEX bars early in the session).
    window_vwap: float | None = None
    # True when the ticker is on the earnings calendar at trigger time.
    # Used downstream to relax confidence and opening-bell gates.
    is_earnings: bool = False

    def summary(self) -> str:
        tag = f"[{self.trigger_type}] " if self.trigger_type != "intraday" else ""
        return (
            f"{tag}{self.ticker}: move={self.price_move_pct:.1%} "
            f"vol_ratio={self.volume_ratio:.1f}x price=${self.price:.2f}"
        )


def _is_valid_ticker(t: str) -> bool:
    """Reject symbols that would cause Alpaca API 400 errors (dashes, phantom dots, etc.)."""
    if not t or len(t) > 6:
        return False
    if not all(c.isalpha() or c == "." for c in t):
        return False
    if not any(c.isalpha() for c in t):  # reject bare "." or ".."
        return False
    return t not in {"CASH", "USD", "XTSLA", "MARGIN_USD"}


def load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python build_watchlist.py` first."
        )
    tickers: list[str] = []
    skipped = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        t = raw.strip().upper()
        if not t or t.startswith("#"):
            continue
        if _is_valid_ticker(t):
            tickers.append(t)
        else:
            skipped += 1
            log.debug("load_watchlist: skipping invalid ticker %r", t)
    if skipped:
        log.warning("load_watchlist: skipped %d invalid ticker(s) from %s", skipped, path)
    if not tickers:
        raise RuntimeError(f"Watchlist {path} is empty")
    return tickers


async def prefetch_adv(
    client: StockHistoricalDataClient,
    tickers: list[str],
    lookback_days: int = config.ADV_LOOKBACK_DAYS,
) -> dict[str, float]:
    """Return mean daily volume per ticker over the last `lookback_days`
    trading days. Tickers with no data are omitted (the trigger check
    treats missing ADV as "don't fire volume trigger")."""

    # Request a calendar-day window slightly larger than trading-day target.
    start = datetime.now(tz=config.MARKET_TZ) - timedelta(days=lookback_days * 2 + 5)
    adv: dict[str, float] = {}

    for i in range(0, len(tickers), _HISTORICAL_BATCH_SIZE):
        batch = tickers[i : i + _HISTORICAL_BATCH_SIZE]
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
            start=start,
            feed=DataFeed.IEX,
        )
        bars = None
        for attempt in range(config.ADV_FETCH_RETRIES):
            try:
                bars = await asyncio.to_thread(client.get_stock_bars, req)
                break
            except Exception:
                if attempt < config.ADV_FETCH_RETRIES - 1:
                    wait_secs = 2 ** attempt
                    log.warning(
                        "ADV prefetch failed (attempt %d/%d) for batch starting %s; retry in %ds",
                        attempt + 1, config.ADV_FETCH_RETRIES, batch[0], wait_secs,
                    )
                    await asyncio.sleep(wait_secs)
                else:
                    log.exception("ADV prefetch gave up for batch starting %s; skipping", batch[0])
        if bars is None:
            continue
        df = bars.df
        if df is None or df.empty:
            continue
        # df has MultiIndex (symbol, timestamp). Take last `lookback_days` rows
        # per symbol and average the `volume` column.
        for sym in batch:
            if sym not in df.index.get_level_values("symbol"):
                continue
            sub = df.xs(sym, level="symbol")
            if sub.empty:
                continue
            recent = sub.tail(lookback_days)
            mean_vol = float(recent["volume"].mean())
            if mean_vol > 0:
                adv[sym] = mean_vol
        log.info(
            "ADV prefetch: %d/%d tickers done", min(i + _HISTORICAL_BATCH_SIZE, len(tickers)), len(tickers)
        )
        # Gentle pace — well under 200 req/min even without this.
        await asyncio.sleep(0.1)

    log.info("ADV prefetch complete: %d tickers with volume baseline", len(adv))
    return adv


async def prefetch_prev_close(
    client: StockHistoricalDataClient,
    tickers: list[str],
) -> dict[str, float]:
    """Return the most recent session's closing price per ticker.

    Fetches 5 calendar days of daily bars (covers 3+ trading days) and takes
    the last close. Called at startup so gap-open detection at 9:30 ET is instant.
    Tickers with no bar data are omitted — they simply won't fire the gap trigger.
    """
    start = datetime.now(tz=config.MARKET_TZ) - timedelta(days=5)
    prev_close: dict[str, float] = {}

    for i in range(0, len(tickers), _HISTORICAL_BATCH_SIZE):
        batch = tickers[i : i + _HISTORICAL_BATCH_SIZE]
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
            start=start,
            feed=DataFeed.IEX,
        )
        bars = None
        for attempt in range(config.ADV_FETCH_RETRIES):
            try:
                bars = await asyncio.to_thread(client.get_stock_bars, req)
                break
            except Exception:
                if attempt < config.ADV_FETCH_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    log.warning("prev_close prefetch gave up for batch starting %s", batch[0])
        if bars is None:
            continue
        df = bars.df
        if df is None or df.empty:
            continue
        for sym in batch:
            if sym not in df.index.get_level_values("symbol"):
                continue
            sub = df.xs(sym, level="symbol")
            if sub.empty:
                continue
            last_close = float(sub["close"].iloc[-1])
            if last_close > 0:
                prev_close[sym] = last_close
        await asyncio.sleep(0.1)

    log.info("prev_close prefetch complete: %d tickers with baseline close", len(prev_close))
    return prev_close


class TriggerDetector:
    """Maintains per-ticker state and decides whether a bar fires a trigger.

    Runs in the Alpaca stream's own event loop (a separate thread). Pushes
    events to the main loop via `loop.call_soon_threadsafe` so the
    asyncio.Queue stays owned by the main loop — crossing loops with
    asyncio primitives directly is unsafe.
    """

    def __init__(
        self,
        adv: dict[str, float],
        calendar: MarketCalendar,
        out_queue: asyncio.Queue[TriggerEvent],
        main_loop: asyncio.AbstractEventLoop,
        catalyst_tickers: set[str] | None = None,
        earnings_calendar: EarningsCalendar | None = None,
        intraday_gappers: set[str] | None = None,
        prev_close: dict[str, float] | None = None,
    ) -> None:
        self._adv = adv
        self._calendar = calendar
        self._out = out_queue
        self._main_loop = main_loop
        self._catalyst_tickers = catalyst_tickers or set()
        self._earnings_calendar = earnings_calendar
        self._intraday_gappers = intraday_gappers if intraday_gappers is not None else set()
        self._prev_close = prev_close or {}
        self._cum_volume: dict[str, int] = {}
        self._price_window: dict[str, deque[tuple[datetime, float]]] = {}
        self._last_trigger: dict[str, datetime] = {}
        self._vol_confirm_count: dict[str, int] = {}
        self._gap_triggered_today: set[str] = set()
        self._session_date: date | None = None
        # Last seen VWAP per ticker (from the most recent bar). Populated as
        # bars arrive and threaded onto TriggerEvent so downstream Gemini
        # scoring can reason about chasing-vs-VWAP entries.
        self._last_vwap: dict[str, float] = {}

    def _maybe_reset_session(self, bar_ts: datetime) -> None:
        bar_date = bar_ts.astimezone(config.MARKET_TZ).date()
        if self._session_date != bar_date:
            log.info("New session detected (%s). Resetting cumulative volumes.", bar_date)
            self._cum_volume.clear()
            self._price_window.clear()
            self._vol_confirm_count.clear()
            self._gap_triggered_today.clear()
            self._last_vwap.clear()
            # Also clear last-trigger timestamps so a ticker that fired near
            # EOD yesterday isn't suppressed by a stale cooldown at today's open.
            self._last_trigger.clear()
            self._session_date = bar_date

    def _update_price_window(self, ticker: str, bar) -> tuple[float, float]:
        window = self._price_window.setdefault(ticker, deque())
        ts = bar.timestamp
        # Push high, low, close so we capture intra-bar extremes too.
        window.append((ts, float(bar.low)))
        window.append((ts, float(bar.high)))
        window.append((ts, float(bar.close)))
        cutoff = ts - timedelta(seconds=config.PRICE_MOVE_WINDOW_SECONDS)
        while window and window[0][0] < cutoff:
            window.popleft()
        prices = [p for _, p in window]
        return min(prices), max(prices)

    def _in_cooldown(self, ticker: str, now: datetime) -> bool:
        last = self._last_trigger.get(ticker)
        if last is None:
            return False
        return (now - last) < timedelta(minutes=config.COOLDOWN_MINUTES)

    def _check_gap_open(self, ticker: str, bar) -> None:
        """Fire a gap_open TriggerEvent if the first open-bar gaps vs prev close.

        Only fires within the first GAP_OPEN_WINDOW_MINUTES after 9:30 ET,
        at most once per ticker per session.
        """
        prev_c = self._prev_close.get(ticker)
        if prev_c is None or prev_c <= 0:
            return
        if ticker in self._gap_triggered_today:
            return

        bar_et = bar.timestamp.astimezone(config.MARKET_TZ)
        bar_minutes = bar_et.hour * 60 + bar_et.minute
        open_minutes = 9 * 60 + 30
        minutes_since_open = bar_minutes - open_minutes
        if not (0 <= minutes_since_open < config.GAP_OPEN_WINDOW_MINUTES):
            return

        open_price = float(bar.open)
        if open_price <= 0:
            return

        gap_pct = (open_price - prev_c) / prev_c
        if gap_pct < config.GAP_OPEN_THRESHOLD:
            return

        self._gap_triggered_today.add(ticker)
        log.info(
            "[GAP OPEN] %s gap=+%.1f%% open=$%.2f prev_close=$%.2f",
            ticker, gap_pct * 100, open_price, prev_c,
        )
        is_earnings = (
            self._earnings_calendar is not None
            and ticker in self._earnings_calendar.get_tickers()
        )
        event = TriggerEvent(
            ticker=ticker,
            price=open_price,
            price_move_pct=gap_pct,
            volume_ratio=0.0,
            cumulative_volume=self._cum_volume.get(ticker, 0),
            expected_volume=0.0,
            triggered_at=bar.timestamp,
            window_high=float(bar.high),
            window_low=float(bar.low),
            trigger_type="gap_open",
            window_vwap=self._last_vwap.get(ticker),
            is_earnings=is_earnings,
        )
        self._main_loop.call_soon_threadsafe(self._out.put_nowait, event)

    async def handle_bar(self, bar) -> None:
        ticker = bar.symbol
        self._maybe_reset_session(bar.timestamp)

        self._cum_volume[ticker] = self._cum_volume.get(ticker, 0) + int(bar.volume)
        window_low, window_high = self._update_price_window(ticker, bar)

        # Capture VWAP from the bar payload (Alpaca minute bars include vwap).
        # getattr is defensive — if a bar variant ever omits the field we just
        # skip the update rather than raising.
        bar_vwap = getattr(bar, "vwap", None)
        if bar_vwap is not None:
            try:
                vwap_f = float(bar_vwap)
                if vwap_f > 0:
                    self._last_vwap[ticker] = vwap_f
            except (TypeError, ValueError):
                pass

        # Gap-open check: fires once per ticker at 9:30 ET if the bar's open
        # price is >= GAP_OPEN_THRESHOLD above the previous session close.
        # Runs before the opening-bell suppression so it is never filtered out.
        self._check_gap_open(ticker, bar)

        # is_market_open uses threading.Lock internally so it's safe from this
        # stream-owned loop.
        try:
            if not await self._calendar.is_market_open():
                return
        except Exception:
            log.warning("market clock check failed for %s; assuming open", ticker)

        # Earnings + large gap-up override: a confirmed earnings catalyst with a
        # gap of 20%+ vs prior close is exactly the setup the opening-bell filter
        # would otherwise wrongly suppress (e.g. NVDA-class beats). Bypass the
        # filter for that combination only.
        is_earnings = (
            self._earnings_calendar is not None
            and ticker in self._earnings_calendar.get_tickers()
        )
        prev_c = self._prev_close.get(ticker)
        gap_up_pct = (float(bar.close) - prev_c) / prev_c if prev_c else 0.0
        earnings_gap_override = is_earnings and gap_up_pct >= 0.20

        try:
            minutes_in = await self._calendar.minutes_since_open()
            if 0 <= minutes_in < config.OPENING_BELL_MINUTES:
                if earnings_gap_override:
                    log.info(
                        "[OPENING BELL] %s bypassed — earnings catalyst with %.0f%% gap",
                        ticker, gap_up_pct * 100,
                    )
                else:
                    bar_et = bar.timestamp.astimezone(config.MARKET_TZ)
                    log.info(
                        "[OPENING BELL] suppressed %s at %s — too close to open",
                        ticker,
                        bar_et.strftime("%H:%M:%S"),
                    )
                    return
        except Exception:
            log.warning("minutes_since_open check failed for %s; skipping opening bell filter", ticker)

        if self._in_cooldown(ticker, bar.timestamp):
            return

        # Price move — catalyst tickers and premarket gappers get a tighter threshold.
        price_move_pct = 0.0
        if window_low > 0:
            price_move_pct = (window_high - window_low) / window_low
        price_threshold = (
            config.PRICE_TRIGGER_CATALYST_PCT
            if (ticker in self._catalyst_tickers or ticker in self._intraday_gappers)
            else config.PRICE_MOVE_THRESHOLD_PCT
        )
        price_fired = price_move_pct > price_threshold

        # Volume spike
        vol_fired = False
        vol_ratio = 0.0
        expected = 0.0
        adv_value = self._adv.get(ticker)
        if adv_value:
            minutes_in = await self._calendar.minutes_since_open()
            if minutes_in > 0:
                expected = adv_value * (minutes_in / config.REGULAR_SESSION_MINUTES)
                if expected > 0:
                    vol_ratio = self._cum_volume[ticker] / expected
                    is_earnings = (
                        self._earnings_calendar is not None
                        and ticker in self._earnings_calendar.get_tickers()
                    )
                    vol_threshold = (
                        config.VOLUME_TRIGGER_EARNINGS_MULTIPLIER
                        if is_earnings
                        else config.VOLUME_RATIO_THRESHOLD
                    )
                    vol_fired = vol_ratio > vol_threshold
                    if vol_fired and is_earnings:
                        log.info("[EARNINGS CATALYST] lowered volume threshold for %s", ticker)

        if not (price_fired or vol_fired):
            self._vol_confirm_count.pop(ticker, None)
            return

        # High-volume / low-price-move triggers are noisy (see KALV 1061×).
        # Require VOLUME_SPIKE_CONFIRM_BARS consecutive qualifying bars before firing.
        if (
            vol_fired
            and not price_fired
            and vol_ratio > config.VOLUME_SPIKE_CONFIRM_VOL_THRESHOLD
            and price_move_pct < config.VOLUME_SPIKE_CONFIRM_MIN_MOVE
        ):
            count = self._vol_confirm_count.get(ticker, 0) + 1
            self._vol_confirm_count[ticker] = count
            if count < config.VOLUME_SPIKE_CONFIRM_BARS:
                log.info(
                    "[CONFIRM] %s vol=%.0fx move=%.2f%% — bar %d/%d, waiting",
                    ticker, vol_ratio, price_move_pct * 100,
                    count, config.VOLUME_SPIKE_CONFIRM_BARS,
                )
                return
            log.info(
                "[CONFIRM] %s confirmed over %d bars — firing",
                ticker, config.VOLUME_SPIKE_CONFIRM_BARS,
            )
        else:
            self._vol_confirm_count.pop(ticker, None)

        event = TriggerEvent(
            ticker=ticker,
            price=float(bar.close),
            price_move_pct=price_move_pct,
            volume_ratio=vol_ratio,
            cumulative_volume=self._cum_volume[ticker],
            expected_volume=expected,
            triggered_at=bar.timestamp,
            window_high=window_high,
            window_low=window_low,
            window_vwap=self._last_vwap.get(ticker),
            is_earnings=is_earnings,
        )
        self._last_trigger[ticker] = bar.timestamp
        log.info("TRIGGER %s", event.summary())
        # Cross-loop hand-off: schedule put_nowait on the main loop.
        self._main_loop.call_soon_threadsafe(self._out.put_nowait, event)


async def run_stream(
    watchlist: list[str],
    adv: dict[str, float],
    calendar: MarketCalendar,
    out_queue: asyncio.Queue[TriggerEvent],
    stop_event: asyncio.Event,
    catalyst_tickers: set[str] | None = None,
    earnings_calendar: EarningsCalendar | None = None,
    intraday_gappers: set[str] | None = None,
    dynamic_sub_queue: "asyncio.Queue[str] | None" = None,
    prev_close: dict[str, float] | None = None,
) -> None:
    """Run the Alpaca bar stream with exponential-backoff reconnect.

    Blocks until `stop_event` is set. alpaca-py's StockDataStream.run() is a
    blocking sync call, so we offload to a thread and race it against the
    stop_event so the service can shut down cleanly on SIGTERM.

    `catalyst_tickers` is the subset of `watchlist` that came from the
    catalyst list — they get a lower price-move threshold in the detector.
    `intraday_gappers` is a shared set populated by premarket_scanner and
    microcap_screener; gappers are subscribed on every reconnect and use the
    catalyst price threshold.
    `dynamic_sub_queue` carries new tickers to subscribe mid-session without
    waiting for the next stream reconnect.
    """

    main_loop = asyncio.get_running_loop()
    detector = TriggerDetector(
        adv, calendar, out_queue, main_loop,
        catalyst_tickers, earnings_calendar, intraday_gappers,
        prev_close=prev_close,
    )

    async def on_bar(bar) -> None:
        try:
            await detector.handle_bar(bar)
        except Exception:
            log.exception("bar handler failed for %s", getattr(bar, "symbol", "?"))

    # Single-element list so _dynamic_subscriber always sees the live stream.
    _active: list[StockDataStream | None] = [None]

    async def _dynamic_subscriber() -> None:
        """Drain dynamic_sub_queue and subscribe new tickers mid-session."""
        if dynamic_sub_queue is None:
            return
        while not stop_event.is_set():
            try:
                ticker = await asyncio.wait_for(dynamic_sub_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            stream = _active[0]
            if stream is None:
                # Stream not yet up or between reconnects; re-queue and wait.
                await dynamic_sub_queue.put(ticker)
                await asyncio.sleep(2.0)
                continue
            try:
                # subscribe_bars is sync-blocking when the stream is running
                # (it sends a subscribe message via run_coroutine_threadsafe).
                await asyncio.to_thread(stream.subscribe_bars, on_bar, ticker)
                log.info("[GAPSCANNER] WS subscribed %s mid-session", ticker)
            except Exception:
                log.exception("[GAPSCANNER] WS subscribe failed for %s", ticker)

    dyn_task = asyncio.create_task(_dynamic_subscriber())
    backoff = 1.0
    try:
        while not stop_event.is_set():
            # Include current gappers on every reconnect so they survive restarts.
            extra = sorted((intraday_gappers or set()) - set(watchlist))
            all_tickers = watchlist + extra

            stream = StockDataStream(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                feed=DataFeed.IEX,
            )
            stream.subscribe_bars(on_bar, *all_tickers)
            _active[0] = stream
            log.info(
                "WS connecting; subscribed to %d tickers (%d gapper(s))",
                len(all_tickers), len(extra),
            )

            # Run the blocking stream in a thread. When it exits (disconnect or
            # error), we land back here and either reconnect or exit cleanly.
            connected_at = asyncio.get_running_loop().time()
            stream_task = asyncio.create_task(asyncio.to_thread(stream.run))
            stop_task = asyncio.create_task(stop_event.wait())

            done, _pending = await asyncio.wait(
                {stream_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            _active[0] = None  # stream is no longer active

            if stop_task in done:
                log.info("stop event received; closing stream")
                try:
                    await asyncio.to_thread(stream.stop)
                except Exception:
                    log.exception("error stopping stream")
                stream_task.cancel()
                return

            # Stream exited on its own. Reconnect with backoff.
            # If the connection ran ≥60s, treat it as healthy and reset the
            # backoff floor — otherwise a single late-day disconnect
            # ratchets the next reconnect delay up to 60s for no reason.
            ran_secs = asyncio.get_running_loop().time() - connected_at
            if ran_secs >= 60.0:
                backoff = 1.0
            exc = stream_task.exception() if stream_task.done() else None
            log.warning("WS stream exited%s after %.0fs; reconnecting in %.1fs",
                        f" with {exc}" if exc else "", ran_secs, backoff)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                return  # stop_event fired during backoff
            except asyncio.TimeoutError:
                pass
            backoff = min(60.0, backoff * 2.0 + random.uniform(0, 0.5))
    finally:
        dyn_task.cancel()


async def _smoke() -> None:
    """Standalone smoke test: load watchlist, prefetch ADV, stream bars.

    Run during market hours. Prints triggers to stdout.
    """
    from alpaca.trading.client import TradingClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    tickers = load_watchlist(config.WATCHLIST_PATH)
    log.info("Loaded %d tickers from watchlist", len(tickers))

    hist = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    adv = await prefetch_adv(hist, tickers)

    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER)
    cal = MarketCalendar(trading)

    queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
    stop = asyncio.Event()

    async def printer() -> None:
        while not stop.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            print("EVENT:", event.summary())

    await asyncio.gather(
        run_stream(tickers, adv, cal, queue, stop),
        printer(),
    )


if __name__ == "__main__":
    asyncio.run(_smoke())
