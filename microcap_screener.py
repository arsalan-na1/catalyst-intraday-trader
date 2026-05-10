"""Micro-cap catalyst screener — parallel async task.

Polls Alpaca's most-actives screener every MICROCAP_POLL_SECONDS while the
market is open, applies a stack of filters, and pushes survivors through the
existing scorer + Telegram pipeline.

Filter rationale (each threshold tunable via .env):

  * Price in [$1.00, $20.00]
        Bottom guards against sub-penny pump-and-dump junk where one print
        moves the tape 50%+ and slippage destroys edge. Top excludes mid-caps
        the main stream already covers.

  * Today's volume >= 4x the trailing 30-day ADV
        Real micro-cap catalysts produce volume blowouts. 4x cleanly
        separates "in play" from normal noise without being so strict that
        legitimate setups get filtered. ADV is computed from the same daily
        bars helper the main bot uses for its watchlist.

  * Market cap < $500M
        Defines "micro-cap" for our purposes (technically nano + micro).
        Computed as price * shares_outstanding from Alpaca asset metadata
        when available. If shares_outstanding is unavailable the market-cap
        filter is skipped; the ticker is still scored and added to
        intraday_gappers so the WebSocket stream watches it for future moves.

  * Not in the main watchlist.txt
        Avoid duplicating alerts the primary stream is already watching.

  * Not alerted in the last MICROCAP_DEDUP_HOURS hours (in-memory set,
    cleared at midnight ET)
        Stops the same hot ticker spamming the chat across multiple polls
        within the same session.

Survivors pass through the existing Scorer (Gemini grounded research +
verdict) — the alert fires regardless of whether a news catalyst was
identified, but the alert message reflects which case applied so a
volume-only signal can be sized accordingly by the trader.

Endpoint note: spec asked for /v2/screener/stocks/most-actives but Alpaca's
production path is /v1beta1/screener/stocks/most-actives. We hit the working
path; if Alpaca ever GAs a v2 screener, swap the constant below.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

import config
from cooldown_store import CooldownStore
from daily_summary import SessionStats
from market_calendar import MarketCalendar
from news import determine_sector, fetch_news, is_analyst_action, is_biotech_catalyst, is_insider_buying
from scorer import Scorer, TriggerContext
from stream import load_watchlist, prefetch_adv
from telegram_handler import TelegramHandler
from trader import Trader

log = logging.getLogger("microcap")

_LOG_PREFIX = "[MICROCAP]"

# Alpaca screener — most-actives by volume. v1beta1 is the live path; spec
# document referenced /v2 but that namespace doesn't exist for screener yet.
_MOST_ACTIVES_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"
_MOST_ACTIVES_TOP_N = 50

# Alpaca snapshot endpoint — used to read today's running volume and daily
# bar. Bulk path keeps us under rate limits when we have ~50 candidates.
_SNAPSHOTS_URL = "https://data.alpaca.markets/v2/stocks/snapshots"


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }


class MicrocapScreener:
    def __init__(
        self,
        trading_client: TradingClient,
        hist_client: StockHistoricalDataClient,
        http_session: aiohttp.ClientSession,
        scorer: Scorer,
        telegram: TelegramHandler,
        trader: Trader,
        calendar: MarketCalendar,
        base_watchlist: set[str],
        stats: SessionStats | None = None,
        cooldown_store: CooldownStore | None = None,
        intraday_gappers: set[str] | None = None,
        dynamic_sub_queue: "asyncio.Queue[str] | None" = None,
        shared_adv: dict[str, float] | None = None,
    ) -> None:
        self._trading = trading_client
        self._hist = hist_client
        self._http = http_session
        self._scorer = scorer
        self._telegram = telegram
        self._trader = trader
        self._calendar = calendar
        self._stats = stats
        self._cooldown_store = cooldown_store
        self._intraday_gappers = intraday_gappers
        self._dynamic_sub_queue = dynamic_sub_queue
        self._shared_adv = shared_adv
        self._base_watchlist = {t.upper() for t in base_watchlist}
        # ticker -> last alert UTC datetime
        self._alerted: dict[str, datetime] = {}
        # Reset trigger: ET date the alerted dict belongs to
        self._alerted_session: date | None = None
        # Cached ADV per symbol; (adv_value, cached_at). Pruned at 24h via
        # _prune_caches so multi-day sessions don't accumulate stale entries.
        self._adv_cache: dict[str, tuple[float, datetime]] = {}
        # Cached market cap (None = checked, no data); (shares_or_None, cached_at).
        self._market_cap_cache: dict[str, tuple[float | None, datetime]] = {}
        # Last Gemini evaluation timestamp per ticker (15-min cooldown);
        # pruned at MICROCAP_DEDUP_HOURS via _prune_caches.
        self._last_evaluated: dict[str, datetime] = {}

    def _prune_caches(self) -> None:
        """Drop cache entries older than their TTL.

        Run at the start of every poll cycle. The other quant-signal caches
        in the project all prune; without this, _adv_cache / _market_cap_cache /
        _last_evaluated would grow monotonically across multi-day sessions
        (bounded by symbols seen, but inconsistent and untidy).
        """
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_dedup = now - timedelta(hours=config.MICROCAP_DEDUP_HOURS)

        before = (len(self._adv_cache), len(self._market_cap_cache), len(self._last_evaluated))
        self._adv_cache = {
            k: v for k, v in self._adv_cache.items() if v[1] >= cutoff_24h
        }
        self._market_cap_cache = {
            k: v for k, v in self._market_cap_cache.items() if v[1] >= cutoff_24h
        }
        self._last_evaluated = {
            k: v for k, v in self._last_evaluated.items() if v >= cutoff_dedup
        }
        after = (len(self._adv_cache), len(self._market_cap_cache), len(self._last_evaluated))
        if before != after:
            log.debug(
                "%s pruned caches: adv %d→%d, mcap %d→%d, last_eval %d→%d",
                _LOG_PREFIX, *(v for pair in zip(before, after) for v in pair),
            )

    # --- public entry point ---

    async def run(self, stop_event: asyncio.Event) -> None:
        log.info("%s screener starting (poll=%ss)", _LOG_PREFIX, config.MICROCAP_POLL_SECONDS)
        while not stop_event.is_set():
            try:
                # Prune every cycle (regardless of market hours) so the
                # caches stay bounded even on holidays / weekends.
                self._prune_caches()
                if await self._calendar.is_market_open():
                    await self._poll_once()
                else:
                    log.debug("%s market closed; skipping poll", _LOG_PREFIX)
            except Exception:
                log.exception("%s poll iteration failed", _LOG_PREFIX)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=config.MICROCAP_POLL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass

    # --- polling ---

    async def _poll_once(self) -> None:
        self._maybe_reset_dedup()

        candidates = await self._fetch_most_actives()
        if not candidates:
            log.info("%s most-actives returned 0 symbols", _LOG_PREFIX)
            return
        log.info("%s most-actives returned %d symbols", _LOG_PREFIX, len(candidates))

        snapshots = await self._fetch_snapshots([c["symbol"] for c in candidates])
        if not snapshots:
            log.warning("%s snapshot fetch returned nothing; skipping poll", _LOG_PREFIX)
            return

        # Ensure ADV is loaded for everything we might score.
        await self._ensure_adv([c["symbol"] for c in candidates])

        for cand in candidates:
            symbol = cand["symbol"]
            try:
                await self._evaluate(symbol, snapshots.get(symbol) or {})
            except Exception:
                log.exception("%s evaluate %s failed", _LOG_PREFIX, symbol)

    async def _evaluate(self, symbol: str, snap: dict) -> None:
        if self._stats and self._stats.halt_new_entries:
            log.info("%s [CIRCUIT BREAKER] halted; skipping %s", _LOG_PREFIX, symbol)
            return
        if symbol in self._base_watchlist:
            log.debug("%s %s already on main watchlist; skip", _LOG_PREFIX, symbol)
            return
        if self._is_recently_alerted(symbol):
            log.debug("%s %s within dedup window; skip", _LOG_PREFIX, symbol)
            return
        if self._cooldown_store and self._cooldown_store.is_in_cooldown(symbol, window_minutes=config.EXIT_COOLDOWN_MINUTES):
            log.debug("%s %s in persistent %dmin cooldown; skip",
                      _LOG_PREFIX, symbol, config.EXIT_COOLDOWN_MINUTES)
            return

        latest_trade = snap.get("latestTrade") or {}
        daily_bar = snap.get("dailyBar") or {}
        prev_daily_bar = snap.get("prevDailyBar") or {}

        price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
        if price <= 0:
            log.debug("%s %s no price data; skip", _LOG_PREFIX, symbol)
            return
        if price < config.MICROCAP_PRICE_MIN or price > config.MICROCAP_PRICE_MAX:
            log.debug(
                "%s %s price $%.2f outside [$%.2f, $%.2f]",
                _LOG_PREFIX, symbol, price,
                config.MICROCAP_PRICE_MIN, config.MICROCAP_PRICE_MAX,
            )
            return

        today_volume = int(daily_bar.get("v") or 0)
        adv_entry = self._adv_cache.get(symbol)
        adv = adv_entry[0] if adv_entry else 0.0
        if adv <= 0:
            log.debug("%s %s no ADV baseline; skip", _LOG_PREFIX, symbol)
            return
        vol_ratio = today_volume / adv if adv > 0 else 0.0
        if vol_ratio < config.MICROCAP_VOL_RATIO_MIN:
            log.debug(
                "%s %s vol_ratio %.2fx below %.1fx",
                _LOG_PREFIX, symbol, vol_ratio, config.MICROCAP_VOL_RATIO_MIN,
            )
            return

        # Dollar volume filter: avoids very low-liquidity movers where slippage
        # is catastrophic even on small fills.
        dollar_volume = price * today_volume
        if dollar_volume < config.MICROCAP_MIN_DOLLAR_VOLUME:
            log.debug(
                "%s %s dollar volume $%.0f below $%.0f minimum",
                _LOG_PREFIX, symbol, dollar_volume, config.MICROCAP_MIN_DOLLAR_VOLUME,
            )
            return

        market_cap = await self._get_market_cap(symbol, price)
        if market_cap is None:
            log.info(
                "%s %s shares_outstanding unavailable; tracking without mcap filter",
                _LOG_PREFIX, symbol,
            )
            # Subscribe to WS so TriggerDetector watches it for further moves.
            if self._intraday_gappers is not None and symbol not in self._intraday_gappers:
                self._intraday_gappers.add(symbol)
                if self._dynamic_sub_queue is not None:
                    await self._dynamic_sub_queue.put(symbol)
                if self._shared_adv is not None:
                    adv_entry2 = self._adv_cache.get(symbol)
                    adv_val = adv_entry2[0] if adv_entry2 else 0.0
                    if adv_val > 0:
                        self._shared_adv[symbol] = adv_val
                log.info("%s subscribed %s to WS (no mcap data)", _LOG_PREFIX, symbol)
            # Fall through — score the ticker without the market-cap gate.
        elif market_cap > config.MICROCAP_MARKET_CAP_MAX:
            log.debug(
                "%s %s mcap $%.0fM > $%.0fM ceiling",
                _LOG_PREFIX, symbol, market_cap / 1e6,
                config.MICROCAP_MARKET_CAP_MAX / 1e6,
            )
            return

        prev_close = float(prev_daily_bar.get("c") or 0.0)
        change_pct = ((price - prev_close) / prev_close) if prev_close > 0 else 0.0

        # --- Opt 3 Stage 1: pre-Gemini filter ---
        # Higher bar than the existing MICROCAP_VOL_RATIO_MIN check above.
        if vol_ratio < config.MICROCAP_VOL_THRESHOLD or abs(change_pct) < config.MICROCAP_PRICE_THRESHOLD:
            log.info(
                "%s %s filtered pre-Gemini (vol=%.1fx < %.1fx or move=%.1f%% < %.0f%%)",
                _LOG_PREFIX, symbol, vol_ratio, config.MICROCAP_VOL_THRESHOLD,
                change_pct * 100, config.MICROCAP_PRICE_THRESHOLD * 100,
            )
            return

        now_dt = datetime.now(timezone.utc)
        last_eval = self._last_evaluated.get(symbol)
        if last_eval is not None and (now_dt - last_eval).total_seconds() < 900:  # 15 minutes
            age_min = (now_dt - last_eval).total_seconds() / 60
            log.debug(
                "%s %s evaluated %.0fmin ago; skipping (15-min cooldown)",
                _LOG_PREFIX, symbol, age_min,
            )
            return

        mcap_str = f"mcap=${market_cap / 1e6:.0f}M" if market_cap is not None else "mcap=unavailable"
        log.info(
            "%s candidate %s price=$%.2f vol=%d (%.1fx ADV) %s change=%.2f%%",
            _LOG_PREFIX, symbol, price, today_volume, vol_ratio,
            mcap_str, change_pct * 100,
        )

        ctx = TriggerContext(
            ticker=symbol,
            price=price,
            price_move_pct=change_pct,
            volume_ratio=vol_ratio,
        )

        try:
            news_items = await fetch_news(symbol, session=self._http)
        except Exception:
            log.exception("%s news fetch failed for %s", _LOG_PREFIX, symbol)
            news_items = []

        # --- Opt 3 Stage 2: no-grounding quick check ---
        # Call Gemini once without Google Search to filter obvious no-trades cheaply.
        # Only proceed to full grounding if should_trade=True AND confidence >= 7.
        try:
            quick_verdict = await self._scorer._score_without_grounding(ctx, news_items)
        except Exception:
            log.exception("%s quick (no-grounding) score failed for %s; proceeding to full score",
                          _LOG_PREFIX, symbol)
            quick_verdict = None

        self._last_evaluated[symbol] = datetime.now(timezone.utc)

        if quick_verdict is not None and not quick_verdict.should_trade:
            log.info(
                "%s %s Stage-2 no-trade (conf=%d): %s",
                _LOG_PREFIX, symbol, quick_verdict.confidence,
                quick_verdict.skip_reason or quick_verdict.catalyst_summary,
            )
            return

        use_full_grounding = quick_verdict is None or quick_verdict.confidence >= 7
        if use_full_grounding:
            log.info(
                "%s %s Stage-2 passed (conf=%s); proceeding to full grounding",
                _LOG_PREFIX, symbol,
                quick_verdict.confidence if quick_verdict else "N/A (quick score failed)",
            )
            verdict, tech, _ = await self._scorer.score(ctx, news_items)
        else:
            # should_trade=True but confidence < 7: enter on no-grounding verdict.
            log.info(
                "%s %s entering on no-grounding verdict (conf=%d < 7)",
                _LOG_PREFIX, symbol, quick_verdict.confidence,
            )
            verdict, tech = quick_verdict, None
        if self._stats is not None:
            if tech is not None:
                self._stats.technicals_fetched += 1
            else:
                self._stats.technicals_failed += 1
        if verdict is None:
            log.warning("%s scorer returned no verdict for %s", _LOG_PREFIX, symbol)
            return
        ctx.technicals = tech

        biotech = is_biotech_catalyst(news_items)
        if biotech:
            min_conf = config.BIOTECH_GEMINI_CONFIDENCE
            min_mag = config.BIOTECH_GEMINI_MAGNITUDE
            log.info("%s [BIOTECH CATALYST] using lowered thresholds for %s",
                     _LOG_PREFIX, symbol)
        else:
            min_conf = config.MIN_GEMINI_CONFIDENCE
            min_mag = config.MIN_GEMINI_MAGNITUDE
            if is_analyst_action(news_items):
                log.info("%s [ANALYST CATALYST] detected for %s", _LOG_PREFIX, symbol)
            elif is_insider_buying(news_items):
                log.info("%s [INSIDER CATALYST] detected for %s", _LOG_PREFIX, symbol)
            else:
                log.info("%s [STANDARD] using standard thresholds for %s",
                         _LOG_PREFIX, symbol)

        # Technical signal gate: bearish technicals raise the bar or hard-skip.
        if verdict.technical_signal == "bearish":
            skip_reason = verdict.skip_reason or ""
            if any(kw in skip_reason.lower() for kw in ("rsi", "overextended", "extended")):
                log.info(
                    "%s [BEARISH TECH SKIP] %s: hard skip — %s",
                    _LOG_PREFIX, symbol, skip_reason,
                )
                return
            min_conf = min(10, min_conf + 2)
            log.info(
                "%s [BEARISH TECH] %s: bearish signal; confidence threshold raised to %d/10",
                _LOG_PREFIX, symbol, min_conf,
            )

        if (verdict.confidence < min_conf
                or verdict.magnitude_estimate < min_mag):
            log.info(
                "%s %s below thresholds: confidence=%d/%d magnitude=%d/%d",
                _LOG_PREFIX, symbol, verdict.confidence, min_conf,
                verdict.magnitude_estimate, min_mag,
            )
            return

        has_news = bool(news_items)
        sector = determine_sector(symbol, news_items)
        bought = False
        try:
            bought = await self._trader.execute_auto_buy(
                ctx,
                verdict,
                http_session=self._http,
                is_microcap=True,
                market_cap=market_cap,
                today_volume=today_volume,
                vol_ratio=vol_ratio,
                has_news=has_news,
                biotech_fast_track=biotech,
                is_biotech=biotech,
                sector=sector,
            )
        except Exception:
            log.exception("%s execute_auto_buy failed for %s", _LOG_PREFIX, symbol)
            return

        if not bought:
            log.info(
                "%s %s: no order placed (capacity limit); dedup/cooldown NOT set",
                _LOG_PREFIX, symbol,
            )
            return

        self._alerted[symbol] = datetime.now(timezone.utc)
        if self._cooldown_store:
            await self._cooldown_store.set_cooldown(symbol)
        log.info(
            "%s ALERTED %s score=%d/%d news=%s biotech=%s",
            _LOG_PREFIX, symbol, verdict.confidence, verdict.magnitude_estimate,
            has_news, biotech,
        )

    # --- helpers ---

    def _maybe_reset_dedup(self) -> None:
        today_et = datetime.now(tz=config.MARKET_TZ).date()
        if self._alerted_session != today_et:
            if self._alerted:
                log.info("%s new ET session %s; clearing %d-entry dedup set",
                         _LOG_PREFIX, today_et, len(self._alerted))
            self._alerted.clear()
            self._alerted_session = today_et

    def _is_recently_alerted(self, symbol: str) -> bool:
        last = self._alerted.get(symbol)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last) < timedelta(hours=config.MICROCAP_DEDUP_HOURS)

    async def _fetch_most_actives(self) -> list[dict]:
        params = {"by": "volume", "top": str(_MOST_ACTIVES_TOP_N)}
        try:
            async with self._http.get(
                _MOST_ACTIVES_URL,
                params=params,
                headers=_alpaca_headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning(
                        "%s most-actives HTTP %s: %s",
                        _LOG_PREFIX, resp.status, body[:200],
                    )
                    return []
                payload = await resp.json()
        except Exception:
            log.exception("%s most-actives request failed", _LOG_PREFIX)
            return []
        return list(payload.get("most_actives", []) or [])

    async def _fetch_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        params = {"symbols": ",".join(symbols), "feed": "iex"}
        try:
            async with self._http.get(
                _SNAPSHOTS_URL,
                params=params,
                headers=_alpaca_headers(),
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning(
                        "%s snapshots HTTP %s: %s",
                        _LOG_PREFIX, resp.status, body[:200],
                    )
                    return {}
                payload = await resp.json()
        except Exception:
            log.exception("%s snapshots request failed", _LOG_PREFIX)
            return {}
        # Endpoint returns either {"snapshots": {...}} or {symbol: {...}}.
        if "snapshots" in payload:
            return dict(payload["snapshots"])
        return {k: v for k, v in payload.items() if isinstance(v, dict)}

    async def _ensure_adv(self, symbols: list[str]) -> None:
        missing = [s for s in symbols if s not in self._adv_cache]
        if not missing:
            return
        try:
            adv = await prefetch_adv(self._hist, missing, lookback_days=30)
        except Exception:
            log.exception("%s ADV prefetch failed", _LOG_PREFIX)
            adv = {}
        cached_at = datetime.now(timezone.utc)
        for sym in missing:
            # Cache 0.0 for symbols with no ADV so we skip them next time too.
            self._adv_cache[sym] = (adv.get(sym, 0.0), cached_at)

    async def _get_market_cap(self, symbol: str, price: float) -> float | None:
        cached_entry = self._market_cap_cache.get(symbol)
        if cached_entry is not None:
            shares = cached_entry[0]
            return shares * price if shares else None
        try:
            asset = await asyncio.to_thread(self._trading.get_asset, symbol)
        except Exception:
            log.debug("%s get_asset(%s) failed", _LOG_PREFIX, symbol, exc_info=True)
            self._market_cap_cache[symbol] = (None, datetime.now(timezone.utc))
            return None
        # alpaca-py's Asset model does not surface shares_outstanding on every
        # tier — fall back to None so the symbol is skipped.
        shares = getattr(asset, "shares_outstanding", None)
        try:
            shares_f = float(shares) if shares is not None else 0.0
        except (TypeError, ValueError):
            shares_f = 0.0
        cached_at = datetime.now(timezone.utc)
        if shares_f <= 0:
            self._market_cap_cache[symbol] = (None, cached_at)
            return None
        self._market_cap_cache[symbol] = (shares_f, cached_at)
        return shares_f * price


async def supervised_run(
    trading_client: TradingClient,
    hist_client: StockHistoricalDataClient,
    http_session: aiohttp.ClientSession,
    scorer: Scorer,
    telegram: TelegramHandler,
    trader: Trader,
    calendar: MarketCalendar,
    watchlist_path: Path,
    stop_event: asyncio.Event,
    stats: SessionStats | None = None,
    cooldown_store: CooldownStore | None = None,
    intraday_gappers: set[str] | None = None,
    dynamic_sub_queue: "asyncio.Queue[str] | None" = None,
    shared_adv: dict[str, float] | None = None,
) -> None:
    """Run the screener with crash-restart so it never takes down the bot."""

    if not config.MICROCAP_ENABLED:
        log.info("%s disabled via MICROCAP_ENABLED; supervisor exiting", _LOG_PREFIX)
        return

    base_watchlist = set(load_watchlist(watchlist_path)) if watchlist_path.exists() else set()
    screener = MicrocapScreener(
        trading_client=trading_client,
        hist_client=hist_client,
        http_session=http_session,
        scorer=scorer,
        telegram=telegram,
        trader=trader,
        calendar=calendar,
        base_watchlist=base_watchlist,
        stats=stats,
        cooldown_store=cooldown_store,
        intraday_gappers=intraday_gappers,
        dynamic_sub_queue=dynamic_sub_queue,
        shared_adv=shared_adv,
    )

    while not stop_event.is_set():
        try:
            await screener.run(stop_event)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s screener crashed; restarting in 60s", _LOG_PREFIX)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60)
                return
            except asyncio.TimeoutError:
                pass
