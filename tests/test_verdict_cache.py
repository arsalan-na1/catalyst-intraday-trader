"""B2 — verdict cache must carry the technicals it was scored with.

Bug: on a verdict-cache hit, score() returned tech=None. The post-score
downtrend / falling-knife gates in bot._process_trigger fail CLOSED on None
(TREND_GATE_FAIL_CLOSED defaults true), so a re-trigger inside the 30-min
verdict-cache window could NEVER act on its cached verdict — even when the
technicals were healthy. Healthy trades were silently dropped.

Fix (cache-the-technicals): store (now, verdict, tech) in the cache; _cache_get
returns (verdict, tech); score() returns the cached technicals on a hit so the
gates evaluate on real data exactly as they would on a fresh score.

These tests prove:
  1. a cache hit returns the real cached technicals (not None);
  2. will_score_call_model tolerates the 3-tuple cache entry (no crash);
  3. a cached bullish verdict + healthy tech is tradeable on re-trigger (the fix);
  4. a cached verdict + downtrend / falling-knife / overbought tech is STILL
     gated out (the fix opened no hole);
  5. the cached-path accept-set matches the gate rules — i.e. a cached
     re-trigger trades iff a fresh score with the same technicals would.

A cache hit short-circuits score() before any enrichment/Gemini call, so a real
Scorer needs no network and no mocking here.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time as _dt_time, timezone
from types import SimpleNamespace

import pytest

import bot
import scorer


# --- builders --------------------------------------------------------------


def _tech(rsi=55.0, trend="uptrend", dist=-5.0):
    return SimpleNamespace(rsi_14=rsi, trend=trend, dist_from_52w_high_pct=dist)


def _buy_verdict():
    return SimpleNamespace(
        should_trade=True, confidence=9, magnitude_estimate=8,
        catalyst_found=True, skip_reason=None,
    )


def _ctx(ticker="AAPL"):
    return scorer.TriggerContext(
        ticker=ticker, price=100.0, price_move_pct=0.10, volume_ratio=5.0,
    )


HEALTHY = _tech(rsi=55.0, trend="uptrend", dist=-5.0)
DOWNTREND = _tech(rsi=55.0, trend="downtrend", dist=-5.0)
FALLING_KNIFE = _tech(rsi=55.0, trend="ranging", dist=-50.0)
OVERBOUGHT = _tech(rsi=85.0, trend="uptrend", dist=-5.0)


# --- unit: score() cache-hit returns the cached technicals -----------------


def test_cache_hit_returns_cached_technicals():
    # (1) The core fix: a hit returns the verdict AND the technicals it was
    # cached with — not None.
    s = scorer.Scorer(hist_client=None)
    ctx = _ctx()
    verdict = _buy_verdict()
    s._cache_put(ctx.ticker, verdict, HEALTHY)

    v, t, q = asyncio.run(s.score(ctx, []))

    assert v is verdict
    assert t is HEALTHY          # not None — the bug was returning None here
    assert q == {}


def test_cache_hit_with_no_tech_returns_none_tech():
    # Consistency: if technicals genuinely weren't available when scored, the
    # cache stores None and the hit returns None — the gates then fail closed,
    # which is the correct (unchanged) fail-safe behavior.
    s = scorer.Scorer(hist_client=None)
    ctx = _ctx()
    verdict = _buy_verdict()
    s._cache_put(ctx.ticker, verdict)  # no tech supplied

    v, t, q = asyncio.run(s.score(ctx, []))

    assert v is verdict
    assert t is None


def test_will_score_call_model_tolerates_three_tuple_cache_entry():
    # (2) Guards the easy-to-miss blast-radius bug: will_score_call_model
    # unpacks the cache entry. With the 3-tuple it must not raise, and a fresh
    # cached entry still means "no billable call".
    s = scorer.Scorer(hist_client=None)
    ctx = _ctx()
    s._cache_put(ctx.ticker, _buy_verdict(), HEALTHY)

    assert s.will_score_call_model(ctx) is False


# --- integration: cached re-trigger through the real gate path -------------


class _StubTrader:
    def __init__(self):
        self.session_blocked_tickers = set()
        self._hist_client = None  # None → Kronos block is skipped
        self.execute_calls = 0

    async def execute_auto_buy(self, ctx, verdict, **kwargs):
        self.execute_calls += 1
        return True


class _StubTelegram:
    async def send_premarket_watch(self, **k):
        return None

    async def send_message(self, *a, **k):
        return None


class _StubCooldown:
    def is_in_cooldown(self, ticker, window_minutes=0):
        return False

    async def set_cooldown(self, ticker):
        return None


def _make_stats():
    return SimpleNamespace(
        alerts_sent=0, insider_bullish_seen=0, revision_strong_seen=0,
        squeeze_setups_seen=0, technicals_failed=0, technicals_fetched=0,
    )


def _make_event():
    return SimpleNamespace(
        ticker="AAPL", price=100.0, price_move_pct=0.10, volume_ratio=6.0,
        trigger_type="intraday", window_vwap=None, is_earnings=False,
        triggered_at=datetime.now(timezone.utc),
    )


def _run_cached(cached_tech, monkeypatch, verdict=None):
    """Drive _process_trigger with a REAL Scorer whose cache is primed with
    `verdict` (default: a bullish buy) + `cached_tech`, so the run exercises the
    verdict-cache hit path end-to-end. Returns the stub trader so the caller can
    assert on execute_calls."""
    async def _news(*a, **k):
        return []
    monkeypatch.setattr(bot, "fetch_news", _news)
    monkeypatch.setattr(bot, "is_biotech_catalyst", lambda news: False)
    monkeypatch.setattr(bot, "is_analyst_action", lambda news: False)
    monkeypatch.setattr(bot, "is_insider_buying", lambda news: False)
    monkeypatch.setattr(bot, "determine_sector", lambda t, news: "tech")
    monkeypatch.setattr(bot, "_MARKET_OPEN", _dt_time(0, 0, 0))

    s = scorer.Scorer(hist_client=None)
    s._cache_put("AAPL", verdict if verdict is not None else _buy_verdict(), cached_tech)
    # Sanity: a primed fresh cache means no billable call is imminent, so the
    # pre-Gemini gate is skipped and score() serves the cached verdict.
    assert s.will_score_call_model(_ctx()) is False

    trader = _StubTrader()
    asyncio.run(bot._process_trigger(
        _make_event(), s, _StubTelegram(), trader, _make_stats(), None, _StubCooldown(),
    ))
    return s, trader


def test_cached_bullish_verdict_with_healthy_tech_trades_on_retrigger(monkeypatch):
    # (3) THE FIX: a cached bullish verdict whose cached technicals are healthy
    # now trades on the re-trigger (previously dropped via fail-closed None).
    s, trader = _run_cached(HEALTHY, monkeypatch)
    assert trader.execute_calls == 1


def test_cached_verdict_with_downtrend_tech_still_gated(monkeypatch):
    # (4) NO HOLE: cached downtrend technicals are still rejected by the
    # downtrend gate — the fix did not bypass the gates on cache hits.
    s, trader = _run_cached(DOWNTREND, monkeypatch)
    assert trader.execute_calls == 0


def test_cached_verdict_with_falling_knife_tech_still_gated(monkeypatch):
    # (4) NO HOLE: cached falling-knife technicals are still rejected.
    s, trader = _run_cached(FALLING_KNIFE, monkeypatch)
    assert trader.execute_calls == 0


@pytest.mark.parametrize(
    "cached_tech, should_trade",
    [
        (HEALTHY, True),        # passes all gates → trades
        (OVERBOUGHT, False),    # RSI ceiling gate → no trade
        (DOWNTREND, False),     # downtrend gate → no trade
        (FALLING_KNIFE, False), # falling-knife gate → no trade
    ],
)
def test_cached_path_accept_set_matches_gate_rules(cached_tech, should_trade, monkeypatch):
    # (5) The cached-path accept-set equals the gate rules applied to the cached
    # technicals: a cached re-trigger trades iff a fresh score with the same
    # technicals would. Only the HEALTHY case is the intended new behavior; the
    # rest stay rejected, so the accept-set is unchanged except that one case.
    s, trader = _run_cached(cached_tech, monkeypatch)
    assert trader.execute_calls == (1 if should_trade else 0)


def test_cached_path_rsi_gate_fails_open_on_missing_rsi(monkeypatch):
    # The RSI ceiling gate fails OPEN (unlike the trend gates which fail closed):
    # cached technicals with rsi_14=None but an otherwise-healthy trend should
    # still trade. Covers the one gate whose missing-data behavior differs.
    s, trader = _run_cached(_tech(rsi=None, trend="uptrend", dist=-5.0), monkeypatch)
    assert trader.execute_calls == 1


def test_cached_verdict_with_none_tech_gated_closed(monkeypatch):
    # When technicals were unavailable at score time the cache holds None; on the
    # re-trigger the trend / falling-knife gates fail CLOSED — the correct, and
    # unchanged, fail-safe. Proven at the integration level, not just by comment.
    s, trader = _run_cached(None, monkeypatch)
    assert trader.execute_calls == 0


def test_cached_non_buy_verdict_not_traded_even_with_healthy_tech(monkeypatch):
    # A cached should_trade=False verdict must hit the [GEMINI SKIP] return even
    # with healthy cached technicals — the fix must not turn cached declines
    # into trades.
    skip_verdict = SimpleNamespace(
        should_trade=False, confidence=9, magnitude_estimate=2,
        catalyst_found=True, skip_reason="weak catalyst",
    )
    s, trader = _run_cached(HEALTHY, monkeypatch, verdict=skip_verdict)
    assert trader.execute_calls == 0


def test_cached_low_confidence_verdict_gated(monkeypatch):
    # A cached should_trade=True verdict below MIN_GEMINI_CONFIDENCE must still be
    # rejected by the bot-side confidence floor on the cached path.
    import config
    monkeypatch.setattr(config, "MIN_GEMINI_CONFIDENCE", 7)
    low_conf = SimpleNamespace(
        should_trade=True, confidence=6, magnitude_estimate=8,
        catalyst_found=True, skip_reason=None,
    )
    s, trader = _run_cached(HEALTHY, monkeypatch, verdict=low_conf)
    assert trader.execute_calls == 0


def test_cache_get_evicts_and_returns_none_past_ttl():
    # The 3-tuple expiry contract: an entry older than VERDICT_CACHE_MINUTES is
    # unpacked cleanly, returns None, and is evicted from the cache.
    import config
    from datetime import timedelta
    s = scorer.Scorer(hist_client=None)
    ctx = _ctx()
    stale_at = datetime.now(timezone.utc) - timedelta(minutes=config.VERDICT_CACHE_MINUTES + 1)
    s._cache[ctx.ticker] = (stale_at, _buy_verdict(), HEALTHY)

    assert s._cache_get(ctx.ticker) is None
    assert ctx.ticker not in s._cache  # expired entry evicted
