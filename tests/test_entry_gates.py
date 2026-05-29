"""Fix 3 — deterministic technical gates run BEFORE the Gemini scoring call.

These tests drive _process_trigger with lightweight stubs (no network, no real
Gemini/Alpaca). They lock the invariant: a candidate that a deterministic
technical gate (RSI / downtrend / falling-knife) will reject must NOT incur a
Gemini scoring call, while the accept/reject outcome is unchanged.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time as _dt_time, timezone
from types import SimpleNamespace

import pytest

import bot


# --- stubs -----------------------------------------------------------------


class _StubScorer:
    def __init__(self, tech, verdict):
        self.tech = tech
        self.verdict = verdict
        self.score_calls = 0
        self.fetch_calls = 0
        self.will_call = True
        self.noted = []
        self._current_regime_label = "unknown"
        self._hist_client = None

    def will_score_call_model(self, ctx):
        return self.will_call

    def note_scored(self, ticker):
        self.noted.append(ticker)

    async def fetch_technicals(self, ctx):
        self.fetch_calls += 1
        return self.tech

    async def score(self, ctx, news, tech=None):
        self.score_calls += 1
        return (
            self.verdict,
            tech if tech is not None else self.tech,
            {"short_interest": None, "insider_score": None, "estimate_revisions": None},
        )


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


def _verdict(should_trade=True, confidence=9):
    return SimpleNamespace(
        should_trade=should_trade, confidence=confidence, magnitude_estimate=8,
        catalyst_found=True, skip_reason=None,
    )


def _run(event, scorer, trader, stats, monkeypatch):
    async def _news(*a, **k):
        return []
    monkeypatch.setattr(bot, "fetch_news", _news)
    monkeypatch.setattr(bot, "is_biotech_catalyst", lambda news: False)
    monkeypatch.setattr(bot, "is_analyst_action", lambda news: False)
    monkeypatch.setattr(bot, "is_insider_buying", lambda news: False)
    monkeypatch.setattr(bot, "determine_sector", lambda t, news: "tech")
    # Force non-premarket regardless of wall-clock so the execute path is reached.
    monkeypatch.setattr(bot, "_MARKET_OPEN", _dt_time(0, 0, 0))
    telegram, cooldown = _StubTelegram(), _StubCooldown()
    asyncio.run(bot._process_trigger(event, scorer, telegram, trader, stats, None, cooldown))


# --- tests -----------------------------------------------------------------


def test_overbought_candidate_skips_gemini(monkeypatch):
    # RSI 85 > RSI_MAX_ENTRY (72) → rejected by the deterministic gate.
    tech = SimpleNamespace(rsi_14=85.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict())
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 0     # Gemini NOT called for a doomed candidate
    assert trader.execute_calls == 0   # outcome unchanged: still rejected


def test_downtrend_candidate_skips_gemini(monkeypatch):
    tech = SimpleNamespace(rsi_14=55.0, trend="downtrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict())
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 0
    assert trader.execute_calls == 0


def test_pre_gemini_reject_marks_scoring_cooldown(monkeypatch):
    # The pre-Gemini reject must start the scoring cooldown (note_scored) so
    # rapid re-triggers are suppressed for 30 min — matching the old behavior
    # where score() ran (and set the cooldown) before the gate rejected.
    tech = SimpleNamespace(rsi_14=85.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict())
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 0
    assert scorer.noted == ["AAPL"]


def test_healthy_candidate_scores_once_and_trades(monkeypatch):
    tech = SimpleNamespace(rsi_14=55.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict())
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 1     # exactly one Gemini scoring call
    assert trader.execute_calls == 1   # accepted, as before


def test_cache_or_cooldown_candidate_is_not_pre_gated(monkeypatch):
    # will_score_call_model() False (verdict cache / cooldown) → the pre-Gemini
    # gate must NOT short-circuit; score() is still called so this path behaves
    # exactly as before. The authoritative post-score gate still rejects the
    # overbought tech, so no trade — accept/reject outcome unchanged.
    tech = SimpleNamespace(rsi_14=85.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict())
    scorer.will_call = False
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 1     # not pre-gated despite overbought tech
    assert trader.execute_calls == 0   # post-score gate still rejects it


def test_technically_rejected_none_tech_fail_closed():
    # Missing technicals → downtrend & falling-knife fail closed (default) → reject.
    assert bot._technically_rejected(None) is True


def test_technically_rejected_healthy_passes():
    tech = SimpleNamespace(rsi_14=55.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    assert bot._technically_rejected(tech) is False


def test_technically_rejected_overbought():
    tech = SimpleNamespace(rsi_14=85.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    assert bot._technically_rejected(tech) is True


def test_technically_rejected_downtrend():
    tech = SimpleNamespace(rsi_14=55.0, trend="downtrend", dist_from_52w_high_pct=-5.0)
    assert bot._technically_rejected(tech) is True


def test_technically_rejected_falling_knife():
    # >40% below 52w high and trend not uptrend → falling-knife reject.
    tech = SimpleNamespace(rsi_14=55.0, trend="ranging", dist_from_52w_high_pct=-50.0)
    assert bot._technically_rejected(tech) is True


def test_gemini_skip_still_rejects_when_technicals_pass(monkeypatch):
    # Technicals are healthy, so Gemini IS consulted; should_trade=False rejects.
    tech = SimpleNamespace(rsi_14=55.0, trend="uptrend", dist_from_52w_high_pct=-5.0)
    scorer = _StubScorer(tech, _verdict(should_trade=False))
    trader = _StubTrader()
    _run(_make_event(), scorer, trader, _make_stats(), monkeypatch)
    assert scorer.score_calls == 1
    assert trader.execute_calls == 0
