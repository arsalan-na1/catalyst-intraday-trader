"""Tests for scorer cost-path behavior (Fixes 1, 2, 4).

All Gemini traffic is mocked — no real API calls. Enrichment fetches
(yfinance / Finnhub / Reddit / Alpaca) are monkeypatched to fast async
stubs so score() never touches the network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import config
import scorer


# --- minimal valid CatalystVerdict payload (only no-default fields required) ---
VALID_VERDICT = {
    "catalyst_found": True,
    "catalyst_summary": "Q3 earnings beat with raised guidance.",
    "catalyst_type": "earnings",
    "magnitude_estimate": 7,
    "confidence": 8,
    "suggested_entry": 100.0,
    "reasoning": "Strong beat, healthy momentum, manageable reversal risk.",
}
VALID_VERDICT_JSON = json.dumps(VALID_VERDICT)

_ENRICHMENT_FNS = [
    "get_technicals", "get_fundamentals", "get_insider_sentiment",
    "get_earnings_calendar", "get_congressional_trades", "get_earnings_surprise",
    "get_reddit_sentiment", "get_short_interest", "get_estimate_revisions",
    "get_insider_score", "get_sector_momentum",
]


def _patch_enrichment(monkeypatch):
    """Replace every enrichment fetch with a fast async stub returning None."""
    async def _none(*a, **k):
        return None
    for name in _ENRICHMENT_FNS:
        monkeypatch.setattr(scorer, name, _none)


def _make_scorer(text=VALID_VERDICT_JSON, usage=None):
    s = scorer.Scorer(hist_client=None)
    resp = SimpleNamespace(text=text)
    if usage is not None:
        resp.usage_metadata = SimpleNamespace(**usage)
    mock_gc = AsyncMock(return_value=resp)
    s._client = MagicMock()
    s._client.aio.models.generate_content = mock_gc
    return s, mock_gc


def _strong_ctx():
    # vol_ratio >= GROUNDING_VOL_THRESHOLD forces the (formerly grounded) branch.
    return scorer.TriggerContext(
        ticker="AAPL", price=100.0, price_move_pct=0.10, volume_ratio=5.0,
    )


# ---------------------------------------------------------------------------
# FIX 1 — wasted grounded research call removed (behavior-neutral verdict)
# ---------------------------------------------------------------------------


def test_grounded_score_makes_single_gemini_call(monkeypatch):
    """The strong-signal path must make exactly ONE Gemini call (the verdict).

    Before Fix 1 it made two (research + verdict). The research output was
    never used in the prompt, so removing it must leave the verdict identical.
    """
    _patch_enrichment(monkeypatch)
    s, mock_gc = _make_scorer()

    verdict, tech, quant = asyncio.run(s.score(_strong_ctx(), []))

    # Exactly one Gemini call — the grounded research call is gone.
    assert mock_gc.await_count == 1
    # That single call must NOT use Google Search grounding.
    _, kwargs = mock_gc.await_args
    cfg = kwargs["config"]
    assert getattr(cfg, "tools", None) in (None, [])
    # Verdict parsed and returned unchanged.
    assert verdict is not None
    assert verdict.confidence == 8
    assert verdict.catalyst_type == "earnings"
    # 3-tuple invariant preserved.
    assert quant is not None and isinstance(quant, dict)


def test_grounded_path_never_bills_grounded_flat_cost(monkeypatch):
    """The expensive $0.016 grounded research charge must never be applied.

    (Cost is now token-based — see Fix 2 tests — but this guards the Fix 1
    intent: the grounded call is gone, so the grounded flat constant is never
    added to the monthly tally.)
    """
    _patch_enrichment(monkeypatch)
    s, _ = _make_scorer(usage={"prompt_token_count": 1500, "candidates_token_count": 300})

    asyncio.run(s.score(_strong_ctx(), []))

    assert s.monthly_cost_estimate < config.GEMINI_COST_PER_GROUNDED_CALL


def test_research_helpers_removed():
    """Dead grounded-research helpers must be gone after Fix 1."""
    s = scorer.Scorer(hist_client=None)
    assert not hasattr(s, "_research_with_grounding")
    assert not hasattr(s, "_build_research_prompt")


def test_verdict_prompt_does_not_depend_on_research(monkeypatch):
    """_build_verdict_prompt must not accept or embed a research argument.

    Locks the behavior-neutrality guarantee: the verdict prompt is built from
    trigger/news/technical context only.
    """
    import inspect
    sig = inspect.signature(scorer.Scorer._build_verdict_prompt)
    assert "research" not in sig.parameters


# ---------------------------------------------------------------------------
# FIX 2 — real token-based cost accounting
# ---------------------------------------------------------------------------


def test_compute_cost_flash_lite():
    cost = scorer.compute_gemini_cost("gemini-2.5-flash-lite", 1000, 500)
    assert cost == pytest.approx(1000 / 1e6 * 0.10 + 500 / 1e6 * 0.40)


def test_compute_cost_flash():
    cost = scorer.compute_gemini_cost("gemini-2.5-flash", 2000, 1000)
    assert cost == pytest.approx(2000 / 1e6 * 0.30 + 1000 / 1e6 * 2.50)


def test_compute_cost_unknown_model_contributes_zero_token_cost():
    assert scorer.compute_gemini_cost("made-up-model", 1000, 1000) == 0.0


def test_compute_cost_handles_none_token_counts():
    assert scorer.compute_gemini_cost("gemini-2.5-flash-lite", None, None) == 0.0


def test_compute_cost_grounding_free_under_quota_by_default():
    # GEMINI_GROUNDING_COST_PER_1K defaults to 0.0 (modeled as under free quota).
    assert scorer.compute_gemini_cost(
        "gemini-2.5-flash-lite", 0, 0, grounded_queries=10
    ) == 0.0


def test_compute_cost_grounding_billable_when_rate_set(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_GROUNDING_COST_PER_1K", 35.0)
    # 2 queries at $35/1k = $0.07
    assert scorer.compute_gemini_cost(
        "gemini-2.5-flash-lite", 0, 0, grounded_queries=2
    ) == pytest.approx(0.07)


def test_extract_usage_reads_token_counts():
    s = scorer.Scorer(hist_client=None)
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1200, candidates_token_count=300, total_token_count=1500))
    assert s._extract_usage(resp) == (1200, 300)


def test_extract_usage_missing_metadata_returns_zero():
    s = scorer.Scorer(hist_client=None)
    assert s._extract_usage(SimpleNamespace(text="x")) == (0, 0)


def test_extract_usage_none_fields_return_zero():
    s = scorer.Scorer(hist_client=None)
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=None, candidates_token_count=None))
    assert s._extract_usage(resp) == (0, 0)


def test_score_accrues_real_token_cost_and_breakdown(monkeypatch):
    """score() must charge the REAL token cost (not the flat $0.002) and record
    a per-model token + dollar breakdown."""
    _patch_enrichment(monkeypatch)
    s, _ = _make_scorer(usage={"prompt_token_count": 1500, "candidates_token_count": 300})

    asyncio.run(s.score(_strong_ctx(), []))

    expected = scorer.compute_gemini_cost(config.GEMINI_MODEL_VERDICT, 1500, 300)
    assert s.monthly_cost_estimate == pytest.approx(expected)
    bucket = s.usage_by_model[config.GEMINI_MODEL_VERDICT]
    assert bucket["calls"] == 1
    assert bucket["prompt_tokens"] == 1500
    assert bucket["output_tokens"] == 300
    assert bucket["cost_usd"] == pytest.approx(expected)


def test_append_performance_persists_per_model_usage(tmp_path, monkeypatch):
    """The per-day, per-model breakdown lands in performance.json under the
    date entry, alongside the top-level monthly_cost_estimate."""
    import json as _json
    import daily_summary

    perf = tmp_path / "performance.json"
    monkeypatch.setattr(config, "PERFORMANCE_FILE", perf)

    stats = SimpleNamespace(
        realized_pnl=0.0, trades_taken=0, spikes_detected=0, trade_records=[],
    )
    usage = {"gemini-2.5-flash-lite": {
        "calls": 3, "prompt_tokens": 4500, "output_tokens": 900, "cost_usd": 0.00081}}
    from datetime import datetime, timezone
    daily_summary._append_performance(
        stats, gemini_calls=3, balance=100000.0, today=datetime.now(timezone.utc),
        monthly_cost_estimate=0.0008, monthly_key="2026-05",
        gemini_usage=usage,
    )
    data = _json.loads(perf.read_text())
    # monthly_cost_estimate is persisted rounded to 4 dp (pre-existing behavior).
    assert data["monthly_cost_estimate"] == pytest.approx(0.0008)
    date_key = [k for k in data if k.count("-") == 2 and len(k) == 10][0]
    assert data[date_key]["gemini_usage"] == usage


# ---------------------------------------------------------------------------
# FIX 3 — score() reuses pre-fetched technicals (no double fetch)
# ---------------------------------------------------------------------------


def test_score_reuses_prefetched_tech_without_refetch(monkeypatch):
    """When tech is passed in, score() must NOT call get_technicals again and
    must return the exact object it was given."""
    _patch_enrichment(monkeypatch)

    called = {"n": 0}

    async def _tracking_get_technicals(*a, **k):
        called["n"] += 1
        return SimpleNamespace(rsi_14=50.0, trend="uptrend", dist_from_52w_high_pct=-5.0)

    monkeypatch.setattr(scorer, "get_technicals", _tracking_get_technicals)

    # hist_client set so score() WOULD fetch technicals if not given them.
    s = scorer.Scorer(hist_client=object())
    mock_gc = AsyncMock(return_value=SimpleNamespace(text=VALID_VERDICT_JSON))
    s._client = MagicMock()
    s._client.aio.models.generate_content = mock_gc

    prefetched = SimpleNamespace(rsi_14=61.0, trend="uptrend", dist_from_52w_high_pct=-8.0)
    verdict, tech, quant = asyncio.run(s.score(_strong_ctx(), [], tech=prefetched))

    assert called["n"] == 0          # never re-fetched
    assert tech is prefetched        # the same object flows back out


def test_score_still_fetches_tech_when_not_provided(monkeypatch):
    """Back-compat: with no tech kwarg and a hist_client, score() fetches as before."""
    _patch_enrichment(monkeypatch)

    called = {"n": 0}

    async def _tracking_get_technicals(*a, **k):
        called["n"] += 1
        return SimpleNamespace(rsi_14=50.0, trend="uptrend", dist_from_52w_high_pct=-5.0)

    monkeypatch.setattr(scorer, "get_technicals", _tracking_get_technicals)

    s = scorer.Scorer(hist_client=object())
    mock_gc = AsyncMock(return_value=SimpleNamespace(text=VALID_VERDICT_JSON))
    s._client = MagicMock()
    s._client.aio.models.generate_content = mock_gc

    verdict, tech, quant = asyncio.run(s.score(_strong_ctx(), []))

    assert called["n"] == 1
    assert tech is not None and tech.rsi_14 == 50.0


def test_will_score_call_model_mirrors_score_guards():
    import time as _time
    from datetime import datetime, timezone

    s = scorer.Scorer(hist_client=None)
    ctx = _strong_ctx()

    assert s.will_score_call_model(ctx) is True          # fresh → billable call

    s._budget_mode = "halted"
    assert s.will_score_call_model(ctx) is False          # halted → no call
    s._budget_mode = "normal"

    s._last_scored[ctx.ticker] = _time.monotonic()
    assert s.will_score_call_model(ctx) is False           # cooldown → no call
    s._last_scored.clear()

    s._cache[ctx.ticker] = (datetime.now(timezone.utc), object())
    assert s.will_score_call_model(ctx) is False           # cache hit → no call


def test_will_score_call_model_true_under_ungrounded_only():
    # Only 'halted' short-circuits scoring; the 80% throttle does not.
    s = scorer.Scorer(hist_client=None)
    s._budget_mode = "ungrounded_only"
    assert s.will_score_call_model(_strong_ctx()) is True


def test_note_scored_starts_cooldown():
    # A pre-Gemini reject must start the 30-min scoring cooldown so repeated
    # triggers are suppressed exactly as the old post-score gate flow did.
    s = scorer.Scorer(hist_client=None)
    ctx = _strong_ctx()
    assert s.will_score_call_model(ctx) is True
    s.note_scored(ctx.ticker)
    assert s.will_score_call_model(ctx) is False


# ---------------------------------------------------------------------------
# FIX 4 — incremental cost-state persistence (survives mid-day restart)
# ---------------------------------------------------------------------------


def test_cost_state_persists_and_restores_same_month(tmp_path):
    path = tmp_path / "gemini_cost.json"

    s1 = scorer.Scorer(hist_client=None)
    s1.monthly_cost_estimate = 1.2345
    s1.usage_by_model = {"gemini-2.5-flash-lite": {
        "calls": 5, "prompt_tokens": 7000, "output_tokens": 1500, "cost_usd": 1.2345}}
    scorer.persist_cost_state(s1, path)
    assert path.exists()

    # Simulate a mid-day restart: a fresh Scorer must resume the true tally.
    s2 = scorer.Scorer(hist_client=None)
    assert s2.monthly_cost_estimate == 0.0
    scorer.restore_cost_state(s2, path)
    assert s2.monthly_cost_estimate == pytest.approx(1.2345)
    assert s2.usage_by_model["gemini-2.5-flash-lite"]["calls"] == 5


def test_cost_state_ignored_across_month_boundary(tmp_path):
    import json as _json
    path = tmp_path / "gemini_cost.json"
    path.write_text(_json.dumps({
        "month": "1999-01", "monthly_cost_estimate": 9.99,
        "usage_day": "1999-01-15", "usage_by_model": {},
    }))
    s = scorer.Scorer(hist_client=None)
    scorer.restore_cost_state(s, path)
    assert s.monthly_cost_estimate == 0.0  # stale month must not carry over


def test_cost_state_restore_reenters_halted_mode(tmp_path):
    import json as _json
    from datetime import datetime as _dt
    month = _dt.now(tz=config.MARKET_TZ).strftime("%Y-%m")
    path = tmp_path / "gemini_cost.json"
    path.write_text(_json.dumps({
        "month": month,
        "monthly_cost_estimate": config.MONTHLY_GEMINI_BUDGET_USD + 1.0,
        "usage_day": "", "usage_by_model": {},
    }))
    s = scorer.Scorer(hist_client=None)
    scorer.restore_cost_state(s, path)
    assert s._budget_mode == "halted"


def test_restore_cost_state_missing_file_is_noop(tmp_path):
    s = scorer.Scorer(hist_client=None)
    scorer.restore_cost_state(s, tmp_path / "does_not_exist.json")
    assert s.monthly_cost_estimate == 0.0  # no crash, no change


def test_restore_cost_state_reenters_ungrounded_only_at_80pct(tmp_path):
    import json as _json
    from datetime import datetime as _dt
    month = _dt.now(tz=config.MARKET_TZ).strftime("%Y-%m")
    path = tmp_path / "gemini_cost.json"
    path.write_text(_json.dumps({
        "month": month,
        "monthly_cost_estimate": config.MONTHLY_GEMINI_BUDGET_USD * 0.85,
        "usage_day": "", "usage_by_model": {},
    }))
    s = scorer.Scorer(hist_client=None)
    scorer.restore_cost_state(s, path)
    assert s._budget_mode == "ungrounded_only"


def test_account_usage_reconciles_flat_reservation():
    # _record_call_cost pre-charges the flat ungrounded constant; _account_usage
    # must back it out and leave only the real token cost (no double count).
    s = scorer.Scorer(hist_client=None)
    s.monthly_cost_estimate = config.GEMINI_COST_PER_UNGROUNDED_CALL
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1000, candidates_token_count=200))
    real = s._account_usage(config.GEMINI_MODEL_VERDICT, resp, is_grounded=False)
    expected = scorer.compute_gemini_cost(config.GEMINI_MODEL_VERDICT, 1000, 200)
    assert real == pytest.approx(expected)
    assert s.monthly_cost_estimate == pytest.approx(expected)


def test_account_usage_unknown_model_records_zero_cost():
    s = scorer.Scorer(hist_client=None)
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1000, candidates_token_count=500))
    real = s._account_usage("made-up-model", resp, is_grounded=False)
    assert real == 0.0
    bucket = s.usage_by_model["made-up-model"]
    assert bucket["calls"] == 1
    assert bucket["cost_usd"] == 0.0


def test_roll_usage_day_resets_on_new_day():
    from datetime import datetime as _dt
    s = scorer.Scorer(hist_client=None)
    s.usage_day = "1999-01-01"
    s.usage_by_model = {"x": {"calls": 1}}
    s._roll_usage_day()
    today = _dt.now(tz=config.MARKET_TZ).strftime("%Y-%m-%d")
    assert s.usage_day == today
    assert s.usage_by_model == {}
    # Same-day idempotence — does not clear a populated bucket.
    s.usage_by_model = {"y": {"calls": 2}}
    s._roll_usage_day()
    assert s.usage_by_model == {"y": {"calls": 2}}
