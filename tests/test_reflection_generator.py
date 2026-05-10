"""Tests for ReflectionGenerator.

All Gemini calls are mocked — no real API traffic, no live network. The
close path must NEVER block on or be affected by Gemini failures, so each
failure path is exercised with an explicit assertion that the store still
receives a record (with reflection_text=None).
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Stub Scorer — exposes the exact attributes/methods ReflectionGenerator uses,
# without touching the real google-genai client.
# ---------------------------------------------------------------------------


class _StubLimiter:
    async def acquire(self) -> None:
        return None


class _StubScorer:
    """Minimal Scorer stand-in. The generator only needs:
        _budget_mode, _check_hourly_limit, _record_call_cost, _limiter,
        _consume_hourly_slot, _client, calls_today, calls_skipped,
        calls_ungrounded.
    """

    def __init__(self) -> None:
        self._budget_mode = "normal"
        self._limiter = _StubLimiter()
        self.calls_today = 0
        self.calls_skipped = 0
        self.calls_ungrounded = 0
        self.cost_calls: list[bool] = []  # records is_grounded values
        self.allow_call = True
        self.allow_hourly = True
        # Mocked GenAI client; _call_gemini calls
        # self._client.aio.models.generate_content(...).
        self._client = MagicMock()
        self._client.aio.models.generate_content = AsyncMock()

    def _check_hourly_limit(self) -> bool:
        return self.allow_hourly

    def _record_call_cost(self, *, is_grounded: bool) -> bool:
        self.cost_calls.append(is_grounded)
        return self.allow_call

    def _consume_hourly_slot(self) -> None:
        return None


def _make_generator(tmp_path: Path):
    from reflection_store import ReflectionStore
    from reflection_generator import ReflectionGenerator

    store = ReflectionStore(path=tmp_path / "reflections.jsonl")
    scorer = _StubScorer()
    # hist_client=None → alpha computation returns None deterministically.
    gen = ReflectionGenerator(store=store, scorer=scorer, hist_client=None)
    return store, scorer, gen


def _trade_kwargs(**overrides):
    now = datetime.now(timezone.utc)
    base = dict(
        ticker="AAPL",
        side="long",
        entry_price=100.0,
        exit_price=105.0,
        opened_at=now - timedelta(minutes=45),
        closed_at=now,
        raw_return_pct=0.05,
        exit_reason="TP +5%",
        entry_rationale_text="Strong Q3 beat with raised guidance.",
        entry_top_signals=["catalyst:earnings-strong", "tech:bullish"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_persists_full_record(tmp_path: Path):
    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._client.aio.models.generate_content.return_value = SimpleNamespace(
            text="Solid catalyst trade — held the right amount, exited on TP without giving back gains."
        )
        await gen.on_trade_closed(**_trade_kwargs())
        return store.get_recent("AAPL")

    out = asyncio.run(_go())
    assert len(out) == 1
    rec = out[0]
    assert rec["ticker"] == "AAPL"
    assert rec["reflection_text"].startswith("Solid catalyst trade")
    assert rec["reflection_model"]  # populated
    assert rec["reflection_ts"]  # always set
    assert rec["raw_return_pct"] == pytest.approx(0.05)
    # alpha is None when hist_client is None
    assert rec["alpha_vs_spy_pct"] is None
    assert rec["trade_id"].startswith("AAPL-")
    assert rec["side"] == "long"
    assert rec["exit_reason"] == "TP +5%"


# ---------------------------------------------------------------------------
# Gemini failure paths — close-path equivalent NEVER blocks; record is always
# persisted with reflection_text=None.
# ---------------------------------------------------------------------------


def test_close_path_does_not_block_on_gemini_failure(tmp_path: Path):
    """on_trade_closed completes promptly even when Gemini raises."""

    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._client.aio.models.generate_content.side_effect = RuntimeError(
            "gemini boom"
        )
        # If the generator surfaces the error, this would raise — instead it
        # must swallow and persist a null-text record.
        await gen.on_trade_closed(**_trade_kwargs())
        return store.get_recent("AAPL")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert out[0]["reflection_text"] is None
    assert out[0]["reflection_model"] is None


def test_reflection_persisted_with_null_text_on_empty_response(tmp_path: Path):
    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._client.aio.models.generate_content.return_value = SimpleNamespace(
            text=""
        )
        await gen.on_trade_closed(**_trade_kwargs())
        return store.get_recent("AAPL")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert out[0]["reflection_text"] is None


def test_skipped_when_budget_halted(tmp_path: Path):
    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._budget_mode = "halted"
        await gen.on_trade_closed(**_trade_kwargs())
        # Record still written (audit trail), but reflection_text=None and
        # the Gemini client is never invoked.
        scorer._client.aio.models.generate_content.assert_not_awaited()
        return store.get_recent("AAPL"), scorer.calls_skipped

    out, skipped = asyncio.run(_go())
    assert len(out) == 1
    assert out[0]["reflection_text"] is None
    assert skipped == 1


def test_skipped_when_hourly_limit_full(tmp_path: Path):
    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer.allow_hourly = False
        await gen.on_trade_closed(**_trade_kwargs())
        scorer._client.aio.models.generate_content.assert_not_awaited()
        return store.get_recent("AAPL")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert out[0]["reflection_text"] is None


def test_disabled_via_config_skips_completely(tmp_path: Path, monkeypatch):
    """REFLECTION_ENABLED=False short-circuits the entire path."""
    import config

    monkeypatch.setattr(config, "REFLECTION_ENABLED", False)

    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        await gen.on_trade_closed(**_trade_kwargs())
        scorer._client.aio.models.generate_content.assert_not_awaited()
        return store.get_recent("AAPL")

    out = asyncio.run(_go())
    assert out == []  # no record persisted when feature off


# ---------------------------------------------------------------------------
# Cost / counter accounting
# ---------------------------------------------------------------------------


def test_successful_call_is_billed_as_ungrounded(tmp_path: Path):
    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._client.aio.models.generate_content.return_value = SimpleNamespace(
            text="ok"
        )
        await gen.on_trade_closed(**_trade_kwargs())
        return scorer.cost_calls, scorer.calls_today, scorer.calls_ungrounded

    cost_calls, calls_today, calls_ungrounded = asyncio.run(_go())
    assert cost_calls == [False]  # exactly one ungrounded billing
    assert calls_today == 1
    assert calls_ungrounded == 1


def test_prompt_includes_rationale_and_signals(tmp_path: Path):
    """Round-trip the captured prompt to verify rationale/signals are included."""

    async def _go():
        store, scorer, gen = _make_generator(tmp_path)
        await store.load()
        scorer._client.aio.models.generate_content.return_value = SimpleNamespace(
            text="ok"
        )
        await gen.on_trade_closed(**_trade_kwargs(
            entry_rationale_text="FDA priority review for lead asset.",
            entry_top_signals=["catalyst:FDA-strong", "squeeze:high"],
        ))
        # First positional arg / kwarg: the prompt is passed as `contents`.
        kwargs = scorer._client.aio.models.generate_content.call_args.kwargs
        prompt = kwargs["contents"]
        return prompt

    prompt = asyncio.run(_go())
    assert "FDA priority review" in prompt
    assert "catalyst:FDA-strong" in prompt
    assert "squeeze:high" in prompt
    assert "AAPL" in prompt
    assert "+5.00%" in prompt
