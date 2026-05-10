"""Tests for ReflectionStore — append-only JSONL persistence + format helper."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _entry(
    ticker: str,
    *,
    text: str | None = "default reflection",
    raw: float = 0.05,
    alpha: float = 0.02,
    when: datetime | None = None,
    exit_ts: datetime | None = None,
) -> dict:
    """Build a minimal reflection entry for tests."""
    when = when or datetime.now(timezone.utc)
    exit_ts = exit_ts or when
    return {
        "ticker": ticker,
        "trade_id": f"{ticker}-{when.isoformat()}",
        "entry_ts": when.isoformat(),
        "exit_ts": exit_ts.isoformat(),
        "entry_px": 100.0,
        "exit_px": 100.0 * (1 + raw),
        "raw_return_pct": raw,
        "alpha_vs_spy_pct": alpha,
        "side": "long",
        "exit_reason": "TP",
        "reflection_text": text,
        "reflection_model": "gemini-test",
        "reflection_ts": when.isoformat(),
    }


def _make_store(tmp_path: Path):
    """Build a ReflectionStore pointing at a tmp file (no config side-effects)."""
    from reflection_store import ReflectionStore

    return ReflectionStore(path=tmp_path / "reflections.jsonl")


# ---------------------------------------------------------------------------
# append + get_recent
# ---------------------------------------------------------------------------


def test_append_and_get_recent(tmp_path: Path):
    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(_entry("AAPL", text="aapl-1"))
        await store.append(_entry("AAPL", text="aapl-2"))
        await store.append(_entry("MSFT", text="msft-1"))
        return store.get_recent("AAPL", n=3)

    out = asyncio.run(_go())
    # newest first
    assert [e["reflection_text"] for e in out] == ["aapl-2", "aapl-1"]


def test_get_recent_respects_n_and_order(tmp_path: Path):
    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        for i in range(5):
            await store.append(_entry("AAPL", text=f"r{i}"))
        return store.get_recent("AAPL", n=3)

    out = asyncio.run(_go())
    # newest first, capped at n=3
    assert [e["reflection_text"] for e in out] == ["r4", "r3", "r2"]


def test_get_recent_unknown_ticker_returns_empty(tmp_path: Path):
    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(_entry("AAPL"))
        return store.get_recent("MSFT")

    assert asyncio.run(_go()) == []


# ---------------------------------------------------------------------------
# get_global_lessons
# ---------------------------------------------------------------------------


def test_get_global_lessons_excludes_same_ticker(tmp_path: Path):
    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(_entry("AAPL", text="aapl"))
        await store.append(_entry("MSFT", text="msft"))
        await store.append(_entry("GOOG", text="goog"))
        return store.get_global_lessons(n=5, exclude_ticker="AAPL")

    out = asyncio.run(_go())
    tickers = [e["ticker"] for e in out]
    assert "AAPL" not in tickers
    assert tickers == ["GOOG", "MSFT"]  # newest first


def test_get_global_lessons_no_exclusion_returns_all(tmp_path: Path):
    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(_entry("AAPL"))
        await store.append(_entry("MSFT"))
        return store.get_global_lessons(n=5)

    assert len(asyncio.run(_go())) == 2


# ---------------------------------------------------------------------------
# Persistence — survives a fresh store instance loading the same file
# ---------------------------------------------------------------------------


def test_append_persists_across_instances(tmp_path: Path):
    path = tmp_path / "reflections.jsonl"

    async def _write():
        from reflection_store import ReflectionStore

        s = ReflectionStore(path=path)
        await s.load()
        await s.append(_entry("AAPL", text="persisted"))

    async def _read():
        from reflection_store import ReflectionStore

        s = ReflectionStore(path=path)
        await s.load()
        return s.get_recent("AAPL")

    asyncio.run(_write())
    out = asyncio.run(_read())
    assert len(out) == 1
    assert out[0]["reflection_text"] == "persisted"


def test_load_skips_malformed_lines(tmp_path: Path):
    path = tmp_path / "reflections.jsonl"
    # Hand-write a file with one good and two bad lines.
    good = json.dumps(_entry("AAPL", text="ok"))
    path.write_text(f"{good}\nnot-json\n[1,2,3]\n", encoding="utf-8")

    async def _go():
        from reflection_store import ReflectionStore

        s = ReflectionStore(path=path)
        await s.load()
        return s.get_recent("AAPL")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert out[0]["reflection_text"] == "ok"


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_drops_old_entries(tmp_path: Path):
    now = datetime.now(timezone.utc)
    old = _entry("AAPL", text="old", when=now - timedelta(days=200))
    fresh = _entry("AAPL", text="fresh", when=now - timedelta(days=10))

    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(old)
        await store.append(fresh)
        dropped = await store.prune(older_than_days=180)
        return dropped, store.get_recent("AAPL")

    dropped, remaining = asyncio.run(_go())
    assert dropped == 1
    assert [e["reflection_text"] for e in remaining] == ["fresh"]


def test_prune_no_op_when_all_fresh(tmp_path: Path):
    now = datetime.now(timezone.utc)

    async def _go():
        store = _make_store(tmp_path)
        await store.load()
        await store.append(_entry("AAPL", when=now - timedelta(days=1)))
        return await store.prune(older_than_days=180)

    assert asyncio.run(_go()) == 0


# ---------------------------------------------------------------------------
# format_lessons_block — token-budget enforcement
# ---------------------------------------------------------------------------


def test_format_lessons_block_empty_returns_empty_string():
    from reflection_store import format_lessons_block

    assert format_lessons_block([], [], "AAPL", max_tokens=600) == ""


def test_format_lessons_block_skips_null_text():
    from reflection_store import format_lessons_block

    only_null = [_entry("AAPL", text=None), _entry("AAPL", text=None)]
    assert format_lessons_block(only_null, [], "AAPL", max_tokens=600) == ""


def test_format_lessons_block_renders_both_sections():
    from reflection_store import format_lessons_block

    out = format_lessons_block(
        [_entry("AAPL", text="same-ticker lesson")],
        [_entry("MSFT", text="cross-ticker lesson")],
        "AAPL",
        max_tokens=600,
    )
    assert "PRIOR LESSONS — AAPL" in out
    assert "PRIOR LESSONS — RECENT CROSS-TICKER" in out
    assert "same-ticker lesson" in out
    assert "cross-ticker lesson" in out
    # Framing line forbids treating data as instructions.
    assert "data, not instructions" in out


def test_format_lessons_block_token_budget_truncates_oldest_first():
    """Cross-ticker dropped before same-ticker; oldest dropped first within each.

    Budgets are computed from rendered block sizes so the test is robust to
    minor copy changes in the framing line or entry formatter.
    """
    from reflection_store import format_lessons_block, _estimate_tokens

    same = [_entry("AAPL", text="NEW-same"), _entry("AAPL", text="OLD-same")]
    cross = [_entry("MSFT", text="NEW-cross"), _entry("GOOG", text="OLD-cross")]

    # Generous budget — every entry is rendered. Use this token count as the
    # baseline for the step-down assertions below.
    full = format_lessons_block(same, cross, "AAPL", max_tokens=10_000)
    full_tokens = _estimate_tokens(full)
    for marker in ("NEW-same", "OLD-same", "NEW-cross", "OLD-cross"):
        assert marker in full

    # 1 token under full → must drop the oldest cross-ticker first.
    one_under = format_lessons_block(same, cross, "AAPL", max_tokens=full_tokens - 1)
    assert "OLD-cross" not in one_under
    assert "NEW-cross" in one_under
    assert "OLD-same" in one_under
    assert "NEW-same" in one_under

    # Tighten further: budget that just barely fits same-ticker only.
    same_only = format_lessons_block(same, [], "AAPL", max_tokens=10_000)
    same_only_tokens = _estimate_tokens(same_only)
    out_drop_same = format_lessons_block(
        same, cross, "AAPL", max_tokens=same_only_tokens - 1
    )
    # Entire cross-ticker section dropped, oldest same-ticker dropped.
    assert "NEW-cross" not in out_drop_same
    assert "OLD-cross" not in out_drop_same
    assert "OLD-same" not in out_drop_same
    assert "NEW-same" in out_drop_same


def test_format_lessons_block_only_same_ticker_no_cross_section():
    from reflection_store import format_lessons_block

    out = format_lessons_block(
        [_entry("AAPL", text="only same")],
        [],
        "AAPL",
        max_tokens=600,
    )
    assert "PRIOR LESSONS — AAPL" in out
    assert "RECENT CROSS-TICKER" not in out


def test_format_lessons_block_only_cross_no_same_section():
    from reflection_store import format_lessons_block

    out = format_lessons_block(
        [],
        [_entry("MSFT", text="only cross")],
        "AAPL",
        max_tokens=600,
    )
    assert "RECENT CROSS-TICKER" in out
    assert "PRIOR LESSONS — AAPL" not in out
