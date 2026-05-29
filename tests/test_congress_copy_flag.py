"""Flag-gating tests for the congress-copy feature.

Prove the hard constraint: with CONGRESS_COPY_ENABLED off (the default) the
feature is fully inert — the scheduler coroutine returns immediately and does no
work (no data refresh, no portfolio load, no disk write), so the bot behaves
byte-identically to today. Also prove the flag-on path actually runs a cycle.

No network, no Alpaca — the cycle is stubbed.
"""

from __future__ import annotations

import asyncio

import pytest

import bot
import config


def test_flag_defaults_off():
    # Off by default: the whole feature is opt-in.
    assert config.CONGRESS_COPY_ENABLED is False


def test_run_congress_copy_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "CONGRESS_COPY_ENABLED", False)

    cycle_calls = []
    load_calls = []

    async def _spy_cycle(*a, **k):
        cycle_calls.append(a)

    monkeypatch.setattr(bot, "_run_congress_cycle", _spy_cycle)
    monkeypatch.setattr(bot.CongressPortfolio, "load",
                        staticmethod(lambda path=None: load_calls.append(path)))

    stop = asyncio.Event()
    asyncio.run(bot.run_congress_copy(None, stop))

    # Disabled → returns immediately: no portfolio loaded, no cycle run, nothing scheduled.
    assert cycle_calls == []
    assert load_calls == []


def test_run_congress_copy_runs_cycle_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "CONGRESS_COPY_ENABLED", True)

    stop = asyncio.Event()
    cycle_calls = []
    sentinel = object()

    async def _spy_cycle(portfolio, session, today=None):
        cycle_calls.append(portfolio)
        stop.set()  # end the daily loop after one cycle

    monkeypatch.setattr(bot, "_run_congress_cycle", _spy_cycle)
    monkeypatch.setattr(bot.CongressPortfolio, "load",
                        staticmethod(lambda path=None: sentinel))

    asyncio.run(bot.run_congress_copy(None, stop))

    assert cycle_calls == [sentinel]  # the loaded portfolio was driven through exactly one cycle


def test_disabled_cycle_writes_no_state(monkeypatch, tmp_path):
    # Belt-and-suspenders: with the flag off nothing is written to the portfolio file.
    monkeypatch.setattr(config, "CONGRESS_COPY_ENABLED", False)
    state = tmp_path / "congress_portfolio.json"
    monkeypatch.setattr(config, "CONGRESS_PORTFOLIO_FILE", state)
    asyncio.run(bot.run_congress_copy(None, asyncio.Event()))
    assert not state.exists()
