"""Pure-function unit tests — no live API calls, no file I/O outside tmp_path.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import math

import pytest


# ---------------------------------------------------------------------------
# _inject_synthetic — bot.py CLI input validation
# ---------------------------------------------------------------------------

def _run_inject(spec: list[str]) -> int:
    """Run _inject_synthetic against a fresh queue inside an event loop.

    Returns the queue size after the call. Raises whatever the function raises.
    """
    from bot import _inject_synthetic
    from stream import TriggerEvent  # noqa: F401  (forces import, sanity check)

    async def _go() -> int:
        queue: asyncio.Queue = asyncio.Queue()
        await _inject_synthetic(queue, spec)
        return queue.qsize()

    return asyncio.run(_go())


def test_inject_synthetic_valid():
    assert _run_inject(["AAPL", "180.0", "0.12", "7.0"]) == 1


def test_inject_synthetic_dot_ticker_valid():
    # Dot-form tickers (e.g. BRK.B, BF.B) must be accepted.
    assert _run_inject(["BRK.B", "400.0", "0.05", "3.0"]) == 1


def test_inject_synthetic_empty_ticker():
    with pytest.raises(ValueError, match="invalid ticker"):
        _run_inject(["", "180.0", "0.12", "7.0"])


def test_inject_synthetic_too_long_ticker():
    with pytest.raises(ValueError, match="invalid ticker"):
        _run_inject(["TOOLONG", "180.0", "0.12", "7.0"])


def test_inject_synthetic_non_alpha_ticker():
    with pytest.raises(ValueError, match="invalid ticker"):
        _run_inject(["AB1", "180.0", "0.12", "7.0"])


def test_inject_synthetic_non_numeric_price():
    with pytest.raises(ValueError, match="must be numeric"):
        _run_inject(["AAPL", "abc", "0.12", "7.0"])


def test_inject_synthetic_zero_price():
    with pytest.raises(ValueError, match="PRICE must be positive"):
        _run_inject(["AAPL", "0", "0.12", "7.0"])


def test_inject_synthetic_negative_price():
    with pytest.raises(ValueError, match="PRICE must be positive"):
        _run_inject(["AAPL", "-1", "0.12", "7.0"])


def test_inject_synthetic_move_out_of_range_high():
    with pytest.raises(ValueError, match="MOVE_PCT must be in"):
        _run_inject(["AAPL", "180.0", "6.0", "7.0"])


def test_inject_synthetic_move_out_of_range_low():
    with pytest.raises(ValueError, match="MOVE_PCT must be in"):
        _run_inject(["AAPL", "180.0", "-1.0", "7.0"])


def test_inject_synthetic_negative_vol():
    with pytest.raises(ValueError, match="VOL_RATIO must be non-negative"):
        _run_inject(["AAPL", "180.0", "0.12", "-1"])


def test_inject_synthetic_nan_price():
    with pytest.raises(ValueError, match="PRICE must be finite"):
        _run_inject(["AAPL", "nan", "0.12", "7.0"])


def test_inject_synthetic_inf_move():
    with pytest.raises(ValueError, match="MOVE_PCT must be finite"):
        _run_inject(["AAPL", "180.0", "inf", "7.0"])


def test_inject_synthetic_neg_inf_vol():
    with pytest.raises(ValueError, match="VOL_RATIO must be finite"):
        _run_inject(["AAPL", "180.0", "0.12", "-inf"])


# ---------------------------------------------------------------------------
# _calc_take_profit — trader.py magnitude → TP tier mapping
# ---------------------------------------------------------------------------

def test_calc_take_profit_tiers():
    import config
    from trader import _calc_take_profit

    # High tier — magnitude at or above TP_MAGNITUDE_HIGH (default 9)
    assert _calc_take_profit(config.TP_MAGNITUDE_HIGH) == config.TP_HIGH_PCT
    assert _calc_take_profit(config.TP_MAGNITUDE_HIGH + 1) == config.TP_HIGH_PCT

    # Mid tier — at or above TP_MAGNITUDE_MID but below high
    assert _calc_take_profit(config.TP_MAGNITUDE_MID) == config.TP_MID_PCT
    assert _calc_take_profit(config.TP_MAGNITUDE_HIGH - 1) == config.TP_MID_PCT

    # Low tier — at or above TP_MAGNITUDE_LOW but below mid
    assert _calc_take_profit(config.TP_MAGNITUDE_LOW) == config.TP_LOW_PCT
    assert _calc_take_profit(config.TP_MAGNITUDE_MID - 1) == config.TP_LOW_PCT

    # Below low — None means "skip the trade"
    assert _calc_take_profit(config.TP_MAGNITUDE_LOW - 1) is None
    assert _calc_take_profit(0) is None


# ---------------------------------------------------------------------------
# CooldownStore — round-trip in tmp_path
# ---------------------------------------------------------------------------

def test_cooldown_store_round_trip(tmp_path):
    from cooldown_store import CooldownStore

    store = CooldownStore(path=tmp_path / "cooldowns.json")
    assert store.is_in_cooldown("AAPL") is False
    asyncio.run(store.set_cooldown("AAPL"))
    assert store.is_in_cooldown("AAPL") is True
    # Window 0 → no time can fit inside a zero-length window → always False.
    assert store.is_in_cooldown("AAPL", window_minutes=0) is False
    # Other tickers remain unaffected.
    assert store.is_in_cooldown("MSFT") is False


def test_cooldown_store_persists_to_disk(tmp_path):
    """A fresh CooldownStore loading the same file should see the entry."""
    from cooldown_store import CooldownStore

    path = tmp_path / "cooldowns.json"
    store_a = CooldownStore(path=path)
    asyncio.run(store_a.set_cooldown("TSLA"))

    store_b = CooldownStore(path=path)
    asyncio.run(store_b.load())
    assert store_b.is_in_cooldown("TSLA") is True


# ---------------------------------------------------------------------------
# _pearson_corr — trader.py static math helper
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]):
    """Call Trader._pearson_corr without needing a Trader instance.

    Python doesn't enforce that `self` is the right type, so passing None
    works as long as the method body doesn't touch self (it doesn't).
    """
    from trader import Trader
    return Trader._pearson_corr(None, xs, ys)


def test_pearson_corr_too_short():
    # len < 10 → None per the function contract.
    assert _pearson([1.0] * 5, [2.0] * 5) is None


def test_pearson_corr_identical_series():
    series = [float(i) for i in range(20)]
    result = _pearson(series, series)
    assert result is not None
    assert math.isclose(result, 1.0, rel_tol=1e-9)


def test_pearson_corr_perfectly_inverse():
    xs = [float(i) for i in range(20)]
    ys = [float(-i) for i in range(20)]
    result = _pearson(xs, ys)
    assert result is not None
    assert math.isclose(result, -1.0, rel_tol=1e-9)


def test_pearson_corr_constant_series():
    # Zero std dev → degenerate → None.
    constant = [5.0] * 20
    varying = [float(i) for i in range(20)]
    assert _pearson(constant, varying) is None
    assert _pearson(varying, constant) is None
    assert _pearson(constant, constant) is None


# ---------------------------------------------------------------------------
# Trader.sync_open_positions — pre-change snapshot compatibility
# ---------------------------------------------------------------------------

def test_sync_open_positions_pre_change_snapshot_defaults_rationale_fields(
    tmp_path, monkeypatch
):
    """A snapshot written before the reflection feature must still load.

    Older snapshots have no `entry_rationale_text` / `entry_top_signals`
    keys. The loader must default them to None / [] rather than KeyError'ing
    or attribute-erroring later when ActivePosition is constructed. This
    is a runtime-only failure mode that py_compile can't catch.
    """
    import json
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    import config
    from daily_summary import SessionStats
    from trader import Trader

    # Re-route both the snapshot READ (sync_open_positions) and WRITE
    # (the trailing _save_positions_snapshot) into tmp_path.
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    # Snapshot in the OLD format — no entry_rationale_text, no entry_top_signals.
    old_snapshot = {
        "AAPL": {
            "qty": 100,
            "entry_price": 180.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "gemini_confidence": 8,
            "gemini_magnitude": 7,
            "technical_signal": "bullish",
            "sector": "Technology",
            "take_profit_pct": 0.10,
            "stop_loss_pct": 0.05,
            "sl_floor": -0.05,
            "max_hold_minutes": 90,
            "hold_strategy": "momentum",
            "regime": "trending",
            "had_factor_signal": True,
        }
    }
    (tmp_path / "open_positions.json").write_text(
        json.dumps(old_snapshot), encoding="utf-8"
    )

    # Mock Alpaca: get_all_positions returns one matching position.
    fake_alpaca_pos = MagicMock()
    fake_alpaca_pos.symbol = "AAPL"
    fake_alpaca_pos.qty = "100"
    fake_alpaca_pos.avg_entry_price = "180.0"

    trading_client = MagicMock()
    trading_client.get_all_positions.return_value = [fake_alpaca_pos]

    telegram = MagicMock()
    stats = SessionStats()
    trader = Trader(trading_client, telegram, stats)

    asyncio.run(trader.sync_open_positions())

    pos = trader._positions["AAPL"]
    # New fields default cleanly when absent from the snapshot.
    assert pos.entry_rationale_text is None
    assert pos.entry_top_signals == []
    # Existing fields still recovered correctly so we know we're testing
    # the right code path.
    assert pos.qty == 100
    assert pos.entry_price == 180.0
    assert pos.gemini_confidence == 8
    assert pos.had_factor_signal is True


def test_sync_open_positions_corrupt_rationale_field_falls_back_to_default(
    tmp_path, monkeypatch
):
    """Defensive: a snapshot with the wrong type for the new fields must
    not crash. Wrong-type values fall back to None / [] rather than
    propagating into ActivePosition where they'd break downstream
    serialization or the reflection prompt."""
    import json
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    import config
    from daily_summary import SessionStats
    from trader import Trader

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    bad_snapshot = {
        "MSFT": {
            "qty": 50,
            "entry_price": 400.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "take_profit_pct": 0.10,
            "stop_loss_pct": 0.05,
            "max_hold_minutes": 90,
            "hold_strategy": "momentum",
            # Wrong types — would crash naive code.
            "entry_rationale_text": 12345,
            "entry_top_signals": "not-a-list",
        }
    }
    (tmp_path / "open_positions.json").write_text(
        json.dumps(bad_snapshot), encoding="utf-8"
    )

    fake_alpaca_pos = MagicMock()
    fake_alpaca_pos.symbol = "MSFT"
    fake_alpaca_pos.qty = "50"
    fake_alpaca_pos.avg_entry_price = "400.0"

    trading_client = MagicMock()
    trading_client.get_all_positions.return_value = [fake_alpaca_pos]

    trader = Trader(trading_client, MagicMock(), SessionStats())
    asyncio.run(trader.sync_open_positions())

    pos = trader._positions["MSFT"]
    assert pos.entry_rationale_text is None
    assert pos.entry_top_signals == []
