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


# ---------------------------------------------------------------------------
# Entry-sizing helpers in trader.py: _apply_atr_floor, _apply_risk_cap,
# _apply_rr_floor. Pure functions — no Alpaca, no I/O.
# ---------------------------------------------------------------------------

def test_atr_floor_binds_widens_tight_sl():
    """A 9% ATR with multiplier 1.5 needs a 13.5% stop; Gemini-set 5% must widen."""
    from trader import _apply_atr_floor
    sl = _apply_atr_floor(
        stop_loss_pct=0.05,
        atr_14_pct=9.0,
        atr_multiplier=1.5,
        sl_max=0.18,
    )
    assert sl == pytest.approx(0.135, abs=1e-9)


def test_atr_floor_clamps_at_sl_max():
    """A 20% ATR × 1.5 = 30% would exceed the 18% ceiling — must clamp to 18%."""
    from trader import _apply_atr_floor
    sl = _apply_atr_floor(0.05, atr_14_pct=20.0, atr_multiplier=1.5, sl_max=0.18)
    assert sl == pytest.approx(0.18, abs=1e-9)


def test_atr_floor_no_atr_returns_input():
    """ATR unavailable → SL unchanged (fail open)."""
    from trader import _apply_atr_floor
    assert _apply_atr_floor(0.05, atr_14_pct=None, atr_multiplier=1.5, sl_max=0.18) == 0.05


def test_atr_floor_does_not_lower_a_wider_stop():
    """A Gemini-set 8% SL stays 8% when the ATR floor would only need 4.5%."""
    from trader import _apply_atr_floor
    assert _apply_atr_floor(0.08, atr_14_pct=3.0, atr_multiplier=1.5, sl_max=0.18) == 0.08


def test_risk_cap_reduces_size_when_stop_widens():
    """Dollar risk = size_pct × stop_pct should stay ≤ RISK_PER_TRADE_PCT."""
    from trader import _apply_risk_cap

    # Gemini wants 10% size. Risk budget 0.5%. Stop 5% → risk_size = 10% (no cap).
    # Stop 10% → risk_size = 5%. Stop 15% → risk_size = 3.33%.
    risk_pct = 0.005
    gemini_size = 0.10

    cap_5 = _apply_risk_cap(gemini_size, risk_pct, stop_loss_pct=0.05, size_min=0.02)
    cap_10 = _apply_risk_cap(gemini_size, risk_pct, stop_loss_pct=0.10, size_min=0.02)
    cap_15 = _apply_risk_cap(gemini_size, risk_pct, stop_loss_pct=0.15, size_min=0.02)

    # Tight stop: no reduction.
    assert cap_5 == pytest.approx(0.10, abs=1e-9)
    # Wider stop: size halved.
    assert cap_10 == pytest.approx(0.05, abs=1e-9)
    assert cap_15 == pytest.approx(0.005 / 0.15, abs=1e-9)

    # Dollar risk at stop stays ≈ constant (≤ 0.5%) when the cap is active.
    for size, sl in [(cap_10, 0.10), (cap_15, 0.15)]:
        assert size * sl <= risk_pct + 1e-9


def test_risk_cap_never_increases_size():
    """
    The risk cap must only reduce — never inflate — the Gemini size.
    Test that applying the risk cap never increases the Gemini size.
    Verify that _apply_risk_cap returns the cap value when the risk-implied size would be larger than the cap.
    Raises AssertionError if the bound is violated.
    """
    from trader import _apply_risk_cap
    # Gemini wants 3% on a 2% stop: risk_implied = 0.5%/2% = 25%, much larger.
    # The function must return the 3%, NOT 25%.
    assert _apply_risk_cap(0.03, 0.005, 0.02, size_min=0.02) == 0.03


def test_risk_cap_honors_size_min_floor():
    """Risk cap must not drive size below SIZE_MIN (broker min allocation)."""
    from trader import _apply_risk_cap
    # 0.5% / 50% = 1% — below the 2% size_min floor.
    capped = _apply_risk_cap(0.10, 0.005, 0.50, size_min=0.02)
    assert capped == pytest.approx(0.02, abs=1e-9)


def test_rr_floor_lifts_tp_when_stop_widens():
    """ATR-widened 12% stop with 2:1 RR floor lifts a 15% TP to 24%."""
    from trader import _apply_rr_floor
    tp = _apply_rr_floor(
        take_profit_pct=0.15,
        stop_loss_pct=0.12,
        min_rr_ratio=2.0,
        tp_max=0.50,
    )
    assert tp == pytest.approx(0.24, abs=1e-9)


def test_rr_floor_reclamps_to_tp_max():
    """If SL × RR exceeds GEMINI_TP_MAX, TP clamps to ceiling."""
    from trader import _apply_rr_floor
    # 0.30 × 2.0 = 0.60 > 0.50 ceiling.
    tp = _apply_rr_floor(0.20, stop_loss_pct=0.30, min_rr_ratio=2.0, tp_max=0.50)
    assert tp == pytest.approx(0.50, abs=1e-9)


def test_rr_floor_leaves_strong_tp_untouched():
    """A TP already above SL × RR ratio stays put."""
    from trader import _apply_rr_floor
    tp = _apply_rr_floor(0.30, stop_loss_pct=0.05, min_rr_ratio=2.0, tp_max=0.50)
    assert tp == pytest.approx(0.30, abs=1e-9)


# ---------------------------------------------------------------------------
# Entry-gate helpers in bot.py: _should_skip_overbought, _should_skip_downtrend,
# _should_skip_falling_knife. Pure functions — no scoring, no Alpaca.
# ---------------------------------------------------------------------------

def test_overbought_gate_rejects_rsi_75():
    from bot import _should_skip_overbought
    assert _should_skip_overbought(75.0, rsi_max=72) is True


def test_overbought_gate_passes_rsi_60():
    from bot import _should_skip_overbought
    assert _should_skip_overbought(60.0, rsi_max=72) is False


def test_overbought_gate_passes_at_boundary():
    from bot import _should_skip_overbought
    # Spec says "> RSI_MAX_ENTRY" so exactly equal must pass.
    assert _should_skip_overbought(72.0, rsi_max=72) is False


def test_overbought_gate_fails_open_when_rsi_none():
    """RSI unavailable must NOT skip — gate fails open."""
    from bot import _should_skip_overbought
    assert _should_skip_overbought(None, rsi_max=72) is False


def test_downtrend_gate_rejects_downtrend():
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend("downtrend", reject_downtrend=True) is True


def test_downtrend_gate_passes_uptrend_and_sideways():
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend("uptrend", True) is False
    assert _should_skip_downtrend("sideways", True) is False


def test_downtrend_gate_disabled():
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend("downtrend", reject_downtrend=False) is False


def test_falling_knife_arec_case_rejected():
    """AREC reproduction: -69% from 52w high, downtrend, RSI 48 → skip."""
    from bot import _should_skip_falling_knife
    # dist_from_52w_high_pct is signed negative below the high.
    assert _should_skip_falling_knife(
        trend="downtrend",
        dist_from_52w_high_pct=-69.0,
        drawdown_threshold_pct=0.40,
    ) is True


def test_falling_knife_uptrend_passes_through():
    """Even at -50% from highs, an uptrend means recovery in progress — allow."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        trend="uptrend",
        dist_from_52w_high_pct=-50.0,
        drawdown_threshold_pct=0.40,
    ) is False


def test_falling_knife_shallow_drawdown_passes():
    """Down 20% from highs, sideways trend — below threshold, allow."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        trend="sideways",
        dist_from_52w_high_pct=-20.0,
        drawdown_threshold_pct=0.40,
    ) is False


def test_falling_knife_no_data_fails_open_when_allowed():
    """Legacy fail-open behavior preserved when fail_closed=False."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        "downtrend", None, 0.40, fail_closed=False
    ) is False


# ---------------------------------------------------------------------------
# _should_skip_unmet_rr: skip the trade when ATR-widened stop pushes the
# required TP above GEMINI_TP_MAX. Shipping a clamped TP would mean a
# silently sub-MIN_RR_RATIO setup — filter, don't downsize.
# ---------------------------------------------------------------------------

def test_rr_unmet_skips_when_required_tp_above_cap():
    """18% SL × 2.0 RR = 36% required TP — above 50% cap? No (36<50). Try 30% SL."""
    from trader import _should_skip_unmet_rr
    # SL widened to 30% by extreme ATR. Required TP = 60%, cap = 50% → skip.
    assert _should_skip_unmet_rr(
        stop_loss_pct=0.30, gemini_tp_max=0.50, min_rr_ratio=2.0
    ) is True


def test_rr_unmet_passes_when_required_tp_fits_under_cap():
    """12% SL × 2.0 RR = 24% — fits under the 50% TP cap → don't skip."""
    from trader import _should_skip_unmet_rr
    assert _should_skip_unmet_rr(
        stop_loss_pct=0.12, gemini_tp_max=0.50, min_rr_ratio=2.0
    ) is False


def test_rr_unmet_passes_at_exact_cap():
    """SL × RR == TP_MAX must pass — we use '>', not '>='."""
    from trader import _should_skip_unmet_rr
    # 0.25 × 2.0 = 0.50 == cap → not skipped.
    assert _should_skip_unmet_rr(0.25, 0.50, 2.0) is False


# ---------------------------------------------------------------------------
# Fail-CLOSED downtrend / falling-knife gates with the TREND_GATE_FAIL_CLOSED
# default ON. RSI gate stays fail-open (covered above).
# ---------------------------------------------------------------------------

def test_downtrend_gate_fails_closed_on_missing_trend():
    """trend=None with fail_closed=True must skip (can't confirm)."""
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend(None, reject_downtrend=True, fail_closed=True) is True


def test_downtrend_gate_fails_closed_on_empty_trend():
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend("", reject_downtrend=True, fail_closed=True) is True


def test_downtrend_gate_fail_open_legacy_preserved():
    """fail_closed=False restores the previous fail-open semantics."""
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend(None, True, fail_closed=False) is False


def test_downtrend_gate_disabled_overrides_fail_closed():
    """REJECT_DOWNTREND=False means the gate is off — fail-closed is irrelevant."""
    from bot import _should_skip_downtrend
    assert _should_skip_downtrend(None, reject_downtrend=False, fail_closed=True) is False


def test_falling_knife_fails_closed_on_missing_drawdown():
    """dist_from_52w_high_pct=None with fail_closed=True must skip."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        trend="downtrend",
        dist_from_52w_high_pct=None,
        drawdown_threshold_pct=0.40,
        fail_closed=True,
    ) is True


def test_falling_knife_fails_closed_on_missing_trend():
    """trend=None with fail_closed=True must skip even with valid drawdown."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        trend=None,
        dist_from_52w_high_pct=-50.0,
        drawdown_threshold_pct=0.40,
        fail_closed=True,
    ) is True


def test_falling_knife_uptrend_still_passes_under_fail_closed():
    """A complete record showing an uptrend recovery must still pass."""
    from bot import _should_skip_falling_knife
    assert _should_skip_falling_knife(
        trend="uptrend",
        dist_from_52w_high_pct=-50.0,
        drawdown_threshold_pct=0.40,
        fail_closed=True,
    ) is False


# ---------------------------------------------------------------------------
# Composition: biotech (Gemini sets 0.05 size in prompt) × risk cap × SIZE_MIN
# floor. Documents the existing MIN behavior — there is NO separate code-side
# biotech cap today (only the Gemini prompt hint at scorer.py:118), so this
# test pins the composition we actually have.
# ---------------------------------------------------------------------------

def test_biotech_prompt_size_then_risk_cap_takes_min():
    """Gemini hints 5% size for biotech. Wide stop should still let risk cap reduce."""
    from trader import _apply_risk_cap
    # Gemini-set 0.05 for biotech; SL = 0.10 (typical biotech ATR-widened).
    # risk_implied = 0.005 / 0.10 = 0.05 → equal to Gemini size, no reduction.
    assert _apply_risk_cap(0.05, 0.005, 0.10, size_min=0.02) == pytest.approx(0.05, abs=1e-9)
    # SL widens to 0.15 (high-ATR biotech). risk_implied = 0.033 < 0.05 → cap wins.
    capped = _apply_risk_cap(0.05, 0.005, 0.15, size_min=0.02)
    assert capped == pytest.approx(0.005 / 0.15, abs=1e-9)
    assert capped < 0.05  # confirms MIN reduced the biotech-sized position


# ---------------------------------------------------------------------------
# _apply_biotech_cap: hard size cap for biotech sector. Turns the prior
# Gemini-prompt-only hint into an enforced clamp.
# Sector value contract (news.determine_sector at news.py:218): one of
# {"tech", "biotech", "energy", "financials", "unknown"}.
# ---------------------------------------------------------------------------

def test_biotech_cap_clamps_oversized_biotech_position():
    """Gemini returns 0.10 size on a biotech name → hard-capped to 0.05."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="biotech",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
    )
    assert capped == pytest.approx(0.05, abs=1e-9)


def test_biotech_cap_leaves_undersized_biotech_position_alone():
    """Cap is a MIN — never inflates a Gemini size already at/below the cap."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.03,
        sector="biotech",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
    )
    assert capped == pytest.approx(0.03, abs=1e-9)


def test_biotech_cap_does_not_touch_non_biotech_sectors():
    """A tech name keeps its full Gemini-allotted size."""
    from trader import _apply_biotech_cap
    for sector in ("tech", "energy", "financials"):
        result = _apply_biotech_cap(
            0.10, sector, frozenset({"biotech"}), 0.05
        )
        assert result == pytest.approx(0.10, abs=1e-9), (
            f"sector={sector!r} should NOT be capped"
        )


def test_biotech_cap_fails_safe_on_unknown_sector():
    """Sector resolution missed (returned 'unknown') → don't cap.

    Spec says: 'if sector is None/unknown, do NOT apply the cap' — we don't
    want to 5%-cap everything on a sector-lookup miss.
    """
    from trader import _apply_biotech_cap
    assert _apply_biotech_cap(
        0.10, "unknown", frozenset({"biotech"}), 0.05
    ) == pytest.approx(0.10, abs=1e-9)


def test_biotech_cap_fails_safe_on_none_sector():
    from trader import _apply_biotech_cap
    assert _apply_biotech_cap(
        0.10, None, frozenset({"biotech"}), 0.05
    ) == pytest.approx(0.10, abs=1e-9)


def test_biotech_cap_honors_custom_sector_set():
    """If BIOTECH_SECTORS is widened (e.g. someone adds 'healthcare'), match too."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        0.10, "healthcare", frozenset({"biotech", "healthcare"}), 0.05
    )
    assert capped == pytest.approx(0.05, abs=1e-9)


def test_biotech_cap_then_risk_cap_compose_min_tightest_wins():
    """Final size = MIN(Gemini, biotech cap, risk cap). Both directions."""
    from trader import _apply_biotech_cap, _apply_risk_cap

    # Scenario A: biotech cap binds tightest.
    #   Gemini = 0.10, biotech cap = 0.05, tight stop (SL=0.05) → risk_implied = 0.10.
    #   After biotech: 0.05. Risk cap leaves it alone. Final 0.05.
    size = _apply_biotech_cap(0.10, "biotech", frozenset({"biotech"}), 0.05)
    size = _apply_risk_cap(size, 0.005, stop_loss_pct=0.05, size_min=0.02)
    assert size == pytest.approx(0.05, abs=1e-9)

    # Scenario B: risk cap binds tighter than biotech.
    #   Gemini = 0.10, biotech cap = 0.05, SL = 0.15 → risk_implied = 0.005/0.15 ≈ 0.033.
    #   After biotech: 0.05. Risk cap reduces to 0.033. Final 0.033.
    size = _apply_biotech_cap(0.10, "biotech", frozenset({"biotech"}), 0.05)
    size = _apply_risk_cap(size, 0.005, stop_loss_pct=0.15, size_min=0.02)
    assert size == pytest.approx(0.005 / 0.15, abs=1e-9)
    assert size < 0.05  # confirms risk cap won

    # Order independence: biotech-cap × risk-cap is commutative because both
    # are pure MINs.  Apply in reverse order, same result.
    rev = _apply_risk_cap(0.10, 0.005, 0.15, size_min=0.02)
    rev = _apply_biotech_cap(rev, "biotech", frozenset({"biotech"}), 0.05)
    assert rev == pytest.approx(0.005 / 0.15, abs=1e-9)


# ---------------------------------------------------------------------------
# _apply_biotech_cap leak fix: the cap must ALSO bind when the trigger fires
# on an FDA/biotech catalyst even if the static sector mapping puts the
# ticker in a non-biotech bucket. determine_sector short-circuits to the
# static _SECTOR_MAP before checking is_biotech_catalyst(news), so a tech-
# mapped ticker with an FDA event would otherwise escape the cap. Decision:
# bind when (sector ∈ biotech_sectors) OR is_biotech.
# Sector-concentration logic intentionally NOT changed — that asks "what
# bucket is this company"; the cap asks "is this catalyst gap-risk".
# ---------------------------------------------------------------------------

def test_biotech_cap_binds_on_tech_sector_with_biotech_catalyst():
    """The leak case: sector='tech' but FDA catalyst flagged → cap binds."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="tech",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=True,
    )
    assert capped == pytest.approx(0.05, abs=1e-9)


def test_biotech_cap_does_not_bind_on_tech_sector_without_biotech_catalyst():
    """Vanilla tech name with no biotech news flag — keeps full Gemini size."""
    from trader import _apply_biotech_cap
    result = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="tech",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=False,
    )
    assert result == pytest.approx(0.10, abs=1e-9)


def test_biotech_cap_still_binds_on_biotech_sector_without_news_flag():
    """Regression guard: sector-match path keeps working when is_biotech=False."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="biotech",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=False,
    )
    assert capped == pytest.approx(0.05, abs=1e-9)


def test_biotech_cap_binds_on_unknown_sector_with_biotech_catalyst():
    """Long-tail FDA name not in _SECTOR_MAP and not detected via news fallback
    sector path — the is_biotech flag still triggers the cap."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="unknown",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=True,
    )
    assert capped == pytest.approx(0.05, abs=1e-9)


def test_biotech_cap_unknown_sector_no_news_flag_still_fails_safe():
    """The original fail-safe must survive: unknown sector + no FDA flag → no cap.
    A sector-lookup miss on a non-biotech name must NOT 5%-cap the trade."""
    from trader import _apply_biotech_cap
    result = _apply_biotech_cap(
        position_size_pct=0.10,
        sector="unknown",
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=False,
    )
    assert result == pytest.approx(0.10, abs=1e-9)


def test_biotech_cap_none_sector_with_biotech_catalyst():
    """Strict None sector (not the string 'unknown') + FDA flag → cap binds.
    Pins the (None, True) cell of the (sector?, is_biotech?) matrix."""
    from trader import _apply_biotech_cap
    capped = _apply_biotech_cap(
        position_size_pct=0.10,
        sector=None,
        biotech_sectors=frozenset({"biotech"}),
        biotech_cap_pct=0.05,
        is_biotech=True,
    )
    assert capped == pytest.approx(0.05, abs=1e-9)
