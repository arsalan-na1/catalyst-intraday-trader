"""Virtual congress-copy portfolio tests (Phase 2) — pure, offline.

Cover sizing (equal-weight vs range-tier on dollar RANGES), the open/should-open
gates (freshness + start-date so we never backfill the historical dump or act on
overly-stale disclosures), mirror-sell exits, mark-to-market P&L, and the atomic
state round-trip. No network, no Alpaca — prices are injected.
"""

from __future__ import annotations

import pytest

import config
from congress_trades import CongressTrade
from congress_portfolio import CongressPortfolio, VirtualPosition, position_key, size_usd_for


def _trade(ticker="AAPL", ttype="buy", member="Jane Doe",
           tx="2026-05-01", disc="2026-05-20", lo=1001.0, hi=15000.0):
    return CongressTrade(
        member=member, chamber="senate", ticker=ticker, transaction_type=ttype,
        transaction_date=tx, disclosure_date=disc, amount_min=lo, amount_max=hi, source="x",
    )


def _portfolio(equity=100_000.0, started="2026-05-15", mode="equal_weight", base=1000.0):
    return CongressPortfolio(starting_equity=equity, started_on=started,
                             sizing_mode=mode, base_usd=base)


# --- sizing -----------------------------------------------------------------


def test_equal_weight_ignores_the_disclosed_range():
    small = _trade(lo=1001.0, hi=15000.0)
    large = _trade(lo=1_000_001.0, hi=5_000_000.0)
    assert size_usd_for(small, "equal_weight", 1000.0) == 1000.0
    assert size_usd_for(large, "equal_weight", 1000.0) == 1000.0  # same $ — range ignored


@pytest.mark.parametrize("hi, expected_mult", [
    (15_000, 1), (50_000, 2), (100_000, 3), (250_000, 4),
    (500_000, 5), (1_000_000, 6), (5_000_000, 7), (50_000_000, 7),
])
def test_range_tier_scales_with_disclosed_upper_bound(hi, expected_mult):
    t = _trade(lo=1001.0, hi=float(hi))
    assert size_usd_for(t, "range_tier", 1000.0) == 1000.0 * expected_mult


def test_equal_weight_same_shares_for_different_ranges():
    p = _portfolio(mode="equal_weight", base=1000.0)
    a = p.open_from_trade(_trade(ticker="AAA", lo=1001, hi=15000), price=100.0, today="2026-05-29")
    b = p.open_from_trade(_trade(ticker="BBB", lo=1_000_001, hi=5_000_000), price=100.0, today="2026-05-29")
    assert a.shares == b.shares == 10.0  # $1000 / $100, regardless of disclosed range


# --- open / should_open gates -----------------------------------------------


def test_open_from_trade_sizes_and_debits_cash():
    p = _portfolio(equity=100_000.0, base=1000.0)
    pos = p.open_from_trade(_trade(), price=50.0, today="2026-05-29")
    assert pos.shares == 20.0                 # 1000 / 50
    assert pos.entry_price == 50.0
    assert pos.entry_date == "2026-05-29"     # acted today, NOT backdated to disclosure
    assert pos.lag_days == 19                 # 2026-05-01 -> 2026-05-20
    assert p.cash == pytest.approx(99_000.0)
    assert p.open[pos.key] is pos


@pytest.mark.parametrize("price", [None, 0.0, -5.0])
def test_open_from_trade_rejects_bad_price(price):
    p = _portfolio()
    assert p.open_from_trade(_trade(), price=price, today="2026-05-29") is None
    assert p.cash == 100_000.0


def test_size_capped_at_available_cash():
    p = _portfolio(equity=500.0, base=1000.0)        # only $500 cash, $1000 target
    pos = p.open_from_trade(_trade(), price=100.0, today="2026-05-29")
    assert pos.shares == 5.0                          # min(1000, 500) / 100
    assert p.cash == pytest.approx(0.0)


def test_should_open_accepts_fresh_new_buy():
    p = _portfolio(started="2026-05-15")
    assert p.should_open(_trade(disc="2026-05-20"), today="2026-05-29", freshness_days=60) is True


def test_should_open_rejects_non_buy():
    p = _portfolio(started="2026-05-15")
    assert p.should_open(_trade(ttype="sell", disc="2026-05-20"), "2026-05-29", 60) is False


def test_should_open_rejects_pre_start_disclosure():
    # Disclosed before we started tracking → never backfill the historical dump.
    p = _portfolio(started="2026-05-15")
    assert p.should_open(_trade(disc="2026-05-10"), "2026-05-29", 60) is False


def test_should_open_rejects_old_transaction_even_if_recently_disclosed():
    # Case 3: the disclosure lag can exceed a year. A trade executed long ago is
    # a dead signal even when disclosed today — freshness is measured on the
    # TRANSACTION date, not on how recently it was disclosed.
    p = _portfolio(started="2026-01-01")
    stale = _trade(tx="2026-01-15", disc="2026-05-20")  # ~134-day-old trade, fresh disclosure
    assert p.should_open(stale, today="2026-05-29", freshness_days=60) is False


def test_should_open_accepts_normal_disclosure_lag():
    # A typical ~40-day-lag trade (transaction within freshness) still opens.
    p = _portfolio(started="2026-05-15")
    fresh = _trade(tx="2026-04-20", disc="2026-05-20")  # 39-day-old trade
    assert p.should_open(fresh, today="2026-05-29", freshness_days=60) is True


def test_should_open_rejects_missing_transaction_date():
    # Fail-closed: can't assess staleness without a transaction date.
    p = _portfolio(started="2026-05-15")
    assert p.should_open(_trade(tx="", disc="2026-05-20"), today="2026-05-29", freshness_days=60) is False


def test_should_open_rejects_future_transaction_date():
    # A future-dated transaction (clock skew / bad data) is rejected by the
    # 0 <= tx_age lower bound, independent of the disclosure-date check.
    p = _portfolio(started="2026-05-15")
    assert p.should_open(_trade(tx="2026-06-10", disc="2026-05-20"), today="2026-05-29", freshness_days=60) is False


@pytest.mark.parametrize("tx, expected", [
    ("2026-03-30", True),    # exactly 60 days before today=2026-05-29 → accept (<=)
    ("2026-03-29", False),   # 61 days → reject
])
def test_should_open_freshness_boundary(tx, expected):
    p = _portfolio(started="2026-01-01")
    assert p.should_open(_trade(tx=tx, disc="2026-05-20"), today="2026-05-29", freshness_days=60) is expected


def test_should_open_rejects_already_seen():
    p = _portfolio(started="2026-05-15")
    t = _trade(disc="2026-05-20")
    p.open_from_trade(t, price=100.0, today="2026-05-29")
    assert p.should_open(t, "2026-05-29", 60) is False  # already open


# --- mirror-sell exits -------------------------------------------------------


def test_mirror_sell_closes_only_same_member_same_ticker():
    p = _portfolio()
    jane_aapl = p.open_from_trade(_trade(ticker="AAPL", member="Jane Doe", disc="2026-05-20"), 100.0, "2026-05-29")
    p.open_from_trade(_trade(ticker="MSFT", member="Jane Doe", disc="2026-05-21"), 100.0, "2026-05-29")
    p.open_from_trade(_trade(ticker="AAPL", member="John Smith", disc="2026-05-22"), 100.0, "2026-05-29")

    closed = p.close_matching_sells(_trade(ticker="AAPL", ttype="sell", member="Jane Doe"), price=120.0, today="2026-05-30")

    assert closed == [jane_aapl.key]              # only Jane's AAPL
    assert jane_aapl.key not in p.open
    assert len(p.open) == 2                        # Jane MSFT + John AAPL remain
    assert p.realized_pnl == pytest.approx(10 * (120.0 - 100.0))  # 10 shares, +$20


def test_mirror_sell_no_holding_is_noop():
    p = _portfolio()
    assert p.close_matching_sells(_trade(ttype="sell"), price=100.0, today="2026-05-30") == []


# --- apply_disclosures end to end -------------------------------------------


def test_apply_disclosures_opens_fresh_buys_and_mirrors_sells():
    p = _portfolio(started="2026-05-15", base=1000.0)
    trades = [
        _trade(ticker="AAPL", member="Jane Doe", disc="2026-05-20"),                 # fresh buy
        _trade(ticker="OLD", member="Jane Doe", disc="2026-05-01"),                  # pre-start → skipped
        _trade(ticker="NOPRICE", member="Jane Doe", disc="2026-05-21"),              # fresh but no price → skipped
    ]
    prices = {"AAPL": 100.0}  # NOPRICE deliberately absent
    opened, closed = p.apply_disclosures(trades, prices, today="2026-05-29", freshness_days=60)

    assert len(opened) == 1
    assert p.open[opened[0]].ticker == "AAPL"
    assert closed == []


def test_non_tradeable_adr_without_price_is_not_opened():
    # Case 4: ADR/OTC names with no Alpaca snapshot price never become positions
    # — the portfolio is virtual and price-gated, so there are no failed/zero-fill
    # orders, just no position.
    p = _portfolio(started="2026-05-15")
    trades = [_trade(ticker="IFNNY", disc="2026-05-20"), _trade(ticker="SFGYY", disc="2026-05-21")]
    opened, closed = p.apply_disclosures(trades, prices={}, today="2026-05-29", freshness_days=60)
    assert opened == [] and p.open == {}


# --- mark to market ----------------------------------------------------------


def test_mark_to_market_equity_and_pnl():
    p = _portfolio(equity=100_000.0, base=1000.0)
    p.open_from_trade(_trade(ticker="AAA", disc="2026-05-20"), price=100.0, today="2026-05-29")  # 10 sh, -$1000 cash
    summary = p.mark_to_market({"AAA": 110.0})                                    # +$100 unrealized
    assert summary["cash"] == pytest.approx(99_000.0)
    assert summary["positions_value"] == pytest.approx(1_100.0)
    assert summary["equity"] == pytest.approx(100_100.0)
    assert summary["unrealized_pnl"] == pytest.approx(100.0)
    assert summary["total_pnl"] == pytest.approx(100.0)
    assert summary["open_positions"] == 1


def test_mark_to_market_missing_price_falls_back_to_entry():
    p = _portfolio(base=1000.0)
    p.open_from_trade(_trade(ticker="AAA", disc="2026-05-20"), price=100.0, today="2026-05-29")
    summary = p.mark_to_market({})  # no price supplied
    assert summary["unrealized_pnl"] == pytest.approx(0.0)       # falls back to entry, not dropped
    assert summary["equity"] == pytest.approx(100_000.0)


def test_range_tier_sizes_bigger_for_bigger_disclosures():
    p = _portfolio(mode="range_tier", base=1000.0)
    small = p.open_from_trade(_trade(ticker="AAA", hi=15_000), price=100.0, today="2026-05-29")    # tier 1 → $1000
    large = p.open_from_trade(_trade(ticker="BBB", hi=5_000_000), price=100.0, today="2026-05-29") # tier 7 → $7000
    assert small.shares == 10.0
    assert large.shares == 70.0


# --- persistence -------------------------------------------------------------


def test_save_load_round_trip_atomic(tmp_path):
    path = tmp_path / "congress_portfolio.json"
    p = _portfolio(started="2026-05-15", base=1000.0)
    p.open_from_trade(_trade(ticker="AAPL", disc="2026-05-20"), price=100.0, today="2026-05-29")
    p.realized_pnl = 42.0
    p.save(path)

    assert path.exists()
    assert not (tmp_path / "congress_portfolio.json.tmp").exists()  # atomic rename, tmp cleaned

    loaded = CongressPortfolio.load(path)
    assert loaded.started_on == "2026-05-15"
    assert loaded.cash == pytest.approx(p.cash)
    assert loaded.realized_pnl == 42.0
    assert set(loaded.open) == set(p.open)
    pos = next(iter(loaded.open.values()))
    assert isinstance(pos, VirtualPosition) and pos.ticker == "AAPL" and pos.shares == 10.0


def test_load_missing_file_starts_fresh(tmp_path):
    p = CongressPortfolio.load(tmp_path / "nope.json")
    assert p.cash == config.CONGRESS_VIRTUAL_EQUITY_USD
    assert p.open == {} and p.closed == []
