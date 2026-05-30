"""Congress data-layer tests (Phase 1) — fixtures only, never the network.

Cover normalization (amount ranges, tickers, types, dates), the two source
parsers, dedup, the 45-day lag, the atomic disk cache round-trip, and the
fail-open fetch (with fetch_with_retry monkeypatched — no real HTTP).
"""

from __future__ import annotations

import asyncio
import json

import pytest

import config
import congress_trades as ct


# --- recorded fixtures (representative of the real schemas) -----------------

SENATE_FIXTURE = [
    {"transaction_date": "01/02/2024", "owner": "Spouse", "ticker": "AAPL",
     "asset_description": "Apple Inc", "asset_type": "Stock", "type": "Purchase",
     "amount": "$1,001 - $15,000", "comment": "--", "senator": "Jane A Doe",
     "disclosure_date": "02/15/2024", "ptr_link": "x"},
    {"transaction_date": "01/03/2024", "owner": "Self", "ticker": "MSFT",
     "asset_type": "Stock", "type": "Sale (Full)", "amount": "$15,001 - $50,000",
     "senator": "John B Smith", "disclosure_date": "02/20/2024"},
    {"transaction_date": "01/04/2024", "ticker": "--", "asset_type": "Stock",
     "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "Jane A Doe",
     "disclosure_date": "02/15/2024"},                       # unknown ticker -> dropped
    {"transaction_date": "01/05/2024", "ticker": "TSLA", "asset_type": "Stock Option",
     "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "X",
     "disclosure_date": "02/16/2024"},                       # option -> dropped
    {"transaction_date": "01/06/2024", "ticker": "NVDA", "asset_type": "Stock",
     "type": "Exchange", "amount": "$1,001 - $15,000", "senator": "Y",
     "disclosure_date": "02/16/2024"},                       # exchange -> dropped
]

FMP_HOUSE_FIXTURE = [
    {"symbol": "GOOGL", "transactionDate": "2024-01-10", "disclosureDate": "2024-02-25",
     "type": "Purchase", "amount": "$15,001 - $50,000", "representative": "Rep Alpha",
     "assetType": "Stock"},
    {"symbol": "BRK.B", "transactionDate": "2024-01-11", "disclosureDate": "2024-02-26",
     "type": "Sale (Partial)", "amount": "$1,001 - $15,000", "representative": "Rep Beta",
     "assetType": "Stock"},
]

FMP_SENATE_FIXTURE = [
    {"symbol": "COST", "transactionDate": "2024-01-12", "disclosureDate": "2024-02-27",
     "type": "Purchase", "amount": "$50,001 - $100,000", "office": "Sen Gamma",
     "assetType": "Stock"},
]


# --- amount range parsing ---------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("$1,001 - $15,000", (1001.0, 15000.0)),
    ("$15,001 - $50,000", (15001.0, 50000.0)),
    ("$1,000,001 - $5,000,000", (1000001.0, 5000000.0)),
    ("Over $50,000,000", (50000000.0, 50000000.0)),
    ("$1,000,001 +", (1000001.0, 1000001.0)),
    ("$5,000", (5000.0, 5000.0)),
    ("-", (0.0, 0.0)),
    ("", (0.0, 0.0)),
    (None, (0.0, 0.0)),
])
def test_parse_amount_range(raw, expected):
    assert ct.parse_amount_range(raw) == expected


def test_amount_range_keeps_both_bounds_not_a_point():
    lo, hi = ct.parse_amount_range("$1,001 - $15,000")
    assert lo != hi  # the range is preserved, never collapsed to one number


# --- ticker / type / date normalization -------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("AAPL", "AAPL"), ("aapl", "AAPL"), ("  msft ", "MSFT"), ("$TSLA", "TSLA"),
    ("BRK.B", "BRK.B"), ("BRK-B", "BRK-B"), ("F", "F"),
    ("OTIS (1)", "OTIS"), ("OTIS (2)", "OTIS"), ("AAPL (10)", "AAPL"),  # split-lot markers stripped
    ("--", None), ("", None), ("N/A", None), (None, None),
    ("AAPL240119C00150000", None),  # option contract symbol
    ("123", None),                  # numeric junk
])
def test_normalize_ticker(raw, expected):
    assert ct.normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Purchase", "buy"), ("purchase", "buy"), ("BUY", "buy"),
    ("Sale", "sell"), ("Sale (Full)", "sell"), ("Sale (Partial)", "sell"), ("sale", "sell"),
    ("Exchange", None), ("", None), (None, None),  # Exchange/other → ignored
])
def test_normalize_transaction_type(raw, expected):
    assert ct.normalize_transaction_type(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("01/02/2024", "2024-01-02"), ("2024-01-02", "2024-01-02"),
    ("1/2/24", "2024-01-02"), ("garbage", ""), ("", ""), (None, ""),
])
def test_parse_date(raw, expected):
    assert ct.parse_date(raw) == expected


@pytest.mark.parametrize("asset_type, expected", [
    ("Stock", True), ("stock", True), ("  Stock ", True),
    ("REIT", False),               # explicit decision: REIT excluded by default (config-opt-in)
    ("Corporate Bond", False),     # a bond must never become a stock buy (case 1)
    ("Stock Option", False), ("Municipal Security", False), ("ETF", False), ("Crypto", False),
    ("Common Stock", False),       # strict allowlist: only configured types (default {"stock"})
    ("", False), (None, False),    # fail-closed: an unknown/missing asset type is NOT copyable
])
def test_is_copyable_asset(asset_type, expected):
    assert ct._is_copyable_asset(asset_type) is expected


# --- source parsers ---------------------------------------------------------


def test_parse_senate_stock_watcher_filters_and_normalizes():
    trades = ct.parse_senate_stock_watcher(SENATE_FIXTURE)
    # Only AAPL (buy) and MSFT (sell) survive: '--', option, and exchange drop.
    assert [(t.ticker, t.transaction_type) for t in trades] == [
        ("AAPL", "buy"), ("MSFT", "sell"),
    ]
    aapl = trades[0]
    assert aapl.chamber == "senate"
    assert aapl.source == "senate_stock_watcher"
    assert aapl.member == "Jane A Doe"
    assert aapl.transaction_date == "2024-01-02"
    assert aapl.disclosure_date == "2024-02-15"
    assert (aapl.amount_min, aapl.amount_max) == (1001.0, 15000.0)


def test_parse_senate_falls_back_to_date_recieved_and_name_parts():
    rec = [{"transaction_date": "01/02/2024", "ticker": "AAPL", "asset_type": "Stock",
            "type": "Purchase", "amount": "$1,001 - $15,000",
            "first_name": "Jane A", "last_name": "Doe", "date_recieved": "01/20/2024"}]
    t = ct.parse_senate_stock_watcher(rec)[0]
    assert t.member == "Jane A Doe"
    assert t.disclosure_date == "2024-01-20"  # date_recieved used when disclosure_date absent


def test_bond_with_equity_ticker_is_never_copied():
    # Case 1: live data has Corporate Bond rows carrying equity tickers
    # (e.g. OTIS/JPM). A bond disclosure must NEVER become a stock buy.
    rows = [
        {"transaction_date": "01/02/2024", "ticker": "OTIS", "asset_type": "Corporate Bond",
         "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "X", "disclosure_date": "02/15/2024"},
        {"transaction_date": "01/02/2024", "ticker": "JPM", "asset_type": "Corporate Bond",
         "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "X", "disclosure_date": "02/15/2024"},
    ]
    assert ct.parse_senate_stock_watcher(rows) == []


def test_fmp_reit_excluded_by_default():
    # Case 1: REIT is not "Stock"; excluded under the default allowlist.
    rows = [{"symbol": "O", "assetType": "REIT", "type": "Purchase", "amount": "$1,001 - $15,000",
             "representative": "X", "transactionDate": "2024-01-02", "disclosureDate": "2024-02-15"}]
    assert ct.parse_fmp_trades(rows, "house") == []


def test_only_stock_assettype_is_copied():
    rows = [
        {"symbol": "AAPL", "assetType": "Stock", "type": "Purchase", "amount": "$1,001 - $15,000",
         "representative": "Rep A", "transactionDate": "2024-01-02", "disclosureDate": "2024-02-15"},
        {"symbol": "OTIS", "assetType": "Corporate Bond", "type": "Purchase", "amount": "$1,001 - $15,000",
         "representative": "Rep A", "transactionDate": "2024-01-02", "disclosureDate": "2024-02-15"},
        {"symbol": "O", "assetType": "REIT", "type": "Purchase", "amount": "$1,001 - $15,000",
         "representative": "Rep A", "transactionDate": "2024-01-02", "disclosureDate": "2024-02-15"},
    ]
    trades = ct.parse_fmp_trades(rows, "house")
    assert [t.ticker for t in trades] == ["AAPL"]  # only the Stock row survives


def test_stock_sale_with_blank_assettype_still_exits():
    # Sell side is lenient (never suppress a risk-reducing exit): a held stock's
    # SALE must still produce a sell trade even when the row's assetType is
    # blank/missing — otherwise the mirror-exit would never fire and the virtual
    # book would be stuck long.
    rows = [{"transaction_date": "01/02/2024", "ticker": "AAPL", "asset_type": "",
             "type": "Sale (Full)", "amount": "$1,001 - $15,000", "senator": "Jane Doe",
             "disclosure_date": "02/15/2024"}]
    trades = ct.parse_senate_stock_watcher(rows)
    assert [(t.ticker, t.transaction_type) for t in trades] == [("AAPL", "sell")]


def test_bond_sale_is_dropped_so_it_cannot_false_close_a_stock():
    # A Corporate Bond SALE must NOT become a sell trade — otherwise it would
    # mirror-close a held stock position of the same ticker+member.
    rows = [{"transaction_date": "01/02/2024", "ticker": "OTIS", "asset_type": "Corporate Bond",
             "type": "Sale (Full)", "amount": "$1,001 - $15,000", "senator": "Jane Doe",
             "disclosure_date": "02/15/2024"}]
    assert ct.parse_senate_stock_watcher(rows) == []


def test_reit_copied_when_opted_in(monkeypatch):
    # The allowlist is config-driven: opting REIT in lets a REIT buy through.
    monkeypatch.setattr(config, "CONGRESS_BUY_ASSET_TYPES", frozenset({"stock", "reit"}))
    rows = [{"symbol": "O", "assetType": "REIT", "type": "Purchase", "amount": "$1,001 - $15,000",
             "representative": "X", "transactionDate": "2024-01-02", "disclosureDate": "2024-02-15"}]
    assert [t.ticker for t in ct.parse_fmp_trades(rows, "house")] == ["O"]


def test_split_lots_same_amount_dedupe_to_one():
    # Case 5: one transaction reported in parts, marked (1)/(2) on the ticker.
    # They normalize to the same symbol and dedupe to a single intended trade.
    rows = [
        {"transaction_date": "01/02/2024", "ticker": "OTIS (1)", "asset_type": "Stock",
         "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "Jane Doe", "disclosure_date": "02/15/2024"},
        {"transaction_date": "01/02/2024", "ticker": "OTIS (2)", "asset_type": "Stock",
         "type": "Purchase", "amount": "$1,001 - $15,000", "senator": "Jane Doe", "disclosure_date": "02/15/2024"},
    ]
    trades = ct.dedupe(ct.parse_senate_stock_watcher(rows))
    assert len(trades) == 1 and trades[0].ticker == "OTIS"


def test_parse_fmp_trades_house_and_senate():
    house = ct.parse_fmp_trades(FMP_HOUSE_FIXTURE, "house")
    assert [(t.ticker, t.transaction_type, t.chamber) for t in house] == [
        ("GOOGL", "buy", "house"), ("BRK.B", "sell", "house"),
    ]
    assert house[0].source == "fmp"
    assert house[0].member == "Rep Alpha"
    assert house[0].disclosure_date == "2024-02-25"
    senate = ct.parse_fmp_trades(FMP_SENATE_FIXTURE, "senate")
    assert senate[0].ticker == "COST" and senate[0].member == "Sen Gamma"


# --- dedup / filters / lag --------------------------------------------------


def test_dedupe_drops_refiled_duplicates():
    trades = ct.parse_senate_stock_watcher(SENATE_FIXTURE * 2)  # every row twice
    deduped = ct.dedupe(trades)
    assert len(deduped) == 2  # AAPL + MSFT, duplicates collapsed


def test_buys_only():
    trades = ct.parse_senate_stock_watcher(SENATE_FIXTURE)
    assert [t.ticker for t in ct.buys_only(trades)] == ["AAPL"]


def test_lag_days_surfaces_disclosure_delay():
    t = ct.parse_senate_stock_watcher(SENATE_FIXTURE)[0]
    assert t.lag_days == 44  # 2024-01-02 traded, 2024-02-15 disclosed
    assert t.to_dict()["lag_days"] == 44


def test_lag_days_zero_when_dates_unknown():
    t = ct.CongressTrade("M", "senate", "AAPL", "buy", "", "", 1001.0, 15000.0, "x")
    assert t.lag_days == 0


# --- atomic disk cache round-trip -------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "congress_trades.json"
    trades = ct.parse_senate_stock_watcher(SENATE_FIXTURE)
    ct.save_trades(trades, path)

    assert path.exists()
    assert not (tmp_path / "congress_trades.json.tmp").exists()  # tmp cleaned (atomic rename)

    loaded = ct.load_trades(path)
    assert [t.to_dict() for t in loaded] == [t.to_dict() for t in trades]


def test_load_trades_missing_file_returns_empty(tmp_path):
    assert ct.load_trades(tmp_path / "nope.json") == []


def test_load_trades_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert ct.load_trades(p) == []


# --- fail-open fetch (monkeypatched HTTP, no network) -----------------------


def _patch_fetch(monkeypatch, handler):
    async def _f(session, url, **kwargs):
        return handler(url)
    monkeypatch.setattr(ct, "fetch_with_retry", _f)


def test_fetch_trades_uses_senate_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(config, "CONGRESS_DATA_API_KEY", "")
    _patch_fetch(monkeypatch, lambda url: (200, json.dumps(SENATE_FIXTURE)))
    trades = asyncio.run(ct.fetch_trades(None))
    assert {t.source for t in trades} == {"senate_stock_watcher"}
    assert {t.ticker for t in trades} == {"AAPL", "MSFT"}


def test_fetch_trades_uses_fmp_when_key_present(monkeypatch):
    monkeypatch.setattr(config, "CONGRESS_DATA_API_KEY", "FMPKEY")

    def handler(url):
        if "house-latest" in url:
            return (200, json.dumps(FMP_HOUSE_FIXTURE))
        if "senate-latest" in url:
            return (200, json.dumps(FMP_SENATE_FIXTURE))
        return (404, "")
    _patch_fetch(monkeypatch, handler)

    trades = asyncio.run(ct.fetch_trades(None))
    assert {t.source for t in trades} == {"fmp"}
    assert {t.ticker for t in trades} == {"GOOGL", "BRK.B", "COST"}


@pytest.mark.parametrize("bad", [None, (500, ""), (200, "not json"), (403, "")])
def test_fetch_trades_fails_open(monkeypatch, bad):
    monkeypatch.setattr(config, "CONGRESS_DATA_API_KEY", "")
    _patch_fetch(monkeypatch, lambda url: bad)
    assert asyncio.run(ct.fetch_trades(None)) == []


def test_refresh_keeps_existing_cache_on_empty_fetch(tmp_path, monkeypatch):
    path = tmp_path / "congress_trades.json"
    # Seed a cache, then a failing fetch must NOT overwrite it.
    ct.save_trades(ct.parse_senate_stock_watcher(SENATE_FIXTURE), path)
    monkeypatch.setattr(config, "CONGRESS_DATA_API_KEY", "")
    _patch_fetch(monkeypatch, lambda url: None)  # fail-open empty

    result = asyncio.run(ct.refresh(None, path))
    assert result == []
    assert len(ct.load_trades(path)) == 2  # prior cache intact
