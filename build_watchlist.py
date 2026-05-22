"""One-shot watchlist generator.

Produces watchlist.txt from five sources, merged and deduplicated:
  1. S&P 500 constituents (Wikipedia)
  2. S&P 400 MidCap constituents (Wikipedia)
  3. S&P 600 SmallCap constituents (Wikipedia)
  4. Russell 2000 constituents (iShares IWM ETF holdings CSV)
  5. FinViz high-volume screener (Average Volume > 1M on NASDAQ + NYSE)

Run manually (monthly is fine — index rebalances are quarterly/annual):

    python build_watchlist.py

Target: 2,000–3,000 unique tickers.

Alpaca IEX WebSocket: no documented symbol limit for minute-bar subscriptions
(confirmed at https://docs.alpaca.markets/docs/websocket-streaming), so no
prioritisation by ADV is required. All tickers are written to watchlist.txt.

Ticker normalisation rules:
  - Dashes → dots (BRK-B → BRK.B, as Alpaca uses dot notation)
  - Uppercase, strip whitespace
  - Max 6 chars, alphabetic + dot only
  - Known non-ticker strings (CASH, USD, etc.) excluded
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("build_watchlist")

# iShares ETF holding CSVs ─ same download pattern across all products
_ISHARES_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Referer": "https://www.ishares.com/",
}

# Generic request headers for Wikipedia + FinViz
HEADERS = {"User-Agent": "Mozilla/5.0 (TradingBot watchlist builder)"}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

# iShares IWM holdings CSV (Russell 2000)
IWM_CSV_URL = (
    "https://www.ishares.com/us/products/239726/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)

OUTPUT_PATH = Path(__file__).resolve().parent / "watchlist.txt"

_INVALID_TICKERS = {"CASH", "USD", "XTSLA", "MARGIN_USD", ""}


def _is_valid_ticker(t: str) -> bool:
    if not t or len(t) > 6:
        return False
    if not all(c.isalpha() or c == "." for c in t):
        return False
    if not any(c.isalpha() for c in t):  # reject bare "." or ".."
        return False
    return t not in _INVALID_TICKERS


# ---------------------------------------------------------------------------
# Source 1: S&P 500 (Wikipedia — table id="constituents")
# ---------------------------------------------------------------------------

def fetch_sp500() -> set[str]:
    log.info("Fetching S&P 500 constituents from Wikipedia...")
    resp = requests.get(SP500_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia")
    tickers: set[str] = set()
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        t = cells[0].get_text(strip=True).replace("-", ".")
        if _is_valid_ticker(t):
            tickers.add(t)
    log.info("  S&P 500: %d tickers", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# Sources 2 & 3: S&P 400 and S&P 600 (Wikipedia — no stable id attribute)
# ---------------------------------------------------------------------------

def _fetch_wikipedia_index(url: str, name: str) -> set[str]:
    """Generic scraper for a Wikipedia index page that has a wikitable with
    a 'Symbol' or 'Ticker' column. Uses pandas read_html for robustness."""
    log.info("Fetching %s from Wikipedia...", name)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as exc:
        log.warning("pd.read_html failed for %s: %s; trying BeautifulSoup fallback", name, exc)
        return _fetch_wikipedia_bs4(resp.text, name)

    for df in tables:
        for col_candidate in ("Symbol", "Ticker", "Ticker symbol"):
            if col_candidate in df.columns:
                tickers: set[str] = set()
                for raw in df[col_candidate].dropna():
                    t = str(raw).strip().upper().replace("-", ".")
                    if _is_valid_ticker(t):
                        tickers.add(t)
                if tickers:
                    log.info("  %s: %d tickers (column '%s')", name, len(tickers), col_candidate)
                    return tickers

    log.warning("  %s: no ticker column found via read_html; trying BS4 fallback", name)
    return _fetch_wikipedia_bs4(resp.text, name)


def _fetch_wikipedia_bs4(html: str, name: str) -> set[str]:
    """BeautifulSoup fallback: find the first wikitable with a Symbol header."""
    soup = BeautifulSoup(html, "lxml")
    tickers: set[str] = set()
    for table in soup.find_all("table"):
        classes = " ".join(table.get("class", []))
        if "wikitable" not in classes:
            continue
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        sym_idx = None
        for candidate in ("Symbol", "Ticker", "Ticker symbol"):
            if candidate in headers:
                sym_idx = headers.index(candidate)
                break
        if sym_idx is None:
            continue
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) > sym_idx:
                t = cells[sym_idx].get_text(strip=True).replace("-", ".")
                if _is_valid_ticker(t):
                    tickers.add(t)
        if tickers:
            log.info("  %s (BS4 fallback): %d tickers", name, len(tickers))
            return tickers
    log.warning("  %s: could not extract any tickers", name)
    return set()


def fetch_sp400() -> set[str]:
    """Return the set of S&P 400 tickers from the Wikipedia index."""
    return _fetch_wikipedia_index(SP400_URL, "S&P 400")


def fetch_sp600() -> set[str]:
    return _fetch_wikipedia_index(SP600_URL, "S&P 600")


# ---------------------------------------------------------------------------
# Source 4: Russell 2000 from iShares IWM ETF holdings CSV
# ---------------------------------------------------------------------------

def fetch_iwm() -> set[str]:
    """Russell 2000 from iShares IWM ETF holdings CSV.

    Uses a browser-like User-Agent and Referer header. Returns an empty set
    and logs a warning on any failure so the rest of the build continues.
    """
    log.info("Fetching Russell 2000 (IWM) holdings from iShares...")
    try:
        resp = requests.get(IWM_CSV_URL, headers=_ISHARES_HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("IWM CSV fetch failed (%s); skipping Russell 2000", exc)
        return set()

    raw = resp.text
    lines = raw.splitlines()
    # iShares CSVs have ~10 junk header rows; find the data header row.
    header_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("Ticker,"):
            header_idx = idx
            break
    if header_idx is None:
        log.warning("IWM CSV: could not find 'Ticker' header row; skipping")
        return set()

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception as exc:
        log.warning("IWM CSV parse failed: %s; skipping", exc)
        return set()

    if "Ticker" not in df.columns:
        log.warning("IWM CSV missing Ticker column; skipping")
        return set()

    tickers: set[str] = set()
    for raw_t in df["Ticker"].dropna():
        t = str(raw_t).strip().upper().replace("-", ".")
        if _is_valid_ticker(t):
            tickers.add(t)

    log.info("  Russell 2000 (IWM): %d tickers", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# Source 5: FinViz high-volume screener (Average Volume > 1M)
# ---------------------------------------------------------------------------

def fetch_finviz_active(max_per_exchange: int = 300) -> set[str]:
    """FinViz screener for stocks with Average Volume > 1M on NASDAQ and NYSE.

    Silently returns an empty set if finvizfinance is not installed.
    """
    try:
        from finvizfinance.screener.overview import Overview
    except ImportError:
        log.warning("finvizfinance not installed; skipping FinViz active screener")
        return set()

    tickers: set[str] = set()
    for exchange in ("NASDAQ", "NYSE"):
        try:
            screener = Overview()
            screener.set_filter(filters_dict={
                "Average Volume": "Over 1M",
                "Exchange": exchange,
            })
            df = screener.screener_view(limit=max_per_exchange, verbose=0, sleep_sec=2)
            if df is not None and not df.empty and "Ticker" in df.columns:
                for t in df["Ticker"].dropna():
                    t = str(t).strip().upper()
                    if _is_valid_ticker(t):
                        tickers.add(t)
            log.info("  FinViz active (%s): running total %d", exchange, len(tickers))
            time.sleep(3)
        except Exception as exc:
            log.warning("FinViz active screener failed for %s: %s", exchange, exc)

    log.info("  FinViz active total: %d tickers", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# Alpaca validation (optional --validate flag)
# ---------------------------------------------------------------------------

def _validate_against_alpaca(tickers: list[str]) -> list[str]:
    """Remove tickers Alpaca doesn't recognise or marks as inactive/untradable.

    Calls get_all_assets() once (one API request) and cross-references against
    the returned set of active tradable US equity symbols — much faster than
    100-ticker batched get_asset() loops for a 2 000+ ticker list.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        log.warning("ALPACA_API_KEY/ALPACA_SECRET_KEY not set; skipping Alpaca validation")
        return tickers

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
    except ImportError:
        log.warning("alpaca-py not installed; skipping Alpaca validation")
        return tickers

    log.info("Validating %d tickers against Alpaca asset registry…", len(tickers))
    client = TradingClient(api_key, api_secret, paper=True)
    try:
        req = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
        )
        assets = client.get_all_assets(req)
    except Exception as exc:
        log.warning("Alpaca get_all_assets failed: %s; skipping validation", exc)
        return tickers

    tradable: set[str] = {a.symbol.upper() for a in assets if getattr(a, "tradable", False)}
    log.info("Alpaca returned %d active tradable US equity symbols", len(tradable))

    validated: list[str] = []
    pruned = 0
    for t in tickers:
        if t in tradable:
            validated.append(t)
        else:
            pruned += 1
            log.debug("  pruned: %s (not in Alpaca active/tradable set)", t)

    log.info(
        "Alpaca validation: kept %d, pruned %d (from %d total)",
        len(validated), pruned, len(tickers),
    )
    return validated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help=(
            "After building the list, remove tickers Alpaca does not recognise "
            "or marks inactive/untradable. Requires ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in .env."
        ),
    )
    args = p.parse_args()

    sp500 = fetch_sp500()
    sp400 = fetch_sp400()
    sp600 = fetch_sp600()

    iwm = fetch_iwm()

    # Small delay between iShares and FinViz to avoid hammering external hosts.
    time.sleep(2)
    finviz_active = fetch_finviz_active()

    all_tickers = sorted(sp500 | sp400 | sp600 | iwm | finviz_active)

    # Filter out any residual non-ticker strings (e.g. the "." phantom)
    all_tickers = [t for t in all_tickers if _is_valid_ticker(t)]

    if args.validate:
        all_tickers = _validate_against_alpaca(all_tickers)
        # Re-apply format filter in case validate returned anything unexpected
        all_tickers = [t for t in all_tickers if _is_valid_ticker(t)]

    log.info(
        "Merged: S&P500=%d S&P400=%d S&P600=%d IWM=%d FinViz=%d → total unique=%d",
        len(sp500), len(sp400), len(sp600), len(iwm), len(finviz_active), len(all_tickers),
    )

    if len(all_tickers) < 1000:
        log.warning(
            "Final count (%d) is lower than expected — check individual source failures above.",
            len(all_tickers),
        )

    OUTPUT_PATH.write_text("\n".join(all_tickers) + "\n", encoding="utf-8")
    log.info("Wrote %d tickers to %s", len(all_tickers), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
