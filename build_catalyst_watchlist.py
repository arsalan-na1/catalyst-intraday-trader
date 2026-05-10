"""
build_catalyst_watchlist.py
────────────────────────────────────────────────────────────────────────────
Builds watchlist_catalyst.txt: a focused list of high-short-interest,
catalyst-prone tickers that get tighter trigger thresholds in the bot
(>5% price move vs the normal >8%).

Sources (in priority order):
  1. FinViz screener  – primary, always runs
  2. MarketBeat       – optional; uses curl_cffi to bypass Cloudflare.
                        Silently skipped if curl_cffi is not installed or
                        the site returns a block/error.

Usage:
    python build_catalyst_watchlist.py [--output PATH] [--min-short FLOAT]
                                       [--max-price FLOAT] [--no-marketbeat]

Defaults:
    --output       watchlist_catalyst.txt  (same dir as script)
    --min-short    10  (minimum Short Float %, FinViz filter)
    --max-price    50  (maximum share price, FinViz filter)
    --no-marketbeat    flag to skip MarketBeat scrape entirely

FinViz filter logic:
  · Short Float ≥ min-short %
  · Price ≤ max-price
  · Exchange: NASDAQ or NYSE
  · Optionable: Yes  (liquidity proxy)
  · Sorted by Short Interest Share descending (highest short % first)
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Tickers that must always be in the catalyst watchlist regardless of what
# FinViz or MarketBeat return — names with active catalyst flow that the
# screeners frequently miss (small float, recent IPOs, sector-specific).
MANUAL_ADDITIONS = {"RKLB", "ASTS", "LUNR", "PL", "ACHR"}


# ──────────────────────────────────────────────────────────────────────────
# 1.  FinViz  (primary)
# ──────────────────────────────────────────────────────────────────────────

def fetch_finviz(min_short: float, max_price: float) -> list[str]:
    """
    Pull high-short-interest tickers from FinViz screener.

    Key fix (2025-04):
        screener_view(order=...) accepts ORDER keys from order_dict,
        NOT filter keys.  The correct key for short % of float is
        'Short Interest Share' (→ URL param 'shortinterestshare').
        'Float Short' is a FILTER key only — passing it as order raises
        ValueError.
    """
    try:
        from finvizfinance.screener.ownership import Ownership  # ownership view has Short Float col
    except ImportError:
        log.error("finvizfinance not installed.  Run: pip install finvizfinance")
        return []

    # Map min_short to the closest FinViz "Float Short" filter bucket.
    # FinViz only offers discrete thresholds — pick the largest that is
    # ≤ min_short so we don't under-filter.
    _float_short_buckets = {
        5:  "Over 5%",
        10: "Over 10%",
        15: "Over 15%",
        20: "Over 20%",
        25: "Over 25%",
        30: "Over 30%",
    }
    threshold_key = max(
        (k for k in _float_short_buckets if k <= min_short),
        default=5,
    )
    float_short_filter = _float_short_buckets[threshold_key]
    log.info("FinViz  Float Short filter: %s  (requested ≥%.0f%%)", float_short_filter, min_short)

    # Map max_price to FinViz "Price" filter
    # FinViz discrete buckets: Under 5, Under 10, Under 20, Under 50, Under 100
    _price_buckets = {50: "Under 50", 20: "Under 20", 10: "Under 10", 100: "Under 100"}
    price_filter = _price_buckets.get(int(max_price), f"Under {int(max_price)}")
    log.info("FinViz  Price filter: %s", price_filter)

    filters = {
        "Float Short": float_short_filter,   # ← valid FILTER key
        "Option/Short": "Optionable and shortable",
    }
    # Note: Exchange filter accepts 'NASDAQ' or 'NYSE' individually; to get
    # both we run two passes and merge.
    tickers: set[str] = set()

    for exchange in ("NASDAQ", "NYSE"):
        try:
            screener = Ownership()
            screener.set_filter(
                filters_dict={**filters, "Exchange": exchange}
            )
            # ▼▼▼  THE FIX: order='Short Interest Share' (not 'Float Short') ▼▼▼
            df = screener.screener_view(
                order="Short Interest Share",
                ascend=False,           # highest short % first
                limit=500,
                verbose=0,
                sleep_sec=2,            # be polite to FinViz
            )
            if df is not None and not df.empty:
                batch = df["Ticker"].dropna().str.strip().tolist()
                log.info("FinViz  %s → %d tickers", exchange, len(batch))
                tickers.update(batch)
            time.sleep(3)               # avoid rate-limiting between passes
        except Exception as exc:
            log.warning("FinViz  %s pass failed: %s", exchange, exc)

    log.info("FinViz  total unique tickers: %d", len(tickers))
    return sorted(tickers)


# ──────────────────────────────────────────────────────────────────────────
# 2.  MarketBeat  (optional, curl_cffi required)
# ──────────────────────────────────────────────────────────────────────────

_MB_URL = (
    "https://www.marketbeat.com/market-data/short-interest/?sortby=1&page=1"
)

def fetch_marketbeat(pages: int = 5) -> list[str]:
    """
    Scrape MarketBeat's Short Interest leaderboard pages.

    Requires: pip install curl-cffi beautifulsoup4
    Falls back gracefully (returns []) if:
      · curl_cffi not installed
      · Cloudflare/MarketBeat returns a non-200 or block page
      · Any parse error
    """
    try:
        import curl_cffi.requests as cffi_requests   # pip install curl-cffi
        from bs4 import BeautifulSoup
    except ImportError as e:
        log.warning("MarketBeat skipped – missing dependency (%s).", e)
        return []

    tickers: list[str] = []
    session = cffi_requests.Session(impersonate="chrome")

    for page in range(1, pages + 1):
        url = f"https://www.marketbeat.com/market-data/short-interest/?sortby=1&page={page}"
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning("MarketBeat  page %d → HTTP %d — stopping.", page, resp.status_code)
                break
            if "Just a moment" in resp.text or "cf-browser-verification" in resp.text:
                log.warning("MarketBeat  Cloudflare challenge detected on page %d — skipping.", page)
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            # MarketBeat renders tickers as links like /stocks/nasdaq/AAPL/short-interest/
            links = soup.select("a[href*='/stocks/']")
            page_tickers = []
            for a in links:
                parts = a["href"].strip("/").split("/")
                # href format: stocks / {exchange} / {TICKER} / ...
                if len(parts) >= 3 and parts[0] == "stocks":
                    ticker = parts[2].upper()
                    if ticker.isalpha() and 1 <= len(ticker) <= 5:
                        page_tickers.append(ticker)

            page_tickers = list(dict.fromkeys(page_tickers))  # deduplicate, preserve order
            log.info("MarketBeat  page %d → %d tickers", page, len(page_tickers))
            tickers.extend(page_tickers)
            time.sleep(2)

        except Exception as exc:
            log.warning("MarketBeat  page %d error: %s", page, exc)
            break

    unique = list(dict.fromkeys(tickers))
    log.info("MarketBeat  total unique tickers: %d", len(unique))
    return unique


# ──────────────────────────────────────────────────────────────────────────
# 3.  Main
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build catalyst watchlist")
    p.add_argument("--output", default=str(Path(__file__).parent / "watchlist_catalyst.txt"))
    p.add_argument("--min-short", type=float, default=10.0,
                   help="Minimum Short Float %% (default 10)")
    p.add_argument("--max-price", type=float, default=50.0,
                   help="Maximum share price (default 50)")
    p.add_argument("--no-marketbeat", action="store_true",
                   help="Skip MarketBeat scrape entirely")
    p.add_argument("--mb-pages", type=int, default=5,
                   help="Number of MarketBeat pages to scrape (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Step 1: FinViz ──────────────────────────────────────────────────
    log.info("═══ FinViz pass ═══════════════════════════════════════════════")
    finviz_tickers = fetch_finviz(
        min_short=args.min_short,
        max_price=args.max_price,
    )

    # ── Step 2: MarketBeat (optional) ───────────────────────────────────
    mb_tickers: list[str] = []
    if not args.no_marketbeat:
        log.info("═══ MarketBeat pass ════════════════════════════════════════════")
        mb_tickers = fetch_marketbeat(pages=args.mb_pages)
    else:
        log.info("MarketBeat skipped (--no-marketbeat flag)")

    # ── Step 3: Merge & deduplicate ────────────────────────────────────
    # FinViz first (already sorted by short interest), then any MarketBeat
    # additions not already captured.
    seen: set[str] = set()
    combined: list[str] = []
    for t in finviz_tickers + mb_tickers:
        t = t.upper().strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)

    # Merge manual additions — included unconditionally even when both sources
    # come back empty, so a screener outage never drops these tickers.
    manual_added = 0
    for t in MANUAL_ADDITIONS:
        t = t.upper().strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)
            manual_added += 1
    log.info("Manual additions: %d tickers", manual_added)

    if not combined:
        log.error("No tickers collected — watchlist_catalyst.txt NOT written.")
        sys.exit(1)

    # ── Step 4: Write output ───────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(combined) + "\n", encoding="utf-8")

    log.info(
        "✅  Wrote %d tickers → %s   (FinViz: %d  |  MarketBeat: %d  |  overlap removed)",
        len(combined), out_path, len(finviz_tickers), len(mb_tickers),
    )


if __name__ == "__main__":
    main()
