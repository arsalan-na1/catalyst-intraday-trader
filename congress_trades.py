"""Congressional-trade data layer — shared, network-isolated, fail-open.

Daily bulk pull of recently *disclosed* congressional stock trades, normalized
to a common ``CongressTrade`` record and cached to disk (atomic tmp->rename).
This is a 45-day-delayed feed by law, so it is pulled at most daily — NEVER on
the intraday hot path. Consumed by the off-by-default virtual congress-copy
portfolio (see docs/specs/congress-copy.md).

Free sources:
  * Financial Modeling Prep (FMP) free tier — House + Senate ``latest``
    endpoints — used when ``config.CONGRESS_DATA_API_KEY`` is set. Full coverage.
  * Senate Stock Watcher — keyless GitHub-raw mirror of ``all_transactions.json``
    — automatic fallback when no FMP key is present. Senate-only, zero setup.

Everything fails open: any network/parse error yields ``[]``, never an
exception, so this can never break the bot. All HTTP goes through
``http_utils.fetch_with_retry``. No Alpaca, no Gemini, no shared state.

The three inherent realities are handled explicitly:
  * 45-day delay  -> both ``transaction_date`` and ``disclosure_date`` are kept
                     and ``lag_days`` is exposed; trades are lagging signals.
  * dollar RANGES -> ``amount_min`` / ``amount_max`` are parsed and kept as
                     bounds; the range is never collapsed to a point estimate.
  * dupes/refiles -> deduped on a stable (member,ticker,date,type,amount) key,
                     and tickers are normalized ('--'/options/blank dropped).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import aiohttp

import config
from http_utils import fetch_with_retry

log = logging.getLogger("congress_trades")

_BUY_WORDS = ("purchase", "buy")
_SELL_WORDS = ("sale", "sell")
# Non-equity asset labels we never copy (the benchmark copies ordinary stock).
_NON_EQUITY_MARKERS = (
    "option", "bond", "note", "fund", "etf", "crypto", "future", "warrant",
    "municipal", "treasur",
)
# AAPL, F, GOOGL, BRK.B, BF.B, BRK-B — letters with an optional class suffix.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?$")


@dataclass(frozen=True)
class CongressTrade:
    member: str
    chamber: str            # "house" | "senate"
    ticker: str             # normalized uppercase equity symbol
    transaction_type: str   # "buy" | "sell"
    transaction_date: str   # ISO yyyy-mm-dd ('' if unknown)
    disclosure_date: str    # ISO yyyy-mm-dd ('' if unknown)
    amount_min: float       # lower bound of the disclosed dollar range
    amount_max: float       # upper bound of the disclosed dollar range
    source: str             # "fmp" | "senate_stock_watcher"

    @property
    def lag_days(self) -> int:
        """Disclosure date minus transaction date in days (>=0); 0 if unknown.

        Surfaces the legally-mandated reporting delay so consumers treat the
        trade as a lagging signal, not a fresh one."""
        t = _iso_to_date(self.transaction_date)
        d = _iso_to_date(self.disclosure_date)
        if t is None or d is None:
            return 0
        return max(0, (d - t).days)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["lag_days"] = self.lag_days
        return out


# --- pure normalization helpers --------------------------------------------


def parse_amount_range(raw) -> tuple[float, float]:
    """Parse a disclosed dollar range, e.g. '$1,001 - $15,000' -> (1001.0, 15000.0).

    Handles single bounds ('Over $50,000,000', '$1,000,001 +', '$15,000 -') by
    using the one value as both bounds, and missing/'-'/None as (0.0, 0.0).
    Never raises. Amounts are RANGES by law, so both bounds are preserved — we
    never invent a point estimate here.
    """
    if not isinstance(raw, str) or not raw.strip():
        return (0.0, 0.0)
    vals: list[float] = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", raw):
        try:
            vals.append(float(token.replace(",", "")))
        except ValueError:
            continue
    if not vals:
        return (0.0, 0.0)
    return (min(vals), max(vals))


def normalize_ticker(raw) -> str | None:
    """Uppercase/clean a ticker; return None for unknown/non-equity symbols."""
    if not isinstance(raw, str):
        return None
    t = raw.strip().upper().lstrip("$").strip()
    if not t or t in ("--", "—", "-", "N/A", "NA", "NONE"):
        return None
    if not _TICKER_RE.match(t):
        return None
    return t


def normalize_transaction_type(raw) -> str | None:
    """Map disclosure transaction types to 'buy' / 'sell' / None (ignore)."""
    if not isinstance(raw, str):
        return None
    low = raw.strip().lower()
    if any(w in low for w in _BUY_WORDS):
        return "buy"
    if any(w in low for w in _SELL_WORDS):
        return "sell"
    return None  # exchange / receive / other — not copied


def _iso_to_date(s) -> date | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def parse_date(raw) -> str:
    """Normalize a date to ISO yyyy-mm-dd; '' if unparseable.

    Accepts MM/DD/YYYY (Senate Stock Watcher) and YYYY-MM-DD (FMP)."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _is_equity(asset_type) -> bool:
    """True unless the asset is explicitly labelled non-equity (option/bond/...).

    Unspecified ('' / None) is allowed — ticker validity is the real gate."""
    if not asset_type or not isinstance(asset_type, str):
        return True
    low = asset_type.strip().lower()
    return not any(marker in low for marker in _NON_EQUITY_MARKERS)


def _member_name(*candidates) -> str:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ""


# --- source parsers (pure: dict records -> CongressTrade) ------------------


def parse_senate_stock_watcher(records: Iterable[dict]) -> list[CongressTrade]:
    """Parse Senate Stock Watcher ``all_transactions.json`` records."""
    out: list[CongressTrade] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        if not _is_equity(r.get("asset_type")):
            continue
        ticker = normalize_ticker(r.get("ticker"))
        ttype = normalize_transaction_type(r.get("type"))
        if ticker is None or ttype is None:
            continue
        lo, hi = parse_amount_range(r.get("amount"))
        member = _member_name(
            r.get("senator"),
            " ".join(x for x in (r.get("first_name"), r.get("last_name")) if x),
            r.get("office"),
        )
        out.append(CongressTrade(
            member=member, chamber="senate", ticker=ticker,
            transaction_type=ttype,
            transaction_date=parse_date(r.get("transaction_date")),
            disclosure_date=parse_date(r.get("disclosure_date") or r.get("date_recieved")),
            amount_min=lo, amount_max=hi, source="senate_stock_watcher",
        ))
    return out


def parse_fmp_trades(records: Iterable[dict], chamber: str) -> list[CongressTrade]:
    """Parse FMP house/senate trading records (defensive about field names)."""
    out: list[CongressTrade] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        if not _is_equity(r.get("assetType") or r.get("asset_type")):
            continue
        ticker = normalize_ticker(r.get("symbol") or r.get("ticker"))
        ttype = normalize_transaction_type(r.get("type"))
        if ticker is None or ttype is None:
            continue
        lo, hi = parse_amount_range(r.get("amount"))
        member = _member_name(
            r.get("representative"), r.get("senator"), r.get("office"),
            " ".join(x for x in (r.get("firstName"), r.get("lastName")) if x),
        )
        out.append(CongressTrade(
            member=member, chamber=chamber, ticker=ticker,
            transaction_type=ttype,
            transaction_date=parse_date(r.get("transactionDate") or r.get("transaction_date")),
            disclosure_date=parse_date(r.get("disclosureDate") or r.get("disclosure_date")),
            amount_min=lo, amount_max=hi, source="fmp",
        ))
    return out


# --- dedup / filters -------------------------------------------------------


def dedupe(trades: Iterable[CongressTrade]) -> list[CongressTrade]:
    """Drop duplicate/refiled rows on a stable identity key (first wins)."""
    seen: set[tuple] = set()
    out: list[CongressTrade] = []
    for t in trades:
        key = (
            t.member.lower(), t.ticker, t.transaction_date,
            t.transaction_type, t.amount_min, t.amount_max,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def buys_only(trades: Iterable[CongressTrade]) -> list[CongressTrade]:
    return [t for t in trades if t.transaction_type == "buy"]


# --- atomic disk cache -----------------------------------------------------


def save_trades(trades: list[CongressTrade], path=None) -> None:
    """Atomically persist trades (tmp->rename). Never raises."""
    try:
        path = Path(path or config.CONGRESS_TRADES_FILE)
        payload = {
            "updated_at": datetime.now(tz=config.MARKET_TZ).isoformat(),
            "count": len(trades),
            "trades": [t.to_dict() for t in trades],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic rename — no partial read on crash
    except Exception:
        log.warning("[CONGRESS] failed to persist trades", exc_info=True)


def load_trades(path=None) -> list[CongressTrade]:
    """Load cached trades. Returns [] if the file is missing/corrupt."""
    path = Path(path or config.CONGRESS_TRADES_FILE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    out: list[CongressTrade] = []
    for d in payload.get("trades", []):
        try:
            out.append(CongressTrade(
                member=d["member"], chamber=d["chamber"], ticker=d["ticker"],
                transaction_type=d["transaction_type"],
                transaction_date=d["transaction_date"],
                disclosure_date=d["disclosure_date"],
                amount_min=float(d["amount_min"]), amount_max=float(d["amount_max"]),
                source=d.get("source", ""),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --- fetch (fail-open) -----------------------------------------------------


async def _fetch_json(session: aiohttp.ClientSession, url: str, label: str):
    res = await fetch_with_retry(session, url)
    if res is None:
        log.warning("[CONGRESS] %s fetch exhausted retries", label)
        return None
    status, text = res
    if status != 200:
        log.warning("[CONGRESS] %s HTTP %s", label, status)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("[CONGRESS] %s returned non-JSON", label)
        return None


async def _fetch_senate_stock_watcher(session) -> list[CongressTrade]:
    data = await _fetch_json(session, config.CONGRESS_SENATE_FALLBACK_URL, "Senate Stock Watcher")
    if not isinstance(data, list):
        return []
    return parse_senate_stock_watcher(data)


async def _fetch_fmp(session) -> list[CongressTrade]:
    key = config.CONGRESS_DATA_API_KEY
    base = config.CONGRESS_FMP_BASE_URL.rstrip("/")
    limit = config.CONGRESS_FMP_LIMIT
    collected: list[CongressTrade] = []
    for chamber, path in (("house", "/stable/house-latest"), ("senate", "/stable/senate-latest")):
        url = f"{base}{path}?page=0&limit={limit}&apikey={key}"
        data = await _fetch_json(session, url, f"FMP {chamber}")
        if isinstance(data, list):
            collected.extend(parse_fmp_trades(data, chamber))
    return collected


async def fetch_trades(session: aiohttp.ClientSession) -> list[CongressTrade]:
    """Pull recent congressional trades from the configured free source.

    Uses FMP (House+Senate) when CONGRESS_DATA_API_KEY is set, else the keyless
    Senate Stock Watcher fallback. Fail-open: returns [] on any error.
    """
    try:
        raw = (await _fetch_fmp(session) if config.CONGRESS_DATA_API_KEY
               else await _fetch_senate_stock_watcher(session))
        return dedupe(raw)
    except Exception:
        log.warning("[CONGRESS] fetch failed; returning no trades (fail-open)", exc_info=True)
        return []


async def refresh(session: aiohttp.ClientSession, path=None) -> list[CongressTrade]:
    """Fetch + persist the latest trades. On a fail-open empty result the
    existing cache is kept untouched. Returns the freshly fetched trades."""
    trades = await fetch_trades(session)
    if trades:
        save_trades(trades, path)
        log.info("[CONGRESS] refreshed %d congressional trades", len(trades))
    else:
        log.info("[CONGRESS] no trades fetched; keeping existing cache")
    return trades


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    import asyncio

    async def _smoke():
        async with aiohttp.ClientSession() as s:
            trades = await fetch_trades(s)
            buys = buys_only(trades)
            print(f"fetched {len(trades)} trades ({len(buys)} buys); "
                  f"source={'fmp' if config.CONGRESS_DATA_API_KEY else 'senate_stock_watcher'}")
            for t in buys[:5]:
                print(f"  {t.disclosure_date} {t.ticker:6} {t.transaction_type:4} "
                      f"${t.amount_min:,.0f}-${t.amount_max:,.0f} lag={t.lag_days}d {t.member}")

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke())
