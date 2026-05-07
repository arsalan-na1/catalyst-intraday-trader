"""News fetcher: Alpaca + SEC EDGAR + Yahoo Finance + Seeking Alpha + FinViz + Reddit.

Fetch strategy (all sources run in parallel via asyncio.gather):
  Primary:   Alpaca news, SEC EDGAR 8-K, earnings-scoped Alpaca search
  Secondary: Yahoo Finance RSS, Seeking Alpha RSS, FinViz news (sync/thread),
             Reddit r/wallstreetbets + r/stocks (posts with ≥100 upvotes)
  Fallback:  NewsAPI.org — only if all primary + secondary return nothing

After gathering, results are merged and deduplicated by headline similarity
(Jaccard token overlap ≥ 0.80) and capped at _MERGED_CAP items.

A 15-second in-process cache prevents hammering the same endpoints when a
ticker triggers multiple pipeline passes in rapid succession.

Diagnostic logging: every fetch logs an INFO line with item counts so empty
results in production are debuggable without re-running with a debugger.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiohttp

import config
from http_utils import fetch_with_retry

log = logging.getLogger("news")

_MAX_ITEMS = 10
_ALPACA_LOOKBACK_HOURS = 48
# EDGAR 7-day window catches Friday-PM 8-Ks before Monday open.
_EDGAR_LOOKBACK_DAYS = 7
_MERGED_CAP = 15

# Per-ticker result cache to avoid hammering endpoints on burst triggers.
_news_cache: dict[str, tuple[float, list["NewsItem"]]] = {}
_NEWS_CACHE_SECONDS = 15.0
# Per-ticker locks: prevents concurrent triggers from all firing the 7-source
# fetch simultaneously for the same ticker (double-checked locking pattern).
_news_fetch_locks: dict[str, asyncio.Lock] = {}

# Reddit: only include posts above this upvote threshold.
_REDDIT_MIN_SCORE = 100

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

SEC_USER_AGENT = "trading-bot contact@example.com"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

_cik_cache: dict[str, str | None] = {}
_ticker_map_lock = asyncio.Lock()
_ticker_map: dict[str, str] | None = None

_SA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

_REDDIT_HEADERS = {
    "User-Agent": "TradingBot/1.0 (automated research tool; not for spam)"
}

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    published_at: datetime | None
    summary: str = ""

    def to_prompt_line(self) -> str:
        ts = self.published_at.isoformat() if self.published_at else "?"
        base = f"- [{ts}] ({self.source}) {self.headline}"
        if self.summary:
            base += f" — {self.summary[:200]}"
        return base


# ---------------------------------------------------------------------------
# Keyword classifiers
# ---------------------------------------------------------------------------

BIOTECH_KEYWORDS: tuple[str, ...] = (
    "phase 1", "phase 2", "phase 3", "phase 2/3", "phase 3b",
    "clinical trial", "topline", "top-line",
    "fda approval", "fda approved", "fda rejected",
    "complete response letter", "accelerated approval", "priority review",
    "rolling review", "advisory committee", "adcom",
    "breakthrough therapy", "fast track designation", "orphan drug",
    "pdufa", "nda", "bla", "inda", "anda", "510(k)",
    "premarket approval", "pma clearance", "rmat designation",
    "primary endpoint", "secondary endpoint", "key secondary endpoint",
    "met its endpoint", "missed its endpoint", "failed to meet",
    "statistically significant", "clinically meaningful",
    "overall survival", "progression-free survival",
    "objective response rate", "complete response rate",
    "disease-free survival", "surrogate endpoint",
    "interim analysis", "futility analysis",
    "dose escalation", "maximum tolerated dose",
    "pharmacokinetics", "compassionate use", "expanded access",
    "safety data", "efficacy data", "ards", "mortality data",
    "new drug application", "biologic license application",
)

ANALYST_UPGRADE_KEYWORDS: tuple[str, ...] = (
    "upgraded to buy", "upgraded to outperform", "upgraded to overweight",
    "upgraded to strong buy", "initiates coverage",
    "initiates with buy", "initiates with outperform", "initiates with overweight",
    "price target raised", "price target increased",
    "raises price target", "boosts price target",
    "pt raised", "pt increased", "double upgrade",
    "analyst upgrade", "reiterate buy", "reiterates buy", "reiterates overweight",
)

INSIDER_BUYING_KEYWORDS: tuple[str, ...] = (
    "insider buying", "insider purchase",
    "director purchased", "ceo purchased", "cfo purchased", "officer purchased",
    "form 4", "13d filing", "13g filing",
    "beneficial ownership", "activist investor", "activist stake",
    "activist position", "takes stake", "acquires stake",
    "acquired stake", "discloses stake",
)


def _normalize_for_match(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower())


def _any_keyword_match(items: list[NewsItem], keywords: tuple[str, ...]) -> bool:
    if not items:
        return False
    for item in items:
        haystack = _normalize_for_match(f"{item.headline} {item.summary}")
        for kw in keywords:
            if kw in haystack:
                return True
    return False


def is_biotech_catalyst(items: list[NewsItem]) -> bool:
    return _any_keyword_match(items, BIOTECH_KEYWORDS)


def is_analyst_action(items: list[NewsItem]) -> bool:
    return _any_keyword_match(items, ANALYST_UPGRADE_KEYWORDS)


def is_insider_buying(items: list[NewsItem]) -> bool:
    return _any_keyword_match(items, INSIDER_BUYING_KEYWORDS)


# ---------------------------------------------------------------------------
# Sector classification (best-effort static map + news fallback)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    # Tech / internet platforms
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "NVDA": "tech", "AMD": "tech", "INTC": "tech",
    "TSLA": "tech", "AMZN": "tech", "CRM": "tech", "ORCL": "tech",
    "ADBE": "tech", "QCOM": "tech", "TXN": "tech", "MU": "tech",
    "AMAT": "tech", "LRCX": "tech", "KLAC": "tech", "MRVL": "tech",
    "SNOW": "tech", "PLTR": "tech", "UBER": "tech", "LYFT": "tech",
    "COIN": "tech", "SQ": "tech", "PYPL": "tech", "SHOP": "tech",
    "NET": "tech", "ZS": "tech", "CRWD": "tech", "PANW": "tech",
    "OKTA": "tech", "DDOG": "tech", "MDB": "tech", "RBLX": "tech",
    "ABNB": "tech", "U": "tech", "PATH": "tech", "NFLX": "tech",
    "SNAP": "tech", "PINS": "tech", "ROKU": "tech", "TWLO": "tech",
    "NOW": "tech", "WDAY": "tech", "ZM": "tech", "DOCU": "tech",
    # Biotech / pharma
    "MRNA": "biotech", "BNTX": "biotech", "PFE": "biotech",
    "JNJ": "biotech", "MRK": "biotech", "ABBV": "biotech",
    "BMY": "biotech", "GILD": "biotech", "BIIB": "biotech",
    "REGN": "biotech", "VRTX": "biotech", "ALNY": "biotech",
    "NBIX": "biotech", "SAGE": "biotech", "INCY": "biotech",
    "SGEN": "biotech", "EXAS": "biotech", "NVAX": "biotech",
    "NKTR": "biotech", "ACAD": "biotech", "PRGO": "biotech",
    "RARE": "biotech", "IONS": "biotech", "MRTX": "biotech",
    "KYMR": "biotech", "ARCT": "biotech", "AGEN": "biotech",
    "RCUS": "biotech", "TGTX": "biotech", "FATE": "biotech",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "EOG": "energy",
    "SLB": "energy", "HAL": "energy", "BKR": "energy", "DVN": "energy",
    "PXD": "energy", "MPC": "energy", "VLO": "energy", "PSX": "energy",
    "OXY": "energy", "FANG": "energy", "EQT": "energy", "AR": "energy",
    "RIG": "energy", "NOV": "energy", "MRO": "energy", "HES": "energy",
    # Financials
    "JPM": "financials", "BAC": "financials", "WFC": "financials",
    "GS": "financials", "MS": "financials", "C": "financials",
    "BX": "financials", "BLK": "financials", "SCHW": "financials",
    "CME": "financials", "ICE": "financials", "SPGI": "financials",
    "MCO": "financials", "AXP": "financials", "V": "financials",
    "MA": "financials", "USB": "financials", "PNC": "financials",
    "TFC": "financials", "COF": "financials", "ALLY": "financials",
    "SOFI": "financials", "HOOD": "financials", "LC": "financials",
}


def determine_sector(ticker: str, news: list[NewsItem]) -> str:
    """Best-effort sector: static map first, biotech news signal as fallback.

    Returns 'unknown' for unrecognised tickers with no biotech news signal.
    Callers that receive 'unknown' should skip sector concentration checks.
    """
    mapped = _SECTOR_MAP.get(ticker.upper())
    if mapped:
        return mapped
    if is_biotech_catalyst(news):
        return "biotech"
    return "unknown"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _headline_tokens(headline: str) -> set[str]:
    return set(_TOKEN_RE.findall(headline.lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _headline_tokens(a)
    tb = _headline_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedupe_by_similarity(items: list[NewsItem], threshold: float = 0.8) -> list[NewsItem]:
    """Remove items whose headline is ≥ threshold Jaccard-similar to an earlier item."""
    result: list[NewsItem] = []
    for item in items:
        if not any(_jaccard(item.headline, r.headline) >= threshold for r in result):
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Alpaca news
# ---------------------------------------------------------------------------

def _alpaca_auth_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }


def _loads_or_empty(body: str) -> dict:
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


async def _fetch_alpaca(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    start = (datetime.now(tz=timezone.utc) - timedelta(hours=_ALPACA_LOOKBACK_HOURS)).isoformat()
    params = {
        "symbols": ticker,
        "start": start,
        "limit": str(_MAX_ITEMS),
        "include_content": "false",
        "sort": "desc",
    }
    result = await fetch_with_retry(
        session, ALPACA_NEWS_URL,
        params=params,
        headers=_alpaca_auth_headers(),
        timeout=aiohttp.ClientTimeout(total=15),
    )
    if result is None:
        log.warning("alpaca news: all retries exhausted for %s", ticker)
        return []
    status, body = result
    if status != 200:
        log.warning("alpaca news HTTP %s for %s: %s", status, ticker, body[:300])
        return []

    log.info("alpaca news raw[%s] (%dB): %s", ticker, len(body), body[:600])
    try:
        data = await asyncio.to_thread(_loads_or_empty, body)
    except Exception:
        log.exception("alpaca news JSON parse failed for %s", ticker)
        return []

    raw = data.get("news") or []
    if not raw:
        log.info("alpaca news empty for %s", ticker)

    items: list[NewsItem] = []
    for article in raw:
        published = None
        ts = article.get("created_at") or article.get("updated_at")
        if ts:
            try:
                published = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        items.append(NewsItem(
            headline=article.get("headline") or "",
            source=article.get("source") or "alpaca",
            url=article.get("url") or "",
            published_at=published,
            summary=(article.get("summary") or "")[:500],
        ))
    return items


# ---------------------------------------------------------------------------
# SEC EDGAR 8-K
# ---------------------------------------------------------------------------

async def _load_ticker_map(session: aiohttp.ClientSession) -> dict[str, str]:
    global _ticker_map
    if _ticker_map is not None:
        return _ticker_map
    async with _ticker_map_lock:
        if _ticker_map is not None:
            return _ticker_map
        result = await fetch_with_retry(
            session, SEC_TICKERS_URL,
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=20),
        )
        if result is None or result[0] != 200:
            if result is not None:
                log.warning("SEC company_tickers HTTP %s: %s", result[0], result[1][:200])
            else:
                log.warning("SEC company_tickers: all retries exhausted")
            _ticker_map = {}
            return _ticker_map
        try:
            payload = json.loads(result[1])
        except Exception:
            log.exception("SEC company_tickers JSON parse failed")
            _ticker_map = {}
            return _ticker_map

        out: dict[str, str] = {}
        for row in payload.values():
            try:
                t = str(row["ticker"]).upper()
                cik = str(int(row["cik_str"])).zfill(10)
                out[t] = cik
            except (KeyError, TypeError, ValueError):
                continue
        _ticker_map = out
        log.info("SEC ticker map loaded: %d entries", len(out))
        return _ticker_map


async def _resolve_cik(ticker: str, session: aiohttp.ClientSession) -> str | None:
    upper = ticker.upper()
    if upper in _cik_cache:
        return _cik_cache[upper]
    mapping = await _load_ticker_map(session)
    cik = mapping.get(upper)
    _cik_cache[upper] = cik
    if cik is None:
        log.warning("SEC ticker map has no CIK for %s", upper)
    else:
        log.info("SEC CIK for %s = %s", upper, cik)
    return cik


async def _fetch_edgar(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    cik = await _resolve_cik(ticker, session)
    if not cik:
        return []
    cik_padded = cik.zfill(10)
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)
    result = await fetch_with_retry(
        session, url,
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=aiohttp.ClientTimeout(total=15),
    )
    if result is None:
        log.warning("SEC submissions: all retries exhausted for %s (CIK %s)", ticker, cik_padded)
        return []
    status, body = result
    if status != 200:
        log.warning("SEC submissions HTTP %s for %s (CIK %s): %s",
                    status, ticker, cik_padded, body[:300])
        return []

    log.info("SEC submissions raw[%s CIK %s] (%dB): %s",
             ticker, cik_padded, len(body), body[:600])

    try:
        data = json.loads(body)
    except Exception:
        log.exception("SEC submissions JSON parse failed for %s", ticker)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accessions = recent.get("accessionNumber", []) or []
    primary_docs = recent.get("primaryDocument", []) or []
    items_field = recent.get("items", []) or []
    acceptances = recent.get("acceptanceDateTime", []) or []
    length = min(len(forms), len(dates), len(accessions), len(primary_docs))

    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=_EDGAR_LOOKBACK_DAYS)).date()
    log.info("SEC %s scanning %d filings; 8-K cutoff=%s", ticker, length, cutoff_date)
    out: list[NewsItem] = []
    for i in range(length):
        if forms[i] != "8-K":
            continue
        try:
            filing_date = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except ValueError:
            continue
        if filing_date < cutoff_date:
            continue

        accession_nodash = accessions[i].replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_nodash}/{primary_docs[i]}"
        )
        item_codes = items_field[i] if i < len(items_field) else ""
        headline = f"SEC 8-K filed {dates[i]}"
        if item_codes:
            headline += f" — items {item_codes}"

        published_at: datetime | None = None
        if i < len(acceptances) and acceptances[i]:
            try:
                published_at = datetime.fromisoformat(acceptances[i].replace("Z", "+00:00"))
            except ValueError:
                pass
        if published_at is None:
            published_at = datetime.combine(filing_date, datetime.min.time(), tzinfo=timezone.utc)

        out.append(NewsItem(
            headline=headline,
            source="SEC EDGAR",
            url=filing_url,
            published_at=published_at,
            summary="",
        ))
    return out


# ---------------------------------------------------------------------------
# Alpaca earnings-scoped search
# ---------------------------------------------------------------------------

async def fetch_earnings_summary(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    start = (datetime.now(tz=timezone.utc) - timedelta(hours=_ALPACA_LOOKBACK_HOURS)).isoformat()
    params = {
        "symbols": ticker,
        "start": start,
        "limit": str(_MAX_ITEMS),
        "include_content": "false",
        "sort": "desc",
    }
    result = await fetch_with_retry(
        session, ALPACA_NEWS_URL,
        params=params,
        headers=_alpaca_auth_headers(),
        timeout=aiohttp.ClientTimeout(total=15),
    )
    if result is None:
        log.warning("alpaca earnings news: all retries exhausted for %s", ticker)
        return []
    status, body = result
    if status != 200:
        log.warning("alpaca earnings news HTTP %s for %s: %s", status, ticker, body[:300])
        return []

    try:
        data = await asyncio.to_thread(_loads_or_empty, body)
    except Exception:
        return []

    raw = data.get("news") or []
    items: list[NewsItem] = []
    for article in raw:
        published = None
        ts = article.get("created_at") or article.get("updated_at")
        if ts:
            try:
                published = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        items.append(NewsItem(
            headline=article.get("headline") or "",
            source=article.get("source") or "alpaca",
            url=article.get("url") or "",
            published_at=published,
            summary=(article.get("summary") or "")[:500],
        ))
    return items


# ---------------------------------------------------------------------------
# Yahoo Finance RSS
# ---------------------------------------------------------------------------

async def _fetch_yahoo_rss(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    result = await fetch_with_retry(session, url, timeout=aiohttp.ClientTimeout(total=10))
    if result is None:
        log.debug("yahoo rss: all retries exhausted for %s", ticker)
        return []
    status, content = result
    if status != 200:
        log.debug("yahoo rss HTTP %s for %s", status, ticker)
        return []

    def _parse(text: str) -> list[NewsItem]:
        try:
            import feedparser  # type: ignore[import]
        except ImportError:
            return []
        feed = feedparser.parse(text)
        items: list[NewsItem] = []
        for entry in feed.entries[:_MAX_ITEMS]:
            headline = (entry.get("title") or "").strip()
            if not headline:
                continue
            pub: datetime | None = None
            parsed = entry.get("published_parsed")
            if parsed:
                try:
                    pub = datetime(*parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
            items.append(NewsItem(
                headline=headline,
                source="Yahoo Finance",
                url=entry.get("link") or "",
                published_at=pub,
                summary=(entry.get("summary") or "")[:500],
            ))
        return items

    return await asyncio.to_thread(_parse, content)


# ---------------------------------------------------------------------------
# Seeking Alpha RSS
# ---------------------------------------------------------------------------

async def _fetch_seeking_alpha(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
    result = await fetch_with_retry(
        session, url,
        headers=_SA_HEADERS,
        timeout=aiohttp.ClientTimeout(total=12),
    )
    if result is None:
        log.debug("seeking alpha: all retries exhausted for %s", ticker)
        return []
    status, content = result
    if status != 200:
        log.debug("seeking alpha HTTP %s for %s", status, ticker)
        return []

    def _parse(text: str) -> list[NewsItem]:
        try:
            import feedparser  # type: ignore[import]
        except ImportError:
            return []
        feed = feedparser.parse(text)
        items: list[NewsItem] = []
        for entry in feed.entries[:_MAX_ITEMS]:
            headline = (entry.get("title") or "").strip()
            if not headline:
                continue
            pub: datetime | None = None
            parsed = entry.get("published_parsed")
            if parsed:
                try:
                    pub = datetime(*parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
            items.append(NewsItem(
                headline=headline,
                source="Seeking Alpha",
                url=entry.get("link") or "",
                published_at=pub,
                summary=(entry.get("summary") or "")[:500],
            ))
        return items

    return await asyncio.to_thread(_parse, content)


# ---------------------------------------------------------------------------
# FinViz news (sync — wrapped in thread)
# ---------------------------------------------------------------------------

async def _fetch_finviz_news(ticker: str) -> list[NewsItem]:
    def _sync() -> list[NewsItem]:
        try:
            from finvizfinance.stock import finvizfinance as FinvizStock  # type: ignore[import]
            stock = FinvizStock(ticker)
            df = stock.ticker_news()
            if df is None or df.empty:
                return []
            items: list[NewsItem] = []
            for _, row in df.iterrows():
                headline = str(row.get("Title", "")).strip()
                if not headline:
                    continue
                items.append(NewsItem(
                    headline=headline,
                    source="FinViz",
                    url=str(row.get("Link", "")),
                    published_at=None,
                    summary="",
                ))
            return items[:_MAX_ITEMS]
        except Exception:
            log.debug("finviz news failed for %s", ticker, exc_info=True)
            return []

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# Reddit (r/wallstreetbets + r/stocks, posts ≥ _REDDIT_MIN_SCORE upvotes)
# ---------------------------------------------------------------------------

async def _fetch_reddit(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    subreddits = ["wallstreetbets", "stocks"]
    items: list[NewsItem] = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {
            "q": ticker,
            "sort": "new",
            "limit": "10",
            "type": "link",
            "restrict_sr": "1",
        }
        result = await fetch_with_retry(
            session, url,
            params=params,
            headers=_REDDIT_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        )
        if result is None:
            log.debug("reddit: all retries exhausted for r/%s %s", sub, ticker)
            continue
        status, body = result
        if status != 200:
            log.debug("reddit HTTP %s for r/%s %s", status, sub, ticker)
            continue
        try:
            data = json.loads(body)
        except Exception:
            log.debug("reddit JSON parse failed for r/%s %s", sub, ticker, exc_info=True)
            continue

        posts = data.get("data", {}).get("children", [])
        for post in posts:
            pd_ = post.get("data", {})
            score = int(pd_.get("score", 0))
            if score < _REDDIT_MIN_SCORE:
                continue
            title = (pd_.get("title") or "").strip()
            if not title:
                continue
            created_utc = pd_.get("created_utc")
            pub: datetime | None = None
            if created_utc:
                try:
                    pub = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                except Exception:
                    pass
            permalink = pd_.get("permalink", "")
            items.append(NewsItem(
                headline=f"[Reddit r/{sub} +{score}] {title}",
                source=f"Reddit/{sub}",
                url=f"https://www.reddit.com{permalink}",
                published_at=pub,
                summary=f"{pd_.get('num_comments', 0)} comments",
            ))
    return items


# ---------------------------------------------------------------------------
# NewsAPI.org fallback
# ---------------------------------------------------------------------------

async def _fetch_newsapi(ticker: str, session: aiohttp.ClientSession) -> list[NewsItem]:
    if not config.NEWSAPI_KEY:
        log.debug("NEWSAPI_KEY not set; skipping fallback")
        return []
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": _MAX_ITEMS,
        "apiKey": config.NEWSAPI_KEY,
    }
    result = await fetch_with_retry(
        session, "https://newsapi.org/v2/everything",
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    )
    if result is None:
        log.warning("newsapi: all retries exhausted for %s", ticker)
        return []
    status, body = result
    if status != 200:
        log.warning("newsapi returned %s for %s: %s", status, ticker, body[:200])
        return []
    try:
        data = json.loads(body)
    except Exception:
        log.exception("newsapi JSON parse failed for %s", ticker)
        return []

    items: list[NewsItem] = []
    for article in data.get("articles", [])[:_MAX_ITEMS]:
        published_raw = article.get("publishedAt")
        published = None
        if published_raw:
            try:
                published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        items.append(NewsItem(
            headline=article.get("title", "") or "",
            source=(article.get("source") or {}).get("name", "newsapi"),
            url=article.get("url", "") or "",
            published_at=published,
            summary=(article.get("description") or "")[:500],
        ))
    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_news(ticker: str, session: aiohttp.ClientSession | None = None) -> list[NewsItem]:
    """Return deduped recent news for `ticker` from all configured sources.

    Results are cached for _NEWS_CACHE_SECONDS seconds per ticker to prevent
    hammering endpoints when the same ticker re-triggers during a burst.

    Per-ticker locking (double-checked pattern) ensures concurrent pipeline
    calls for the same ticker share a single in-flight request rather than
    each independently firing all 7 HTTP sources.
    """
    # Fast-path cache check without acquiring the lock.
    cached = _news_cache.get(ticker)
    if cached:
        cached_at, cached_items = cached
        if time.monotonic() - cached_at < _NEWS_CACHE_SECONDS:
            log.debug("news cache hit for %s (%d items)", ticker, len(cached_items))
            return cached_items

    # Serialize concurrent fetches for the same ticker.
    lock = _news_fetch_locks.setdefault(ticker, asyncio.Lock())
    async with lock:
        # Re-check inside the lock — another waiter may have just populated it.
        cached = _news_cache.get(ticker)
        if cached:
            cached_at, cached_items = cached
            if time.monotonic() - cached_at < _NEWS_CACHE_SECONDS:
                log.debug("news cache hit for %s (%d items, post-lock)", ticker, len(cached_items))
                return cached_items

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        try:
            results = await asyncio.gather(
                _fetch_alpaca(ticker, session),
                _fetch_edgar(ticker, session),
                fetch_earnings_summary(ticker, session),
                _fetch_yahoo_rss(ticker, session),
                _fetch_seeking_alpha(ticker, session),
                _fetch_finviz_news(ticker),
                _fetch_reddit(ticker, session),
                return_exceptions=True,
            )

            source_names = ["alpaca", "edgar", "earnings", "yahoo_rss", "seeking_alpha", "finviz", "reddit"]
            all_items: list[NewsItem] = []
            counts: dict[str, int] = {}
            for name, result in zip(source_names, results):
                if isinstance(result, list):
                    counts[name] = len(result)
                    all_items.extend(result)
                else:
                    counts[name] = 0
                    if isinstance(result, BaseException):
                        log.warning("news source '%s' raised: %r", name, result)

            log.info(
                "news fetch %s: %s",
                ticker,
                " ".join(f"{k}={v}" for k, v in counts.items()),
            )

            merged = _dedupe_by_similarity(all_items)
            final = merged[:_MERGED_CAP]

            if not final:
                log.info("all sources empty for %s; trying newsapi fallback", ticker)
                final = (await _fetch_newsapi(ticker, session))[:_MERGED_CAP]

            now_m = time.monotonic()
            _news_cache[ticker] = (now_m, final)
            # Evict entries older than the cache TTL to prevent unbounded growth.
            stale = [k for k, (t, _) in _news_cache.items() if now_m - t >= _NEWS_CACHE_SECONDS]
            for k in stale:
                del _news_cache[k]
            return final
        finally:
            if own_session:
                await session.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

async def _smoke() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    items = await fetch_news(ticker)
    print(f"got {len(items)} items for {ticker}")
    for item in items:
        print(item.to_prompt_line())


if __name__ == "__main__":
    asyncio.run(_smoke())
