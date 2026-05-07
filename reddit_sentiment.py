"""Reddit sentiment via PRAW (read-only).

Searches r/wallstreetbets, r/stocks, and r/investing for posts mentioning the
ticker ($TICKER or plain TICKER) in the last 24 hours. Counts mentions, scores
sentiment from keyword frequency, and surfaces the top post by upvotes.

Results cached for 30 minutes — Reddit data goes stale faster than fundamentals.
Returns {} on missing credentials or any error.
"""

from __future__ import annotations

import asyncio
import logging
import time

import config

log = logging.getLogger("reddit_sentiment")

_CACHE_SECONDS  = 30 * 60   # 30 minutes
_TIMEOUT_SECONDS = 20.0
_POSTS_PER_SUB  = 25

_cache: dict[str, tuple[float, dict]] = {}

_BULLISH: frozenset[str] = frozenset({
    "buy", "calls", "call", "moon", "breakout", "yolo", "long",
    "bull", "bullish", "rocket", "squeeze", "rip", "gap",
})
_BEARISH: frozenset[str] = frozenset({
    "puts", "put", "short", "crash", "dump", "avoid",
    "bear", "bearish", "sell", "fade", "drop", "sink",
})

_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]


def _sentiment_score(text: str) -> float:
    """Normalize bull/bear keyword ratio to [-1.0, 1.0]."""
    words = text.lower().split()
    bull = sum(1 for w in words if w in _BULLISH)
    bear = sum(1 for w in words if w in _BEARISH)
    total = bull + bear
    return 0.0 if total == 0 else (bull - bear) / total


def _fetch_sync(ticker: str) -> dict:
    import praw  # lazy import — only used when credentials are configured

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        check_for_async=False,  # we run inside asyncio.to_thread, not a coroutine
    )

    query        = f"${ticker} OR {ticker}"
    ticker_upper = ticker.upper()

    mention_count    = 0
    sentiment_sum    = 0.0
    top_title:  str | None = None
    top_upvotes = 0

    for sub_name in _SUBREDDITS:
        try:
            results = list(
                reddit.subreddit(sub_name).search(
                    query, time_filter="day", sort="new", limit=_POSTS_PER_SUB
                )
            )
        except Exception:
            log.debug("praw search failed for r/%s %s", sub_name, ticker, exc_info=True)
            continue

        for post in results:
            title = post.title or ""
            body  = post.selftext or ""
            text  = f"{title} {body}"
            # Skip posts where the ticker doesn't actually appear in the content
            # (PRAW search can return loosely related results).
            if ticker_upper not in text.upper() and f"${ticker_upper}" not in text.upper():
                continue

            mention_count += 1
            sentiment_sum += _sentiment_score(text)

            upvotes = post.score or 0
            if upvotes > top_upvotes:
                top_upvotes = upvotes
                top_title   = title

    if mention_count == 0:
        return {
            "mention_count": 0,
            "sentiment_score": 0.0,
            "top_post_title": None,
            "top_post_upvotes": 0,
        }

    avg = max(-1.0, min(1.0, sentiment_sum / mention_count))
    return {
        "mention_count": mention_count,
        "sentiment_score": round(avg, 3),
        "top_post_title": top_title,
        "top_post_upvotes": top_upvotes,
    }


async def get_reddit_sentiment(ticker: str) -> dict:
    """Return Reddit mention count and sentiment for ticker (last 24h).

    Searches r/wallstreetbets, r/stocks, r/investing. Returns {} when
    PRAW credentials are absent or any error occurs.
    """
    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        return {}

    now    = time.monotonic()
    cached = _cache.get(ticker)
    if cached and now - cached[0] < _CACHE_SECONDS:
        log.debug("reddit sentiment cache hit for %s", ticker)
        return cached[1]

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync, ticker),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("reddit sentiment timed out for %s (>%.0fs)", ticker, _TIMEOUT_SECONDS)
        return {}
    except Exception:
        log.warning("reddit sentiment failed for %s", ticker, exc_info=True)
        return {}

    _cache[ticker] = (now, data)
    log.info(
        "reddit sentiment %s: mentions=%s score=%s top='%s'",
        ticker,
        data.get("mention_count"),
        data.get("sentiment_score"),
        (data.get("top_post_title") or "")[:60],
    )
    return data
