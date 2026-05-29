"""Shared HTTP fetch helper with exponential-backoff retry."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

log = logging.getLogger("http_utils")

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 1.5,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> tuple[int, str] | None:
    """GET url with exponential-backoff retry on transient errors.

    Returns (status_code, response_text) on a completed response — including
    non-retried codes (e.g. 400, 404) so callers can inspect and log them.
    Returns None only after all retries are exhausted.

    Retries on: HTTP 429/500/502/503/504 or aiohttp.ClientError.
    On 429: honours Retry-After header when parseable as float seconds.
    Backoff: waits backoff**attempt seconds before each retry.

    max_bytes: when set, a response whose Content-Length exceeds it (or whose
    body exceeds it when the length is undeclared/chunked) is dropped — the call
    returns (status, "") so size-unbounded external sources can't OOM the
    process. Defaults to None (unbounded, unchanged behavior).
    """
    for attempt in range(retries + 1):
        try:
            async with session.get(url, **kwargs) as resp:
                status = resp.status
                if status in _RETRYABLE_STATUSES and attempt < retries:
                    wait = backoff ** attempt
                    if status == 429:
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                wait = float(ra)
                            except ValueError:
                                pass
                    log.warning(
                        "HTTP %s from %s (attempt %d/%d); retrying in %.1fs",
                        status, url, attempt + 1, retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                if max_bytes is not None:
                    declared = resp.content_length
                    if declared is not None and declared > max_bytes:
                        log.warning(
                            "%s response is %d bytes (> max_bytes %d); dropping",
                            url, declared, max_bytes,
                        )
                        return status, ""
                    body = await resp.content.read(max_bytes + 1)
                    if len(body) > max_bytes:
                        log.warning(
                            "%s response body exceeded max_bytes %d; dropping",
                            url, max_bytes,
                        )
                        return status, ""
                    return status, body.decode("utf-8", errors="replace")
                return status, await resp.text()
        except aiohttp.ClientError as exc:
            if attempt < retries:
                wait = backoff ** attempt
                log.warning(
                    "ClientError from %s (attempt %d/%d): %r; retrying in %.1fs",
                    url, attempt + 1, retries, exc, wait,
                )
                await asyncio.sleep(wait)
            else:
                log.error(
                    "ClientError from %s after %d retries: %r",
                    url, retries, exc,
                )
    return None
