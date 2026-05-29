"""fetch_with_retry max_bytes cap — guards against unbounded external bodies."""

from __future__ import annotations

import asyncio

import pytest

import http_utils


class _FakeContent:
    def __init__(self, body: bytes):
        self._body = body

    async def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


class _FakeResp:
    def __init__(self, status: int, body: bytes, content_length=None):
        self.status = status
        self._body = body
        self.content_length = content_length
        self.content = _FakeContent(body)
        self.headers = {}

    async def text(self) -> str:
        return self._body.decode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    def get(self, url, **kwargs):
        return self._resp


def test_no_cap_returns_full_body():
    body = b"x" * 100
    sess = _FakeSession(_FakeResp(200, body))
    status, text = asyncio.run(http_utils.fetch_with_retry(sess, "http://x"))
    assert status == 200 and len(text) == 100


def test_oversized_body_dropped_when_length_undeclared():
    body = b"x" * 100  # content_length None → must cap on the actual read
    sess = _FakeSession(_FakeResp(200, body, content_length=None))
    status, text = asyncio.run(http_utils.fetch_with_retry(sess, "http://x", max_bytes=10))
    assert status == 200 and text == ""  # dropped → callers fail open


def test_oversized_body_dropped_on_declared_content_length():
    body = b"x" * 100
    sess = _FakeSession(_FakeResp(200, body, content_length=100))
    status, text = asyncio.run(http_utils.fetch_with_retry(sess, "http://x", max_bytes=10))
    assert status == 200 and text == ""


def test_within_cap_returns_body():
    body = b"hello"
    sess = _FakeSession(_FakeResp(200, body, content_length=5))
    status, text = asyncio.run(http_utils.fetch_with_retry(sess, "http://x", max_bytes=1000))
    assert status == 200 and text == "hello"
