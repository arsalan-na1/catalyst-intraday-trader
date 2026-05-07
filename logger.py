"""Structured logging setup.

Daily-rotating file handler under ~/trading-bot/logs/ plus a stderr handler so
journald captures output when running under systemd. Call setup_logging() once
at process start from bot.py.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-20s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called twice (e.g. in tests).
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Daily rotation, 14 days retained. Rotation happens at midnight local time.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "bot.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # stderr handler so journald picks up output under systemd.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Quiet noisy libs.
    for noisy in ("httpx", "urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
