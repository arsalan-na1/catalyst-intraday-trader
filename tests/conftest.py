"""Test fixtures and import-time environment setup.

config.py calls _required("ALPACA_API_KEY") etc. at module load. When tests
run on a host where the real .env is absent (e.g. CI, a fresh checkout), the
import would crash before any test is collected. Set placeholder env vars
*before* anything else imports config.

Tests must not make live API calls — these placeholders look valid enough to
satisfy _required and config._int/_bool parsers but are not real credentials.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root on sys.path so `import bot`, `import trader`, etc. work.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("ALPACA_API_KEY", "test-alpaca-key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test-alpaca-secret")
os.environ.setdefault("ALPACA_PAPER", "true")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1234567890:test-bot-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "0")
