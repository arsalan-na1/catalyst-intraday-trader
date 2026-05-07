"""Centralized configuration.

Loads secrets from `.env` and exposes typed constants plus tunable strategy
thresholds. Import this module once from bot.py and pass values explicitly
into components — avoids scattered os.getenv() calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Project root is the directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
WATCHLIST_PATH = PROJECT_ROOT / "watchlist.txt"

load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}. See .env.example.")
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# --- Secrets ---
ALPACA_API_KEY = _required("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _required("ALPACA_SECRET_KEY")
ALPACA_PAPER = _bool("ALPACA_PAPER", True)

GEMINI_API_KEY = _required("GEMINI_API_KEY")

TELEGRAM_BOT_TOKEN = _required("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = int(_required("TELEGRAM_CHAT_ID"))

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # optional — fallback is best-effort

# --- Alternative data (optional — features degrade gracefully when absent) ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "trading-bot/1.0")

# --- Timezone ---
MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_SESSION_MINUTES = 390  # 09:30–16:00 ET

# --- Strategy thresholds (overridable via .env) ---
PRICE_MOVE_THRESHOLD_PCT = _float("PRICE_MOVE_THRESHOLD_PCT", 0.08)
PRICE_TRIGGER_CATALYST_PCT = _float("PRICE_TRIGGER_CATALYST_PCT", 0.05)
PRICE_MOVE_WINDOW_SECONDS = _int("PRICE_MOVE_WINDOW_SECONDS", 120)
VOLUME_RATIO_THRESHOLD = _float("VOLUME_RATIO_THRESHOLD", 5.0)
VOLUME_TRIGGER_EARNINGS_MULTIPLIER = _float("VOLUME_TRIGGER_EARNINGS_MULTIPLIER", 2.5)
COOLDOWN_MINUTES = _int("COOLDOWN_MINUTES", 5)
EXIT_COOLDOWN_MINUTES = _int("EXIT_COOLDOWN_MINUTES", 60)  # post-close re-entry block

# --- Premarket gap scanner ---
# PREMARKET_GAP_PCT is in percentage units: 15.0 = 15% premarket gap.
PREMARKET_GAP_PCT    = _float("PREMARKET_GAP_PCT",    15.0)
PREMARKET_POLL_SECONDS = _int("PREMARKET_POLL_SECONDS", 120)
OPENING_BELL_MINUTES = _int("OPENING_BELL_MINUTES", 3)

# --- Gap-open trigger ---
# Fires once per ticker at market open (9:30 ET) if the first bar's OPEN price
# is >= GAP_OPEN_THRESHOLD above the previous session close. DO NOT lower below
# 0.15 — smaller gaps are merger-arb noise and will generate false signals.
GAP_OPEN_THRESHOLD     = _float("GAP_OPEN_THRESHOLD",     0.15)
GAP_OPEN_WINDOW_MINUTES = _int("GAP_OPEN_WINDOW_MINUTES", 2)
VOLUME_SPIKE_CONFIRM_BARS = _int("VOLUME_SPIKE_CONFIRM_BARS", 2)
VOLUME_SPIKE_CONFIRM_VOL_THRESHOLD = _float("VOLUME_SPIKE_CONFIRM_VOL_THRESHOLD", 20.0)
VOLUME_SPIKE_CONFIRM_MIN_MOVE = _float("VOLUME_SPIKE_CONFIRM_MIN_MOVE", 0.005)

POSITION_SIZE_PCT = _float("POSITION_SIZE_PCT", 0.10)
# TAKE_PROFIT_PCT is deprecated — take-profit is now dynamic, set per-position from
# Gemini's magnitude score via _calc_take_profit() in trader.py. This variable is kept
# so existing .env files don't error, but it is no longer read by any code path.
TAKE_PROFIT_PCT = _float("TAKE_PROFIT_PCT", 0.15)
STOP_LOSS_PCT = _float("STOP_LOSS_PCT", 0.05)        # fixed hard floor; never dynamic
MAX_POSITIONS = _int("MAX_POSITIONS", 5)

# --- Dynamic take-profit magnitude tiers ---
# magnitude ≥ TP_MAGNITUDE_HIGH → TP_HIGH_PCT; ≥ TP_MAGNITUDE_MID → TP_MID_PCT;
# ≥ TP_MAGNITUDE_LOW → TP_LOW_PCT; below TP_MAGNITUDE_LOW → skip trade entirely.
TP_MAGNITUDE_HIGH = _int("TP_MAGNITUDE_HIGH", 9)
TP_HIGH_PCT       = _float("TP_HIGH_PCT",       0.25)
TP_MAGNITUDE_MID  = _int("TP_MAGNITUDE_MID",  7)
TP_MID_PCT        = _float("TP_MID_PCT",        0.15)
TP_MAGNITUDE_LOW  = _int("TP_MAGNITUDE_LOW",  5)
TP_LOW_PCT        = _float("TP_LOW_PCT",        0.08)

# --- Day-trader session limits ---
EOD_CLOSE_HOUR = _int("EOD_CLOSE_HOUR", 15)          # 3:45 PM ET close-all
EOD_CLOSE_MINUTE = _int("EOD_CLOSE_MINUTE", 45)
NO_NEW_POSITIONS_AFTER_HOUR = _int("NO_NEW_POSITIONS_AFTER_HOUR", 15)    # 3:30 PM ET cutoff
NO_NEW_POSITIONS_AFTER_MINUTE = _int("NO_NEW_POSITIONS_AFTER_MINUTE", 30)

# --- Stale trigger guard ---
# Skip a queued trigger if it's older than STALE_TRIGGER_MINUTES AND the
# current price has already reversed more than STALE_TRIGGER_REVERSAL_PCT
# from the trigger price — the move is over.
STALE_TRIGGER_MINUTES = _int("STALE_TRIGGER_MINUTES", 5)
STALE_TRIGGER_REVERSAL_PCT = _float("STALE_TRIGGER_REVERSAL_PCT", 0.03)

MIN_GEMINI_CONFIDENCE = _int("MIN_GEMINI_CONFIDENCE", 7)
MIN_GEMINI_MAGNITUDE = _int("MIN_GEMINI_MAGNITUDE", 7)

# --- Biotech catalyst fast-track ---
# Raised from 4→6 to guard against binary FDA gap risk: rejection/CRL can
# crater a stock 60–80% before the stop loss can fire. Requires a more
# confident Gemini read before entering, and uses a halved position size
# (BIOTECH_POSITION_SIZE_PCT) to bound the max dollar loss per event.
BIOTECH_GEMINI_CONFIDENCE = _int("BIOTECH_GEMINI_CONFIDENCE", 6)
BIOTECH_GEMINI_MAGNITUDE = _int("BIOTECH_GEMINI_MAGNITUDE", 6)
BIOTECH_POSITION_SIZE_PCT = _float("BIOTECH_POSITION_SIZE_PCT", 0.05)

# --- Tiered circuit breakers ---
# Daily DD > DAILY_LOSS_LIMIT_PCT → halve position sizes for rest of session.
# Daily DD > DAILY_HALT_PCT       → close all positions + halt new entries.
# Weekly DD > WEEKLY_WARN_PCT     → halve position sizes for rest of week.
# Weekly DD > WEEKLY_HALT_PCT     → close all positions + halt for week.
DAILY_LOSS_LIMIT_PCT = _float("DAILY_LOSS_LIMIT_PCT", 0.02)
DAILY_HALT_PCT       = _float("DAILY_HALT_PCT",       0.03)
WEEKLY_WARN_PCT      = _float("WEEKLY_WARN_PCT",      0.05)
WEEKLY_HALT_PCT      = _float("WEEKLY_HALT_PCT",      0.07)

# Peak equity drawdown lock: if current equity < peak * (1 - PEAK_DRAWDOWN_PCT),
# write state/trading_halted.lock and refuse to start until the file is removed.
PEAK_DRAWDOWN_PCT  = _float("PEAK_DRAWDOWN_PCT", 0.10)
PEAK_EQUITY_FILE   = STATE_DIR / "peak_equity.json"
TRADING_HALT_LOCK  = STATE_DIR / "trading_halted.lock"

# --- Bid-ask spread filter ---
# Skip the trade if the current spread exceeds MAX_SPREAD_PCT; fail open on quote errors.
MAX_SPREAD_PCT = _float("MAX_SPREAD_PCT", 0.005)

# --- Correlation filter ---
# If a candidate ticker's 60-day return correlation with any open position > REJECT:
#   skip the trade.  If > REDUCE: halve position_size_pct.
CORRELATION_REJECT = _float("CORRELATION_REJECT", 0.85)
CORRELATION_REDUCE = _float("CORRELATION_REDUCE", 0.70)

# --- Entry retrace guard ---
# In execute_auto_buy(), fetch the current price and compute how much of the
# original spike remains. If the ratio drops below this threshold (i.e. 40%+
# of the move has already retraced), skip the trade to avoid chasing fades.
ENTRY_RETRACE_THRESHOLD = _float("ENTRY_RETRACE_THRESHOLD", 0.6)

# --- Position hold limit ---
# Default fallback for snapshot recovery of positions opened before the
# Gemini-driven hold-time feature. New positions use verdict.max_hold_minutes.
MAX_HOLD_MINUTES = _int("MAX_HOLD_MINUTES", 90)

# --- Gemini-driven parameter hard limits ---
# Gemini sets TP, SL, position size, and hold time per trade.
# These are absolute floors/ceilings enforced in trader.py regardless of what
# Gemini returns — they prevent catastrophic values without removing Gemini control.
GEMINI_TP_MIN   = _float("GEMINI_TP_MIN",   0.05)
GEMINI_TP_MAX   = _float("GEMINI_TP_MAX",   0.50)
GEMINI_SL_MIN   = _float("GEMINI_SL_MIN",   0.02)
GEMINI_SL_MAX   = _float("GEMINI_SL_MAX",   0.10)
GEMINI_SIZE_MIN = _float("GEMINI_SIZE_MIN", 0.02)
GEMINI_SIZE_MAX = _float("GEMINI_SIZE_MAX", 0.12)
GEMINI_HOLD_MIN = _int("GEMINI_HOLD_MIN", 30)
GEMINI_HOLD_MAX = _int("GEMINI_HOLD_MAX", 390)

# --- Concurrency / rate limiting ---
GEMINI_CONCURRENCY = _int("GEMINI_CONCURRENCY", 2)
PIPELINE_CONCURRENCY = _int("PIPELINE_CONCURRENCY", 3)
VERDICT_CACHE_MINUTES = _int("VERDICT_CACHE_MINUTES", 30)

# --- Open-position re-evaluation ---
POSITION_EVAL_INTERVAL_MINUTES = _int("POSITION_EVAL_INTERVAL_MINUTES", 15)
POSITION_EVAL_MIN_HOLD_MINUTES = _int("POSITION_EVAL_MIN_HOLD_MINUTES", 10)
POSITION_EVAL_SEMAPHORE_LIMIT  = _int("POSITION_EVAL_SEMAPHORE_LIMIT",  2)

# --- Micro-cap screener ---
MICROCAP_ENABLED = _bool("MICROCAP_ENABLED", True)
MICROCAP_POLL_SECONDS = _int("MICROCAP_POLL_SECONDS", 180)
MICROCAP_PRICE_MIN = _float("MICROCAP_PRICE_MIN", 1.00)
MICROCAP_PRICE_MAX = _float("MICROCAP_PRICE_MAX", 20.00)
MICROCAP_VOL_RATIO_MIN = _float("MICROCAP_VOL_RATIO_MIN", 4.0)
MICROCAP_MARKET_CAP_MAX = _float("MICROCAP_MARKET_CAP_MAX", 500_000_000.0)
MICROCAP_DEDUP_HOURS = _int("MICROCAP_DEDUP_HOURS", 6)
MICROCAP_MIN_DOLLAR_VOLUME = _float("MICROCAP_MIN_DOLLAR_VOLUME", 500_000.0)

# --- Gemini cost controls ---
# Skip Google Search grounding for signals below these thresholds (uses ~10× cheaper ungrounded call).
GROUNDING_VOL_THRESHOLD   = _float("GROUNDING_VOL_THRESHOLD",   3.0)
GROUNDING_PRICE_THRESHOLD = _float("GROUNDING_PRICE_THRESHOLD", 0.03)
# Hard cap on total Gemini API calls per hour across all modules.
MAX_GEMINI_CALLS_PER_HOUR = _int("MAX_GEMINI_CALLS_PER_HOUR", 60)
# Microcap screener: minimum thresholds before any Gemini call is attempted.
MICROCAP_VOL_THRESHOLD   = _float("MICROCAP_VOL_THRESHOLD",   5.0)
MICROCAP_PRICE_THRESHOLD = _float("MICROCAP_PRICE_THRESHOLD", 0.05)

# --- External limits (known free-tier ceilings) ---
GEMINI_REQUESTS_PER_MINUTE = 15
ALPACA_HISTORICAL_RPS = 3  # conservative — 200 req/min budget
ADV_LOOKBACK_DAYS = 20
ADV_FETCH_RETRIES = _int("ADV_FETCH_RETRIES", 3)

# --- Operational cadence ---
POSITION_POLL_SECONDS = 15
PNL_UPDATE_SECONDS = 5 * 60
DAILY_SUMMARY_HOUR = 16
DAILY_SUMMARY_MINUTE = 5
HEARTBEAT_INTERVAL_MINUTES = _int("HEARTBEAT_INTERVAL_MINUTES", 30)
PERFORMANCE_FILE = PROJECT_ROOT / "performance.json"

# --- Quantitative signal cache TTLs ---
SHORT_INTEREST_CACHE_HOURS = _int("SHORT_INTEREST_CACHE_HOURS", 4)
ESTIMATE_REVISION_CACHE_HOURS = _int("ESTIMATE_REVISION_CACHE_HOURS", 6)
INSIDER_CACHE_HOURS = _int("INSIDER_CACHE_HOURS", 12)

# --- Gemini model selection ---
# Research uses gemini-2.5-flash: native Google Search grounding,
# highest-stakes call in the pipeline.
# Verdict/eval uses gemini-2.5-flash-lite: 5x cheaper output tokens,
# full response_schema support confirmed, research context already
# provided so quality risk is minimal.
GEMINI_MODEL_RESEARCH = os.getenv(
    "GEMINI_MODEL_RESEARCH", "gemini-2.5-flash"
)
GEMINI_MODEL_VERDICT = os.getenv(
    "GEMINI_MODEL_VERDICT", "gemini-2.5-flash-lite"
)

# --- Gemini monthly budget cap ---
# When monthly_cost_estimate hits 80% of MONTHLY_GEMINI_BUDGET_USD, the scorer
# switches to ungrounded-only (skips the Google Search research call). At 100%
# all Gemini calls halt for the rest of the month — bot keeps running on
# TP/SL/timeout rules. Reset on the 1st of each calendar month.
MONTHLY_GEMINI_BUDGET_USD       = _float("MONTHLY_GEMINI_BUDGET_USD", 30.0)
GEMINI_COST_PER_GROUNDED_CALL   = _float("GEMINI_COST_PER_GROUNDED_CALL", 0.016)
GEMINI_COST_PER_UNGROUNDED_CALL = _float("GEMINI_COST_PER_UNGROUNDED_CALL", 0.002)
