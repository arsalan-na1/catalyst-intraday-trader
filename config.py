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
NEAR_TP_EXIT_PCT   = _float("NEAR_TP_EXIT_PCT", 0.01)   # close when within 1% of TP
NEAR_TP_MIN_HOLD_MINUTES = _int("NEAR_TP_MIN_HOLD_MINUTES", 20)  # must hold at least this long first
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
MIN_GEMINI_CONFIDENCE_EARNINGS = _int("MIN_GEMINI_CONFIDENCE_EARNINGS", 6)
MIN_GEMINI_MAGNITUDE = _int("MIN_GEMINI_MAGNITUDE", 7)

# Hard RSI ceiling: bot rejects entries when technicals report RSI above
# this value, regardless of Gemini's verdict. Tightened from 82 → 72 so we
# stop buying late-stage breakouts that have already exhausted the move.
# Gemini's system prompt already says RSI > 85 → should_trade=False, but the
# model has been observed overriding its own rule, so this is a bot-side
# floor. Fail open when technicals are unavailable.
RSI_MAX_ENTRY = _int("RSI_MAX_ENTRY", 72)

# Trend filter: skip entries when the daily trend is "downtrend" (price below
# SMA50 AND SMA50 below SMA200). Catches the obvious "bottom-fishing into a
# falling stock" mistake before Gemini's catalyst-bias can override technicals.
REJECT_DOWNTREND = _bool("REJECT_DOWNTREND", True)

# Falling-knife gate: skip when price is more than this fraction below the
# 52-week high AND the trend is not "uptrend". The combination is the AREC
# pattern (down 69% from highs, downtrend, RSI 48) — a stock that has been
# punished hard and has not yet recovered, no matter how good the catalyst
# headline looks.
FALLING_KNIFE_DRAWDOWN_PCT = _float("FALLING_KNIFE_DRAWDOWN_PCT", 0.40)

# When trend or 52-week-high distance is missing, fail CLOSED on the
# downtrend and falling-knife gates — we can't confirm the stock is not a
# falling knife, so don't trade it. The overbought (RSI) gate stays
# fail-open because RSI absence is common and doesn't signal a hidden risk.
# Set to False to restore the legacy fail-open behavior on these two gates.
TREND_GATE_FAIL_CLOSED = _bool("TREND_GATE_FAIL_CLOSED", True)

# --- Biotech catalyst fast-track ---
# Raised from 4→6 to guard against binary FDA gap risk: rejection/CRL can
# crater a stock 60–80% before the stop loss can fire. Requires a more
# confident Gemini read before entering, and uses a halved position size
# (BIOTECH_POSITION_SIZE_PCT) to bound the max dollar loss per event.
BIOTECH_GEMINI_CONFIDENCE = _int("BIOTECH_GEMINI_CONFIDENCE", 6)
BIOTECH_GEMINI_MAGNITUDE = _int("BIOTECH_GEMINI_MAGNITUDE", 6)
BIOTECH_POSITION_SIZE_PCT = _float("BIOTECH_POSITION_SIZE_PCT", 0.05)

# Hard biotech size cap enforced in trader._apply_biotech_cap. When the
# sector returned by news.determine_sector() falls in this set, the
# position size is clamped down to BIOTECH_POSITION_SIZE_PCT regardless
# of what Gemini set.
#
# The sector contract from news.determine_sector() is exactly one of:
# {"tech", "biotech", "energy", "financials", "unknown"} — there is no
# "healthcare" label. Pharma/biotech tickers in news._SECTOR_MAP all map
# to "biotech". If a future sector label (e.g. "healthcare") is added,
# include it here so the cap continues to apply.
BIOTECH_SECTORS = frozenset({"biotech"})

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
# Raised 0.10 → 0.18 so the ATR floor (1.5× daily ATR) isn't immediately
# clamped back down on high-ATR names — a 9% ATR stock needs a 13.5% stop.
GEMINI_SL_MAX   = _float("GEMINI_SL_MAX",   0.18)
GEMINI_SIZE_MIN = _float("GEMINI_SIZE_MIN", 0.02)
GEMINI_SIZE_MAX = _float("GEMINI_SIZE_MAX", 0.12)
GEMINI_HOLD_MIN = _int("GEMINI_HOLD_MIN", 30)
GEMINI_HOLD_MAX = _int("GEMINI_HOLD_MAX", 390)

# ATR-anchored stop-loss floor. After clamping Gemini's stop_loss_pct, the
# trader enforces a minimum stop distance of ATR_SL_MULTIPLIER × daily ATR%
# so a routine intraday wiggle doesn't stop the trade out. Raised 0.5 → 1.5:
# a tight half-ATR stop gets hit by ordinary noise on the same day. Capped
# at GEMINI_SL_MAX so a very-volatile name doesn't end up with a
# wider-than-ceiling stop.
ATR_SL_MULTIPLIER = _float("ATR_SL_MULTIPLIER", 1.5)

# Constant fractional-risk sizing: cap shares so the dollar loss at the stop
# never exceeds RISK_PER_TRADE_PCT of equity. Equivalent to scaling
# position_size_pct down to RISK_PER_TRADE_PCT / stop_loss_pct. Only ever
# reduces the Gemini-set size — never increases it.
RISK_PER_TRADE_PCT = _float("RISK_PER_TRADE_PCT", 0.005)

# Reward-to-risk floor. After the final stop_loss_pct (including ATR floor),
# the trader lifts take_profit_pct to at least stop_loss_pct × MIN_RR_RATIO
# (then reclamps to GEMINI_TP_MAX). Keeps the setup's RR intact when the ATR
# floor widens the stop.
MIN_RR_RATIO = _float("MIN_RR_RATIO", 2.0)

# --- Concurrency / rate limiting ---
GEMINI_CONCURRENCY = _int("GEMINI_CONCURRENCY", 2)
PIPELINE_CONCURRENCY = _int("PIPELINE_CONCURRENCY", 3)
VERDICT_CACHE_MINUTES = _int("VERDICT_CACHE_MINUTES", 30)

# --- Open-position re-evaluation ---
POSITION_EVAL_INTERVAL_MINUTES = _int("POSITION_EVAL_INTERVAL_MINUTES", 15)
POSITION_EVAL_MIN_HOLD_MINUTES = _int("POSITION_EVAL_MIN_HOLD_MINUTES", 10)
POSITION_EVAL_SEMAPHORE_LIMIT  = _int("POSITION_EVAL_SEMAPHORE_LIMIT",  2)
# Force a Gemini re-eval whenever an open position's unrealized P&L drops
# below -FORCED_REEVAL_LOSS_PCT, bypassing the news-fingerprint skip. A
# bleeding position should always get a fresh look even when no new headlines
# have appeared.
FORCED_REEVAL_LOSS_PCT = _float("FORCED_REEVAL_LOSS_PCT", 0.015)

# --- Micro-cap screener ---
MICROCAP_ENABLED = _bool("MICROCAP_ENABLED", True)
MICROCAP_POLL_SECONDS = _int("MICROCAP_POLL_SECONDS", 180)
MICROCAP_PRICE_MIN = _float("MICROCAP_PRICE_MIN", 1.00)
MICROCAP_PRICE_MAX = _float("MICROCAP_PRICE_MAX", 20.00)
MICROCAP_VOL_RATIO_MIN = _float("MICROCAP_VOL_RATIO_MIN", 4.0)
MICROCAP_MARKET_CAP_MAX = _float("MICROCAP_MARKET_CAP_MAX", 500_000_000.0)
# Fail-closed: when shares_outstanding is unavailable, skip the ticker rather
# than scoring it without the market-cap filter. The filter exists to block
# illiquid sub-$300M names; bypassing on missing data defeats the purpose on
# exactly the tickers it targets. Set to False only to restore the old
# fail-open behavior (not recommended for live trading).
MICROCAP_REQUIRE_MCAP = _bool("MICROCAP_REQUIRE_MCAP", True)
MICROCAP_DEDUP_HOURS = _int("MICROCAP_DEDUP_HOURS", 6)
MICROCAP_MIN_DOLLAR_VOLUME = _float("MICROCAP_MIN_DOLLAR_VOLUME", 500_000.0)

# --- Gemini cost controls ---
# Below these thresholds a trigger is bucketed as a "weak" signal. NOTE: since
# the grounded Google-Search research call was removed, both buckets now make
# the SAME single verdict call — these thresholds only select which counter
# (calls_grounded vs calls_ungrounded) is incremented for telemetry.
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
# Incremental Gemini cost-state sidecar (Fix 4): flushed every
# COST_PERSIST_SECONDS so a mid-day restart resumes the true monthly tally
# instead of drifting back to the last daily-summary snapshot.
GEMINI_COST_STATE_FILE = STATE_DIR / "gemini_cost.json"
COST_PERSIST_SECONDS = _int("COST_PERSIST_SECONDS", 60)

# --- Quantitative signal cache TTLs ---
SHORT_INTEREST_CACHE_HOURS = _int("SHORT_INTEREST_CACHE_HOURS", 4)
ESTIMATE_REVISION_CACHE_HOURS = _int("ESTIMATE_REVISION_CACHE_HOURS", 6)
INSIDER_CACHE_HOURS = _int("INSIDER_CACHE_HOURS", 12)

# --- Gemini model selection ---
# Single scoring call: gemini-2.5-flash-lite (cheap output tokens, full
# response_schema support). The former grounded research stage
# (gemini-2.5-flash + Google Search) was removed — its output was never used
# in the verdict prompt — so GEMINI_MODEL_RESEARCH no longer exists.
GEMINI_MODEL_VERDICT = os.getenv(
    "GEMINI_MODEL_VERDICT", "gemini-2.5-flash-lite"
)

# --- Kronos secondary confirmation ---
# When enabled, after a Gemini buy verdict has cleared all gates, fetch the
# last 60 one-minute bars and ask Kronos-mini for a short-horizon continuation
# probability. If the fraction of predicted closes above the last known close
# is below KRONOS_MIN_PROB, the buy is downgraded to a skip (no cooldown).
# Failure-modes (model unavailable, fetch error, inference None) fall through
# silently and the trade proceeds on the Gemini verdict alone.
USE_KRONOS      = _bool("USE_KRONOS", True)
KRONOS_MIN_PROB = _float("KRONOS_MIN_PROB", 0.45)

# --- Gemini monthly budget cap ---
# When monthly_cost_estimate hits 80% of MONTHLY_GEMINI_BUDGET_USD, the scorer
# switches to ungrounded-only (skips the Google Search research call). At 100%
# all Gemini calls halt for the rest of the month — bot keeps running on
# TP/SL/timeout rules. Reset on the 1st of each calendar month.
MONTHLY_GEMINI_BUDGET_USD       = _float("MONTHLY_GEMINI_BUDGET_USD", 30.0)
# Flat per-call estimates. Retained ONLY as a conservative pre-call budget
# reservation in Scorer._record_call_cost; the real monthly tally is now
# reconciled to token-based cost (see GEMINI_TOKEN_PRICING below).
GEMINI_COST_PER_GROUNDED_CALL   = _float("GEMINI_COST_PER_GROUNDED_CALL", 0.016)
GEMINI_COST_PER_UNGROUNDED_CALL = _float("GEMINI_COST_PER_UNGROUNDED_CALL", 0.002)

# --- Real per-token Gemini pricing (USD per 1,000,000 tokens) ---
# Source of truth for actual cost, computed from response.usage_metadata.
# Add a model string here if GEMINI_MODEL_* is pointed at a new model, or the
# call's token cost silently drops to $0 (a warning is logged).
GEMINI_TOKEN_PRICING = {
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}
# Google Search grounding billing. Real rate is $35 / 1,000 queries after the
# first GEMINI_GROUNDING_FREE_PER_DAY queries/day (shared quota). Modeled as
# $0 (assume under quota) by default — set GEMINI_GROUNDING_COST_PER_1K=35 to
# bill it. NOTE: the grounded research call was removed (see scorer.py), so
# grounded_queries is 0 in practice; this is kept for future re-introduction.
GEMINI_GROUNDING_COST_PER_1K  = _float("GEMINI_GROUNDING_COST_PER_1K", 0.0)
GEMINI_GROUNDING_FREE_PER_DAY = _int("GEMINI_GROUNDING_FREE_PER_DAY", 1500)

# --- Post-trade reflection loop ---
# After each trade closes, a one-paragraph reflection is generated via Gemini
# and persisted to state/reflections.jsonl. On the next entry decision for
# the same ticker, the most recent reflections are injected into the verdict
# prompt as PRIOR LESSONS so the bot learns from prior wins/losses on that name.
# Reflection generation runs in the background — the close path never awaits it.
REFLECTION_ENABLED                  = _bool("REFLECTION_ENABLED", True)
REFLECTION_MODEL                    = os.getenv("REFLECTION_MODEL", GEMINI_MODEL_VERDICT)
REFLECTION_PRUNE_DAYS               = _int("REFLECTION_PRUNE_DAYS", 180)
REFLECTION_MAX_PER_TICKER_INJECTED  = _int("REFLECTION_MAX_PER_TICKER_INJECTED", 3)
REFLECTION_MAX_GLOBAL_INJECTED      = _int("REFLECTION_MAX_GLOBAL_INJECTED", 5)
REFLECTION_INJECTED_TOKEN_BUDGET    = _int("REFLECTION_INJECTED_TOKEN_BUDGET", 600)

# --- Congressional-trade copy: shared data layer (off-by-default feature) ---
# See docs/specs/congress-copy.md. The virtual copy portfolio (Phase 2) is
# gated by CONGRESS_COPY_ENABLED; this Phase-1 block is just the data layer
# (congress_trades.py), which is inert until that portfolio is wired in.
#
# An FMP free-tier key enables full House+Senate coverage. Leave it BLANK to run
# keyless on the Senate Stock Watcher fallback (Senate-only). To turn on full
# coverage later, paste your free FMP key into .env as CONGRESS_DATA_API_KEY=...
CONGRESS_DATA_API_KEY        = os.getenv("CONGRESS_DATA_API_KEY", "")
CONGRESS_TRADES_FILE         = STATE_DIR / "congress_trades.json"
CONGRESS_FMP_BASE_URL        = os.getenv("CONGRESS_FMP_BASE_URL", "https://financialmodelingprep.com")
CONGRESS_FMP_LIMIT           = _int("CONGRESS_FMP_LIMIT", 250)  # only sent on a paid tier (see CONGRESS_FMP_PAID_TIER)
# FMP's FREE tier returns HTTP 402 when pagination params (page/limit) are sent
# to the /stable/{house,senate}-latest endpoints — the bare keyed call works.
# Default off: send only the apikey. Set true with a paid key to use page/limit.
CONGRESS_FMP_PAID_TIER       = _bool("CONGRESS_FMP_PAID_TIER", False)
# Allowlist of disclosure assetType values the copy-buy acts on (lowercased,
# comma-separated). Default "stock" only — so Corporate Bond / REIT / options /
# ETFs are NOT copied, and a bond row carrying an equity ticker (e.g. OTIS/JPM)
# can never become a stock buy. Fail-closed: an unknown/blank assetType is not
# copyable. Add "reit" etc. here to opt those in.
CONGRESS_BUY_ASSET_TYPES = frozenset(
    a.strip().lower()
    for a in os.getenv("CONGRESS_BUY_ASSET_TYPES", "stock").split(",")
    if a.strip()
)
# Hard ceiling on a single congress-data response body. The keyless Senate dump
# is a few MB today, but it's an operator-set external URL — cap it so a
# repointed/compromised mirror can't OOM the Pi. Fail-open: an oversized body is
# dropped (treated as no data).
CONGRESS_MAX_FETCH_BYTES     = _int("CONGRESS_MAX_FETCH_BYTES", 50 * 1024 * 1024)
CONGRESS_SENATE_FALLBACK_URL = os.getenv(
    "CONGRESS_SENATE_FALLBACK_URL",
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_transactions.json",
)

# Virtual copy portfolio (Phase 2). OFF by default: when CONGRESS_COPY_ENABLED is
# false the scheduler task is not started and this feature is fully inert, so the
# bot behaves byte-identically to today. The portfolio places NO Alpaca orders and
# shares no state with the scalper — it marks to market against live prices in its
# own state file.
CONGRESS_COPY_ENABLED     = _bool("CONGRESS_COPY_ENABLED", False)
CONGRESS_SIZING_MODE      = os.getenv("CONGRESS_SIZING_MODE", "equal_weight")  # equal_weight | range_tier
CONGRESS_EQUAL_WEIGHT_USD = _float("CONGRESS_EQUAL_WEIGHT_USD", 1000.0)   # virtual $ per disclosed buy (range_tier base unit)
CONGRESS_VIRTUAL_EQUITY_USD = _float("CONGRESS_VIRTUAL_EQUITY_USD", 100000.0)  # virtual account size
CONGRESS_PORTFOLIO_FILE   = STATE_DIR / "congress_portfolio.json"
CONGRESS_FRESHNESS_DAYS   = _int("CONGRESS_FRESHNESS_DAYS", 60)  # max TRANSACTION age (days) to copy: a trade executed longer ago than this is a dead signal even if just disclosed (lag can exceed a year)
CONGRESS_REFRESH_HOURS    = _int("CONGRESS_REFRESH_HOURS", 24)   # daily cadence — never the intraday hot path
