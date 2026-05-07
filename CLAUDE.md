# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Syntax-check every file (no test suite — this is the primary correctness check)
python3 -m py_compile bot.py stream.py news.py scorer.py technicals.py trader.py \
    telegram_handler.py daily_summary.py microcap_screener.py market_calendar.py \
    earnings_calendar.py config.py logger.py cooldown_store.py http_utils.py \
    position_monitor.py build_watchlist.py build_catalyst_watchlist.py \
    fundamentals.py finnhub_data.py reddit_sentiment.py premarket_scanner.py \
    short_interest.py estimate_revisions.py sector_momentum.py market_regime.py

# End-to-end pipeline test (works outside market hours; skips real stream)
python bot.py --inject-trigger AAPL 180.0 0.12 7.0

# Module smoke tests (each file's __main__ block)
python market_calendar.py          # safe anytime
python news.py AAPL                # safe anytime
python scorer.py                   # burns 1 Gemini request
python stream.py                   # only useful during market hours

# Rebuild watchlists (run monthly, market closed only)
python build_watchlist.py
python build_catalyst_watchlist.py

# Deployment (run from Mac)
# Step 1: Sync + restart + watch logs (run every time code changes)
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='logs' --exclude='.env' \
    --exclude='watchlist.txt' --exclude='watchlist_catalyst.txt' \
    ~/Projects/Trading-Bot/ aso@192.168.1.183:/home/aso/trading-bot/ && \
ssh aso@192.168.1.183 "sudo systemctl restart trading-bot && journalctl -u trading-bot -f"

# Step 2: Install new packages (only when requirements.txt changed)
ssh aso@192.168.1.183 "cd ~/trading-bot && source .venv/bin/activate && pip install -r requirements.txt"

# Step 3: Rebuild watchlists on Pi (market closed only)
ssh aso@192.168.1.183 "cd ~/trading-bot && source .venv/bin/activate && python3 build_watchlist.py && python3 build_catalyst_watchlist.py"

# Other useful Pi commands
ssh aso@192.168.1.183 "sudo systemctl status trading-bot"
ssh aso@192.168.1.183 "sudo systemctl stop trading-bot"
ssh aso@192.168.1.183 "cat ~/trading-bot/state/trade_log.jsonl"
ssh aso@192.168.1.183 "cat ~/trading-bot/state/open_positions.json"
```

All config is in `.env` (see `.env.example`). `python3 -m py_compile` is the only automated check — run it on every file you touch before considering work done.

---

## Infrastructure

- **Hardware:** Raspberry Pi 5
- **OS:** Debian, systemd service (`trading-bot.service`)
- **Python:** 3.13, venv at `~/trading-bot/.venv`
- **Pi IP:** 192.168.1.183, user: `aso`
- **Bot directory:** `/home/aso/trading-bot/`
- **Alpaca:** paper trading, IEX data feed (`ALPACA_PAPER=true`)
- **Gemini:** google-genai SDK, grounded research mode
- **Telegram bot:** @SBSPTbot

---

## Architecture

The bot is a **single asyncio event loop** running all tasks concurrently via `asyncio.gather` in `bot.py:main()`. No threading except blocking I/O wrapped in `asyncio.to_thread`.

### Signal pipeline

```
Alpaca WebSocket (stream.py TriggerDetector)
    → asyncio.Queue[TriggerEvent]  (cross-loop via call_soon_threadsafe)
    → pipeline_consumer (bot.py)
        → fetch_news → scorer.score (2 Gemini calls)
        → trader.execute_auto_buy
```

The stream runs `StockDataStream.run()` in a worker thread (`asyncio.to_thread`). `TriggerDetector` posts events to the main loop's queue via `main_loop.call_soon_threadsafe`. This cross-loop hand-off is the only safe way to write to shared state from the stream thread.

### Two-stage Gemini scoring (scorer.py)

Google Search grounding is incompatible with `response_schema` in a single call. Scoring always requires two API calls:
1. **Research call** — `tools=[google_search]`, free-form text, no schema. Uses `_MODEL_RESEARCH` (default `gemini-2.5-flash`).
2. **Verdict call** — `response_schema=CatalystVerdict`, no tools. Uses `_MODEL_VERDICT` (default `gemini-2.5-flash-lite`, ~5× cheaper output tokens).

Both count toward the shared `RateLimiter` (15 RPM). Entry scoring uses `_sem`; open-position re-evaluation uses `_position_sem` (separate so re-evals never queue behind new entries).

### Monthly Gemini budget cap (scorer.py)

`Scorer._record_call_cost(is_grounded: bool)` is invoked inside `_research_with_grounding`, `_verdict_from_research`, and `score_open_position` before each Gemini API call. Adds the per-call cost (config: `GEMINI_COST_PER_GROUNDED_CALL` / `GEMINI_COST_PER_UNGROUNDED_CALL`) to `scorer.monthly_cost_estimate`.

State machine:
- `normal` → `ungrounded_only` at ≥80% of `MONTHLY_GEMINI_BUDGET_USD` (skips Google Search grounding for the rest of the month).
- `ungrounded_only` → `halted` at ≥100% (all Gemini calls return early; bot continues on TP/SL/timeout rules only).

`scorer._alert_callback` is set from `bot.py` to push a Telegram message on each transition. `monthly_cost_estimate` and current month string are saved at the top level of `performance.json`; `bot.py` loads them on startup if the stored month matches the current month. The 1st-of-month reset happens inside `run_daily_summary`.

### Market regime detector (market_regime.py)

`MarketRegimeDetector` fits a 3-state Gaussian HMM (`hmmlearn`) on SPY daily bars (400 calendar days lookback). Features per bar: 5-day momentum, 10-day annualized volatility, SMA20/SMA50 ratio. States are mapped to `trending` / `ranging` / `volatile` by sorting on the volatility mean. `get_regime()` is cached for 60 minutes and never raises — any failure leaves the previous label intact (or returns `"unknown"`).

A background coroutine `run_regime_refresher` calls `get_regime()` every 60 minutes and syncs the result to `scorer._current_regime_label` (which is appended to the verdict prompt). `Trader._do_execute_buy` applies regime adjustments after the late-session sizing block:

| Regime | size_multiplier | hold_multiplier |
|--------|-----------------|-----------------|
| trending | 1.0 | 1.2 |
| ranging | 0.7 | 0.8 |
| volatile | 0.5 | 0.6 |
| unknown | 1.0 | 1.0 |

The trader's apply step is wrapped in try/except. `unknown` produces no change. **Regime detection is fail-open and never blocks a trade.**

**`scorer.score()` returns a 3-tuple: `(CatalystVerdict | None, Technicals | None, dict)`** — the third element is `quant_signals` with keys `short_interest`, `estimate_revisions`, `insider_score`, `sector_momentum`. Returns `{}` on cache/cooldown hits. All callers must unpack 3 values.

### Quantitative factor modules

Three new modules feed signals into the Gemini verdict prompt before entry and re-evaluation:

- **`short_interest.py`** — yfinance short float %, days to cover, squeeze_score ("high"/"medium"/"low"). Cache: `SHORT_INTEREST_CACHE_HOURS` (default 4h).
- **`estimate_revisions.py`** — Finnhub `recommendation_trends` + `company_earnings`: EPS surprise avg, trend, analyst revision, composite revision_score (−2 to +2). Cache: `ESTIMATE_REVISION_CACHE_HOURS` (default 6h). Requires `FINNHUB_API_KEY`.
- **`sector_momentum.py`** — SPDR ETF % change via Alpaca hist_client. Cache: 5 minutes. Returns `None` when `hist_client` is `None` or sector not in map.
- **`finnhub_data.get_insider_score()`** — individual insider buy/sell transactions (not MSPR). Cache: `INSIDER_CACHE_HOURS` (default 12h).

All four are fetched via `asyncio.gather()` inside `scorer.score()` and `score_open_position()`. All fail open — if unavailable, their line is silently omitted from the Gemini prompt. Sector momentum is fetched serially after the gather because it needs `fundamentals.sector` first.

### Active position monitoring (position_monitor.py)

Re-evaluates every open position with Gemini every 15 minutes. For each position:
- Fetches fresh news since entry
- Fetches updated technicals
- Gets current price from Alpaca
- Calls `scorer.score_open_position()` which returns: `hold`, `exit`, or `raise_target`
- On `exit` → calls `trader._close_position()` with reason `"gemini_exit"`
- On `raise_target` → updates `pos.take_profit_pct` and sends Telegram notification
- On `hold` → logs at DEBUG, no Telegram message
- Never evaluates positions younger than 10 minutes
- Uses `_position_sem` (limit 2) — never blocks new entry scoring

### Close / P&L accounting flow (trader.py)

`_close_position` only submits the Alpaca market-sell order. It never writes P&L or trade logs. The **fill event from TradingStream is the single source of truth**:

```
_close_position → submits order, sets pos.close_submitted=True, schedules _close_fallback(60s)
TradingStream SELL fill event → _handle_sell_fill → P&L, TradeRecord, trade_log.jsonl, Telegram, snapshot
60s fallback (_close_fallback) → same accounting with pre-close P&L estimate if fill never arrives
```

`_closing_in_progress: set[str]` prevents duplicate close orders across monitor ticks. On Alpaca error `40310000` ("insufficient qty available") a close order is already pending — the bot stays in `_closing_in_progress` and waits for the fill event rather than retrying.

### Watchlist tiers

- `watchlist.txt` — main list (~1,869 tickers), price-move threshold 8% (`PRICE_MOVE_THRESHOLD_PCT`)
- `watchlist_catalyst.txt` — FDA/biotech/squeeze names (~787 tickers), threshold 5% (`PRICE_TRIGGER_CATALYST_PCT`)
- `microcap_screener.py` — independent polling of Alpaca most-actives endpoint every 3 min

### State files (`state/` directory)

| File | Written by | Read by |
|------|-----------|---------|
| `open_positions.json` | `_save_positions_snapshot` (tmp→rename, atomic) | `sync_open_positions` on startup |
| `cooldowns.json` | `CooldownStore.set_cooldown` | `CooldownStore.is_in_cooldown` |
| `trade_log.jsonl` | `_write_trade_log` (append) | Telegram `/trades` command |

`sync_open_positions()` is called in `bot.py:main()` before tasks start. It overlays Alpaca's live positions with the snapshot to recover `entry_price`, `take_profit_pct`, Gemini scores, and `sector`.

### Key invariants

**`execute_auto_buy` returns `bool`** — `True` only when an order is submitted. Callers must only call `cooldown_store.set_cooldown` when it returns `True`. Capacity rejections (MAX_POSITIONS, sector limit) return `False` without setting cooldown so the signal can retry when capacity frees up. Pass `quant_signals=quant` (the third element from `scorer.score()`) to enable factor signal tracking; it sets `ActivePosition.had_factor_signal` and increments `stats.factor_signals_traded`.

**Factor signal tracking** — `ActivePosition.had_factor_signal` is `True` when any of: squeeze_score in ("high","medium"), insider_signal=="bullish", or revision_score≥1 at entry time. `stats.factor_signals_correct` is incremented in `_handle_sell_fill` when the reason starts with "TP" and `had_factor_signal` is True. `stats.squeeze_setups_seen`, `stats.insider_bullish_seen`, `stats.revision_strong_seen` are incremented per-trigger in `bot._process_trigger` (regardless of trade outcome).

**Gemini-driven trade parameters** — Gemini sets TP, SL, position size, hold time, and hold strategy per trade via `CatalystVerdict`. Hard floors/ceilings enforced in `trader.py`:
- `take_profit_pct`: 0.05–0.50 (Gemini sets based on catalyst strength)
- `stop_loss_pct`: 0.02–0.10 (Gemini sets based on volatility/thesis risk)
- `position_size_pct`: 0.02–0.12 (Gemini scales with confidence; biotech gets 0.02–0.05)
- `max_hold_minutes`: 30–390 (Gemini sets per `hold_strategy`: momentum/catalyst/swing)
- `should_trade`: Gemini's explicit trade/no-trade decision replaces confidence threshold

**Gap-open trigger** — at 9:30 ET (±`GAP_OPEN_WINDOW_MINUTES`), if first bar's open price is ≥ `GAP_OPEN_THRESHOLD` (15%) above the previous session close, emits a `TriggerEvent(trigger_type="gap_open")`. Fires once per ticker per session. Previous close prefetched at startup via `prefetch_prev_close()`.

**Position re-evaluation** — `position_monitor.py` wakes every 10 minutes and can now return 6 actions: `hold`, `exit`, `raise_target`, `tighten_sl` (raise SL floor to lock in gains), `adjust_tp` (raise or lower TP), `add_time` (extend hold window). The `sl_floor` field on `ActivePosition` tracks the current stop floor and can be raised above zero by `tighten_sl`.

**Circuit breaker** (`SessionStats.halt_new_entries`) is checked in `pipeline_consumer` and `microcap_screener._evaluate` before every entry. It trips when `realized_pnl ≤ -(DAILY_LOSS_LIMIT_PCT × opening_equity)`. It resets at 4:05 PM ET via `run_daily_summary`.

**Entry retrace guard** — before submitting a buy, fetches current price and skips if less than 60% of the original move remains (`ENTRY_RETRACE_THRESHOLD`).

**Hold timeout** — only fires when the position is profitable (`plpc >= 0`). Losing positions are never forced out by the clock — the SL floor or EOD close handles them.

**All external HTTP calls** must go through `http_utils.fetch_with_retry`. Never use raw `session.get` outside of the specific one-shot price check helpers.

**Blocking I/O** — every call to the Alpaca SDK (which is sync) must be wrapped in `asyncio.to_thread`. File reads/writes in hot paths must also use `asyncio.to_thread`.

### Concurrency guards

- `RateLimiter` (sliding window, 15 RPM) — shared across all Gemini calls
- `_sem` (GEMINI_CONCURRENCY, default 2) — new-entry scoring only
- `_position_sem` (POSITION_EVAL_SEMAPHORE_LIMIT, default 2) — open-position re-evaluation only
- `_closing_in_progress: set[str]` — prevents duplicate Alpaca close orders
- `_close_fallback_tasks: dict[str, Task]` — one per in-flight close; cancelled on fill event

### Config thresholds (all overridable via .env)

| Setting | Default | Notes |
|---------|---------|-------|
| PRICE_TRIGGER_PCT | 8% | Price move to trigger pipeline |
| PRICE_TRIGGER_CATALYST_PCT | 5% | Lower bar for catalyst watchlist |
| VOLUME_TRIGGER_MULTIPLIER | 5x | Volume vs ADV to trigger |
| VOLUME_TRIGGER_EARNINGS_MULTIPLIER | 2.5x | Lower bar on earnings day |
| GAP_OPEN_THRESHOLD | 15% | Min gap vs prev close to fire gap-open trigger |
| GAP_OPEN_WINDOW_MINUTES | 2 | Window after 9:30 ET to detect gaps |
| OPENING_BELL_MINUTES | 3 | Suppress intraday triggers at open (gap-open bypasses this) |
| MAX_POSITIONS | 5 | Max concurrent positions |
| DAILY_LOSS_LIMIT_PCT | 2% | Circuit breaker threshold |
| ENTRY_RETRACE_THRESHOLD | 0.6 | Skip if 40%+ of move already retraced |
| POSITION_EVAL_INTERVAL_MINUTES | 15 | Gemini re-eval frequency |
| **Gemini-driven limits (hard floors/ceilings)** | | |
| GEMINI_TP_MIN / GEMINI_TP_MAX | 5% / 50% | Take-profit range Gemini can set |
| GEMINI_SL_MIN / GEMINI_SL_MAX | 2% / 10% | Stop-loss range Gemini can set |
| GEMINI_SIZE_MIN / GEMINI_SIZE_MAX | 2% / 12% | Position size range Gemini can set |
| GEMINI_HOLD_MIN / GEMINI_HOLD_MAX | 30 / 390min | Hold time range Gemini can set |
| SHORT_INTEREST_CACHE_HOURS | 4h | yfinance short interest cache TTL |
| ESTIMATE_REVISION_CACHE_HOURS | 6h | Finnhub estimate revisions cache TTL |
| INSIDER_CACHE_HOURS | 12h | Finnhub insider transactions cache TTL |
| GEMINI_MODEL_RESEARCH | gemini-2.5-flash | Model used for grounded research call |
| GEMINI_MODEL_VERDICT | gemini-2.5-flash-lite | Model used for verdict / position-eval calls |
| MONTHLY_GEMINI_BUDGET_USD | 30.0 | Monthly Gemini hard cost cap |
| GEMINI_COST_PER_GROUNDED_CALL | 0.016 | Cost charged per grounded call |
| GEMINI_COST_PER_UNGROUNDED_CALL | 0.002 | Cost charged per ungrounded call |

`config._required()` raises `RuntimeError` at import time for missing secrets — the bot will not start without valid Alpaca, Gemini, and Telegram credentials. `ALPACA_PAPER=true` must be set for paper trading; flipping to `false` routes real orders.

---

## Known Non-Issues (do not re-fix these)

- **NASDAQ earnings API timeout on startup** — always times out on Pi at boot time; Yahoo Finance fallback loads 60+ tickers successfully. Expected behavior, not a bug.
- **BF.B in watchlist** — dot-form (`BF.B`) is valid on Alpaca. The dash-form (`BF-B`) was the bad one and has been removed from both watchlists.
- **MarketBeat skipped in build_catalyst_watchlist.py** — `curl_cffi` not installed on Pi; FinViz alone provides 787 catalyst tickers which is sufficient.
- **1 invalid ticker warning on watchlist_catalyst.txt** — was `BF-B`, now removed. If this warning reappears, grep for `[^A-Z.]` pattern in the watchlist file.

---

## MCP Tools & Plugins — MANDATORY USAGE RULES

These are not optional. Claude MUST follow these rules on every task, no exceptions:

### Before writing ANY code:
1. If the code touches alpaca-py, google-genai, aiohttp, finnhub-python, or tenacity — use Context7 FIRST to verify current API syntax. Do not rely on training data.
2. If you need to check a live URL, scrape docs, or verify an endpoint is still active — use Firecrawl.
3. After using Context7 or Firecrawl, explicitly state what you found before writing code.

### Verification checklist (run before marking any task done):
- [ ] Did I use Context7 for any SDK/library calls?
- [ ] Did I py_compile every file I touched?
- [ ] Did I verify no blocking I/O in async functions?
- [ ] Did I check for race conditions on shared state?

### Available tools:

- **Context7** — look up current docs before writing any code touching:
  alpaca-py, google-genai, aiohttp, finnhub-python, tenacity, hmmlearn.
  Prevents use of deprecated or renamed API methods.
  Invoke: "use context7 to find current docs for [library]"

- **Firecrawl** — use for scraping live URLs, verifying API endpoints are still active,
  researching news sources, or checking current library changelogs.
  Invoke: "use firecrawl to scrape [url]"

NEVER guess at API syntax. NEVER skip py_compile.
NEVER use training data alone for any Alpaca, Gemini, or Finnhub API calls.
