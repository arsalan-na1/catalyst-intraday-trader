# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Syntax-check every file (run on every file you touch before declaring work done)
python3 -m py_compile bot.py stream.py news.py scorer.py technicals.py trader.py \
    telegram_handler.py daily_summary.py microcap_screener.py market_calendar.py \
    earnings_calendar.py config.py logger.py cooldown_store.py http_utils.py \
    position_monitor.py build_watchlist.py build_catalyst_watchlist.py \
    fundamentals.py finnhub_data.py reddit_sentiment.py premarket_scanner.py \
    short_interest.py estimate_revisions.py sector_momentum.py market_regime.py

# Run unit tests (245 tests: pure functions in test_core.py + mocked scorer/cost/entry-gate suites)
python -m pytest tests/ -v

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

All config is in `.env` (see `.env.example`). Two automated checks: `python3 -m py_compile` (syntax) and `python -m pytest tests/` (245 unit tests: pure functions plus mocked scorer/cost-accounting/entry-gate suites). Run both on any change before considering work done.

---

## Infrastructure

- **Hardware:** Raspberry Pi 5
- **OS:** Debian, systemd service (`trading-bot.service`)
- **Python:** 3.13, venv at `~/trading-bot/.venv`
- **Pi IP:** 192.168.1.183, user: `aso`
- **Bot directory:** `/home/aso/trading-bot/`
- **Alpaca:** paper trading, IEX data feed (`ALPACA_PAPER=true`)
- **Gemini:** google-genai SDK, single structured-verdict call (no grounding)
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
        → kronos_scorer.get_continuation_prob (optional gate)
        → trader.execute_auto_buy
```

The stream runs `StockDataStream.run()` in a worker thread (`asyncio.to_thread`). `TriggerDetector` posts events to the main loop's queue via `main_loop.call_soon_threadsafe`. This cross-loop hand-off is the only safe way to write to shared state from the stream thread.

### Single-call Gemini scoring (scorer.py)

Scoring is a **single** Gemini call — `response_schema=CatalystVerdict`, no tools — using `_MODEL_VERDICT` (default `gemini-2.5-flash-lite`). It runs through `_verdict_from_research`.

> **History:** there used to be a leading "research" call (`tools=[google_search]`, `gemini-2.5-flash`) whose free-form output was passed to the verdict prompt. The verdict prompt **never actually inserted that text**, so the grounded call (the priciest in the pipeline) was pure cost with zero effect on decisions. It was removed, along with `_research_with_grounding`, `_build_research_prompt`, `_RESEARCH_SYSTEM_INSTRUCTION`, `_MODEL_RESEARCH`, and `GEMINI_MODEL_RESEARCH`. `score()` still keeps a "strong vs weak" signal split (`GROUNDING_VOL_THRESHOLD` / `GROUNDING_PRICE_THRESHOLD`), but both buckets now make the same single verdict call — the split only selects which telemetry counter (`calls_grounded` vs `calls_ungrounded`) increments.

The call counts toward the shared `RateLimiter` (15 RPM). Entry scoring uses `_sem`; open-position re-evaluation uses `_position_sem` (separate so re-evals never queue behind new entries). `score()` accepts an optional `tech=` kwarg so the caller's pre-Gemini gate technicals are reused (no double fetch) — see *Pre-Gemini reject gate* below.

### Monthly Gemini budget cap — real token accounting (scorer.py)

Cost is now measured from **real token usage**, not a flat per-call guess:
- `compute_gemini_cost(model, prompt_tokens, output_tokens, grounded_queries=)` prices a call from `config.GEMINI_TOKEN_PRICING` ($/1M tokens). Grounding is billed only when `GEMINI_GROUNDING_COST_PER_1K` is non-zero (default 0 — the grounded call is gone anyway).
- `Scorer._record_call_cost(is_grounded)` runs **before** each call as a conservative flat *reservation* (gates the budget, drives the state machine). `Scorer._account_usage(model, response, …)` runs **after** each call: it reads `response.usage_metadata`, computes the real cost, and reconciles `monthly_cost_estimate` (subtracts the reservation, adds the real cost). It also accumulates a per-day, per-model breakdown in `scorer.usage_by_model`. Wired into every Gemini call site (verdict, `score_open_position`, reflection, self-improvement, startup ping).

### Pre-Gemini reject gate (bot._process_trigger)

The deterministic technical reject gates (RSI ceiling, downtrend, falling-knife) run **before** the Gemini scoring call so a doomed candidate never incurs the cost. Flow: when `scorer.will_score_call_model(ctx)` is True (i.e. score() would make a real billable call — not a verdict-cache / 30-min-cooldown / halted hit), `bot._process_trigger` calls `scorer.fetch_technicals(ctx)`, runs `bot._technically_rejected(tech)`, and on reject calls `scorer.note_scored(ticker)` (starts the 30-min cooldown so re-triggers are suppressed) and returns — no Gemini call. The surviving technicals are passed to `score(tech=…)` to avoid a second fetch.

The **post-score** inline gates in `_process_trigger` are deliberately kept as the authority, so the set of candidates ultimately accepted/rejected is unchanged — only doomed *fresh* candidates skip the scoring call. (The pre-gate intentionally does **not** fire for cache/cooldown hits, so the existing verdict-cache behavior is preserved.)

State machine: `normal` → `ungrounded_only` at ≥80% of `MONTHLY_GEMINI_BUDGET_USD` → `halted` at ≥100% (all Gemini calls return early; bot runs on TP/SL/timeout rules only). `scorer._alert_callback` pushes a Telegram message on each transition.

**Persistence (drift-proof):** `monthly_cost_estimate` + month + today's `usage_by_model` are flushed to **`state/gemini_cost.json`** every `COST_PERSIST_SECONDS` (default 60s) by `run_cost_persister` (atomic tmp→rename), and also written into `performance.json` at the daily summary (`_append_performance(gemini_usage=…)`). On startup `restore_cost_state` reloads `gemini_cost.json` if it is for the current month (re-entering `ungrounded_only`/`halted`), so a mid-day restart resumes the true tally instead of drifting back to the last daily snapshot. The 1st-of-month reset happens inside `run_daily_summary`.

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

### Kronos secondary confirmation (kronos_scorer.py)

After Gemini approves a buy but before `execute_auto_buy` submits the order, `bot._process_trigger` calls `kronos_scorer.get_continuation_prob()` if `USE_KRONOS=true` (default). This loads NeoQuasar's Kronos-mini model lazily from `/home/aso/Kronos`, runs an inference over the last 60 one-minute bars (fetched via the trader's existing Alpaca historical data client), and returns the fraction of the next 10 predicted bars that close above the current price.

If the probability is below `KRONOS_MIN_PROB` (default 0.45), the buy is downgraded to a skip — no order is submitted, but no cooldown is set so the next trigger for the same ticker is still eligible. If the probability is at or above the threshold, the buy proceeds normally.

Fail-open: every error path (model not loaded, frequency inference failure, inference exception, fewer than 20 input bars, missing OHLCV columns) returns `None` and the gate is bypassed. Set `USE_KRONOS=false` in `.env` to disable the model entirely (skips the lazy import too — useful when `/home/aso/Kronos` isn't checked out).

### Active position monitoring (position_monitor.py)

Re-evaluates every open position with Gemini every `POSITION_EVAL_INTERVAL_MINUTES` (default 15min). For each position:
- Fetches fresh news since entry
- Fetches updated technicals
- Gets current price from Alpaca
- Calls `scorer.score_open_position()` which returns one of six actions: `hold`, `exit`, `raise_target`, `tighten_sl`, `adjust_tp`, `add_time`
- On `exit` → calls `trader._close_position()` with reason `"gemini_exit — <reason>"`
- On `raise_target` → updates `pos.take_profit_pct` (must exceed current TP) and sends Telegram notification
- On `tighten_sl` → raises `pos.sl_floor` to lock in gains (must exceed current floor)
- On `adjust_tp` → revises `pos.take_profit_pct` (clamped to GEMINI_TP_MIN/MAX, can be raised or lowered)
- On `add_time` → extends `pos.max_hold_minutes` (capped at GEMINI_HOLD_MAX)
- On `hold` → logs at DEBUG, no Telegram message
- Never evaluates positions younger than `POSITION_EVAL_MIN_HOLD_MINUTES` (default 10min)
- Bypasses the news-fingerprint skip when current P&L is below `-FORCED_REEVAL_LOSS_PCT` (default −1.5%)
- Uses `_position_sem` (limit 2) — never blocks new entry scoring

### Close / P&L accounting flow (trader.py)

`_close_position` only submits the Alpaca market-sell order. It never writes P&L or trade logs. The **fill event from TradingStream is the single source of truth**:

```
_close_position → submits order, sets pos.close_submitted=True, schedules _close_fallback(60s)
TradingStream SELL fill event → _handle_sell_fill → P&L, TradeRecord, trade_log.jsonl, Telegram, snapshot
60s fallback (_close_fallback) → same accounting with pre-close P&L estimate if fill never arrives
```

`_closing_in_progress: set[str]` prevents duplicate close orders across monitor ticks. On Alpaca error `40310000` ("insufficient qty available") a close order is already pending — the bot stays in `_closing_in_progress` and waits for the fill event rather than retrying.

### Nightly self-improvement (self_improvement.py)

Runs once per trading day at 4:30 PM ET via `run_nightly_self_improvement` in `bot.py`, after EOD close completes (15:45 ET) so all sell fills have settled. Reads `state/trade_log.jsonl`, computes coarse win-rate stats grouped by exit reason, Gemini confidence, Gemini magnitude, and `technical_signal`, then asks Gemini (one verdict call, routed through `Scorer._record_call_cost` so it counts toward the monthly budget) to extract patterns and produce a structured insights JSON.

Output goes to `state/insights.json`. The top of `scorer.score()` loads this file on every entry call (if present and `< INSIGHTS_FRESH_DAYS = 7` days old) and appends the lessons / favor_patterns / avoid_patterns to the verdict prompt, so Gemini learns from what has actually worked and failed in this account.

Self-skips on weekends and when fewer than `MIN_TRADES_FOR_ANALYSIS = 5` trades exist in the log. Any failure (file missing, JSON parse error, Gemini call failure) logs a warning and returns silently — never crashes the bot. Routes through the same monthly budget gate, so when the bot is `halted` the nightly call is skipped just like scoring traffic.

### Watchlist tiers

- `watchlist.txt` — main list (~1,869 tickers), price-move threshold 8% (`PRICE_MOVE_THRESHOLD_PCT`)
- `watchlist_catalyst.txt` — FDA/biotech/squeeze names (~787 tickers), threshold 5% (`PRICE_TRIGGER_CATALYST_PCT`)
- `microcap_screener.py` — independent polling of Alpaca most-actives endpoint every 3 min

**Risk-filter posture: fail closed.** When inputs to a risk filter are missing, the filter must block — not pass — the candidate. The microcap market-cap filter is the reference case: `MICROCAP_REQUIRE_MCAP=true` (default) skips any ticker whose `shares_outstanding` is unavailable, because the filter exists to block illiquid sub-$300M names and bypassing on missing data defeats the purpose on exactly the tickers it targets. Same principle applies to any future risk filter — never let absent data downgrade a safety check.

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
- `stop_loss_pct`: 0.02–0.18 (Gemini sets based on volatility/thesis risk; ceiling raised from 0.10 → 0.18 so the ATR floor isn't immediately reclamped on high-ATR names)
- `position_size_pct`: 0.02–0.12 (Gemini scales with confidence). **Biotech is hard-capped at `BIOTECH_POSITION_SIZE_PCT` (default 5%)** by `trader._apply_biotech_cap`. The cap binds when EITHER `sector ∈ BIOTECH_SECTORS` OR `is_biotech` (the `is_biotech_catalyst(news)` flag threaded from `bot._process_trigger`). The OR closes the case where `news.determine_sector()` short-circuits to the static `_SECTOR_MAP` before checking the FDA-news fallback — a tech-mapped ticker firing on an FDA event would otherwise escape the cap. Sector-concentration logic is **deliberately not** tied to `is_biotech`: that asks "what bucket is this company?" while the cap asks "is this catalyst gap-risk?" Sector "unknown" with no FDA news flag is **not** capped (fail-safe).
- `max_hold_minutes`: 30–390 (Gemini sets per `hold_strategy`: momentum/catalyst/swing)
- `should_trade`: Gemini's explicit trade/no-trade decision replaces confidence threshold

**Gap-open trigger** — at 9:30 ET (±`GAP_OPEN_WINDOW_MINUTES`), if first bar's open price is ≥ `GAP_OPEN_THRESHOLD` (15%) above the previous session close, emits a `TriggerEvent(trigger_type="gap_open")`. Fires once per ticker per session. Previous close prefetched at startup via `prefetch_prev_close()`.

**Position re-evaluation** — `position_monitor.py` wakes every `POSITION_EVAL_INTERVAL_MINUTES` (default 15min) and can return 6 actions: `hold`, `exit`, `raise_target`, `tighten_sl` (raise SL floor to lock in gains), `adjust_tp` (raise or lower TP), `add_time` (extend hold window). The `sl_floor` field on `ActivePosition` tracks the current stop floor and can be raised above zero by `tighten_sl`.

**Circuit breaker** (`SessionStats.halt_new_entries`) is checked in `pipeline_consumer` and `microcap_screener._evaluate` before every entry. It trips when `realized_pnl ≤ -(DAILY_LOSS_LIMIT_PCT × opening_equity)`. It resets at 4:05 PM ET via `run_daily_summary`.

**Entry retrace guard** — before submitting a buy, fetches current price and skips if less than 60% of the original move remains (`ENTRY_RETRACE_THRESHOLD`).

**Hold timeout** — only fires when the position is profitable (`plpc >= 0.001`, ~10 bps; the strict-positive threshold avoids float-precision races where Alpaca returns 0.0 at the tick instant and the SELL fills slightly negative). When it fires, the deadline is pushed 30 minutes out and a `_timeout_reeval` task routes through the same Gemini-driven path as the regular monitor (hold / exit / raise_target / tighten_sl / adjust_tp / add_time). If the Gemini call fails, fallback closes profitable positions only with reason `timeout_fallback — review unavailable`. Losing positions are never forced out by the clock — the SL floor or EOD close handles them.

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
| RSI_MAX_ENTRY | 72 | Bot-side hard ceiling on RSI for entries (rejects regardless of Gemini verdict). Lowered from 82 — renamed env var, old `MAX_ENTRY_RSI` no longer read. |
| REJECT_DOWNTREND | true | Skip entries when daily trend == "downtrend" (price < SMA50 AND SMA50 < SMA200) |
| FALLING_KNIFE_DRAWDOWN_PCT | 0.40 | Skip when >40% below 52-week high AND trend is not "uptrend" |
| TREND_GATE_FAIL_CLOSED | true | When trend or 52w-high distance is missing, the downtrend + falling-knife gates SKIP (can't confirm safety). RSI gate stays fail-open. Logs `[TREND DATA MISSING]`. Set false to restore legacy fail-open. |
| MAX_POSITIONS | 5 | Max concurrent positions |
| DAILY_LOSS_LIMIT_PCT | 2% | Circuit breaker threshold |
| ENTRY_RETRACE_THRESHOLD | 0.6 | Skip if 40%+ of move already retraced |
| POSITION_EVAL_INTERVAL_MINUTES | 15 | Gemini re-eval frequency |
| FORCED_REEVAL_LOSS_PCT | 1.5% | Bypass news-fingerprint skip and force a Gemini re-look when P&L drops below this |
| **Gemini-driven limits (hard floors/ceilings)** | | |
| GEMINI_TP_MIN / GEMINI_TP_MAX | 5% / 50% | Take-profit range Gemini can set |
| GEMINI_SL_MIN / GEMINI_SL_MAX | 2% / 18% | Stop-loss range Gemini can set. Max raised 10% → 18% so the ATR floor isn't immediately reclamped on high-ATR names. |
| ATR_SL_MULTIPLIER | 1.5 | SL must cover at least this × daily ATR%; capped at GEMINI_SL_MAX. Raised 0.5 → 1.5 — a half-ATR stop gets hit by ordinary noise. |
| RISK_PER_TRADE_PCT | 0.005 | Constant fractional-risk sizing: caps `position_size_pct` to `RISK_PER_TRADE_PCT / stop_loss_pct` so dollar risk at stop ≤ 0.5% of equity. Only ever reduces the Gemini-set size. |
| MIN_RR_RATIO | 2.0 | Final TP lifted to `max(TP, SL × MIN_RR_RATIO)`. If `SL × MIN_RR_RATIO > GEMINI_TP_MAX`, the trade is SKIPPED (`[RR UNMET]`) rather than shipped at degraded RR — keeps the reward:risk floor real, doesn't silently clamp it away. |
| GEMINI_SIZE_MIN / GEMINI_SIZE_MAX | 2% / 12% | Position size range Gemini can set |
| BIOTECH_POSITION_SIZE_PCT | 0.05 | Hard cap applied by `trader._apply_biotech_cap`. Binds when `sector ∈ BIOTECH_SECTORS` OR `is_biotech` (the `is_biotech_catalyst(news)` flag). FDA names gap through stops — was a soft Gemini-prompt hint, now enforced in code. |
| BIOTECH_SECTORS | `{"biotech"}` | Sector strings (from `news.determine_sector`) that trigger the biotech size cap. The function's value set is `{"tech", "biotech", "energy", "financials", "unknown"}`; only `"biotech"` qualifies by default. Add `"healthcare"` etc. if future labels are introduced. The OR-`is_biotech` path covers the case where a non-biotech-mapped ticker fires on an FDA catalyst. |
| GEMINI_HOLD_MIN / GEMINI_HOLD_MAX | 30 / 390min | Hold time range Gemini can set |
| SHORT_INTEREST_CACHE_HOURS | 4h | yfinance short interest cache TTL |
| ESTIMATE_REVISION_CACHE_HOURS | 6h | Finnhub estimate revisions cache TTL |
| INSIDER_CACHE_HOURS | 12h | Finnhub insider transactions cache TTL |
| GEMINI_MODEL_VERDICT | gemini-2.5-flash-lite | Model used for the single verdict / position-eval call (the only scoring model; `GEMINI_MODEL_RESEARCH` was removed) |
| MONTHLY_GEMINI_BUDGET_USD | 30.0 | Monthly Gemini hard cost cap (now driven by real token cost) |
| GEMINI_TOKEN_PRICING | flash 0.30/2.50, flash-lite 0.10/0.40 | $/1M input/output per model — source of truth for real cost |
| GEMINI_GROUNDING_COST_PER_1K | 0.0 | $/1k Search-grounding queries; 0 = under free quota. Real rate $35; grounding is currently unused |
| GEMINI_COST_PER_GROUNDED_CALL | 0.016 | Flat pre-call **reservation** only (reconciled to real token cost post-call) |
| GEMINI_COST_PER_UNGROUNDED_CALL | 0.002 | Flat pre-call **reservation** only (reconciled to real token cost post-call) |
| COST_PERSIST_SECONDS | 60 | Interval for flushing the running cost tally to `state/gemini_cost.json` |
| **Kronos secondary confirmation** | | |
| USE_KRONOS | true | Toggle Kronos-mini gate after Gemini buy approval; false skips the lazy import |
| KRONOS_MIN_PROB | 0.45 | Minimum continuation probability to allow the buy through |
| **Microcap screener** | | |
| MICROCAP_REQUIRE_MCAP | true | Fail-closed: skip ticker when `shares_outstanding` is unavailable. Set false only to restore the old fail-open behavior. |

`config._required()` raises `RuntimeError` at import time for missing secrets — the bot will not start without valid Alpaca, Gemini, and Telegram credentials. `ALPACA_PAPER=true` must be set for paper trading; flipping to `false` routes real orders.

---

## Known Non-Issues (do not re-fix these)

- **NASDAQ earnings API timeout on startup** — always times out on Pi at boot time; Yahoo Finance fallback loads 60+ tickers successfully. Expected behavior, not a bug.
- **BF.B in watchlist** — dot-form (`BF.B`) is valid on Alpaca. The dash-form (`BF-B`) was the bad one and has been removed from both watchlists.
- **MarketBeat skipped in build_catalyst_watchlist.py** — `curl_cffi` not installed on Pi; FinViz alone provides 787 catalyst tickers which is sufficient.
- **1 invalid ticker warning on watchlist_catalyst.txt** — was `BF-B`, now removed. If this warning reappears, grep for `[^A-Z.]` pattern in the watchlist file.

---

## MCP Tools & Plugins — MANDATORY USAGE RULES

These are not optional. Claude MUST follow these rules on every task, no exceptions.

### Available MCP servers

- **Context7** (HTTP transport at `mcp.context7.com` + a plugin-bundled variant) — current docs for any library. Use FIRST whenever code touches: alpaca-py, google-genai, aiohttp, finnhub-python, hmmlearn, yfinance, praw, python-telegram-bot, or any other SDK in `requirements.txt`. Invoke: "use context7 to find current docs for [library]"
- **Web search** — fallback when Context7 doesn't have what's needed (current market structure, regulatory changes, breaking news, SDK migration notices not yet indexed in published docs).

There is no Firecrawl in this setup — earlier versions of this file referenced it; ignore those references. There is no GitHub MCP either; commits, tags, and pushes are done manually after each session. XcodeBuildMCP is registered globally but does not apply to this repo (Python only, no Xcode).

### Available plugins

Two plugins load via `--plugin-dir` at session launch:

- **agent-skills** at `~/projects/agent-skills` — 21 named skills + 3 specialized agents (code-reviewer, security-auditor, test-engineer).
- **ruflo** at `~/projects/ruflo` — ~30 domain-specific sub-plugins.

Skill names matter. Reference them by exact name in prompts when applicable.

### Skills most relevant to this repo

| When working on… | Use skill | Use agent |
|---|---|---|
| Anything touching secrets, HMAC, .env permissions, external API auth | `security-and-hardening` | `security-auditor` |
| Live-log triage, bot crashes, fill-event accounting gaps, ghost-position recovery | `debugging-and-error-recovery` | — |
| Stream/scorer/microcap hot paths, cache eviction, async concurrency tuning | `performance-optimization` | — |
| Adding tests to `tests/test_core.py` (pure functions only — no mocks, no live calls) | `test-driven-development` | `test-engineer` |
| Retiring an old API surface, env var, or behavior (e.g., the heartbeat URL rename) | `deprecation-and-migration` | — |
| Multi-phase changes — ship one step at a time, verify, move on | `incremental-implementation` | — |
| Final review of every changed file before declaring work done | `code-review-and-quality` | `code-reviewer` |
| Breaking down a vague feature request before any code is written | `planning-and-task-breakdown` | — |

For ruflo plugins, only invoke one when the task name explicitly maps to its sub-plugin name (e.g., ruflo-cost-tracker for Gemini budget work, ruflo-observability for journal/log changes, ruflo-market-data for Alpaca/yfinance/Finnhub fetch refactors). Do not fan out across many at once.

### Before writing ANY code

1. If the code touches a third-party SDK, use Context7 FIRST to verify current API syntax. Do not rely on training data.
2. After consulting Context7, explicitly state what you found before writing code.
3. Identify which agent-skills skill applies (table above) and invoke it.

### Verification checklist (run before marking any task done)

- [ ] Did I use Context7 for any SDK/library calls?
- [ ] Did I run `python3 -m py_compile` on every file I touched?
- [ ] Did I run `python -m pytest tests/ -v` if my change could affect any function covered by the suite?
- [ ] Did I verify no blocking I/O in async functions (Alpaca SDK calls, file I/O on hot paths)?
- [ ] Did I check for race conditions on shared state (`_closing_in_progress`, `_close_fallback_tasks`, cooldown dicts)?
- [ ] Did I run `code-review-and-quality` on every changed file?

NEVER guess at API syntax. NEVER skip py_compile. NEVER use training data alone for any Alpaca, Gemini, Finnhub, or yfinance API calls.
