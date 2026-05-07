# HANDOFF — Trading Bot

Last updated: 2026-05-06

This file is the running operational record for the bot. CLAUDE.md is for
agents working on the code; HANDOFF.md is for the human (or next agent)
who picks up where the last session left off.

---

## What Was Built

### 2026-05-06 session (this session)

- **Premarket squeeze verification** — confirmed and dated:
  `[SQUEEZE SETUP]` log fires when `squeeze_score=="high"` AND gap ≥
  `PREMARKET_GAP_PCT`; ticker added to both `intraday_gappers` and
  `dynamic_sub_queue`; ADV prefetched via `prefetch_adv`. Verified
  comment block prepended to `premarket_scanner.py`.
- **Model cost split** — `GEMINI_MODEL_RESEARCH` (default
  `gemini-2.5-flash`) and `GEMINI_MODEL_VERDICT` (default
  `gemini-2.5-flash-lite`) added to `config.py`. Scorer uses
  `_MODEL_RESEARCH` only for `_research_with_grounding` and
  `_MODEL_VERDICT` for every verdict / position-eval call. ~5× cheaper
  output tokens on the verdict side.
- **Gemini $30/month budget cap** — `MONTHLY_GEMINI_BUDGET_USD` (default
  $30) tracked across calls via `Scorer._record_call_cost`. At 80%
  budget switches to `ungrounded_only`; at 100% switches to `halted`
  (all Gemini calls skipped, bot runs on TP/SL/timeout only). State
  persists across restarts in `performance.json` top-level
  `monthly_cost_estimate` and `monthly_cost_month` keys. Resets on the
  1st of each calendar month inside `run_daily_summary`. Telegram
  alerts fire at both transitions; heartbeat logs current cost / mode.
- **HMM market regime detector** — new `market_regime.py` fits a
  3-state Gaussian HMM to SPY daily bars (5d momentum, 10d annualized
  vol, SMA20/SMA50 ratio) over 400 calendar days. Refreshed every
  60 minutes by the `run_regime_refresher` background task.
  `Trader._do_execute_buy` applies `size_multiplier` and
  `hold_multiplier` (trending=1.0/1.2, ranging=0.7/0.8,
  volatile=0.5/0.6, unknown=1.0/1.0). Scorer sees the label in the
  verdict prompt as `Market regime (HMM/SPY): {label}`. Always
  fail-open — "unknown" never blocks a trade.
- **Security & robustness fixes (Part 1)**:
  - Atomic writes (tmp→rename) for `cooldowns.json`, `performance.json`,
    `peak_equity.json`.
  - `peak_equity.json` read/write moved to `asyncio.to_thread` (was sync
    on main loop at startup).
  - Telegram bot token redacted from `getMe` error messages — token
    is in URL path so naive `log.error("%s", e)` could leak it via
    aiohttp `ClientError` traceback.
  - CLI `--inject-trigger` validates ticker shape, numeric parsing, and
    sane bounds for price / move / volume.
  - `premarket_scanner` rejects non-numeric `change_percent` from the
    Alpaca movers payload instead of raising `ValueError`.
  - `trader.sync_open_positions` skips Alpaca rows with non-numeric or
    zero/negative qty/avg_entry_price.
  - `trader._monitor_tick` wraps `get_all_positions` in
    `asyncio.wait_for(timeout=15)` so a frozen Alpaca call can't stall
    the monitor.
  - `trader._do_execute_buy` rejects NaN/inf in Gemini-returned
    take_profit/stop_loss/position_size before clamping.
  - `short_interest._fetch_sync` defensively coerces yfinance numerics
    and rejects non-positive values.

### Earlier sessions (carried forward)

- **ESPR session block** — acquisition/merger keywords on the close
  reason mark a ticker as session-blocked (`session_blocked_tickers`),
  cleared at EOD or daily reset.
- **Duplicate catalyst guard** — second entry in the same sector
  within 5 minutes is rejected (`_sector_last_entry`).
- **Gemini-driven trade params** — TP / SL / size / hold are set per
  trade by Gemini and clamped to `GEMINI_*_MIN/MAX`.
- **Position re-eval action set** — `hold`, `exit`, `raise_target`,
  `tighten_sl`, `adjust_tp`, `add_time`.
- **Two-stage Gemini scoring** — research call (grounded) + verdict
  call (schema). Cost tracked separately in `calls_grounded` /
  `calls_ungrounded` / `calls_skipped`.
- **Tiered circuit breakers** — daily warn / daily halt / weekly warn
  / weekly halt + peak-drawdown lock file.
- **Persistent cooldowns**, **stale-trigger guard**,
  **entry-retrace guard**, **bid-ask spread filter**,
  **correlation filter**, **profitable-only hold timeout**.

---

## File Map

| File | Purpose |
|------|---------|
| `bot.py` | Entrypoint; wires asyncio tasks + signal handling. |
| `stream.py` | Alpaca WS bar stream + TriggerDetector + ADV/prev-close prefetch. |
| `scorer.py` | Two-stage Gemini scoring + budget tracker + position re-eval. |
| `trader.py` | Order execution, monitoring, EOD close, fill stream, regime adjustments. |
| `position_monitor.py` | Periodic Gemini re-eval of open positions. |
| `news.py` | Multi-source news fetcher + sector classifier + biotech/analyst/insider keyword check. |
| `technicals.py` | RSI / SMA / ATR / 52w range from Alpaca daily bars. |
| `fundamentals.py` | yfinance market-cap / float / sector / 52w / target. |
| `finnhub_data.py` | Insider sentiment, congressional trades, earnings. |
| `short_interest.py` | yfinance short float, days to cover, squeeze score. |
| `estimate_revisions.py` | Finnhub analyst revisions + EPS surprise. |
| `sector_momentum.py` | SPDR sector ETF % change via Alpaca. |
| `reddit_sentiment.py` | PRAW r/wsb + r/stocks + r/investing. |
| `daily_summary.py` | EOD recap + performance.json + monthly cost reset. |
| `cooldown_store.py` | Persistent per-ticker cooldown JSON. |
| `microcap_screener.py` | Micro-cap most-actives screener. |
| `premarket_scanner.py` | 9:00-9:29 ET premarket gap scanner. |
| `earnings_calendar.py` | NASDAQ → Yahoo fallback for today's earnings tickers. |
| `market_calendar.py` | Cached `/clock` wrapper around Alpaca. |
| `market_regime.py` | **NEW** — HMM-based SPY regime detector. |
| `telegram_handler.py` | Outbound messages + `/status`, `/positions`, `/close`, `/watchlist`, `/trades`. |
| `http_utils.py` | `fetch_with_retry`. |
| `logger.py` | TimedRotatingFileHandler + stderr. |
| `config.py` | Centralized env-driven config. |
| `build_watchlist.py` | One-shot watchlist builder (S&P 500/400/600, IWM, FinViz). |
| `build_catalyst_watchlist.py` | One-shot catalyst (high-short) watchlist builder. |

---

## Critical Technical Invariants

- **`scorer.score()` returns `(verdict, technicals, quant_signals)` 3-tuple** —
  unchanged. `quant_signals` is `{}` on cache/cooldown hits.
- **`execute_auto_buy` returns `bool`** — `True` only when an order is
  submitted. Callers set persistent cooldown only on `True`.
- **Gemini model constants** — `_MODEL_RESEARCH` (research call only)
  vs `_MODEL_VERDICT` (every other Gemini call). Both read from
  `config.GEMINI_MODEL_RESEARCH` / `config.GEMINI_MODEL_VERDICT`.
- **Monthly budget gate** — `Scorer._record_call_cost(is_grounded)` is
  called inside `_research_with_grounding`, `_verdict_from_research`,
  and `score_open_position` before any Gemini API request. Returns
  False → call is skipped; the `_budget_mode` may transition to
  `ungrounded_only` (≥80%) or `halted` (≥100%). `_alert_callback` is
  set from `bot.py` to push a Telegram message on each transition.
- **Regime detector fail-open** — `MarketRegimeDetector.get_regime()`
  catches all exceptions internally; on any failure the cached label
  is returned (or `"unknown"` when the cache is empty). The trader's
  apply step is also wrapped in try/except. **Regime detection never
  blocks a trade.**
- **Atomic state writes** — `open_positions.json`, `cooldowns.json`,
  `performance.json`, `peak_equity.json` all use the tmp→rename
  pattern.
- **Single-source-of-truth for fills** — TradingStream `fill` event
  triggers `_handle_sell_fill` which writes P&L, trade log, and
  Telegram. The 60-second fallback handles missing fill events.
- **Stream trigger thresholds in `stream.py`** are the operational
  contract — never edit `PRICE_MOVE_THRESHOLD_PCT`,
  `PRICE_TRIGGER_CATALYST_PCT`, `VOLUME_RATIO_THRESHOLD`,
  `GAP_OPEN_THRESHOLD`, or `ALPACA_PAPER` without explicit
  authorisation.

---

## Deployment

### Sync + restart

```bash
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='logs' \
    --exclude='.env' --exclude='watchlist.txt' --exclude='watchlist_catalyst.txt' \
    ~/Projects/Trading-Bot/ aso@192.168.1.183:/home/aso/trading-bot/ && \
ssh aso@192.168.1.183 "sudo systemctl restart trading-bot && journalctl -u trading-bot -f"
```

### After requirements.txt changed (this session adds hmmlearn + numpy)

```bash
ssh aso@192.168.1.183 "cd ~/trading-bot && source .venv/bin/activate && pip install -r requirements.txt"
```

> ⚠️ `hmmlearn` is the only non-trivial new dependency. It pulls in
> `scipy` and (transitively) BLAS bindings. On a clean Pi venv expect
> a 1–2 minute build. It is preinstalled on most setups.

### New `.env` vars (all have defaults; nothing breaks if absent)

| Var | Default | Purpose |
|-----|---------|---------|
| `GEMINI_MODEL_RESEARCH` | `gemini-2.5-flash` | Grounded research model. |
| `GEMINI_MODEL_VERDICT` | `gemini-2.5-flash-lite` | Verdict / position-eval model. |
| `MONTHLY_GEMINI_BUDGET_USD` | `30.0` | Monthly hard cap. |
| `GEMINI_COST_PER_GROUNDED_CALL` | `0.016` | Cost charged per grounded call. |
| `GEMINI_COST_PER_UNGROUNDED_CALL` | `0.002` | Cost charged per ungrounded call. |

---

## Known Open Items / Watch List

- Monthly cost is an **estimate** based on flat per-call constants. If
  actual Gemini billing diverges materially, tune
  `GEMINI_COST_PER_*_CALL` to match.
- `MarketRegimeDetector` re-fits the HMM on every refresh; could be
  optimised to update incrementally if SPY history grows large.
- `stream.py` has no native rate-limit handler if Alpaca starts
  throttling minute-bar subscriptions on a very large watchlist —
  watchlist size is ~2,600 today which is fine.
