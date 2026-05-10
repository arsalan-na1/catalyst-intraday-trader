# HANDOFF — Trading Bot

Last updated: 2026-05-10

This file is the running operational record for the bot. CLAUDE.md is for
agents working on the code; HANDOFF.md is for the human (or next agent)
who picks up where the last session left off.

---

## What Was Built

### 2026-05-10 second session — post-trade reflection loop

TradingAgents-style learning loop: after each trade closes, compute the
realized outcome plus alpha vs SPY, generate a one-paragraph Gemini
reflection, persist it, and on the next entry decision for the same ticker
inject the most recent reflections (same-ticker + cross-ticker) into the
verdict prompt under a `PRIOR LESSONS` section.

**New files**

- `reflection_store.py` — append-only JSONL at `state/reflections.jsonl`
  (mirrors `_write_trade_log`'s open-append-close pattern). Methods:
  `load`, `append`, `get_recent(ticker, n)`, `get_global_lessons(n,
  exclude_ticker)`, `prune(older_than_days)`. A single `asyncio.Lock`
  guards `append` and `prune` — same single-writer guarantee as
  `cooldown_store`, with the lock as defense-in-depth for the rare case
  where two reflection background tasks finish near-simultaneously.
  Module-level `format_lessons_block(same, cross, ticker, max_tokens)`
  renders the injectable block and enforces the token budget by dropping
  oldest cross-ticker first, then oldest same-ticker. Uses a `len // 4`
  char-based token estimate (no `tiktoken` dependency).

- `reflection_generator.py` — `ReflectionGenerator.on_trade_closed(...)`.
  Computes alpha vs SPY by pulling SPY 1-min IEX bars over the holding
  window via the existing `hist_client`, then calls Gemini through the
  scorer's monthly budget gate (mirrors `self_improvement._call_gemini`).
  On Gemini failure, persists the record with `reflection_text=None` so
  every closed trade has an audit-trail entry. The generator NEVER raises
  out to the caller — `on_trade_closed` swallows all exceptions. Uses
  `config.REFLECTION_MODEL` (defaults to `GEMINI_MODEL_VERDICT`),
  `temperature=0.3`, `max_output_tokens=200`, capped to 600 chars before
  storage.

- `tests/test_reflection_store.py` (15 tests) — append+get_recent,
  ordering and N-cap, cross-ticker exclusion, persistence across
  instances, malformed-line skip on load, prune drops old + no-op on
  fresh, format helper empty/null-text/sections/token-budget.

- `tests/test_reflection_generator.py` (8 tests) — happy path, close
  path doesn't block on Gemini failure, null-text persisted on empty
  response, budget halted skip, hourly-limit-full skip,
  `REFLECTION_ENABLED=False` short-circuit, ungrounded billing,
  prompt-content round-trip. Gemini is mocked via `AsyncMock`; no real
  API traffic.

**Wiring changes**

- `trader.py` — `ActivePosition` gained `entry_rationale_text` (≤200
  chars from `verdict.reasoning`) and `entry_top_signals` (up to 3 short
  tags derived from catalyst quality + quant signals + technical_signal
  via the new `_derive_top_signals` helper). Both are persisted in
  `_save_positions_snapshot` and restored in `sync_open_positions` so a
  restart preserves them. `Trader` gained `_reflection_generator: Any`
  attribute and a `_dispatch_reflection` helper. `_handle_sell_fill` and
  `_close_fallback` invoke `_dispatch_reflection` after
  `_write_trade_log` completes — fire-and-forget via `_spawn_background`,
  the close path itself never awaits it. The ghost-position branch in
  `_monitor_tick` does NOT dispatch a reflection (no useful input).

- `scorer.py` — `Scorer` gained `_reflection_store: Any` attribute and a
  `_get_reflections_block(ticker)` helper. `_build_verdict_prompt` now
  emits a `reflections_section` immediately after the `RECENT INSIGHTS`
  line; the section is omitted entirely when the helper returns "" (per
  spec: do not inject "no prior lessons"). `_build_position_eval_prompt`
  has a NOTE comment documenting the deferred v2 decision (see Deferred
  / v2 below).

- `bot.py` — constructs `ReflectionStore` and `ReflectionGenerator` in
  `main()`, calls `await store.load()` and an at-startup
  `store.prune(REFLECTION_PRUNE_DAYS)`, then assigns
  `scorer._reflection_store = store` and
  `trader._reflection_generator = generator`. All gated on
  `config.REFLECTION_ENABLED`.

- `config.py` — six new constants:
  `REFLECTION_ENABLED` (default True),
  `REFLECTION_MODEL` (defaults to `GEMINI_MODEL_VERDICT`),
  `REFLECTION_PRUNE_DAYS` (180),
  `REFLECTION_MAX_PER_TICKER_INJECTED` (3),
  `REFLECTION_MAX_GLOBAL_INJECTED` (5),
  `REFLECTION_INJECTED_TOKEN_BUDGET` (600).

**Verification**

- `python3 -m py_compile` clean across the full project list (29 files).
- `python -m pytest tests/ -v` — 44 passed (21 original + 15 new
  reflection_store + 8 new reflection_generator).
- Pi deploy + live paper-trade validation deferred to next session.

### Deferred / v2

- **Reflection injection into position-eval prompt.** v1 injects PRIOR
  LESSONS into `_build_verdict_prompt` only. Adding it to
  `_build_position_eval_prompt` needs a filtering design first — same
  ticker may have many reflections, and an open position should probably
  weight loss-only and recency-weighted entries differently than entry
  decisions do. There's a NOTE comment at the top of
  `_build_position_eval_prompt` flagging this so the decision is
  discoverable when v2 picks it up.

### 2026-05-10 — docs only: added recent vars to .env.example

Added the `# --- Kronos secondary confirmation ---` section to `.env.example`
with `USE_KRONOS=true` and `KRONOS_MIN_PROB=0.45` (defaults match `config.py`).
Verified `MAX_ENTRY_RSI`, `FORCED_REEVAL_LOSS_PCT`, and `ATR_SL_MULTIPLIER`
were already present. No code changes.

### 2026-05-07 third session — defensive hardening + first test suite

Seven targeted changes derived from the prior session's audit. All deployed
to the Pi and verified live; 21/21 unit tests pass.

**Security**

- `telegram_handler.py` — added `_redact()` helper that strips
  `config.TELEGRAM_BOT_TOKEN` from arbitrary text. Replaced every
  HTTP-touching `log.exception(...)` with `log.error("...: %s",
  _redact(repr(e)))`. The token is in the Bot API URL path that
  python-telegram-bot embeds in every request, so any chained
  httpx exception's `__str__` would otherwise leak it via the
  traceback. Sites changed: `stop`, `send_message`, `_reply`,
  `/trades` handler, `/close` handler. `/regime` (pure compute) and
  the `/trades` JSON parse log retain `exc_info=True` — both safe.

**Robustness**

- `scorer.py` — both `asyncio.gather()` enrichment fan-outs in
  `score()` (10 sources) and `score_open_position()` (3 sources) now
  use `return_exceptions=True`. Each result is checked: exceptions
  are logged with the source name (`[SCORER] enrichment fetch
  '<name>' raised for <ticker>: <repr>`) and substituted with `None`.
  All downstream consumers already handle `None`/`{}` for missing
  signals, so no other code changes were needed. The 3-tuple
  `(verdict, technicals, quant_signals)` invariant holds on all five
  return paths of `score()` (verified manually).

- `finnhub_data.py` — `get_congressional_trades()` now latches a
  module-level `_congressional_403: bool` flag the first time the
  endpoint returns HTTP 403 (Finnhub free tier doesn't include this
  data). On the latch it logs once at WARNING `[FINNHUB]
  Congressional trades endpoint returned 403 — not available on this
  plan. Disabling for session.` Subsequent calls return `{}`
  immediately. Detection uses `getattr(e, "status_code", None)` so
  it works whether or not `FinnhubAPIException` is the exact wrapping
  exception.

- `microcap_screener.py` — `_adv_cache`, `_market_cap_cache`, and
  `_last_evaluated` were unbounded. Cache values now carry
  `datetime` timestamps: `dict[str, tuple[float, datetime]]` for the
  first two, `dict[str, datetime]` for `_last_evaluated`. New method
  `_prune_caches()` evicts entries older than 24h (adv/mcap) or
  `MICROCAP_DEDUP_HOURS` (eval). Called at the start of every poll
  cycle inside `MicrocapScreener.run()`, regardless of market hours,
  so multi-day weekends don't accumulate stale entries either.
  `import time` was removed from the file (no remaining users).

- `stream.py` — `TriggerDetector._maybe_reset_session()` now also
  clears `self._last_trigger` alongside the other per-ticker dicts.
  Without this clear, a ticker that fired near EOD yesterday could
  have its first trigger of the next session suppressed by a stale
  cooldown timestamp.

- `bot.py` — `_inject_synthetic` now rejects NaN and inf for
  PRICE/MOVE_PCT/VOL_RATIO via `math.isfinite()`. NaN comparisons
  always return `False`, so previously `--inject-trigger AAPL nan
  0.12 7.0` would slip past every bounds check and propagate into
  the pipeline. Mirrors the `_safe_float` guard already in
  `trader._do_execute_buy`.

**Tests**

- New `tests/` directory containing `__init__.py`, `conftest.py`,
  and `test_core.py`. `pytest.ini` at the project root sets
  `testpaths = tests`. `pytest>=8.0.0` added to `requirements.txt`.
- `conftest.py` injects placeholder env vars before `config` loads
  so the tests run on any host without a real `.env`.
- 21 tests covering pure functions only — no mocks, no live API
  calls, no file I/O outside `tmp_path`:
  - `_inject_synthetic` (14 tests): valid input, dot-form ticker,
    empty/long/non-alpha tickers, non-numeric price, zero/negative
    price, out-of-range move (high & low), negative vol, NaN price,
    inf move, -inf vol.
  - `_calc_take_profit` (1 test, all four magnitude tiers).
  - `CooldownStore` (2 tests): in-memory round-trip + on-disk
    round-trip across two store instances using the same `tmp_path`.
  - `Trader._pearson_corr` (4 tests): too-short series → None,
    identical → ~1.0, perfectly inverse → ~-1.0, constant series →
    None (zero std-dev guard).
- All 21 tests pass on the Pi venv:
  `python -m pytest tests/ -v` → `21 passed in 1.37s`.

**Deployment notes for the next session**

- This session adds `pytest` as a new top-level dependency. On a
  fresh Pi venv: `pip install -r requirements.txt`. Already
  installed on `aso@192.168.1.183` (pytest 9.0.3 from piwheels).
- `tests/` is included in the rsync command in CLAUDE.md (no
  excludes target it). State files / .env / watchlists remain
  excluded.
- Service was restarted on the Pi after sync and reached steady
  state cleanly: all four self-check ✅ lines, no new error
  patterns. Pre-existing benign warnings (NASDAQ earnings API
  timeout, BF.B-style invalid ticker line, `hmmlearn` non-
  convergence) unchanged.

### 2026-05-07 second session — live-log driven hardening

Seven changes derived from analyzing live trading logs from the
2026-05-07 first session.

**Hard entry gates (bot-side, independent of Gemini)**

- `bot.py` + `config.py` — new `MAX_ENTRY_RSI` (default 82). After the
  `should_trade` check in `_process_trigger`, the bot now hard-rejects
  any entry where `verdict.confidence < MIN_GEMINI_CONFIDENCE` and any
  entry with RSI above `MAX_ENTRY_RSI`. Both gates fail open when the
  relevant data is unavailable. INTC, SABR, MRP and PCT all traded at
  confidence 6 today despite `MIN_GEMINI_CONFIDENCE=7` because Gemini
  set `should_trade=True` and the threshold helper in scorer was never
  called by the live pipeline; INTC also entered at RSI 86 with a
  bearish technical signal because Gemini overrode its own RSI > 85
  rule.

**Trader bug fixes**

- `trader.py:_monitor_tick` — timeout guard tightened from `plpc >= 0`
  to `plpc >= 0.001` (~10 bps). MRP closed at -0.07% via the timeout
  branch today: Alpaca returned `unrealized_plpc=0.0` at the tick
  instant, the close fired, and the position was slightly negative by
  the time the SELL filled. The new threshold avoids that
  float-precision race.
- `trader.py:run_trading_stream` — BUY-fill Telegram notification now
  fires only on `event == "fill"`, not `event == "partial_fill"`. CORZ
  produced 7 separate `✅ Filled` messages today (one per partial fill).
  `pos.entry_price` is still updated silently on partial fills.
- `trader.py:_monitor_tick` — ghost-position accounting gap fixed.
  Today RUN was submitted (`trades_taken` incremented, buy notification
  sent) but the TradingStream fill event never arrived; `_monitor_tick`
  detected the position missing from Alpaca and the old code silently
  popped it from tracking with no `trade_records` append, no
  `trade_log.jsonl` entry and no Telegram notification — leaving
  `trades_taken=8` against 7 recap entries with zero audit trail. The
  branch now writes a synthesized zero-P&L `TradeRecord` (with the
  position's `regime` tag), appends to `trade_log.jsonl`, and sends a
  `⚠️ disappeared from Alpaca…` Telegram warning before cleanup. When
  `entry_price` was never set (buy fill never confirmed), no record is
  written but a loud warning is logged.

**Position-monitor behaviour**

- `trader.py:_monitor_tick` + `position_monitor.py` — the hold-time
  timeout no longer auto-closes. Instead it pushes the deadline 30
  minutes out and schedules a new `_timeout_reeval` coroutine that
  routes through the same Gemini-driven path the regular monitor
  uses, so Gemini chooses between hold / exit / raise_target /
  tighten_sl / adjust_tp / add_time. If the call fails, fallback closes
  only profitable positions (`plpc >= 0.001`) under the reason
  `timeout_fallback — review unavailable`. Today three positions (MRP,
  SABR, CORZ) closed at near-breakeven via the old hard timeout after
  4–5 hours.
  - `Trader` gained `_monitor_scorer`, `_monitor_telegram`,
    `_monitor_http_session`, `_monitor_news_fingerprints` attributes;
    `run_position_monitor` populates them at startup.
  - `_timeout_reeval` lives in `position_monitor.py`; trader.py
    imports it lazily inside `_monitor_tick` to dodge the module-load
    circular (`position_monitor` already imports `Trader` at top
    level).
  - Fail-open: if any of the four refs is still None when timeout
    fires (rare — only between Trader construction and
    `run_position_monitor` start), the trader falls back to the
    original hard close.
- `position_monitor.py` + `config.py` — new `FORCED_REEVAL_LOSS_PCT`
  (default 0.015 = 1.5%). When an open position's unrealized P&L drops
  below this threshold, the per-position monitor now bypasses the
  `_news_fingerprints` skip and forces a fresh Gemini look anyway.
  PCT bled -2.10% over 5h 48m, INTC -1.68% over 6h 8m, NVST -2.16%
  today — all without any new headlines, so the fingerprint
  optimization silently skipped re-looks while the positions
  deteriorated.

**Accounting invariants verified**

- Every `trades_taken` increment now has a matching trade-records
  append path:
  1. Normal exit → `_handle_sell_fill` appends `TradeRecord`.
  2. Stream miss after submitted close → `_close_fallback` appends
     `TradeRecord` after 60 s.
  3. Position vanishes from Alpaca with no close submitted but with
     known `entry_price` → `_monitor_tick` ghost branch appends
     synthesized zero-P&L `TradeRecord`.
  4. Position vanishes with no `entry_price` (buy never confirmed) →
     no record (intentional; loud warning logged).

**New config (config.py + .env.example)**

| Var | Default | Purpose |
|-----|---------|---------|
| `MAX_ENTRY_RSI` | `82` | Bot-side hard ceiling on RSI for entries. |
| `FORCED_REEVAL_LOSS_PCT` | `0.015` | Force a Gemini re-look (bypass news fingerprint) when P&L < `-FORCED_REEVAL_LOSS_PCT`. |

No new required env vars. Existing `.env` files keep working.

### 2026-05-07 first session

**Bug fixes**

- `market_regime.py` — replaced deprecated `datetime.utcnow()` with
  `datetime.now(timezone.utc)` (3 sites: cache age check, double-checked
  cache check after lock, and `_last_updated` write).
- `scorer.py` — split the hourly call counter into a non-mutating
  `_check_hourly_limit()` and a new `_consume_hourly_slot()`. Slots are
  now appended only after every gate (hourly cap, budget mode, halt) has
  passed AND immediately before the actual Gemini request fires, so
  blocked calls no longer inflate the hourly counter. All three Gemini
  callers (`_research_with_grounding`, `_verdict_from_research`,
  `score_open_position`) updated.
- `scorer.py` — `_record_call_cost` no longer adds the cost when it
  blocks the call. The cost is now committed only on the success path,
  so the monthly cost estimate stops drifting up on `halted`-mode
  rejections. Mode-transition alert messages now reference the
  pre-rejection cost (the more accurate figure).
- `position_monitor.py` — removed redundant `import config as _cfg`
  blocks inside `adjust_tp` and `add_time` branches; both now use the
  module-level `import config`.

**Code fixes**

- `bot.py` — startup Gemini ping now uses `config.GEMINI_MODEL_VERDICT`
  (was hardcoded to `gemini-2.5-flash`) and routes through
  `scorer._record_call_cost(is_grounded=False)` so its cost is tracked.
  Skips with a warning when `_budget_mode == "halted"`.
- `trader.py` + `daily_summary.py` — `_sector_last_entry` is now cleared
  alongside `session_blocked_tickers` in both `Trader.run_eod_close()`
  and `daily_summary.run_daily_summary()`. A late-session sector entry
  no longer blocks duplicate-catalyst guard at the open of the next
  session.

**New features**

- `stream.py` + `scorer.py` + `bot.py` — VWAP context threaded through
  the trigger pipeline. `TriggerDetector` captures `bar.vwap` (defensive
  `getattr`) into a per-ticker `_last_vwap` dict; `TriggerEvent` and
  `TriggerContext` both gained `window_vwap: float | None = None`. The
  scorer's verdict prompt appends a "VWAP (session): $X.XX — price is
  ±N% vs VWAP" line to the technicals section, and the system
  instruction tells Gemini to lower confidence by 1–2 additional points
  when price is more than 15% above VWAP (chasing extended moves).
  Synthetic `--inject-trigger` CLI path is unchanged (window_vwap
  defaults to None).
- `trader.py` — ATR-anchored SL floor: after clamping Gemini's
  `stop_loss_pct` to `GEMINI_SL_MIN/MAX`, the trader enforces a
  minimum stop distance of `ATR_SL_MULTIPLIER × technicals.atr_14_pct`,
  capped at `GEMINI_SL_MAX`. Fail-open when technicals or ATR are
  unavailable.
- `daily_summary.py` + `trader.py` + `bot.py` — regime tag on every
  `TradeRecord`. `bot._process_trigger` reads
  `scorer._current_regime_label` and passes it through `execute_auto_buy`
  → `_do_execute_buy` → `ActivePosition.regime`. On close, `_handle_sell_fill`
  and `_close_fallback` write the regime onto the appended `TradeRecord`.
  EOD recap groups today's trades by regime (count, win rate, avg P&L%)
  and includes the breakdown in both the Telegram message and the
  server log.
- `telegram_handler.py` + `bot.py` — `/regime` command shows the HMM
  label, size/hold multipliers, and time since last refresh. The
  `MarketRegimeDetector` is wired through `telegram.set_context(
  regime_detector=…)`.
- `telegram_handler.py` — `/perf` command shows session-to-date trades
  (W/L counts, win rate, realized P&L), grouping by close-reason prefix
  and by regime, plus factor-signal correctness counters from
  `SessionStats`.

**New config (config.py + .env.example)**

| Var | Default | Purpose |
|-----|---------|---------|
| `ATR_SL_MULTIPLIER` | `0.5` | SL must cover at least this × daily ATR%; capped at `GEMINI_SL_MAX`. |

No new required env vars; existing `.env` files keep working.

### 2026-05-06 session

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
