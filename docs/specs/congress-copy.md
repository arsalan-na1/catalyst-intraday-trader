# Spec: Congressional-trade copy (virtual paper portfolio)

## Objective
Benchmark a congressional-trade copy strategy against the live scalper, head-to-head,
in paper. Build a **separate, fully-virtual** portfolio that copies recently *disclosed*
congressional BUYs. It is **off by default** (`CONGRESS_COPY_ENABLED=false`) and, when
off, the bot behaves byte-identically to today.

This is **architecture (b)** from the Phase-0 proposal: a strategy isolated from the
scalper — its own state file, its own mark-to-market P&L, **no Alpaca orders, no shared
state**. The confluence-signal option (a) — wiring congress data into the Gemini verdict
prompt — is explicitly deferred to a separate, separately-flagged follow-on. (The
existing `finnhub_data.get_congressional_trades` Finnhub path is **premium** — it 403s on
a free key — so it is not reused; only its *shape*, cached/fail-open/`to_thread`, is.)

## Hard invariants (must never break)
PAPER only (`ALPACA_PAPER` never flipped); `scorer.score()` returns its 3-tuple;
`execute_auto_buy` returns bool; Alpaca calls in `asyncio.to_thread`; state writes atomic;
**no scalper threshold / sizing / entry-exit change.** Flag off ⇒ no new coroutine starts,
no new network, no new disk; the 130-test baseline is unchanged.

## Data layer — `congress_trades.py` (Phase 1)
Daily bulk pull of recently disclosed trades, normalized + cached to disk. **Never on the
intraday hot path** (the feed is 45-day-delayed by law). Everything **fails open** (returns
`[]`, never raises). All HTTP via `http_utils.fetch_with_retry`.

- **Primary source — Financial Modeling Prep (FMP) free tier** (House + Senate `latest`
  endpoints), used when `CONGRESS_DATA_API_KEY` is set. Full coverage. Free tier 250
  calls/day; one daily pull is trivially within it.
- **Fallback — Senate Stock Watcher** (keyless GitHub-raw mirror of `all_transactions.json`),
  used automatically when no FMP key is present. Senate-only, zero setup.
- **To enable full (House+Senate) coverage:** paste an FMP free-tier key into `.env` as
  `CONGRESS_DATA_API_KEY=...`. With it blank, the feature runs keyless on Senate data.

Normalized record `CongressTrade`: `member, chamber, ticker, transaction_type (buy|sell),
transaction_date, disclosure_date, amount_min, amount_max, source` + `lag_days` property.

### The three inherent realities
- **45-day delay:** retain BOTH dates; expose `lag_days`; the virtual portfolio enters at
  the **disclosure date** (when a copier could act), never the trade date.
- **Dollar RANGES:** parse `amount` to `amount_min`/`amount_max` bounds; never collapse to a
  point. (Range-tier sizing is then a one-flag switch later.)
- **Dedup + ticker normalization:** dedup on `(member, ticker, transaction_date, type,
  amount)`; normalize tickers (uppercase, class suffixes ok, drop `--`/options/blank/
  non-equity); copy only BUYs of equities.

## Virtual portfolio — `congress_portfolio.py` (Phase 2, gated)
Gated by `CONGRESS_COPY_ENABLED` (default false). On the daily pull, open a virtual
position per new disclosed BUY, **equal-weight** (`CONGRESS_SIZING_MODE=equal_weight`,
`CONGRESS_EQUAL_WEIGHT_USD` per position against `CONGRESS_VIRTUAL_EQUITY_USD`). Entry
marked at the disclosure date; mark-to-market against live prices. Own state file
(`state/congress_portfolio.json`, atomic). **No Alpaca orders.** A daily scheduler in
`bot.main()` runs this only when the flag is on.

## Sizing (decided)
Equal-weight (fixed `$` per disclosed buy) — lowest-variance, most comparable benchmark;
sidesteps range-collapse. `amount_min`/`amount_max` are persisted so range-tier weighting
is a later one-flag switch. **Not** Autopilot-proportional (can't faithfully replicate it
without each member's full portfolio + their algorithm; collapsing ranges adds noise).

## Testing
All tests use **recorded fixtures — never the network.** Phase 1: parsing/normalization/
dedup/amount-range/lag/atomic-cache + fail-open fetch (monkeypatched). Phase 2: flag-off =
byte-identical (coroutine not scheduled; 130 baseline unchanged); flag-on portfolio
behavior; equal-weight sizing on ranges; stale-data (lag) handling.

## Boundaries
- **Always:** fail open; atomic writes; daily (not hot-path) pull; tests offline.
- **Ask first:** any paid tier/key beyond FMP free; touching the scalper path.
- **Never:** flip `ALPACA_PAPER`; submit Alpaca orders from the copy portfolio; change
  scalper thresholds/sizing/gates.
