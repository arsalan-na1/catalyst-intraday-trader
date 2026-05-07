# Catalyst Intraday Trader

An autonomous, event-driven intraday trading bot that detects high-momentum catalyst events in real time, evaluates them with Google Gemini AI, and executes paper trades on Alpaca. Built to run continuously on a Raspberry Pi 5.

---

## What It Does

The bot monitors ~2,600 tickers via the Alpaca WebSocket bar stream. When a stock moves ≥8% with 5× normal volume — or a biotech/FDA ticker moves ≥5% — it fires a scoring pipeline:

1. **Gemini research call** (grounded web search) — fetches and summarizes the news catalyst
2. **Gemini verdict call** (structured schema) — produces a `should_trade` decision plus per-trade TP, SL, position size, and hold time
3. **Quantitative factor overlay** — short interest squeeze score, analyst estimate revisions, insider sentiment, sector ETF momentum
4. **Market regime filter** — HMM-detected SPY regime (trending / ranging / volatile) scales position size and hold time
5. **Order execution** — Alpaca paper market order with Gemini-set parameters, clamped to hard risk limits
6. **Continuous re-evaluation** — every 15 minutes, Gemini re-scores open positions and can trigger early exit, raise targets, or extend hold time

Positions are closed by take-profit, stop-loss, Gemini exit signal, or EOD sweep at 3:45 PM ET. All fills are confirmed from the Alpaca TradingStream event (not from order submission).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.13, asyncio single event loop |
| Brokerage | [Alpaca](https://alpaca.markets) — paper trading, IEX data feed |
| AI scoring | Google Gemini 2.5 Flash (research) + Gemini 2.5 Flash-Lite (verdict) |
| Alerts | Telegram bot (`python-telegram-bot`) |
| Quantitative data | yfinance (short interest, fundamentals), Finnhub (earnings revisions, insider trades), Reddit PRAW (social sentiment) |
| Regime detection | `hmmlearn` Gaussian HMM on SPY daily bars |
| News | NewsAPI, RSS feeds, BeautifulSoup scraping |
| HTTP | `aiohttp` with retry logic (`http_utils.fetch_with_retry`) |
| Configuration | `python-dotenv`, all secrets in `.env` |
| Infrastructure | Raspberry Pi 5, Debian, `systemd` service |

---

## Architecture

```
Alpaca WebSocket bar stream (stream.py)
    ↓  TriggerEvent (price ≥8% AND volume ≥5× ADV)
asyncio.Queue
    ↓
pipeline_consumer (bot.py)
    ├─ fetch_news (news.py)
    ├─ fetch_technicals (technicals.py)
    ├─ fetch_fundamentals (fundamentals.py)
    ├─ quant signals: short_interest / estimate_revisions / insider / sector_momentum
    └─ scorer.score() → 2 Gemini API calls
           research call: google_search grounding → free-form text
           verdict call: response_schema=CatalystVerdict → structured JSON
                ↓
        trader.execute_auto_buy()
                ↓
        Alpaca market buy order
                ↓
        TradingStream FILL event → P&L accounting + Telegram alert

Background tasks (asyncio.gather):
    ├─ position_monitor: re-scores every open position every 15 min
    ├─ premarket_scanner: 9:00–9:29 ET gap + squeeze pre-screen
    ├─ microcap_screener: polls Alpaca most-actives every 3 min
    ├─ run_regime_refresher: refits HMM every 60 min
    └─ run_daily_summary: EOD recap + circuit breaker reset
```

The WebSocket stream runs in a worker thread (`asyncio.to_thread`). All cross-thread writes use `main_loop.call_soon_threadsafe` — no shared mutable state accessed directly from the stream thread.

---

## Key Features

### Gemini AI Integration
Two-stage scoring on every trigger. The first call uses Google Search grounding to pull live news; the second call enforces a Pydantic schema (`CatalystVerdict`) to produce a structured trade decision. Google Search grounding is incompatible with `response_schema` in a single call, hence the split.

Gemini sets per-trade parameters:
- `take_profit_pct` — 5–50%, scaled to catalyst strength
- `stop_loss_pct` — 2–10%, scaled to thesis risk
- `position_size_pct` — 2–12% of equity
- `max_hold_minutes` — 30–390 min
- `hold_strategy` — `momentum` / `catalyst` / `swing`
- `should_trade` — explicit trade/no-trade boolean

Hard limits are enforced in `trader.py` regardless of what Gemini returns.

### HMM Market Regime Detection
`market_regime.py` fits a 3-state Gaussian HMM on 400 calendar days of SPY daily bars using three features: 5-day momentum, 10-day annualized volatility, and SMA20/SMA50 ratio. States are labelled `trending` / `ranging` / `volatile` by sorting on volatility mean. The regime is refreshed every 60 minutes and passed to Gemini's verdict prompt. The trader applies size and hold multipliers:

| Regime | Size | Hold |
|--------|------|------|
| trending | 1.0× | 1.2× |
| ranging | 0.7× | 0.8× |
| volatile | 0.5× | 0.6× |
| unknown | 1.0× | 1.0× |

Regime detection is always fail-open — a detection failure never blocks a trade.

### Quantitative Factor Signals
Four factors are fetched in parallel before every entry and position re-evaluation:

- **Short interest** (`short_interest.py`) — yfinance short float %, days to cover, squeeze score (high/medium/low). Cache: 4h.
- **Estimate revisions** (`estimate_revisions.py`) — Finnhub analyst revision trend + EPS surprise composite. Cache: 6h.
- **Insider sentiment** (`finnhub_data.get_insider_score()`) — individual buy/sell transaction scoring. Cache: 12h.
- **Sector momentum** (`sector_momentum.py`) — SPDR ETF % change via Alpaca. Cache: 5 min.

All four fail open — if unavailable, their line is silently omitted from the Gemini prompt.

### Circuit Breakers
- **Daily loss limit** — if realized P&L falls below 2% of opening equity, `halt_new_entries` trips; existing positions are managed normally. Resets at 4:05 PM ET.
- **Daily halt** — at 3% loss, all positions are closed and trading halts for the session.
- **Weekly warn / halt** — at 5% / 7% weekly drawdown, position sizes halve / session halts.
- **Peak drawdown lock** — if equity falls 10% below its all-time peak, writes `state/trading_halted.lock` and exits the process.

### Entry Guards
- **Entry retrace** — skips the buy if ≥40% of the original spike has already reversed
- **Bid-ask spread** — skips if spread > 0.5%
- **Correlation filter** — halves size if new position correlates >70% with an open one; rejects if >85%
- **Stale trigger** — skips queued events older than 5 min if the price has reversed ≥3%
- **Sector dedup** — second entry in the same sector within 5 minutes is rejected
- **Session block** — tickers with acquisition/merger exit reasons are blocked for the rest of the session

### WebSocket Streaming
`stream.py` subscribes to Alpaca minute bars for ~2,600 tickers (main watchlist) and ~787 catalyst tickers (biotech/FDA/high-short). `TriggerDetector` tracks rolling price history per ticker and fires a `TriggerEvent` when both price-move and volume thresholds are met simultaneously. Gap-open triggers fire once at 9:30 ET for any ticker gapping ≥15% vs the prior session close.

### Gemini Cost Controls
Monthly budget cap enforced in `scorer.py`:
- At 80% of `MONTHLY_GEMINI_BUDGET_USD` (default $30): switches to ungrounded calls only (no Google Search)
- At 100%: halts all Gemini calls for the rest of the month; bot continues on TP/SL/timeout rules

Cost state persists across restarts in `performance.json` and resets on the 1st of each month.

---

## Setup

### Prerequisites
- Python 3.13+
- Raspberry Pi 5 (or any Linux host with systemd)
- Alpaca paper trading account — [alpaca.markets](https://alpaca.markets)
- Google AI Studio API key — [aistudio.google.com](https://aistudio.google.com/app/apikey)
- Telegram bot token — create via [@BotFather](https://t.me/BotFather)
- (Optional) Finnhub, NewsAPI, Reddit PRAW credentials

### 1. Clone and install

```bash
git clone https://github.com/arsalan-na1/catalyst-intraday-trader.git
cd catalyst-intraday-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
chmod 600 .env
# Edit .env and fill in all required values
```

All required keys:

| Variable | Source |
|----------|--------|
| `ALPACA_API_KEY` | Alpaca → Paper Trading → View API Keys |
| `ALPACA_SECRET_KEY` | Same as above |
| `ALPACA_PAPER` | Set to `true` for paper trading |
| `GEMINI_API_KEY` | Google AI Studio |
| `TELEGRAM_BOT_TOKEN` | @BotFather |
| `TELEGRAM_CHAT_ID` | Your numeric Telegram chat ID |

Optional (for richer signals): `FINNHUB_API_KEY`, `NEWSAPI_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`.

**API keys go in `.env` only — never committed.**

### 3. Build watchlists

```bash
python build_watchlist.py          # ~1,869 tickers (S&P 500/400/600 + IWM + FinViz)
python build_catalyst_watchlist.py # ~787 tickers (biotech/FDA/high-short)
```

Run monthly, market closed only.

### 4. Test the pipeline

```bash
# Syntax check
python3 -m py_compile bot.py stream.py scorer.py trader.py position_monitor.py

# End-to-end dry run (works outside market hours)
python bot.py --inject-trigger AAPL 180.0 0.12 7.0
```

### 5. Deploy as a systemd service (Raspberry Pi)

```bash
# Sync code to Pi
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='logs' --exclude='.env' \
    --exclude='watchlist.txt' --exclude='watchlist_catalyst.txt' \
    ~/Projects/Trading-Bot/ aso@<PI_IP>:/home/aso/trading-bot/

# Install dependencies on Pi
ssh aso@<PI_IP> "cd ~/trading-bot && source .venv/bin/activate && pip install -r requirements.txt"

# Install and start the service
ssh aso@<PI_IP> "sudo cp ~/trading-bot/trading-bot.service /etc/systemd/system/ && \
    sudo systemctl daemon-reload && sudo systemctl enable trading-bot && sudo systemctl start trading-bot"

# Watch logs
ssh aso@<PI_IP> "journalctl -u trading-bot -f"
```

---

## Project Structure

```
.
├── bot.py                   # Entrypoint — asyncio main loop, task orchestration
├── stream.py                # Alpaca WebSocket stream + TriggerDetector
├── scorer.py                # Two-stage Gemini scoring + budget tracker
├── trader.py                # Order execution, fill handling, EOD close
├── position_monitor.py      # Periodic Gemini re-evaluation of open positions
├── news.py                  # Multi-source news fetcher + sector classifier
├── technicals.py            # RSI / SMA / ATR from Alpaca daily bars
├── fundamentals.py          # Market cap / float / sector via yfinance
├── finnhub_data.py          # Insider sentiment, congressional trades, earnings
├── short_interest.py        # Short float %, days to cover, squeeze score
├── estimate_revisions.py    # Analyst revision trend + EPS surprise
├── sector_momentum.py       # SPDR sector ETF momentum
├── market_regime.py         # HMM-based SPY regime detector
├── reddit_sentiment.py      # WSB / r/stocks / r/investing sentiment
├── microcap_screener.py     # Micro-cap most-actives screener
├── premarket_scanner.py     # 9:00–9:29 ET premarket gap + squeeze scanner
├── daily_summary.py         # EOD recap + monthly cost reset
├── telegram_handler.py      # Telegram bot + /status /positions /close /trades
├── market_calendar.py       # Alpaca market clock wrapper
├── earnings_calendar.py     # NASDAQ → Yahoo earnings calendar fallback
├── cooldown_store.py        # Persistent per-ticker cooldown (JSON)
├── http_utils.py            # fetch_with_retry (aiohttp)
├── logger.py                # TimedRotatingFileHandler + stderr
├── config.py                # All env-driven configuration with defaults
├── build_watchlist.py       # One-shot main watchlist builder
├── build_catalyst_watchlist.py  # One-shot catalyst watchlist builder
├── trading-bot.service      # systemd unit file
├── .env.example             # Credential template (copy to .env)
└── requirements.txt
```

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Current session stats, circuit breaker state, Gemini budget |
| `/positions` | Open positions with entry price, P&L, hold time |
| `/close <TICKER>` | Manually close a position |
| `/watchlist` | Number of tickers being monitored |
| `/trades` | Recent trade log |

---

## Configuration

All thresholds are overridable via `.env`. See `.env.example` for the full list. Key defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `PRICE_MOVE_THRESHOLD_PCT` | 8% | Price move to trigger main watchlist |
| `PRICE_TRIGGER_CATALYST_PCT` | 5% | Trigger threshold for catalyst watchlist |
| `VOLUME_RATIO_THRESHOLD` | 5× | Volume vs ADV required to trigger |
| `MAX_POSITIONS` | 5 | Maximum concurrent positions |
| `DAILY_LOSS_LIMIT_PCT` | 2% | Circuit breaker threshold |
| `MONTHLY_GEMINI_BUDGET_USD` | $30 | Monthly Gemini hard cost cap |
| `GEMINI_MODEL_RESEARCH` | `gemini-2.5-flash` | Grounded research call |
| `GEMINI_MODEL_VERDICT` | `gemini-2.5-flash-lite` | Structured verdict call |

---

## Disclaimer

This bot is for **educational and paper trading purposes only**. It does not constitute financial advice. Past paper trading performance does not predict live trading results. Never risk money you cannot afford to lose.
