# Setup Walkthrough — Raspberry Pi 5 Trading Bot

Everything below assumes a fresh Pi 5. If your Pi is already flashed, skip to [Step 2](#step-2--first-boot--ssh).

---

## Step 1 — Flash Raspberry Pi OS (headless)

1. Download **Raspberry Pi Imager**: <https://www.raspberrypi.com/software/>
2. Insert your microSD card.
3. Choose:
   - **Device**: Raspberry Pi 5
   - **OS**: Raspberry Pi OS (64-bit) — Bookworm
   - **Storage**: your microSD
4. Click the **gear icon** (or `Ctrl+Shift+X`) for advanced options:
   - Set hostname: `tradingpi` (or whatever you like)
   - Enable SSH: yes → "Use password authentication" (or paste a public key)
   - Set username: `aso` — **don't change this**; the systemd unit (`User=aso`, `WorkingDirectory=/home/aso/trading-bot`) expects it. To use a different username, edit `trading-bot.service` first.
   - Set password
   - Configure Wi-Fi: SSID + password + country
   - Set locale/timezone (safe to use your local tz — we handle market tz in code)
5. Write the image. Boot the Pi.

## Step 2 — First boot & SSH

```bash
ssh aso@tradingpi.local   # or use IP from your router
```

Update and install OS-level deps:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-pip git
```

## Step 3 — Create accounts & collect API keys

You'll need five sets of credentials. Keep these in a password manager while you work.

### 3a. Alpaca paper account

1. Sign up at <https://alpaca.markets>.
2. Confirm email.
3. In the dashboard, switch the top-right toggle to **Paper Trading**.
4. Open the right-side panel: "API Keys" → **Generate**.
5. Copy both: `APCA_API_KEY_ID` (key) and `APCA_API_SECRET_KEY` (secret).

### 3b. Google AI Studio (Gemini)

1. Go to <https://aistudio.google.com/app/apikey>.
2. Sign in with a Google account.
3. Click **Create API key** → copy the key.

Free tier gives 15 RPM on Gemini 2.5 Flash — matches our bot's rate limit.

### 3c. Telegram bot

1. Open Telegram, search for `@BotFather`.
2. Send `/newbot` → follow prompts → receive **bot token** (`1234567890:ABC...`).
3. Send any message to your new bot (search its username, tap "Start").
4. In a browser, visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Find `"chat":{"id": 123456789, ...}` in the JSON — that number is your `TELEGRAM_CHAT_ID`.

### 3d. NewsAPI.org (fallback)

1. Register at <https://newsapi.org/register>.
2. Copy your API key. Free tier = 100 requests/day — sufficient for a fallback.

## Step 4 — Clone/copy the project

Option A — from this local Mac to the Pi:

```bash
# On your Mac, from /Users/aso/Projects/:
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='logs' --exclude='.env' \
    --exclude='watchlist.txt' --exclude='watchlist_catalyst.txt' \
    Trading-Bot/ aso@tradingpi.local:/home/aso/trading-bot/
```

Option B — via git (once you push the project to your own repo):

```bash
# On the Pi:
git clone https://github.com/<you>/trading-bot.git /home/aso/trading-bot
```

## Step 5 — Create venv and install dependencies

```bash
cd /home/aso/trading-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This will take several minutes on first run (pandas compiles slowly on ARM).

## Step 6 — Configure secrets

```bash
cp .env.example .env
nano .env         # paste in each value from Step 3
chmod 600 .env    # lock it down — contains paper keys and bot tokens
```

## Step 7 — Generate the watchlist

```bash
source .venv/bin/activate
python build_watchlist.py
python build_catalyst_watchlist.py    # second list — biotech/FDA/high-short tickers
wc -l watchlist.txt watchlist_catalyst.txt   # roughly 1,800 + 800 lines
```

Re-run both monthly — index rebalances happen quarterly and annually.

## Step 8 — Smoke-test modules

Each module has a `_smoke()` runnable. Do these in order:

```bash
source .venv/bin/activate

# Calendar (safe anytime)
python market_calendar.py

# News (safe anytime — tries AAPL by default)
python news.py AAPL

# Scorer (uses Gemini quota — spends 1 request)
python scorer.py

# Stream (only useful during market hours, 9:30–16:00 ET)
python stream.py
```

End-to-end test without waiting for a real spike:

```bash
python bot.py --inject-trigger AAPL 180.0 0.12 7.0
```

You should see an alert in Telegram within ~30 seconds.

## Step 9 — Install the systemd service

```bash
sudo cp /home/aso/trading-bot/trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot
```

Verify:

```bash
sudo systemctl status trading-bot
journalctl -u trading-bot -f --since "5 minutes ago"
```

You should see "Bot online" and your Telegram chat should have the 🟢 online message.

## Step 10 — Operations cheat sheet

```bash
# Restart after code or .env changes:
sudo systemctl restart trading-bot

# Stop temporarily:
sudo systemctl stop trading-bot

# Disable auto-start on boot:
sudo systemctl disable trading-bot

# Live journald logs:
journalctl -u trading-bot -f

# Live app logs (more detail, rotated daily):
tail -f /home/aso/trading-bot/logs/bot.log

# Check last 50 errors:
journalctl -u trading-bot -p err -n 50

# Update watchlist (monthly):
cd /home/aso/trading-bot && source .venv/bin/activate && python build_watchlist.py && sudo systemctl restart trading-bot
```

## Telegram commands once running

| Command | Description |
|---------|-------------|
| `/status` | Uptime, today's stats, open positions |
| `/positions` | Detailed open-position list with entry price and hold time |
| `/close TICKER` | Manually close a specific position at market price |
| `/watchlist` | Number of tickers being monitored |
| `/trades` | Last 5 closed trades from `state/trade_log.jsonl` |
| `/regime` | Current HMM SPY regime label and size/hold multipliers |
| `/perf` | Session P&L, win rate, by-reason and by-regime breakdown |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: Missing required env var` | `.env` missing keys — recheck Step 6 |
| Bot never alerts | Confirm market is open; run `--inject-trigger` to exercise the pipeline |
| Telegram `Forbidden: bot was blocked` | You blocked the bot — unblock and `/start` again |
| `ModuleNotFoundError` at service start | venv path wrong in service file, or `pip install` didn't hit the venv. `which python` should show `/home/aso/trading-bot/.venv/bin/python` when venv is active |
| Repeated WS disconnects | Confirm Alpaca keys are **paper** keys if `ALPACA_PAPER=true`; live keys won't authenticate against paper endpoints |
| `get_news` returns empty | Expected for many mid-caps; NewsAPI fallback handles these |

## Security notes

- `.env` is `chmod 600` and listed in `.gitignore`. Never commit it.
- Paper trading only — `ALPACA_PAPER=true`. Flipping to `false` routes real orders.
- The bot has **write access** to your paper account; it does not need brokerage-level credentials or real-money keys.
