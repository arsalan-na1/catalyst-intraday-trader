"""Virtual congress-copy paper portfolio (off by default; see docs/specs).

A self-contained, FULLY VIRTUAL portfolio that copies recently disclosed
congressional BUYs, for head-to-head benchmarking against the scalper. It places
NO Alpaca orders and shares NO state with the live bot: positions are tracked
here, marked to market against injected prices, and persisted to its own state
file (atomic tmp->rename).

All logic in this module is pure and price-injected — it never touches the
network or Alpaca. The bot layer supplies current prices and the daily cadence
(see bot.run_congress_copy), so this module is fully unit-testable offline.

Design decisions (from docs/specs/congress-copy.md):
  * Entry is recorded at FIRST-SEEN (when a copier could actually act on the
    disclosure), at the current price — never backdated to a historical price,
    which would fabricate a backtest. The transaction/disclosure dates and the
    disclosure lag are retained for transparency.
  * Only disclosures on/after the portfolio's start date AND within
    CONGRESS_FRESHNESS_DAYS are opened, so the first run does not buy the entire
    historical dump at today's prices.
  * Sizing is equal-weight (fixed $ per disclosed buy) by default; range_tier
    (scaling by the disclosed amount bounds) is a config switch. The disclosed
    dollar RANGE is always kept as amount_min/amount_max — never collapsed.
  * A member's disclosed SELL of a held ticker mirrors out that member's
    position(s) in it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import config
from congress_trades import CongressTrade

log = logging.getLogger("congress_portfolio")

# range_tier size multipliers keyed by the upper bound of the disclosed range.
_RANGE_TIERS: tuple[tuple[float, int], ...] = (
    (15_000, 1), (50_000, 2), (100_000, 3), (250_000, 4),
    (500_000, 5), (1_000_000, 6), (5_000_000, 7),
)


def position_key(t: CongressTrade) -> str:
    """Stable identity for a disclosed trade (dedup + open/closed lookup)."""
    return f"{t.member.lower()}|{t.ticker}|{t.transaction_date}|{t.disclosure_date}"


def _today_iso() -> str:
    return datetime.now(tz=config.MARKET_TZ).date().isoformat()


def size_usd_for(trade: CongressTrade, mode: str, base_usd: float) -> float:
    """Target dollar size for a disclosed buy under the chosen sizing mode.

    equal_weight: fixed `base_usd` regardless of the disclosed range (lowest
    variance, most comparable). range_tier: `base_usd` x a tier derived from the
    disclosed range's upper bound — bigger disclosed buys carry more weight
    without the false precision of collapsing the range to a point estimate.
    """
    if mode == "range_tier":
        hi = trade.amount_max or trade.amount_min
        mult = _RANGE_TIERS[-1][1]
        for ceiling, m in _RANGE_TIERS:
            if hi <= ceiling:
                mult = m
                break
        return base_usd * mult
    return base_usd  # equal_weight (default)


@dataclass
class VirtualPosition:
    key: str
    member: str
    ticker: str
    shares: float
    entry_price: float
    entry_date: str         # ISO date we opened (acted on the disclosure)
    transaction_date: str   # when the member traded
    disclosure_date: str    # when it was disclosed
    lag_days: int           # disclosure - transaction: the staleness we acted on
    amount_min: float
    amount_max: float

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price


class CongressPortfolio:
    """A virtual long-only equal-weight portfolio of copied congressional buys."""

    def __init__(
        self,
        starting_equity: float,
        started_on: str,
        cash: float | None = None,
        sizing_mode: str = "equal_weight",
        base_usd: float = 1000.0,
        open_positions: dict[str, VirtualPosition] | None = None,
        closed: list[dict] | None = None,
        realized_pnl: float = 0.0,
    ) -> None:
        self.starting_equity = starting_equity
        self.started_on = started_on            # ISO date; pre-start disclosures never open
        self.cash = starting_equity if cash is None else cash
        self.sizing_mode = sizing_mode
        self.base_usd = base_usd
        self.open: dict[str, VirtualPosition] = open_positions or {}
        self.closed: list[dict] = closed or []
        self.realized_pnl = realized_pnl

    # --- core mechanics (pure: caller supplies prices + today) --------------

    def _seen(self, key: str) -> bool:
        return key in self.open or any(c.get("key") == key for c in self.closed)

    def should_open(self, trade: CongressTrade, today: str, freshness_days: int) -> bool:
        """A disclosed buy is openable iff it is a new (unseen) BUY that:
          * was disclosed on/after we started tracking (no historical backfill),
          * was disclosed no later than today (no future / clock-skew rows), and
          * was EXECUTED within `freshness_days` of today.

        Freshness is measured on the TRANSACTION date, not the disclosure date:
        the legal disclosure lag can exceed a year, so a trade executed long ago
        is a dead signal even when it was only just disclosed. Fail-closed on a
        missing or unparseable transaction/disclosure date."""
        if trade.transaction_type != "buy":
            return False
        if self._seen(position_key(trade)):
            return False
        disc, tx = trade.disclosure_date, trade.transaction_date
        if not disc or disc < self.started_on:
            return False
        try:
            today_d = date.fromisoformat(today)
            disc_age = (today_d - date.fromisoformat(disc)).days
            tx_age = (today_d - date.fromisoformat(tx)).days
        except (ValueError, TypeError):
            return False  # missing/unparseable date → can't assess staleness → fail closed
        if disc_age < 0:                      # disclosed in the future (clock skew)
            return False
        return 0 <= tx_age <= freshness_days   # executed recently enough, not future-dated

    def open_from_trade(self, trade: CongressTrade, price: float | None, today: str) -> VirtualPosition | None:
        if price is None or price <= 0:
            return None
        target = size_usd_for(trade, self.sizing_mode, self.base_usd)
        size = min(target, self.cash)
        if size <= 0:
            return None
        shares = size / price
        pos = VirtualPosition(
            key=position_key(trade), member=trade.member, ticker=trade.ticker,
            shares=shares, entry_price=price, entry_date=today,
            transaction_date=trade.transaction_date, disclosure_date=trade.disclosure_date,
            lag_days=trade.lag_days, amount_min=trade.amount_min, amount_max=trade.amount_max,
        )
        self.cash -= shares * price
        self.open[pos.key] = pos
        return pos

    def _close(self, key: str, price: float, today: str, reason: str) -> None:
        pos = self.open.pop(key)
        proceeds = pos.shares * price
        pnl = proceeds - pos.cost_basis
        self.cash += proceeds
        self.realized_pnl += pnl
        rec = asdict(pos)
        rec.update(exit_price=price, exit_date=today, realized_pnl=pnl, exit_reason=reason)
        self.closed.append(rec)

    def close_matching_sells(self, sell: CongressTrade, price: float | None, today: str) -> list[str]:
        """Mirror a member's disclosed SELL: close that member's open
        position(s) in the sold ticker."""
        if price is None or price <= 0:
            return []
        member = sell.member.lower()
        keys = [k for k, p in self.open.items()
                if p.ticker == sell.ticker and p.member.lower() == member]
        for k in keys:
            self._close(k, price, today, reason="mirror_sell")
        return keys

    def apply_disclosures(
        self,
        trades: Iterable[CongressTrade],
        prices: dict[str, float],
        today: str | None = None,
        freshness_days: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """Open new fresh BUYs and mirror member SELLs using injected prices.

        Returns (opened_keys, closed_keys). Trades for which no price was
        supplied are skipped (open) / no-op (sell of an unheld ticker)."""
        today = today or _today_iso()
        if freshness_days is None:
            freshness_days = config.CONGRESS_FRESHNESS_DAYS
        opened: list[str] = []
        closed: list[str] = []
        # Deterministic order; buys then sells within a (date,ticker) so a same-
        # cycle buy+sell of the same name nets out predictably.
        ordered = sorted(
            trades,
            key=lambda t: (t.disclosure_date, t.ticker, 0 if t.transaction_type == "buy" else 1),
        )
        for t in ordered:
            price = prices.get(t.ticker)
            if t.transaction_type == "buy":
                if self.should_open(t, today, freshness_days):
                    pos = self.open_from_trade(t, price, today)
                    if pos is not None:
                        opened.append(pos.key)
            elif t.transaction_type == "sell":
                closed.extend(self.close_matching_sells(t, price, today))
        return opened, closed

    def mark_to_market(self, prices: dict[str, float]) -> dict:
        """Summary stats against injected prices. Positions with no supplied
        price fall back to their entry price (0 unrealized) rather than dropping."""
        positions_value = 0.0
        unrealized = 0.0
        for pos in self.open.values():
            price = prices.get(pos.ticker, pos.entry_price)
            positions_value += pos.shares * price
            unrealized += pos.shares * (price - pos.entry_price)
        equity = self.cash + positions_value
        return {
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(positions_value, 2),
            "open_positions": len(self.open),
            "closed_positions": len(self.closed),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(equity - self.starting_equity, 2),
            "total_return_pct": round(equity / self.starting_equity - 1.0, 6) if self.starting_equity else 0.0,
        }

    # --- persistence (atomic) ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "starting_equity": self.starting_equity,
            "started_on": self.started_on,
            "cash": self.cash,
            "sizing_mode": self.sizing_mode,
            "base_usd": self.base_usd,
            "realized_pnl": self.realized_pnl,
            "open": {k: asdict(v) for k, v in self.open.items()},
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CongressPortfolio":
        open_positions = {}
        for k, v in (d.get("open") or {}).items():
            try:
                open_positions[k] = VirtualPosition(**v)
            except TypeError:
                # Schema drift in a persisted position — skip it but make the
                # drop visible (its cash was already debited on a prior run, so
                # silently shrinking the book would hide the discrepancy).
                log.warning("[CONGRESS] dropping unloadable persisted position %r", k)
                continue
        return cls(
            starting_equity=float(d.get("starting_equity", config.CONGRESS_VIRTUAL_EQUITY_USD)),
            started_on=d.get("started_on") or _today_iso(),
            cash=float(d.get("cash", d.get("starting_equity", config.CONGRESS_VIRTUAL_EQUITY_USD))),
            sizing_mode=d.get("sizing_mode", "equal_weight"),
            base_usd=float(d.get("base_usd", config.CONGRESS_EQUAL_WEIGHT_USD)),
            open_positions=open_positions,
            closed=list(d.get("closed") or []),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
        )

    def save(self, path=None) -> None:
        """Atomically persist (tmp->rename). Never raises."""
        try:
            path = Path(path or config.CONGRESS_PORTFOLIO_FILE)
            payload = self.to_dict()
            payload["updated_at"] = datetime.now(tz=config.MARKET_TZ).isoformat()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            log.warning("[CONGRESS] failed to persist portfolio", exc_info=True)

    @classmethod
    def load(cls, path=None) -> "CongressPortfolio":
        """Load the portfolio, or construct a fresh one (started today) if the
        file is missing/corrupt."""
        path = Path(path or config.CONGRESS_PORTFOLIO_FILE)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(d)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return cls(
                starting_equity=config.CONGRESS_VIRTUAL_EQUITY_USD,
                started_on=_today_iso(),
                sizing_mode=config.CONGRESS_SIZING_MODE,
                base_usd=config.CONGRESS_EQUAL_WEIGHT_USD,
            )
