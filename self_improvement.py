"""Nightly self-improvement: analyze trade history and write actionable lessons.

Runs once per trading day after EOD close. Reads state/trade_log.jsonl,
computes coarse win-rate stats grouped by exit reason, Gemini confidence,
Gemini magnitude, and technical_signal, then asks Gemini to extract patterns
and produce a structured insights JSON saved to state/insights.json.

The output is consumed by scorer.py — the top of `score()` loads insights.json
(if present and < 7 days old) and appends the lessons / favor_patterns /
avoid_patterns to the Gemini verdict prompt so the model can learn from
what has actually worked and failed in this account.

Failure mode: any error (file missing, JSON parse, Gemini call failure)
logs a warning and returns silently. Never crashes the bot.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types as genai_types

import config

if TYPE_CHECKING:
    from scorer import Scorer

log = logging.getLogger("self_improvement")

TRADE_LOG_PATH = config.STATE_DIR / "trade_log.jsonl"
INSIGHTS_PATH = config.STATE_DIR / "insights.json"
MIN_TRADES_FOR_ANALYSIS = 5
INSIGHTS_FRESH_DAYS = 7


_SYSTEM_INSTRUCTION = """You are a quantitative trading coach reviewing a small
intraday-event-driven trading bot's recent trade log. The bot scores each
candidate via a separate Gemini call (catalyst confidence + magnitude +
technical_signal) and exits via TP / SL / timeout / Gemini re-eval.

Your job: identify ACTIONABLE patterns from the data — which catalyst /
confidence / magnitude / technical-signal combinations consistently win or
lose, and what the bot should change. Be terse, concrete, and grounded in
what the numbers actually show. Do NOT invent generic trading advice.

If the data is too thin or too uniform to support a pattern, say so and
return short lists. It is better to return one solid lesson than five
speculative ones.

Return valid JSON only matching exactly this schema:
{
  "summary": "2-3 sentence high-level takeaway",
  "lessons": ["lesson 1", "lesson 2"],
  "avoid_patterns": ["pattern 1"],
  "favor_patterns": ["pattern 1"],
  "param_suggestions": {
    "confidence_min_suggested": <int 1-10 or null>,
    "magnitude_min_suggested": <int 1-10 or null>
  }
}

Hard limits: lessons ≤ 8 items, avoid_patterns ≤ 6, favor_patterns ≤ 6.
Each list item ≤ 200 characters."""


@dataclass
class _GroupStats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    pnl_pct_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return self.pnl_pct_sum / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
        }


def _read_trades() -> list[dict[str, Any]]:
    if not TRADE_LOG_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with TRADE_LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("[self_improvement] skipping malformed trade_log row")
    return rows


def _bucket(value: Any) -> str:
    """Coerce a possibly-null field into a stable bucket key."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _confidence_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if v <= 5:
        return "low(1-5)"
    if v <= 7:
        return "mid(6-7)"
    return "high(8-10)"


def _magnitude_bucket(value: Any) -> str:
    return _confidence_bucket(value)  # same 1–10 scale, same buckets


def _exit_reason_bucket(value: Any) -> str:
    if not value:
        return "unknown"
    s = str(value).lower()
    if s.startswith("tp") or "take profit" in s:
        return "TP"
    if s.startswith("sl") or "stop loss" in s or "stop-loss" in s:
        return "SL"
    if "timeout" in s:
        return "timeout"
    if "eod" in s:
        return "EOD"
    if "gemini_exit" in s or "near-tp" in s:
        return "discretionary"
    return s[:32]


def _compute_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    by_exit: dict[str, _GroupStats] = defaultdict(_GroupStats)
    by_conf: dict[str, _GroupStats] = defaultdict(_GroupStats)
    by_mag: dict[str, _GroupStats] = defaultdict(_GroupStats)
    by_tech: dict[str, _GroupStats] = defaultdict(_GroupStats)

    overall = _GroupStats()

    for t in trades:
        pnl_pct = float(t.get("pnl_pct") or 0.0)
        is_win = pnl_pct > 0

        for groups, key in (
            (by_exit, _exit_reason_bucket(t.get("exit_reason"))),
            (by_conf, _confidence_bucket(t.get("gemini_confidence"))),
            (by_mag, _magnitude_bucket(t.get("gemini_magnitude"))),
            (by_tech, _bucket(t.get("technical_signal"))),
        ):
            g = groups[key]
            g.n += 1
            g.pnl_pct_sum += pnl_pct
            if is_win:
                g.wins += 1
            else:
                g.losses += 1

        overall.n += 1
        overall.pnl_pct_sum += pnl_pct
        if is_win:
            overall.wins += 1
        else:
            overall.losses += 1

    return {
        "overall": overall.to_dict(),
        "by_exit_reason": {k: v.to_dict() for k, v in by_exit.items()},
        "by_gemini_confidence": {k: v.to_dict() for k, v in by_conf.items()},
        "by_gemini_magnitude": {k: v.to_dict() for k, v in by_mag.items()},
        "by_technical_signal": {k: v.to_dict() for k, v in by_tech.items()},
    }


def _build_user_prompt(trades: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    # Keep raw row count modest — Gemini context is paid per-token. The aggregate
    # stats already capture the patterns; we include a tail of recent rows so the
    # model can spot ticker-level recurrence without ingesting the whole history.
    recent_tail = trades[-30:]
    return (
        "Recent trade history (most-recent 30 rows shown for context):\n"
        + json.dumps(recent_tail, default=str, indent=2)
        + "\n\nAggregate statistics across ALL "
        + str(len(trades))
        + " trades:\n"
        + json.dumps(stats, indent=2)
        + "\n\nReturn the insights JSON now."
    )


async def _call_gemini(
    prompt: str, scorer: "Scorer | None" = None
) -> dict[str, Any] | None:
    """Single ungrounded Gemini call. Same client + model pattern as scorer.py.

    When `scorer` is provided, the call is routed through the scorer's monthly
    budget gate (`_record_call_cost`), shared RPM limiter, hourly cap, and call
    counters — so the cost shows up in daily summary and is suppressed when the
    monthly budget is exhausted (halted) the same way every other Gemini call is.

    Standalone mode (scorer=None) is preserved for the __main__ smoke test.
    """
    if scorer is not None:
        if scorer._budget_mode == "halted":
            log.info("[self_improvement] Gemini halted (monthly budget); skipping nightly run")
            scorer.calls_skipped += 1
            return None
        if not scorer._check_hourly_limit():
            scorer.calls_skipped += 1
            return None
        # Ungrounded call — same accounting as scorer's verdict path.
        if not scorer._record_call_cost(is_grounded=False):
            scorer.calls_skipped += 1
            return None
        await scorer._limiter.acquire()
        scorer._consume_hourly_slot()
        scorer.calls_today += 1
        scorer.calls_ungrounded += 1
        client = scorer._client
    else:
        client = genai.Client(api_key=config.GEMINI_API_KEY)

    try:
        response = await client.aio.models.generate_content(
            model=config.GEMINI_MODEL_VERDICT,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=1500,
            ),
        )
    except Exception:
        log.exception("[self_improvement] Gemini call failed")
        return None

    # Real token accounting (Fix 2). Standalone mode (scorer=None) has no tally.
    if scorer is not None and response is not None and hasattr(scorer, "_account_usage"):
        scorer._account_usage(config.GEMINI_MODEL_VERDICT, response, is_grounded=False)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        log.warning("[self_improvement] Gemini returned empty text")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("[self_improvement] Gemini response was not valid JSON: %s", text[:300])
        return None


def _normalize_insights(parsed: dict[str, Any]) -> dict[str, Any]:
    """Defensive: clamp list lengths and string lengths so a runaway response
    can't bloat the verdict prompt that consumes this file."""
    def _strs(v: Any, max_items: int) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for x in v[:max_items]:
            if isinstance(x, str) and x.strip():
                out.append(x.strip()[:200])
        return out

    summary = parsed.get("summary")
    if not isinstance(summary, str):
        summary = ""
    summary = summary.strip()[:600]

    param = parsed.get("param_suggestions") or {}
    if not isinstance(param, dict):
        param = {}

    def _opt_int(v: Any) -> int | None:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        return iv if 1 <= iv <= 10 else None

    return {
        "summary": summary,
        "lessons": _strs(parsed.get("lessons"), 8),
        "avoid_patterns": _strs(parsed.get("avoid_patterns"), 6),
        "favor_patterns": _strs(parsed.get("favor_patterns"), 6),
        "param_suggestions": {
            "confidence_min_suggested": _opt_int(param.get("confidence_min_suggested")),
            "magnitude_min_suggested": _opt_int(param.get("magnitude_min_suggested")),
        },
    }


async def run_nightly_analysis(scorer: "Scorer | None" = None) -> bool:
    """Run the full pipeline: read trades → stats → Gemini → write insights.json.

    When `scorer` is provided, the Gemini call is gated by the scorer's monthly
    budget and counted in its daily call totals.

    Returns True if insights.json was written, False otherwise. Never raises.
    """
    try:
        trades = _read_trades()
    except Exception:
        log.exception("[self_improvement] failed to read trade log")
        return False

    if len(trades) < MIN_TRADES_FOR_ANALYSIS:
        log.info(
            "[self_improvement] only %d trade(s) logged (min %d); skipping",
            len(trades), MIN_TRADES_FOR_ANALYSIS,
        )
        return False

    try:
        stats = _compute_stats(trades)
    except Exception:
        log.exception("[self_improvement] failed to compute stats")
        return False

    prompt = _build_user_prompt(trades, stats)
    parsed = await _call_gemini(prompt, scorer=scorer)
    if parsed is None:
        return False

    insights = _normalize_insights(parsed)
    insights["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
    insights["trades_analyzed"] = len(trades)

    try:
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = INSIGHTS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(insights, indent=2))
        tmp.replace(INSIGHTS_PATH)
    except Exception:
        log.exception("[self_improvement] failed to write insights.json")
        return False

    log.info(
        "[self_improvement] insights.json written — %d lesson(s), %d avoid, %d favor",
        len(insights["lessons"]),
        len(insights["avoid_patterns"]),
        len(insights["favor_patterns"]),
    )
    return True


def load_insights_if_fresh(path: Path = INSIGHTS_PATH, max_age_days: int = INSIGHTS_FRESH_DAYS) -> dict[str, Any] | None:
    """Read state/insights.json if present and fresh. Returns None on miss."""
    try:
        if not path.exists():
            return None
        raw = path.read_text()
        data = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    except Exception:
        log.exception("[self_improvement] failed to load insights.json")
        return None

    ts = data.get("generated_at")
    if not isinstance(ts, str):
        return None
    try:
        gen_at = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if gen_at.tzinfo is None:
        gen_at = gen_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(tz=timezone.utc) - gen_at).total_seconds() / 86400.0
    if age_days > max_age_days:
        return None
    return data


def format_insights_block(insights: dict[str, Any]) -> str:
    """Render insights as a prompt block to append to Gemini verdict prompts."""
    parts: list[str] = ["\n\n--- LEARNED FROM PAST TRADES ---"]
    lessons = insights.get("lessons") or []
    if lessons:
        parts.append("Lessons:")
        parts.extend(f"- {x}" for x in lessons)
    avoid = insights.get("avoid_patterns") or []
    if avoid:
        parts.append("Avoid:")
        parts.extend(f"- {x}" for x in avoid)
    favor = insights.get("favor_patterns") or []
    if favor:
        parts.append("Favor:")
        parts.extend(f"- {x}" for x in favor)
    return "\n".join(parts) if len(parts) > 1 else ""


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ok = asyncio.run(run_nightly_analysis())
    print("OK" if ok else "no insights produced")
