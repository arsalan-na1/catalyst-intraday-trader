"""Kronos-mini secondary confirmation signal.

Loads NeoQuasar's Kronos-mini model lazily (on first call) and exposes a single
public function: ``get_continuation_prob(ohlcv_df, horizon=10) -> Optional[float]``.

The returned value is the fraction of predicted close prices over the next
``horizon`` bars that sit above the last known close — i.e. how often the
model thinks price keeps grinding higher in the immediate future.

Fail mode: every error path (missing local Kronos checkout, model load
failure, pandas frequency-inference failure, inference exception, too few
input bars) returns ``None`` silently so the caller can fall back to the
existing logic without blocking trades.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import pandas as pd

log = logging.getLogger("kronos_scorer")

_KRONOS_PATH = "/home/aso/Kronos"

_predictor = None
_import_failed = False
_warned = False


def _ensure_loaded() -> bool:
    """Load Kronos-mini + tokenizer once; return True iff predictor is usable."""
    global _predictor, _import_failed, _warned

    if _import_failed:
        return False
    if _predictor is not None:
        return True

    try:
        if _KRONOS_PATH not in sys.path:
            sys.path.insert(0, _KRONOS_PATH)
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        _predictor = KronosPredictor(model, tokenizer)
        log.info("[KRONOS] Kronos-mini loaded")
        return True
    except Exception as exc:
        _import_failed = True
        if not _warned:
            log.warning(
                "[KRONOS] failed to load Kronos-mini: %r — disabled for this session",
                exc,
            )
            _warned = True
        return False


def get_continuation_prob(
    ohlcv_df: pd.DataFrame,
    horizon: int = 10,
) -> Optional[float]:
    """Predict the next ``horizon`` bars and return the fraction above the last close.

    Returns a float in [0.0, 1.0] or None on any failure.
    """
    if ohlcv_df is None or len(ohlcv_df) < 20:
        return None

    if not _ensure_loaded():
        return None

    df = ohlcv_df.iloc[-min(200, len(ohlcv_df)):].copy()

    try:
        idx = pd.DatetimeIndex(df.index)
        freq = pd.infer_freq(idx)
        if freq is None:
            return None
        future_index = pd.date_range(
            start=idx[-1], periods=horizon + 1, freq=freq,
        )[1:]
        if len(future_index) == 0:
            return None
    except Exception:
        log.debug("[KRONOS] frequency inference failed", exc_info=True)
        return None

    x_ts = pd.Series(idx)
    y_ts = pd.Series(future_index)

    try:
        last_close = float(df["close"].iloc[-1])
    except Exception:
        log.debug("[KRONOS] could not read last close", exc_info=True)
        return None

    try:
        pred = _predictor.predict(df=df, x_timestamp=x_ts, y_timestamp=y_ts)
    except Exception:
        log.debug("[KRONOS] prediction failed", exc_info=True)
        return None

    try:
        pred_close = pred["close"]
        n = len(pred_close)
        if n == 0:
            return None
        above = sum(1 for v in pred_close if float(v) > last_close)
        return above / n
    except Exception:
        log.debug("[KRONOS] result parsing failed", exc_info=True)
        return None
