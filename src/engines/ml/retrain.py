"""Incremental ML refresh after /verify — lightweight XGBoost update."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date as date_type
from pathlib import Path
from typing import Any

from src.db import ConclusionRecord, get_session
from src.engines.ml.train import _load_training_rows, _prepare_dataset

logger = logging.getLogger(__name__)

INCREMENTAL_MIN_ROWS = 5
VERIFY_COUNT_THRESHOLD = 3
_SYNC_BUDGET_SEC = 5.0

_MODEL_PATH = Path(
    os.environ.get("ML_MODEL_PATH", "/data/ml/hypothesis_xgb.json")
)


def _verify_triggers_retrain(trading_date: date_type) -> tuple[bool, str]:
    session = get_session()
    try:
        records = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == trading_date,
                ConclusionRecord.verification.isnot(None),
                ConclusionRecord.verification != "",
            )
            .all()
        )
    finally:
        session.close()

    verify_count = len(records)
    key_wrong = any(
        r.part_id in ("P10", "P17") and r.verification == "错" for r in records
    )
    if verify_count >= VERIFY_COUNT_THRESHOLD:
        return True, f"verify_count={verify_count}"
    if key_wrong:
        return True, "P10_or_P17_wrong"
    return False, f"verify_count={verify_count}"


def _train_and_save(rows: list[dict[str, Any]]) -> dict[str, Any]:
    X_data, y_data, feature_names = _prepare_dataset(rows)
    if len(X_data) < INCREMENTAL_MIN_ROWS:
        return {
            "status": "skipped",
            "reason": f"only {len(X_data)} labeled rows after parsing",
            "min_rows": INCREMENTAL_MIN_ROWS,
        }

    try:
        import numpy as np
        from xgboost import XGBClassifier
    except ImportError as exc:
        return {"status": "skipped", "reason": f"ML dependency missing: {exc}"}

    X_np = np.array(X_data, dtype=float)
    y_np = np.array(y_data, dtype=int)

    model = XGBClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.12,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_np, y_np)

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(_MODEL_PATH))

    accuracy = float(np.mean(model.predict(X_np) == y_np))
    return {
        "status": "ok",
        "n_rows": len(X_data),
        "train_accuracy": round(accuracy, 3),
        "model_path": str(_MODEL_PATH),
        "features": feature_names,
    }


def maybe_retrain_after_verify(trading_date: date_type) -> dict[str, Any]:
    """
    Run incremental XGBoost refresh after verify learning refresh.

    Triggers when verify count for the date meets threshold or P10/P17 marked wrong.
    Skips when labeled training rows are below INCREMENTAL_MIN_ROWS.
    """
    should_run, trigger = _verify_triggers_retrain(trading_date)
    if not should_run:
        logger.info(
            "ML retrain skipped for %s: %s (below verify threshold)",
            trading_date.isoformat(),
            trigger,
        )
        return {"status": "skipped", "reason": trigger}

    rows = _load_training_rows()
    if len(rows) < INCREMENTAL_MIN_ROWS:
        logger.info(
            "ML retrain skipped for %s: %d training rows (min %d)",
            trading_date.isoformat(),
            len(rows),
            INCREMENTAL_MIN_ROWS,
        )
        return {
            "status": "skipped",
            "reason": f"training_rows={len(rows)}",
            "min_rows": INCREMENTAL_MIN_ROWS,
            "trigger": trigger,
        }

    start = time.monotonic()
    result = _train_and_save(rows)
    elapsed = time.monotonic() - start
    result["trigger"] = trigger
    result["elapsed_sec"] = round(elapsed, 2)

    if result.get("status") == "ok":
        logger.info(
            "Incremental ML retrain for %s: %d rows, acc=%.3f, %.1fs",
            trading_date.isoformat(),
            result.get("n_rows"),
            result.get("train_accuracy", 0),
            elapsed,
        )
    return result


def maybe_retrain_after_verify_async(trading_date: date_type) -> dict[str, Any]:
    """Wait up to _SYNC_BUDGET_SEC for retrain; continue in background if still running."""
    should_run, trigger = _verify_triggers_retrain(trading_date)
    if not should_run:
        return {"status": "skipped", "reason": trigger, "async": False}

    rows = _load_training_rows()
    if len(rows) < INCREMENTAL_MIN_ROWS:
        return {
            "status": "skipped",
            "reason": f"training_rows={len(rows)}",
            "async": False,
        }

    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            holder["result"] = maybe_retrain_after_verify(trading_date)
        except Exception as exc:
            logger.exception("ML retrain failed for %s", trading_date.isoformat())
            holder["result"] = {"status": "error", "reason": str(exc)}

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"ml-retrain-{trading_date.isoformat()}",
    )
    thread.start()
    thread.join(timeout=_SYNC_BUDGET_SEC)

    if not thread.is_alive():
        result = holder.get("result", {"status": "error", "reason": "no result"})
        result["async"] = False
        return result

    return {
        "status": "queued",
        "reason": f"training exceeded {_SYNC_BUDGET_SEC}s budget",
        "async": True,
        "trigger": trigger,
    }
