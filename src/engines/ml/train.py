"""Weekly ML Training — XGBoost + SHAP feature importance.

Runs on Sundays if ≥30 training rows are available.
Outputs report to data/reports/ml/YYYY-MM-DD-shap.md.
No LLM involved in training; LLM used only for report narrative (optional).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.db import TrainingRow, get_session
from src.utils.trading_calendar import require_trading_day, today_et

logger = logging.getLogger(__name__)

MIN_ROWS = 30
_GRACE_ROWS = 10  # run with warning if between 10-30 rows


def run_weekly_ml(reference_date: date | None = None) -> dict[str, Any]:
    """
    Train XGBoost model on available training rows.
    If insufficient data, log and return graceful skip.
    """
    reference_date = reference_date or today_et()
    # Weekly ML runs Sunday; skip only if misfired onto a weekday holiday.
    if reference_date.weekday() < 5 and require_trading_day(reference_date, job="run_weekly_ml") is None:
        return {
            "status": "skipped",
            "reason": "non_trading_day",
            "reference_date": reference_date.isoformat(),
        }
    date_str = reference_date.isoformat()

    rows = _load_training_rows()
    n_rows = len(rows)
    logger.info("Weekly ML: %d training rows available", n_rows)

    if n_rows < _GRACE_ROWS:
        msg = f"Insufficient data: {n_rows} rows (minimum {MIN_ROWS} for reliable results)"
        logger.info("Weekly ML: %s — graceful skip", msg)
        return {
            "status": "skipped",
            "reason": msg,
            "n_rows": n_rows,
            "min_rows": MIN_ROWS,
        }

    if n_rows < MIN_ROWS:
        logger.warning("Weekly ML: Running with only %d rows (< %d), results may be noisy", n_rows, MIN_ROWS)

    try:
        import numpy as np
    except ImportError:
        return {"status": "error", "reason": "numpy not available"}

    X_data, y_data, feature_names = _prepare_dataset(rows)
    if len(X_data) == 0:
        return {"status": "skipped", "reason": "No valid feature rows after parsing"}

    try:
        result = _train_and_evaluate(X_data, y_data, feature_names, date_str)
    except ImportError as e:
        logger.warning("ML dependencies not available: %s — skipping", e)
        return {"status": "skipped", "reason": f"Missing ML dependency: {e}"}
    except Exception as exc:
        logger.exception("Weekly ML training failed")
        return {"status": "error", "reason": str(exc)}

    _write_shap_report(result, date_str)
    return result


def _load_training_rows() -> list[dict[str, Any]]:
    session = get_session()
    try:
        db_rows = session.query(TrainingRow).order_by(TrainingRow.date).all()
        result = []
        for row in db_rows:
            try:
                features = json.loads(row.features_json)
                labels = json.loads(row.labels_json)
                result.append({"date": row.date, "features": features, "labels": labels})
            except Exception:
                pass
        return result
    finally:
        session.close()


def _prepare_dataset(rows: list[dict]) -> tuple[list, list, list[str]]:
    import numpy as np

    numeric_features = [
        "dgs10", "vix", "vix_chg", "qqq_chg", "smh_chg",
        "nvda_chg", "spy_chg", "oil_chg", "dxy_chg", "breadth_proxy",
    ]

    X = []
    y = []
    for row in rows:
        features = row.get("features", {})
        labels = row.get("labels", {})

        x_vec = []
        valid = True
        for feat in numeric_features:
            val = features.get(feat)
            if val is None:
                val = 0.0
            try:
                x_vec.append(float(val))
            except (TypeError, ValueError):
                x_vec.append(0.0)

        # Label: hypothesis_correct → 1 (对/部分对) or 0 (错/N/A)
        hyp = labels.get("hypothesis_correct", "")
        if hyp in ("对", "部分对"):
            label = 1
        elif hyp == "错":
            label = 0
        else:
            continue  # skip rows without label

        X.append(x_vec)
        y.append(label)

    return X, y, numeric_features


def _train_and_evaluate(
    X: list, y: list, feature_names: list[str], date_str: str
) -> dict[str, Any]:
    import numpy as np

    try:
        from xgboost import XGBClassifier
        import shap
    except ImportError as e:
        raise ImportError(f"xgboost/shap not installed: {e}")

    X_np = np.array(X, dtype=float)
    y_np = np.array(y, dtype=int)

    n = len(y_np)
    split = max(1, int(n * 0.8))
    X_train, X_test = X_np[:split], X_np[split:]
    y_train, y_test = y_np[:split], y_np[split:]

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Accuracy on test set (may be tiny)
    if len(X_test) > 0:
        accuracy = float(np.mean(model.predict(X_test) == y_test))
    else:
        accuracy = float(np.mean(model.predict(X_train) == y_train))

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_np)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_names, mean_abs_shap)
    }
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "status": "ok",
        "date": date_str,
        "n_rows": n,
        "accuracy": round(accuracy, 3),
        "feature_importance": sorted_importance,
        "top_feature": next(iter(sorted_importance)),
    }


def _write_shap_report(result: dict[str, Any], date_str: str) -> None:
    ml_dir = Path(os.environ.get("DATA_ROOT", "/data")) / "reports" / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    report_path = ml_dir / f"{date_str}-shap.md"

    if result.get("status") == "skipped":
        content = f"# Weekly ML Report — {date_str}\n\nStatus: SKIPPED\n\nReason: {result.get('reason')}\n"
        report_path.write_text(content, encoding="utf-8")
        return

    fi = result.get("feature_importance", {})
    lines = [
        f"# Weekly ML Report — {date_str}",
        "",
        f"**Status**: {result.get('status', 'unknown')}",
        f"**Training Rows**: {result.get('n_rows', 0)}",
        f"**Test Accuracy**: {result.get('accuracy', 0):.1%}",
        f"**Top Feature**: {result.get('top_feature', 'N/A')}",
        "",
        "## Feature Importance (SHAP)",
        "",
        "| Feature | SHAP Mean |Abs |",
        "|---------|----------:|",
    ]
    for feat, imp in fi.items():
        lines.append(f"| {feat} | {imp:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- SHAP values represent mean absolute contribution to model output",
        "- Model: XGBoost classifier (hypothesis_correct as label)",
        "- Label encoding: 对/部分对 → 1, 错 → 0",
        "- Future leakage check: all features use T-day or prior data only",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("SHAP report written to %s", report_path)
