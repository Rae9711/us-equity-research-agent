"""Calibration Engine — estimates Expected Value and Risk/Reward for S4.

Uses historical training rows + verify outcomes to calibrate confidence.
Non-LLM. Falls back to simple heuristic when insufficient history.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.db import TrainingRow, get_session

logger = logging.getLogger(__name__)

_MIN_HISTORY = 5  # minimum rows needed for statistical calibration


def compute_ev(
    morning_total: int,
    confidence: float,
    regime_label: str,
    bayesian_ai_weight: float = 0.35,
) -> dict[str, Any]:
    """
    Compute Expected Return, Expected Loss, and Risk/Reward ratio.

    Uses historical win rate from training rows if available, else heuristic.
    """
    hist_win_rate = _historical_win_rate(regime_label)

    if hist_win_rate is not None:
        win_rate = hist_win_rate
    else:
        # Heuristic: map total score → win_rate
        win_rate = _heuristic_win_rate(morning_total, confidence, bayesian_ai_weight)

    # Parameterize expected move based on regime
    regime_factor = _regime_factor(regime_label)
    expected_return = round(win_rate * regime_factor * 4.5 / 100, 4)  # as decimal
    expected_loss = round((1 - win_rate) * regime_factor * 2.0 / 100, 4)

    risk_reward = (
        round(expected_return / expected_loss, 2) if expected_loss > 0 else 99.0
    )

    return {
        "expected_return": expected_return,
        "expected_loss": -abs(expected_loss),
        "risk_reward": risk_reward,
        "win_rate_estimate": round(win_rate, 3),
        "calibration_source": "historical" if hist_win_rate is not None else "heuristic",
    }


def _heuristic_win_rate(total: int, confidence: float, ai_weight: float) -> float:
    base = 0.45
    score_contribution = min(0.20, max(-0.15, total * 0.025))
    confidence_contribution = (confidence - 0.5) * 0.20
    ai_contribution = (ai_weight - 0.35) * 0.30
    return min(0.80, max(0.25, base + score_contribution + confidence_contribution + ai_contribution))


def _regime_factor(regime: str) -> float:
    mapping = {
        "AI Expansion": 1.3,
        "Macro Fear": 0.7,
        "Liquidity Driven": 1.1,
        "Range": 0.8,
    }
    return mapping.get(regime, 1.0)


def _historical_win_rate(regime_label: str) -> float | None:
    """Estimate win rate from historical training rows with actual_driver labels."""
    session = get_session()
    try:
        rows = session.query(TrainingRow).all()
    finally:
        session.close()

    if len(rows) < _MIN_HISTORY:
        return None

    correct = 0
    total = 0
    for row in rows:
        try:
            labels = json.loads(row.labels_json)
        except Exception:
            continue
        hyp = labels.get("hypothesis_correct")
        if hyp in ("对", "部分对"):
            correct += 1
            total += 1
        elif hyp == "错":
            total += 1

    if total < _MIN_HISTORY:
        return None

    return correct / total
