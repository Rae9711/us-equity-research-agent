"""S7 Attribution Engine — decomposes QQQ daily move into driver contributions.

Non-LLM. Uses regression-style heuristic:
- Collect returns of proxy instruments for each driver bucket
- Weight by correlation heuristic
- Normalize to sum = 1.0
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.schemas.market_case import AttributionModel, FeaturesModel
from src.utils.paths import raw_data_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)


def _safe_float(d: dict, *keys: str, default: float = 0.0) -> float:
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)  # type: ignore[assignment]
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _pct_chg(cur: float | None, prev: float | None) -> float:
    if cur is None or prev is None or prev == 0:
        return 0.0
    return (cur - prev) / abs(prev) * 100.0


def compute_attribution(
    trading_date: date | None = None,
    features: FeaturesModel | None = None,
) -> tuple[AttributionModel, str, str]:
    """
    Returns (attribution, actual_driver, surprise_hint).

    Attribution weights sum to 1.0.
    actual_driver: the bucket with largest share.
    surprise_hint: text description of biggest mismatch vs expected.
    """
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()

    # Pull raw scores for each bucket
    ai_score = _ai_driver_score(features) if features else 0.0
    bond_score = _bond_driver_score(features) if features else 0.0
    oil_score = _oil_driver_score(features) if features else 0.0
    macro_score = _macro_driver_score(features) if features else 0.0

    scores = {
        "ai": abs(ai_score),
        "bond": abs(bond_score),
        "oil": abs(oil_score),
        "macro": abs(macro_score),
    }
    total = sum(scores.values())

    if total < 0.001:
        # No signal — equal distribution
        attr = AttributionModel(ai=0.25, bond=0.25, oil=0.25, macro=0.15, other=0.10)
        return attr, "Unknown", "No clear driver signals detected"

    other_frac = 0.05
    scale = (1.0 - other_frac) / total
    ai_w = scores["ai"] * scale
    bond_w = scores["bond"] * scale
    oil_w = scores["oil"] * scale
    macro_w = scores["macro"] * scale

    attr = AttributionModel(
        ai=round(ai_w, 3),
        bond=round(bond_w, 3),
        oil=round(oil_w, 3),
        macro=round(macro_w, 3),
        other=round(other_frac, 3),
    )

    actual_driver = max(scores, key=lambda k: scores[k])
    driver_map = {"ai": "AI/Semiconductor", "bond": "Bond/Rates", "oil": "Oil/Geo", "macro": "Macro"}
    actual_driver_label = driver_map.get(actual_driver, actual_driver)

    surprise = _generate_surprise_hint(attr, features)

    return attr, actual_driver_label, surprise


def _ai_driver_score(f: FeaturesModel) -> float:
    score = 0.0
    if f.nvda_chg is not None:
        score += f.nvda_chg * 1.5
    if f.smh_chg is not None:
        score += f.smh_chg * 1.2
    if f.qqq_chg is not None and f.spy_chg is not None:
        # QQQ outperformance vs SPY is AI signal
        score += (f.qqq_chg - f.spy_chg) * 2.0
    return score


def _bond_driver_score(f: FeaturesModel) -> float:
    score = 0.0
    if f.dgs10 is not None:
        # High rates = negative for tech
        if f.dgs10 > 4.5:
            score += (f.dgs10 - 4.5) * 8.0
    return score


def _oil_driver_score(f: FeaturesModel) -> float:
    if f.oil_chg is not None:
        return abs(f.oil_chg) * 0.5
    return 0.0


def _macro_driver_score(f: FeaturesModel) -> float:
    score = 0.0
    if f.vix_chg is not None:
        score += abs(f.vix_chg) * 0.3
    if f.dxy_chg is not None:
        score += abs(f.dxy_chg) * 1.0
    return score


def _generate_surprise_hint(attr: AttributionModel, f: FeaturesModel | None) -> str:
    if f is None:
        return "No intraday data available for surprise assessment"

    surprises = []

    if attr.bond < 0.10 and f.dgs10 is not None and f.dgs10 > 4.6:
        surprises.append(f"Bond mattered less than expected despite 10Y={f.dgs10:.2f}%")

    if attr.ai > 0.60:
        surprises.append("AI/Semiconductor dominated more than typical day")

    if attr.oil < 0.05 and f.oil_chg is not None and abs(f.oil_chg) > 2.0:
        surprises.append(f"Oil moved {f.oil_chg:.1f}% but market largely ignored it")

    return "; ".join(surprises) if surprises else "No major surprises vs typical driver mix"
