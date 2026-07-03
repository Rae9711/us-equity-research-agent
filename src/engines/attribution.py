"""S7 Attribution Engine — decomposes QQQ daily move into driver contributions.

Non-LLM. Supports split attribution when multiple drivers contributed (e.g. NFP 40% + Chip 60%).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.schemas.market_case import AttributionModel, FeaturesModel
from src.utils.paths import morning_json_path, raw_data_path
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


def _load_morning_driver(date_str: str) -> str:
    path = morning_json_path(date_str)
    if not path.exists():
        return ""
    try:
        morning = json.loads(path.read_text(encoding="utf-8"))
        return ((morning.get("parts") or {}).get("P10") or {}).get("judgment", "")
    except Exception:
        return ""


def _macro_release_signal(date_str: str) -> float:
    from src.utils.trading_calendar import employment_situation_date

    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        return 0.0

    if employment_situation_date(d.year, d.month) == d:
        return 1.0

    path = raw_data_path(date_str)
    if not path.exists():
        return 0.0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    macro = raw.get("macro") or {}
    if macro.get("nfp_release_today"):
        return 1.0
    from src.research.rules import catalysts_on_date

    calendar = macro.get("economic_calendar") or []
    catalysts = catalysts_on_date(calendar, date_str)
    if any(c.get("name") == "NFP" for c in catalysts):
        return 1.0
    if catalysts:
        return 0.85
    return 0.0


def compute_attribution(
    trading_date: date | None = None,
    features: FeaturesModel | None = None,
) -> tuple[AttributionModel, str, str, dict[str, float]]:
    """
    Returns (attribution, actual_driver, surprise_hint, driver_splits).

    When multiple drivers are active, driver_splits sums to ~1.0 with named keys.
    """
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()

    ai_score = abs(_ai_driver_score(features)) if features else 0.0
    bond_score = abs(_bond_driver_score(features)) if features else 0.0
    oil_score = abs(_oil_driver_score(features)) if features else 0.0
    macro_score = abs(_macro_driver_score(features)) if features else 0.0

    macro_release = _macro_release_signal(date_str)
    if macro_release > 0:
        macro_score = max(macro_score, macro_release * 3.0)

    chip_active = bool(
        features
        and ai_score > 1.5
        and (
            (features.smh_chg is not None and features.smh_chg <= -2.0)
            or (features.nvda_chg is not None and features.nvda_chg <= -2.0)
        )
    )
    nfp_active = macro_release >= 0.85 or "nfp" in _load_morning_driver(date_str).lower()
    if not nfp_active and trading_date:
        from src.utils.trading_calendar import employment_situation_date

        nfp_active = employment_situation_date(trading_date.year, trading_date.month) == trading_date

    named: dict[str, float] = {}
    if nfp_active and chip_active:
        nfp_w = 0.40
        chip_w = 0.60
        named = {"NFP": nfp_w, "AI Chip Selloff": chip_w}
        macro_w = nfp_w
        ai_w = chip_w
        bond_w = bond_score * 0.05
        oil_w = oil_score * 0.05
        other_frac = 0.0
        actual_driver = "NFP + AI Chip Selloff"
    else:
        scores = {
            "ai": ai_score,
            "bond": bond_score,
            "oil": oil_score,
            "macro": macro_score,
        }
        total = sum(scores.values())
        if total < 0.001:
            attr = AttributionModel(
                ai=0.25, bond=0.25, oil=0.25, macro=0.15, other=0.10, splits={}
            )
            return attr, "Unknown", "No clear driver signals detected", {}

        other_frac = 0.05
        scale = (1.0 - other_frac) / total
        ai_w = scores["ai"] * scale
        bond_w = scores["bond"] * scale
        oil_w = scores["oil"] * scale
        macro_w = scores["macro"] * scale
        top = max(scores, key=lambda k: scores[k])
        driver_map = {
            "ai": "AI/Semiconductor",
            "bond": "Bond/Rates",
            "oil": "Oil/Geo",
            "macro": "Macro",
        }
        actual_driver = driver_map.get(top, top)
        if nfp_active:
            actual_driver = "NFP"
            named = {"NFP": macro_w + ai_w * 0.1}
        elif chip_active:
            actual_driver = "AI Chip Selloff"
            named = {"AI Chip Selloff": ai_w + macro_w * 0.1}

    attr = AttributionModel(
        ai=round(ai_w, 3),
        bond=round(bond_w, 3),
        oil=round(oil_w, 3),
        macro=round(macro_w, 3),
        other=round(other_frac, 3),
        splits={k: round(v, 3) for k, v in named.items()},
    )

    surprise = _generate_surprise_hint(attr, features, named)
    return attr, actual_driver, surprise, named


def _ai_driver_score(f: FeaturesModel) -> float:
    score = 0.0
    if f.nvda_chg is not None:
        score += f.nvda_chg * 1.5
    if f.smh_chg is not None:
        score += f.smh_chg * 1.2
    if f.qqq_chg is not None and f.spy_chg is not None:
        score += (f.qqq_chg - f.spy_chg) * 2.0
    return score


def _bond_driver_score(f: FeaturesModel) -> float:
    score = 0.0
    if f.dgs10 is not None and f.dgs10 > 4.5:
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


def _generate_surprise_hint(
    attr: AttributionModel,
    f: FeaturesModel | None,
    splits: dict[str, float],
) -> str:
    if f is None:
        return "No intraday data available for surprise assessment"

    surprises = []
    if splits.get("NFP") and splits.get("AI Chip Selloff"):
        surprises.append(
            f"Split attribution: NFP {splits['NFP']:.0%} + Chip Selloff {splits['AI Chip Selloff']:.0%}"
        )

    if attr.bond < 0.10 and f.dgs10 is not None and f.dgs10 > 4.6:
        surprises.append(f"Bond mattered less than expected despite 10Y={f.dgs10:.2f}%")

    if attr.ai > 0.60:
        surprises.append("AI/Semiconductor dominated more than typical day")

    if attr.oil < 0.05 and f.oil_chg is not None and abs(f.oil_chg) > 2.0:
        surprises.append(f"Oil moved {f.oil_chg:.1f}% but market largely ignored it")

    return "; ".join(surprises) if surprises else "No major surprises vs typical driver mix"
