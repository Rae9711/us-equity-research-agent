"""R0 Regime Engine — rule-based (non-LLM) market regime classifier.

Input:  FeaturesModel
Output: RegimeModel (label + confidence)

Rules loaded from config/rules/regime.yaml. First-match priority.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.schemas.market_case import FeaturesModel, RegimeModel

logger = logging.getLogger(__name__)

_RULES_PATH = Path(os.environ.get("REGIME_RULES_PATH", "/workspace/config/rules/regime.yaml"))
if not _RULES_PATH.exists():
    _RULES_PATH = Path(__file__).parent.parent.parent / "config" / "rules" / "regime.yaml"


def _load_rules() -> dict[str, Any]:
    if _RULES_PATH.exists():
        with open(_RULES_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    logger.warning("Regime rules file not found at %s", _RULES_PATH)
    return {}


def _check_ai_expansion(f: FeaturesModel, cond: dict) -> bool:
    smh_gt_qqq = False
    if f.smh_chg is not None and f.qqq_chg is not None:
        smh_gt_qqq = f.smh_chg > f.qqq_chg

    nvda_ok = False
    nvda_gte = cond.get("any", [{}])[1].get("nvda_chg_gte") if len(cond.get("any", [])) > 1 else None
    if nvda_gte is not None and f.nvda_chg is not None:
        nvda_ok = f.nvda_chg >= nvda_gte

    any_ok = smh_gt_qqq or nvda_ok

    vix_ok = True
    vix_lte = cond.get("vix_lte")
    if vix_lte is not None and f.vix is not None:
        vix_ok = f.vix <= vix_lte

    dgs_ok = True
    dgs_lte = cond.get("dgs10_lte")
    if dgs_lte is not None and f.dgs10 is not None:
        dgs_ok = f.dgs10 <= dgs_lte

    return any_ok and vix_ok and dgs_ok


def _check_macro_fear(f: FeaturesModel, cond: dict) -> bool:
    vix_ok = False
    dgs_ok = False
    for item in cond.get("any", []):
        if "vix_gte" in item and f.vix is not None:
            vix_ok = f.vix >= item["vix_gte"]
        if "dgs10_gte" in item and f.dgs10 is not None:
            dgs_ok = f.dgs10 >= item["dgs10_gte"]

    any_ok = vix_ok or dgs_ok

    qqq_ok = True
    qqq_lte = cond.get("qqq_chg_lte")
    if qqq_lte is not None and f.qqq_chg is not None:
        qqq_ok = f.qqq_chg <= qqq_lte

    return any_ok and qqq_ok


def _check_liquidity(f: FeaturesModel, cond: dict) -> bool:
    breadth_ok = True
    bp_gte = cond.get("breadth_proxy_gte")
    if bp_gte is not None and f.breadth_proxy is not None:
        breadth_ok = f.breadth_proxy >= bp_gte
    elif bp_gte is not None:
        breadth_ok = False

    qqq_ok = True
    qqq_gte = cond.get("qqq_chg_gte")
    if qqq_gte is not None and f.qqq_chg is not None:
        qqq_ok = f.qqq_chg >= qqq_gte

    return breadth_ok and qqq_ok


_RULE_CHECKERS = {
    "AI Expansion": _check_ai_expansion,
    "Macro Fear": _check_macro_fear,
    "Liquidity Driven": _check_liquidity,
}


def classify_regime(features: FeaturesModel) -> RegimeModel:
    """Classify market regime using rule-based logic. No LLM involved."""
    rules = _load_rules()
    regimes_cfg = rules.get("regimes", [])
    default_label = rules.get("default", "Range")
    default_conf = rules.get("default_confidence", 0.50)

    for regime_def in regimes_cfg:
        label = regime_def.get("label", "")
        confidence = regime_def.get("confidence", 0.60)
        conditions = regime_def.get("conditions", {})

        checker = _RULE_CHECKERS.get(label)
        if checker is None:
            continue

        try:
            if checker(features, conditions):
                logger.info("R0 Regime: %s (conf=%.2f)", label, confidence)
                return RegimeModel(label=label, confidence=confidence)
        except Exception as exc:
            logger.warning("Regime rule check failed for %s: %s", label, exc)

    logger.info("R0 Regime: %s (default, conf=%.2f)", default_label, default_conf)
    return RegimeModel(label=default_label, confidence=default_conf)
