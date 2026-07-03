"""P17 Hypothesis Engine — builds a verifiable hypothesis from Morning Research output.

Non-LLM logic derives the hypothesis statement and evidence from structured data.
The final statement phrasing may be polished by LLM (in llm layer), but the
evidence/counter lists and confidence are computed here.
"""

from __future__ import annotations

import logging
from datetime import date

from src.schemas.market_case import FeaturesModel, HypothesisModel, RegimeModel

logger = logging.getLogger(__name__)


def build_hypothesis(
    trading_date: date,
    features: FeaturesModel,
    regime: RegimeModel,
    morning_bias: str = "Neutral",
    morning_total: int = 0,
    morning_parts: dict | None = None,
) -> HypothesisModel:
    """
    Derive a verifiable P17 Hypothesis from Morning Research outputs.

    Rules (non-LLM):
    - Select primary driver based on regime + top-scoring morning parts
    - Build evidence list from feature values
    - Build counter_evidence from opposing signals
    - Compute confidence from morning_total + regime confidence
    """
    morning_parts = morning_parts or {}
    date_str = trading_date.isoformat()

    evidence: list[str] = []
    counter_evidence: list[str] = []

    primary_driver = _select_primary_driver(regime, features, morning_parts)
    statement = _build_statement(primary_driver, regime, morning_bias)

    evidence, counter_evidence = _collect_evidence(features, regime, primary_driver)

    confidence = _compute_confidence(
        regime.confidence, morning_total, len(evidence), len(counter_evidence)
    )

    hyp_id = f"H-{date_str}-001"

    return HypothesisModel(
        id=hyp_id,
        statement=statement,
        evidence=evidence,
        counter_evidence=counter_evidence,
        confidence=round(confidence, 2),
        status="待验证",
    )


def _select_primary_driver(
    regime: RegimeModel,
    features: FeaturesModel,
    morning_parts: dict,
) -> str:
    """Pick the single primary driver for today's hypothesis."""
    p10 = (morning_parts.get("P10") or {}).get("judgment", "")
    if "NFP" in p10:
        return "NFP"
    if "Chip" in p10 or "Semiconductor" in p10:
        return "AI Chip Selloff"

    p13 = (morning_parts.get("P13") or {}).get("catalysts") or []
    for c in p13:
        if c.get("name") == "NFP":
            return "NFP"

    p3 = (morning_parts.get("P3") or {}).get("judgment", "")
    if "Employment" in p3 or "Nonfarm" in p3 or "NFP" in p3 or "ADP" in p3:
        return "NFP"
    if "FOMC" in p3 and "FOMC" in p10:
        return "Fed"
    if "CPI" in p3 or "PCE" in p3:
        return "Inflation"

    if regime.label == "AI Expansion":
        if features.smh_chg is not None and features.smh_chg <= -3:
            return "AI Chip Selloff"
        return "AI/Semiconductor"
    if regime.label == "Macro Fear":
        return "Macro"
    if regime.label == "Liquidity Driven":
        return "Risk Appetite"
    return "Macro"


def _build_statement(driver: str, regime: RegimeModel, bias: str) -> str:
    if driver == "NFP":
        return "弱 NFP 能否抵消 AI 抛售？（Macro 利率利好 vs 半导体风险）"
    if driver == "AI Chip Selloff":
        return "AI Chip Selloff dominates today despite macro tailwinds?"
    if driver == "AI/Semiconductor":
        return "AI/Semiconductor leads index today; macro is secondary variable"
    if driver == "Employment":
        return f"Employment data is today's primary market driver under {regime.label} regime"
    if driver == "Fed":
        return f"Fed policy signal dominates price action today"
    if driver == "Risk Appetite":
        return f"Broad risk appetite shift drives cross-asset moves today"
    return f"{driver} is today's primary market driver"


def _collect_evidence(
    features: FeaturesModel,
    regime: RegimeModel,
    driver: str,
) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    counter: list[str] = []

    # VIX signal
    if features.vix is not None:
        if features.vix < 18:
            evidence.append(f"VIX={features.vix:.1f} (low fear)")
        elif features.vix > 25:
            counter.append(f"VIX={features.vix:.1f} (elevated fear)")

    if features.vix_chg is not None:
        if features.vix_chg < -3:
            evidence.append(f"VIX↓ {features.vix_chg:.1f}% (risk-on)")
        elif features.vix_chg > 5:
            counter.append(f"VIX↑ {features.vix_chg:.1f}% (risk-off pressure)")

    # NVDA / SMH for AI regime
    if driver == "AI/Semiconductor":
        if features.nvda_chg is not None:
            if features.nvda_chg >= 1.0:
                evidence.append(f"NVDA +{features.nvda_chg:.1f}%")
            elif features.nvda_chg <= -1.0:
                counter.append(f"NVDA {features.nvda_chg:.1f}%")
        if features.smh_chg is not None:
            if features.smh_chg >= 1.0:
                evidence.append(f"SMH +{features.smh_chg:.1f}%")
            elif features.smh_chg <= -1.0:
                counter.append(f"SMH {features.smh_chg:.1f}%")

    # Bond signal
    if features.dgs10 is not None:
        if features.dgs10 > 4.7:
            counter.append(f"10Y={features.dgs10:.2f}% (elevated rates)")
        elif features.dgs10 < 4.2:
            evidence.append(f"10Y={features.dgs10:.2f}% (supportive rates)")

    # DXY
    if features.dxy_chg is not None:
        if features.dxy_chg > 0.3:
            counter.append(f"DXY↑ {features.dxy_chg:.2f}% (dollar strength headwind)")
        elif features.dxy_chg < -0.3:
            evidence.append(f"DXY↓ {features.dxy_chg:.2f}% (dollar weakness tailwind)")

    # Breadth
    if features.breadth_proxy is not None:
        if features.breadth_proxy >= 0.7:
            evidence.append(f"Breadth {features.breadth_proxy:.0%} (broad participation)")
        elif features.breadth_proxy <= 0.35:
            counter.append(f"Breadth {features.breadth_proxy:.0%} (narrow market)")

    # If no evidence collected, add regime label as generic evidence
    if not evidence:
        evidence.append(f"Regime={regime.label} (conf={regime.confidence:.0%})")

    return evidence, counter


def _compute_confidence(
    regime_conf: float,
    morning_total: int,
    n_evidence: int,
    n_counter: int,
) -> float:
    base = regime_conf * 0.5
    score_contrib = min(0.25, max(-0.15, morning_total * 0.025))
    evidence_ratio = n_evidence / max(1, n_evidence + n_counter)
    evidence_contrib = (evidence_ratio - 0.5) * 0.30
    return min(0.92, max(0.35, base + score_contrib + evidence_contrib + 0.25))
