"""Step 7 — Evening Review (4:10 PM ET).

Attribution Engine + Decision Log + Market Case finalization.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.db.market_case_service import load_case, save_case, update_case
from src.engines.attribution import compute_attribution
from src.features.build import build_features
from src.steps.base import save_step_result
from src.utils.paths import morning_json_path, step_json_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)


def run_step7_evening(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Step 7 Evening Review for %s", date_str)

    features = build_features(trading_date)
    attr, actual_driver, surprise_hint = compute_attribution(trading_date, features)

    # Load morning hypothesis to check correctness
    morning: dict[str, Any] = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    morning_driver_p10 = ((morning.get("parts") or {}).get("P10") or {}).get("judgment", "")
    morning_hypothesis = morning.get("hypothesis", {})
    hyp_id = morning_hypothesis.get("id", "") if isinstance(morning_hypothesis, dict) else ""
    hyp_statement = morning_hypothesis.get("statement", "") if isinstance(morning_hypothesis, dict) else ""

    hypothesis_correct = _assess_hypothesis(
        morning_driver=morning_driver_p10,
        actual_driver=actual_driver,
        morning_hypothesis=morning_hypothesis,
    )

    # Load S4 for trade decision outcome
    s4: dict[str, Any] = {}
    if step_json_path(4, date_str).exists():
        s4 = json.loads(step_json_path(4, date_str).read_text(encoding="utf-8"))
    should_trade = s4.get("should_trade", False)
    trade_instrument = s4.get("instrument")

    # Build attribution summary text
    attr_summary = (
        f"AI {attr.ai:.0%} · Bond {attr.bond:.0%} · "
        f"Oil {attr.oil:.0%} · Macro {attr.macro:.0%} · Other {attr.other:.0%}"
    )

    body_lines = [
        "## Step 7 — Evening Review (Attribution Engine)",
        "",
        "### 7.1 Outcome Attribution",
        "",
        f"| Driver | Contribution |",
        f"|--------|-------------:|",
        f"| AI / Semiconductor | {attr.ai:.0%} |",
        f"| Bond / Rates | {attr.bond:.0%} |",
        f"| Oil / Geo | {attr.oil:.0%} |",
        f"| Macro Data | {attr.macro:.0%} |",
        f"| Other | {attr.other:.0%} |",
        "",
        "### 7.2 True Driver",
        "",
        f"- Morning/Hypothesis: {morning_driver_p10 or 'N/A'}",
        f"- **Actual: {actual_driver}**",
        f"- Hypothesis result: **{hypothesis_correct}**",
        "",
        "### 7.3 Surprise",
        "",
        f"- {surprise_hint}",
        "",
        "### 7.4 Decision Log",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Today's Driver | {actual_driver} |",
        f"| Hypothesis | {hyp_id} · {hypothesis_correct} |",
        f"| Attribution | {attr_summary} |",
        f"| Surprise | {surprise_hint} |",
        f"| Trade Decision | {'Trade' if should_trade else 'No Trade'} |",
    ]

    judgment = (
        f"真正 Driver：{actual_driver} · "
        f"Attribution：AI {attr.ai:.0%} · Bond {attr.bond:.0%} · "
        f"Hypothesis：{hypothesis_correct}"
    )
    one_liner = f"{actual_driver} 主导；Hypothesis {hypothesis_correct}"

    conclusion = {
        "part_id": "S7",
        "judgment": judgment,
        "confidence": None,
        "one_liner": one_liner[:256],
    }

    payload = save_step_result(
        7,
        trading_date,
        step_id="S7",
        job_id="evening_review",
        conclusion=conclusion,
        body_md="\n".join(body_lines),
        extra={
            "attribution": attr.model_dump(),
            "actual_driver": actual_driver,
            "hypothesis_correct": hypothesis_correct,
            "surprise": surprise_hint,
        },
    )

    # Update Market Case
    update_case(date_str, {
        "attribution": attr.model_dump(),
        "labels": {
            "actual_driver": actual_driver,
            "hypothesis_correct": hypothesis_correct,
        },
        "surprise": surprise_hint,
        "intraday": {
            **(load_case(date_str).intraday or {}),
            "s7": {"actual_driver": actual_driver, "attribution": attr.model_dump()},
        },
    })

    return payload


def _assess_hypothesis(
    morning_driver: str,
    actual_driver: str,
    morning_hypothesis: dict | None,
) -> str:
    if not morning_driver:
        return "N/A"

    if actual_driver == "Unknown":
        return "N/A"

    morning_driver_lower = morning_driver.lower()
    actual_lower = actual_driver.lower()

    # Direct match
    if any(kw in morning_driver_lower for kw in actual_lower.split("/")):
        return "对"

    # Partial match (e.g., morning said AI, actual was AI/Semi)
    overlap_keywords = ["ai", "semiconductor", "macro", "employment", "bond", "fed", "risk"]
    for kw in overlap_keywords:
        if kw in morning_driver_lower and kw in actual_lower:
            return "部分对"

    # Regime-level partial match
    if morning_hypothesis and isinstance(morning_hypothesis, dict):
        statement = (morning_hypothesis.get("statement") or "").lower()
        if any(kw in statement for kw in actual_lower.split("/")):
            return "部分对"

    return "错"
