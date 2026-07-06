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
from src.utils.data_freshness import guard_fresh_raw
from src.utils.driver_match import driver_match_level
from src.utils.paths import morning_json_path, raw_data_path, step_json_path
from src.utils.scenario_match import assess_scenarios
from src.utils.trading_calendar import require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)


def run_step7_evening(trading_date: date | None = None) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step7_evening")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    date_str = trading_date.isoformat()
    logger.info("Step 7 Evening Review for %s", date_str)

    _, stale = guard_fresh_raw(trading_date, step="run_step7_evening")
    if stale:
        return stale

    features = build_features(trading_date)
    attr, actual_driver, surprise_hint, driver_splits = compute_attribution(trading_date, features)

    # Load morning hypothesis to check correctness
    morning: dict[str, Any] = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    morning_driver_p10 = ((morning.get("parts") or {}).get("P10") or {}).get("driver") or ""
    if not morning_driver_p10:
        morning_driver_p10 = ((morning.get("parts") or {}).get("P10") or {}).get("judgment", "")
    morning_driver_type = (
        ((morning.get("parts") or {}).get("P10") or {}).get("driver_type")
        or morning.get("driver_type")
    )
    morning_hypothesis = morning.get("hypothesis", {})
    hyp_id = morning_hypothesis.get("id", "") if isinstance(morning_hypothesis, dict) else ""
    hyp_statement = morning_hypothesis.get("statement", "") if isinstance(morning_hypothesis, dict) else ""

    hypothesis_correct = _assess_hypothesis(
        morning_driver=morning_driver_p10,
        actual_driver=actual_driver,
        morning_hypothesis=morning_hypothesis,
        morning_driver_type=morning_driver_type,
    )

    p15 = (morning.get("parts") or {}).get("P15") or {}
    raw: dict[str, Any] = {}
    raw_path = raw_data_path(date_str)
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    scenario_result = assess_scenarios(
        p15_judgment=p15.get("judgment") or "",
        p15_body=p15.get("body_md") or "",
        features=features,
        raw=raw,
    )
    scenario_primary = scenario_result["scenario_primary"]
    scenario_actual = scenario_result["scenario_actual"]
    scenario_correct = scenario_result["scenario_correct"]

    # Load S4 for trade decision outcome
    s4: dict[str, Any] = {}
    if step_json_path(4, date_str).exists():
        s4 = json.loads(step_json_path(4, date_str).read_text(encoding="utf-8"))
    should_trade = s4.get("should_trade", False)
    trade_instrument = s4.get("instrument")

    # Build attribution summary text
    split_txt = (
        " · ".join(f"{k} {v:.0%}" for k, v in driver_splits.items())
        if driver_splits
        else ""
    )
    if driver_splits:
        attr_summary = split_txt
    else:
        attr_summary = (
            f"AI {attr.ai:.0%} · Bond {attr.bond:.0%} · "
            f"Oil {attr.oil:.0%} · Macro {attr.macro:.0%} · Other {attr.other:.0%}"
        )

    lesson = _build_lesson(
        actual_driver=actual_driver,
        driver_splits=driver_splits,
        features=features,
        morning=morning,
        hypothesis_correct=hypothesis_correct,
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
        "### 7.2b Scenario (P15)",
        "",
        f"- Morning primary: **{scenario_primary or 'N/A'}**",
        f"- Actual played out: **{scenario_actual}**",
        f"- Scenario hit: **{'✓' if scenario_correct else '✗'}**",
        f"- Scores: {scenario_result.get('scenario_scores') or {}}",
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
        f"| Driver Splits | {split_txt if driver_splits else '—'} |",
        f"| Hypothesis | {hyp_id} · {hypothesis_correct} |",
        f"| Scenario (P15) | 预测 {scenario_primary or 'N/A'} → 实际 {scenario_actual} · {'命中' if scenario_correct else '未命中'} |",
        f"| Attribution | {attr_summary} |",
        f"| Surprise | {surprise_hint} |",
        f"| Trade Decision | {'Trade' if should_trade else 'No Trade'} |",
    ]

    scenario_hit = "命中" if scenario_correct else "未命中"
    judgment = (
        f"真正 Driver：{actual_driver} · "
        f"Attribution：AI {attr.ai:.0%} · Bond {attr.bond:.0%} · "
        f"Hypothesis：{hypothesis_correct} · "
        f"Scenario {scenario_primary or '?'}→{scenario_actual} {scenario_hit}"
    )
    one_liner = (
        f"{actual_driver} 主导；Scenario {scenario_primary or '?'}→{scenario_actual} {scenario_hit}"
    )

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
            "driver_splits": driver_splits,
            "hypothesis_correct": hypothesis_correct,
            "surprise": surprise_hint,
            "lesson": lesson,
            "scenario_primary": scenario_primary,
            "scenario_actual": scenario_actual,
            "scenario_correct": scenario_correct,
            "scenario_scores": scenario_result.get("scenario_scores"),
        },
    )

    # Update Market Case
    update_case(date_str, {
        "features": features.model_dump(),
        "attribution": attr.model_dump(),
        "labels": {
            "actual_driver": actual_driver,
            "agent_driver": morning_driver_p10 or None,
            "agent_driver_type": morning_driver_type,
            "driver_splits": driver_splits,
            "hypothesis_correct": hypothesis_correct,
            "scenario_primary": scenario_primary,
            "scenario_actual": scenario_actual,
            "scenario_correct": scenario_correct,
        },
        "surprise": surprise_hint,
        "lesson": lesson,
        "intraday": {
            **(load_case(date_str).intraday or {}),
            "s7": {"actual_driver": actual_driver, "attribution": attr.model_dump()},
        },
    })

    return payload


def _build_lesson(
    *,
    actual_driver: str,
    driver_splits: dict[str, float],
    features: Any,
    morning: dict[str, Any],
    hypothesis_correct: str,
) -> str:
    regime = (morning.get("r0") or {}).get("label") or "AI Expansion"
    smh = features.smh_chg
    qqq = features.qqq_chg
    parts: list[str] = []

    if driver_splits.get("NFP") and driver_splits.get("AI Chip Selloff"):
        parts.append(
            "当长期 AI Expansion Regime 与短期 Macro 利率利好（弱 NFP）冲突时，"
            "半导体风险仍可主导 Nasdaq；Dow/价值股可因利率改善走强"
        )
    elif "Chip" in actual_driver or "Semiconductor" in actual_driver:
        parts.append(
            "AI Expansion Regime 下，单日 SMH/NVDA 抛售可压过 Macro 利好，"
            "指数分化（Dow+ / QQQ-）比「大盘涨跌」更有信息量"
        )
    elif "NFP" in actual_driver:
        parts.append("NFP 日优先看 Macro Driver 链条：数据 → 利率 → 成长/价值轮动")

    if smh is not None and qqq is not None and smh <= -3 and qqq <= -0.5:
        parts.append(f"SMH {smh:.1f}% + QQQ {qqq:.1f}%：半导体拖累纳指")

    if hypothesis_correct == "错":
        parts.append("Morning Hypothesis 被 Chip/Macro 冲突推翻——复盘 P10/P15 权重")

    if not parts:
        parts.append(f"{regime} 背景下 {actual_driver} 主导；关注 Regime vs Driver 分离")

    return "；".join(parts)


def _assess_hypothesis(
    morning_driver: str,
    actual_driver: str,
    morning_hypothesis: dict | None,
    morning_driver_type: str | None = None,
) -> str:
    level = driver_match_level(
        morning_driver,
        actual_driver,
        morning_driver_type=morning_driver_type,
    )
    if level != "错":
        return level

    if morning_hypothesis and isinstance(morning_hypothesis, dict):
        statement = (morning_hypothesis.get("statement") or "").lower()
        actual_lower = (actual_driver or "").lower()
        if any(kw in statement for kw in actual_lower.replace("/", " ").split()):
            return "部分对"

    return "错"
