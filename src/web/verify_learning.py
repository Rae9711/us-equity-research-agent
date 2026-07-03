"""Apply /verify human corrections to Market Case labels and refresh Step 8 learning.

Priority: human verify > S7 auto-labels for training (hypothesis, scenario, driver).
"""

from __future__ import annotations

import logging
import re
from datetime import date as date_type
from typing import Any

from src.db import ConclusionRecord
from src.db.market_case_service import load_case, save_case
from src.db.session import get_session
from src.web.verify_order import conclusion_sort_key

logger = logging.getLogger(__name__)

_VERIFY_LESSON_MARKER = "【核对纠正】"
_LEARNING_PART_IDS = frozenset({"P2", "P3", "P10", "P15", "P17", "S7"})

_DRIVER_PATTERNS: list[tuple[str, str]] = [
    (r"\bNFP\b|非农|nonfarm|employment situation", "NFP"),
    (r"\bFOMC\b|联储|fed meeting", "Fed"),
    (r"半导体|semiconductor|\bchip\b|SMH|NVDA|AI Chip", "AI Chip Selloff"),
    (r"bond|利率|10Y|yield", "Bond/Rates"),
    (r"oil|原油|XLE|energy", "Oil/Geo"),
    (r"macro|CPI|inflation|就业", "Macro"),
    (r"risk appetite|风险偏好|liquidity", "Risk Appetite"),
]


def _load_verified_records(trading_date: date_type) -> list[ConclusionRecord]:
    session = get_session()
    try:
        return (
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


def _get_conclusion(trading_date: date_type, part_id: str) -> ConclusionRecord | None:
    session = get_session()
    try:
        return (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == trading_date,
                ConclusionRecord.part_id == part_id,
            )
            .first()
        )
    finally:
        session.close()


def extract_driver_from_text(text: str | None) -> str | None:
    """Best-effort driver name from user judgment / notes."""
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    for pattern, driver in _DRIVER_PATTERNS:
        if re.search(pattern, t, re.I):
            return driver
    if len(t) <= 80 and "Agent" not in t[:20]:
        return t
    return None


def _format_lesson_line(date_str: str, record: ConclusionRecord) -> str:
    agent = (record.judgment or record.one_liner or "—").strip()
    user = (record.user_judgment or record.notes or "").strip()
    status = record.verification or "错"
    part = record.part_id
    if user:
        return f"{date_str}: {part} Agent 说「{agent[:100]}」→ 用户纠正：{user[:200]}"
    return f"{date_str}: {part} Agent 判断{status} — {agent[:120]}"


def build_verify_lesson_block(date_str: str, records: list[ConclusionRecord]) -> str | None:
    lines: list[str] = []
    for record in sorted(records, key=lambda r: conclusion_sort_key(r.part_id)):
        if record.part_id not in _LEARNING_PART_IDS:
            continue
        if record.verification not in ("错", "部分对"):
            continue
        lines.append(_format_lesson_line(date_str, record))
    if not lines:
        return None
    return _VERIFY_LESSON_MARKER + "\n" + "\n".join(lines)


def _strip_verify_lesson(lesson: str | None) -> str:
    if not lesson:
        return ""
    if _VERIFY_LESSON_MARKER not in lesson:
        return lesson.strip()
    before, after = lesson.split(_VERIFY_LESSON_MARKER, 1)
    rest = after.strip()
    if rest and "\n\n" in rest:
        _, remainder = rest.split("\n\n", 1)
        base = before.strip()
        if base and remainder.strip():
            return f"{base}\n\n{remainder.strip()}"
        return remainder.strip() or base
    return before.strip()


def apply_verify_to_market_case(trading_date: date_type) -> dict[str, Any]:
    """Merge /verify overrides into Market Case labels and lesson text."""
    date_str = trading_date.isoformat()
    records = _load_verified_records(trading_date)
    case = load_case(date_str)
    labels = case.labels.model_dump()
    applied: list[str] = []

    for record in records:
        ver = record.verification
        if not ver:
            continue
        if record.part_id == "P17":
            labels["hypothesis_correct"] = ver
            applied.append(f"P17→{ver}")
        elif record.part_id == "P15":
            labels["scenario_correct"] = ver == "对"
            applied.append(f"P15→{'命中' if ver == '对' else '未命中'}")
        elif record.part_id == "S7" and ver == "错":
            corrected = extract_driver_from_text(record.user_judgment or record.notes)
            if corrected:
                labels["actual_driver"] = corrected
                applied.append(f"S7 driver→{corrected}")

    verify_block = build_verify_lesson_block(date_str, records)
    base_lesson = _strip_verify_lesson(case.lesson)
    if verify_block:
        case.lesson = verify_block + (f"\n\n{base_lesson}" if base_lesson else "")
        applied.append("playbook_lesson")
    elif base_lesson != (case.lesson or "").strip():
        case.lesson = base_lesson or None

    from src.schemas.market_case import LabelsModel

    case.labels = LabelsModel.model_validate(labels)
    save_case(case)

    return {
        "date": date_str,
        "applied": applied,
        "verify_lesson": bool(verify_block),
        "labels": labels,
    }


def bayesian_driver_for_date(trading_date: date_type, case_actual_driver: str | None) -> tuple[str, bool]:
    """
    Resolve driver for Bayesian update and whether to run the update.

    When S7 is verified wrong without a corrected driver, skip Bayesian so we
    do not reinforce a bad S7 label.
    """
    driver = case_actual_driver or "Unknown"
    s7 = _get_conclusion(trading_date, "S7")
    if s7 and s7.verification == "错":
        corrected = extract_driver_from_text(s7.user_judgment or s7.notes)
        if corrected:
            return corrected, True
        logger.info(
            "S7 verified wrong on %s without corrected driver — skipping Bayesian",
            trading_date.isoformat(),
        )
        return driver, False
    return driver, True


def refresh_learning_from_verify(trading_date: date_type) -> dict[str, Any]:
    """Apply verify labels then re-run Step 8 learning for the date."""
    apply_result = apply_verify_to_market_case(trading_date)
    from src.jobs.step8_learning import run_step8_learning

    step8 = run_step8_learning(trading_date)
    return {
        "apply": apply_result,
        "step8_one_liner": (step8.get("conclusion") or {}).get("one_liner"),
        "refreshed": True,
    }
