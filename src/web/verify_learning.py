"""Apply /verify human corrections to Market Case labels and refresh Step 8 learning.

Priority: human verify > S7 auto-labels for training (hypothesis, scenario, driver).
"""

from __future__ import annotations

import logging
import re
from datetime import date as date_type
from typing import Any, Literal

from src.db import ConclusionRecord
from src.db.market_case_service import load_case, save_case
from src.db.session import get_session
from src.web.verify_order import conclusion_sort_key

logger = logging.getLogger(__name__)

_VERIFY_LESSON_MARKER = "【核对纠正】"
_LEARNING_PART_IDS = frozenset({"P2", "P3", "P10", "P15", "P17", "S7"})

BayesianMode = Literal["confirm", "downweight", "skip"]

_DRIVER_PATTERNS: list[tuple[str, str]] = [
    (r"\bNFP\b|非农|nonfarm|employment situation", "NFP"),
    (r"\bFOMC\b|联储|fed meeting", "Fed"),
    (r"半导体|semiconductor|\bchip\b|SMH|NVDA|AI Chip", "AI Chip Selloff"),
    (r"bond|利率|10Y|yield", "Bond/Rates"),
    (r"oil|原油|XLE|energy", "Oil/Geo"),
    (r"macro|CPI|inflation|就业", "Macro"),
    (r"risk appetite|风险偏好|liquidity", "Risk Appetite"),
]

_COMPOSITE_SEP = re.compile(r"\s*[+＋/、,，]\s*")


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


def _match_driver_token(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None
    for pattern, driver in _DRIVER_PATTERNS:
        if re.search(pattern, t, re.I):
            return driver
    if len(t) <= 80 and "Agent" not in t[:20]:
        return t
    return None


def extract_drivers_from_text(text: str | None) -> list[str]:
    """Parse one or more driver names from user judgment / notes."""
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
    parts = _COMPOSITE_SEP.split(raw) if _COMPOSITE_SEP.search(raw) else [raw]
    seen: set[str] = set()
    drivers: list[str] = []
    for part in parts:
        matched = _match_driver_token(part)
        if matched and matched not in seen:
            seen.add(matched)
            drivers.append(matched)
    if not drivers:
        single = _match_driver_token(raw)
        if single:
            drivers.append(single)
    return drivers


def extract_driver_from_text(text: str | None) -> str | None:
    """Best-effort single driver string (composite joined with ' + ')."""
    drivers = extract_drivers_from_text(text)
    if not drivers:
        return None
    if len(drivers) == 1:
        return drivers[0]
    return " + ".join(drivers)


def _p10_agent_text(record: ConclusionRecord | None) -> str:
    if not record:
        return ""
    return (record.judgment or record.one_liner or "").strip()


def _morning_p10_driver(date_str: str) -> str:
    from src.web.homepage import load_morning

    morning = load_morning(date_str)
    if not morning:
        return ""
    p10 = (morning.get("parts") or {}).get("P10") or {}
    return (p10.get("judgment") or p10.get("one_liner") or "").strip()


def _driver_hint_texts(trading_date: date_type, case_lesson: str | None) -> list[str]:
    """Ordered fallback texts for inferring corrected driver after S7 wrong."""
    texts: list[str] = []
    for part_id in ("S7", "P10", "P17", "P15"):
        rec = _get_conclusion(trading_date, part_id)
        if not rec:
            continue
        for field in (rec.user_judgment, rec.notes):
            if field and str(field).strip():
                texts.append(str(field).strip())
    if case_lesson and _VERIFY_LESSON_MARKER in case_lesson:
        texts.append(case_lesson)
    return texts


def _user_verified_learning_parts(trading_date: date_type) -> bool:
    for record in _load_verified_records(trading_date):
        if record.part_id in _LEARNING_PART_IDS:
            return True
    return False


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

    p10_record = next((r for r in records if r.part_id == "P10"), None)
    morning_driver = _morning_p10_driver(date_str) or _p10_agent_text(p10_record)
    if morning_driver:
        labels["agent_driver"] = morning_driver

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
                labels["corrected_driver_from_verify"] = corrected
                applied.append(f"S7 driver→{corrected}")
        elif record.part_id == "P10" and ver in ("错", "部分对"):
            corrected = extract_driver_from_text(record.user_judgment or record.notes)
            if corrected:
                labels["corrected_driver_from_verify"] = corrected
                labels["actual_driver"] = corrected
                applied.append(f"P10 driver→{corrected}")
            elif morning_driver:
                labels.setdefault("agent_driver", morning_driver)

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


def bayesian_driver_for_date(
    trading_date: date_type,
    case_actual_driver: str | None,
) -> tuple[str, bool, BayesianMode]:
    """
    Resolve driver for Bayesian update, whether to run, and update mode.

    When S7 is verified wrong without a corrected driver on S7 itself, fall back to
    P10 verify judgment, other user notes, and playbook lesson text. If still no
    driver, use P10「部分对」or any learning-part verify to down-weight the morning
    driver instead of skipping entirely.
    """
    driver = case_actual_driver or "Unknown"
    date_str = trading_date.isoformat()
    s7 = _get_conclusion(trading_date, "S7")
    p10 = _get_conclusion(trading_date, "P10")

    if s7 and s7.verification == "错":
        case = load_case(date_str)
        corrected_label = (
            case.labels.corrected_driver_from_verify if case.labels else None
        )
        if corrected_label:
            return corrected_label, True, "confirm"

        for text in _driver_hint_texts(trading_date, case.lesson):
            corrected = extract_driver_from_text(text)
            if corrected:
                return corrected, True, "confirm"

        morning = (
            _morning_p10_driver(date_str)
            or _p10_agent_text(p10)
            or (case.labels.agent_driver if case.labels else None)
            or ""
        ).strip()

        if p10 and p10.verification == "部分对" and morning:
            logger.info(
                "S7 wrong on %s — P10 部分对, down-weighting morning driver",
                date_str,
            )
            return morning, True, "downweight"

        if _user_verified_learning_parts(trading_date) and morning:
            logger.info(
                "S7 wrong on %s — user verified, down-weighting morning driver",
                date_str,
            )
            return morning, True, "downweight"

        logger.info(
            "S7 verified wrong on %s without driver hint — skipping Bayesian confirm",
            date_str,
        )
        return driver, False, "skip"

    return driver, True, "confirm"


def reference_driver_for_agent_quality(
    trading_date: date_type,
    date_str: str,
    p10_verification: str | None,
    case_labels: dict[str, Any],
    s7_actual: str | None,
) -> str | None:
    """Ground truth for Agent Driver quality: verify-corrected or S7 actual."""
    if p10_verification in ("错", "部分对"):
        corrected = case_labels.get("corrected_driver_from_verify")
        if corrected:
            return corrected
        p10 = _get_conclusion(trading_date, "P10")
        if p10:
            corrected = extract_driver_from_text(p10.user_judgment or p10.notes)
            if corrected:
                return corrected
    if p10_verification == "对":
        return s7_actual
    return s7_actual


def refresh_learning_from_verify(trading_date: date_type) -> dict[str, Any]:
    """Apply verify labels then re-run Step 8 learning for the date."""
    apply_result = apply_verify_to_market_case(trading_date)
    from src.jobs.step8_learning import run_step8_learning

    step8 = run_step8_learning(trading_date)

    ml_result: dict[str, Any] = {"status": "skipped"}
    try:
        from src.engines.ml.retrain import maybe_retrain_after_verify_async

        ml_result = maybe_retrain_after_verify_async(trading_date)
    except Exception as exc:
        logger.warning("ML retrain after verify failed for %s: %s", trading_date, exc)

    return {
        "apply": apply_result,
        "step8_one_liner": (step8.get("conclusion") or {}).get("one_liner"),
        "ml_retrain": ml_result,
        "refreshed": True,
    }
