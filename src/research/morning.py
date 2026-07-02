from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from pytz import timezone

from src.db import ConclusionRecord, DailyRun
from src.db.session import get_session
from src.llm.anthropic_client import AnthropicClient
from src.research.context import build_research_context
from src.research.prompts import MORNING_SYSTEM
from src.research.parts_meta import PART_ORDER
from src.research.report import render_morning_report
from src.research.rules import compute_rule_parts
from src.utils.paths import morning_json_path, morning_report_path, raw_data_path
from src.utils.trading_calendar import ET, today_et

logger = logging.getLogger(__name__)

RULE_PART_IDS = {"P4", "P5", "P6", "P7", "P9", "P11", "P13"}


def _load_raw(trading_date: date) -> dict[str, Any]:
    path = raw_data_path(trading_date.isoformat())
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_parts(
    llm_parts: dict[str, Any], rule_parts: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for pid in PART_ORDER:
        llm_p = (llm_parts.get(pid) or {}) if llm_parts else {}
        rule_p = (rule_parts.get(pid) or {}) if rule_parts else {}
        if pid in RULE_PART_IDS and rule_p:
            merged[pid] = {**llm_p, **rule_p}
        else:
            merged[pid] = llm_p or rule_p
    return merged


def _call_llm(context: dict[str, Any], rule_bundle: dict[str, Any]) -> dict[str, Any]:
    client = AnthropicClient()
    user_prompt = (
        "请完成今日 Morning Research。\n\n"
        f"## research_context\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"## rule_parts（勿改 judgment/分数）\n"
        f"{json.dumps(rule_bundle.get('parts', {}), ensure_ascii=False, indent=2)}"
    )
    return client.complete_json(MORNING_SYSTEM, user_prompt)


def run_morning_research(
    trading_date: date | None = None,
    *,
    skip_llm: bool = False,
) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Morning research starting for %s", date_str)

    raw = _load_raw(trading_date)
    if not raw.get("data_ready"):
        logger.warning("Raw data not fully ready: %s", raw.get("missing"))

    context = build_research_context(raw)
    rule_bundle = compute_rule_parts(raw)

    llm_parts: dict[str, Any] = {}
    if not skip_llm:
        try:
            llm_response = _call_llm(context, rule_bundle)
            llm_parts = llm_response.get("parts") or {}
        except Exception:
            logger.exception("LLM morning research failed; using rule parts only")
            skip_llm = True

    parts = _merge_parts(llm_parts, rule_bundle.get("parts") or {})
    if skip_llm and not llm_parts:
        for pid in PART_ORDER:
            if pid not in parts:
                parts[pid] = {
                    "judgment": "N/A",
                    "confidence": None,
                    "one_liner": "LLM 未运行或失败",
                    "body_md": "",
                }

    payload: dict[str, Any] = {
        "generated_at": datetime.now(ET).isoformat(),
        "trading_date": date_str,
        "prior_trading_day": raw.get("prior_trading_day"),
        "data_ready": raw.get("data_ready"),
        "llm_used": not skip_llm and bool(llm_parts),
        "bias": rule_bundle.get("bias"),
        "total_score": rule_bundle.get("total"),
        "parts": parts,
    }

    report_md = render_morning_report(
        date_str,
        parts,
        {"bias": rule_bundle.get("bias"), "total": rule_bundle.get("total")},
    )
    morning_report_path(date_str).write_text(report_md, encoding="utf-8")
    morning_json_path(date_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote morning report to %s", morning_report_path(date_str))

    _persist_morning(payload)
    return payload


def _persist_morning(payload: dict[str, Any]) -> None:
    trading_date = date.fromisoformat(payload["trading_date"])
    session = get_session()
    try:
        session.add(
            DailyRun(
                trading_date=trading_date,
                step_id="morning_research",
                status="ok" if payload.get("llm_used") else "partial",
                message=f"Bias: {payload.get('bias')} · Total: {payload.get('total_score')}",
            )
        )
        for pid in PART_ORDER:
            part = payload["parts"].get(pid) or {}
            existing = (
                session.query(ConclusionRecord)
                .filter(
                    ConclusionRecord.trading_date == trading_date,
                    ConclusionRecord.part_id == pid,
                )
                .first()
            )
            if existing:
                existing.judgment = part.get("judgment", "")
                existing.confidence = part.get("confidence")
                existing.one_liner = part.get("one_liner", "")
            else:
                session.add(
                    ConclusionRecord(
                        trading_date=trading_date,
                        part_id=pid,
                        judgment=part.get("judgment", ""),
                        confidence=part.get("confidence"),
                        one_liner=part.get("one_liner", ""),
                    )
                )
        session.commit()
    finally:
        session.close()
