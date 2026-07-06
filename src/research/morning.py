from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from pytz import timezone

from src.db import ConclusionRecord, DailyRun
from src.db.market_case_service import load_case, save_case
from src.db.session import get_session
from src.llm.anthropic_client import AnthropicClient
from src.research.context import build_research_context
from src.research.prompts import MORNING_SYSTEM
from src.research.parts_meta import PART_ORDER
from src.research.report import render_morning_report
from src.research.rules import compute_rule_parts
from src.utils.data_freshness import guard_fresh_raw
from src.utils.paths import morning_json_path, morning_report_path, raw_data_path
from src.utils.trading_calendar import ET, require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)

RULE_PART_IDS = {"P1", "P2", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P13", "P15"}


def _is_missing(part: dict[str, Any]) -> bool:
    """True when a Part has no usable judgment/one_liner content."""
    if not part:
        return True
    judgment = str(part.get("judgment") or "").strip()
    one_liner = str(part.get("one_liner") or "").strip()
    body = str(part.get("body_md") or "").strip()
    if judgment in ("", "—", "-", "N/A") and one_liner in ("", "—", "-", "N/A") and not body:
        return True
    return False


def _regime_one_liner(label: str) -> str:
    hints = {
        "AI Expansion": "AI 扩张 Regime；Macro 数据日可暂时 dominate 日内 Driver",
        "Macro Fear": "宏观恐惧主导，Risk-off 优先",
        "Liquidity Driven": "流动性驱动，Risk Appetite 为主",
        "Range": "无明显 Regime 信号，按 Range 处理",
    }
    return hints.get(label, f"Regime={label}")


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
    d = require_trading_day(trading_date, job="run_morning_research")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    date_str = trading_date.isoformat()
    logger.info("Morning research starting for %s", date_str)

    raw, stale = guard_fresh_raw(trading_date, step="run_morning_research")
    if stale:
        return stale

    # Step 1a: R0 Regime Engine (must run before P1-P16)
    regime_model = None
    try:
        from src.features.build import build_features
        from src.engines.regime import classify_regime
        features = build_features(trading_date)
        regime_model = classify_regime(features)
        logger.info("R0 Regime: %s (conf=%.2f)", regime_model.label, regime_model.confidence)
    except Exception:
        logger.exception("R0 Regime Engine failed; using default")
        from src.schemas.market_case import RegimeModel, FeaturesModel
        regime_model = RegimeModel(label="Range", confidence=0.50)
        features = FeaturesModel()

    context = build_research_context(raw)
    # Inject regime context so LLM can use it
    context["r0_regime"] = {"label": regime_model.label, "confidence": regime_model.confidence}
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
    # Guarantee every Part has a non-empty row even when the LLM failed or
    # returned partial output — the rule engine already fills P4/P5/P6/P7/P8/
    # P9/P11/P13, so this only covers the narrative Parts (P1/P2/P3/P10/P12/
    # P14/P15/P16).
    fallback_reason = (
        "LLM 未运行或失败，仅规则引擎输出"
        if skip_llm or not llm_parts
        else "LLM 未返回该 Part，已回填占位"
    )
    for pid in PART_ORDER:
        if _is_missing(parts.get(pid) or {}):
            parts[pid] = {
                "judgment": "N/A",
                "confidence": None,
                "one_liner": fallback_reason,
                "body_md": "",
            }

    # Add R0 as a part for display purposes
    parts["R0"] = {
        "judgment": f"Regime：{regime_model.label}",
        "confidence": regime_model.confidence,
        "one_liner": _regime_one_liner(regime_model.label),
    }

    # Step 1c: P17 Hypothesis (after P1-P16)
    hypothesis_model = None
    try:
        from src.engines.hypothesis import build_hypothesis
        morning_total = rule_bundle.get("total") or 0
        morning_bias = rule_bundle.get("bias") or "Neutral"
        hypothesis_model = build_hypothesis(
            trading_date=trading_date,
            features=features,
            regime=regime_model,
            morning_bias=morning_bias,
            morning_total=morning_total,
            morning_parts=parts,
        )
        parts["P17"] = {
            "judgment": f"Hypothesis：{hypothesis_model.id} · {hypothesis_model.status}",
            "confidence": hypothesis_model.confidence,
            "one_liner": hypothesis_model.statement[:100],
            "hypothesis": hypothesis_model.model_dump(),
        }
        logger.info("P17 Hypothesis: %s (conf=%.2f)", hypothesis_model.id, hypothesis_model.confidence)
    except Exception:
        logger.exception("P17 Hypothesis Engine failed")
        parts["P17"] = {
            "judgment": "Hypothesis：N/A · 待验证",
            "confidence": None,
            "one_liner": "Hypothesis 引擎未生成",
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
        "r0": {"label": regime_model.label, "confidence": regime_model.confidence} if regime_model else None,
        "hypothesis": hypothesis_model.model_dump() if hypothesis_model else None,
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

    # Update Market Case with morning data
    try:
        from src.db.market_case_service import update_case
        update_case(date_str, {
            "regime": regime_model.model_dump() if regime_model else {},
            "hypothesis": hypothesis_model.model_dump() if hypothesis_model else {},
            "morning": {"bias": rule_bundle.get("bias"), "total": rule_bundle.get("total")},
            "features": features.model_dump() if features else {},
        })
    except Exception:
        logger.exception("Failed to update Market Case from morning research")

    _persist_morning(payload)
    return payload


def resync_morning_from_disk(trading_date: date | None = None) -> dict[str, Any]:
    """Re-render morning report + conclusions from existing morning.json (no LLM)."""
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    path = morning_json_path(date_str)
    if not path.exists():
        raise FileNotFoundError(f"No morning.json for {date_str}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = payload.get("parts") or {}
    report_md = render_morning_report(
        date_str,
        parts,
        {
            "bias": payload.get("bias"),
            "total": payload.get("total_score"),
        },
    )
    morning_report_path(date_str).write_text(report_md, encoding="utf-8")
    _persist_morning(payload)
    logger.info("Resynced morning index for %s from disk", date_str)
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
        for pid in ["R0"] + PART_ORDER + ["P17"]:
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
