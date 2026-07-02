"""L3 Playbook Engine — stores and retrieves reusable Market Case patterns.

No LLM involved. Uses cosine-like feature similarity for retrieval.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from typing import Any

from src.db import PlaybookCase, get_session
from src.schemas.market_case import FeaturesModel, MarketCaseModel

logger = logging.getLogger(__name__)


def _next_case_id(session) -> str:
    count = session.query(PlaybookCase).count()
    return f"Case-{count + 1}"


def store_case(case: MarketCaseModel) -> str | None:
    """
    Store a Market Case in the Playbook if it has a notable lesson/surprise.
    Returns the new case_id or None if not stored.
    """
    if not case.lesson and not case.surprise:
        return None
    if not case.regime.label or case.regime.label == "Unknown":
        return None

    pattern = {
        "date": case.date,
        "regime": case.regime.label,
        "regime_confidence": case.regime.confidence,
        "actual_driver": case.labels.actual_driver,
        "hypothesis_correct": case.labels.hypothesis_correct,
        "features_snapshot": {
            "vix": case.features.vix,
            "qqq_chg": case.features.qqq_chg,
            "smh_chg": case.features.smh_chg,
            "nvda_chg": case.features.nvda_chg,
            "dgs10": case.features.dgs10,
            "breadth_proxy": case.features.breadth_proxy,
        },
        "attribution": {
            "ai": case.attribution.ai,
            "bond": case.attribution.bond,
        },
    }

    session = get_session()
    try:
        existing = (
            session.query(PlaybookCase)
            .filter(PlaybookCase.trading_date == case.date)
            .first()
        )
        if existing:
            existing.pattern_json = json.dumps(pattern, ensure_ascii=False)
            existing.lesson = case.lesson
            existing.surprise = case.surprise
            session.commit()
            logger.info("Updated playbook case %s for %s", existing.case_id, case.date)
            return existing.case_id

        case_id = _next_case_id(session)
        session.add(PlaybookCase(
            case_id=case_id,
            trading_date=case.date,
            pattern_json=json.dumps(pattern, ensure_ascii=False),
            lesson=case.lesson,
            surprise=case.surprise,
        ))
        session.commit()
        logger.info("Stored new playbook case %s for %s", case_id, case.date)
        return case_id
    except Exception as exc:
        session.rollback()
        logger.error("Failed to store playbook case: %s", exc)
        return None
    finally:
        session.close()


def find_similar(
    features: FeaturesModel,
    regime_label: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k similar playbook cases by feature similarity.
    Prefer same regime.
    """
    session = get_session()
    try:
        all_cases = session.query(PlaybookCase).all()
    finally:
        session.close()

    scored = []
    for pc in all_cases:
        try:
            pattern = json.loads(pc.pattern_json)
        except Exception:
            continue
        sim = _similarity(features, pattern, regime_label)
        scored.append((sim, pc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, pc in scored[:top_k]:
        try:
            pattern = json.loads(pc.pattern_json)
        except Exception:
            pattern = {}
        results.append({
            "case_id": pc.case_id,
            "date": pc.trading_date,
            "similarity": round(sim, 3),
            "lesson": pc.lesson,
            "surprise": pc.surprise,
            "pattern": pattern,
        })
    return results


def _similarity(features: FeaturesModel, pattern: dict, regime_label: str) -> float:
    score = 0.0

    # Regime match bonus
    if pattern.get("regime") == regime_label:
        score += 0.4

    fs = pattern.get("features_snapshot", {})

    # VIX proximity
    if features.vix is not None and fs.get("vix") is not None:
        vix_diff = abs(features.vix - fs["vix"])
        score += max(0.0, 0.15 - vix_diff * 0.01)

    # QQQ direction match
    if features.qqq_chg is not None and fs.get("qqq_chg") is not None:
        if (features.qqq_chg > 0) == (fs["qqq_chg"] > 0):
            score += 0.15

    # NVDA/SMH direction match
    if features.nvda_chg is not None and fs.get("nvda_chg") is not None:
        if (features.nvda_chg > 0) == (fs["nvda_chg"] > 0):
            score += 0.10
    if features.smh_chg is not None and fs.get("smh_chg") is not None:
        if (features.smh_chg > 0) == (fs["smh_chg"] > 0):
            score += 0.10

    # DGS10 proximity
    if features.dgs10 is not None and fs.get("dgs10") is not None:
        dgs_diff = abs(features.dgs10 - fs["dgs10"])
        score += max(0.0, 0.10 - dgs_diff * 0.1)

    return min(1.0, score)
