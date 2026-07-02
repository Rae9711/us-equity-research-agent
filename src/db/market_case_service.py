"""Service layer for reading and writing Market Cases (DB + filesystem)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.db import MarketCase, get_session
from src.schemas.market_case import MarketCaseModel
from src.utils.paths import report_dir

logger = logging.getLogger(__name__)


def _case_file_path(date_str: str) -> Path:
    return report_dir(date_str) / "case.json"


def load_case(date_str: str) -> MarketCaseModel:
    """Load existing case from DB or return empty skeleton."""
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            return MarketCaseModel.model_validate_json(row.case_json)
    except Exception as exc:
        logger.warning("Could not load case from DB for %s: %s", date_str, exc)
    finally:
        session.close()

    fp = _case_file_path(date_str)
    if fp.exists():
        try:
            return MarketCaseModel.model_validate(json.loads(fp.read_text()))
        except Exception as exc:
            logger.warning("Could not load case from file for %s: %s", date_str, exc)

    return MarketCaseModel(date=date_str)


def save_case(case: MarketCaseModel) -> None:
    """Dual-write: filesystem JSON + DB."""
    date_str = case.date
    case_json = case.model_dump_json(indent=2)

    fp = _case_file_path(date_str)
    fp.write_text(case_json, encoding="utf-8")

    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            row.case_json = case_json
        else:
            session.add(MarketCase(date=date_str, case_json=case_json))
        session.commit()
        logger.info("Market Case saved for %s", date_str)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save Market Case to DB for %s: %s", date_str, exc)
    finally:
        session.close()


def update_case(date_str: str, updates: dict[str, Any]) -> MarketCaseModel:
    """Load, merge updates, and save."""
    case = load_case(date_str)
    data = case.model_dump()
    _deep_merge(data, updates)
    updated = MarketCaseModel.model_validate(data)
    save_case(updated)
    return updated


def _deep_merge(base: dict, overlay: dict) -> None:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
