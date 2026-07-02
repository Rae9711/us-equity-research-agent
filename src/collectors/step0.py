from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from pytz import timezone

from src.collectors.macro import collect_macro
from src.collectors.market import collect_market
from src.collectors.news import collect_news
from src.collectors.options import collect_options
from src.collectors.sector import collect_sectors
from src.collectors.stocks import collect_stocks
from src.db import ConclusionRecord, DailyRun
from src.db.session import get_session
from src.utils.paths import raw_data_path
from src.utils.trading_calendar import ET, prior_trading_day, today_et

logger = logging.getLogger(__name__)


def _flatten_missing(checklist: dict[str, Any], prefix: str) -> list[str]:
    missing: list[str] = []
    for key, ok in checklist.items():
        if not ok:
            missing.append(f"{prefix}.{key}")
    return missing


def collect_step0(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    prior_day = prior_trading_day(trading_date)
    collected_at = datetime.now(ET).isoformat()

    logger.info("Step 0 collect starting for %s", trading_date)

    market = collect_market()
    macro = collect_macro()
    sector = collect_sectors()
    stocks = collect_stocks()
    news = collect_news()
    options = collect_options()

    checklist = {
        "macro": macro.get("checklist", {}),
        "market": market.get("checklist", {}),
        "sector": sector.get("checklist", {}),
        "stocks": stocks.get("checklist", {}),
        "news": news.get("checklist", {}),
        "options": options.get("checklist", {}),
    }

    missing: list[str] = []
    for section, items in checklist.items():
        missing.extend(_flatten_missing(items, section))

    data_ready = len(missing) == 0

    payload: dict[str, Any] = {
        "collected_at": collected_at,
        "trading_date": trading_date.isoformat(),
        "prior_trading_day": prior_day.isoformat(),
        "data_ready": data_ready,
        "missing": missing,
        "checklist": checklist,
        "market": market,
        "macro": macro,
        "sector": sector,
        "stocks": stocks,
        "news": news,
        "options": options,
        "conclusion": {
            "part_id": "Step0",
            "judgment": f"数据就绪：{'YES' if data_ready else 'NO'}",
            "confidence": None,
            "one_liner": (
                "全部 Raw Data 已更新，可进入 Morning Research"
                if data_ready
                else (
                    f"缺失项：{', '.join(missing[:8])}"
                    + ("…" if len(missing) > 8 else "")
                )
            ),
        },
    }

    out_path = raw_data_path(trading_date.isoformat())
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote raw data to %s (ready=%s)", out_path, data_ready)
    return payload


def _persist_run(payload: dict[str, Any]) -> None:
    trading_date = date.fromisoformat(payload["trading_date"])
    conclusion = payload["conclusion"]
    session = get_session()
    try:
        run = DailyRun(
            trading_date=trading_date,
            step_id="collect_raw",
            status="ok" if payload["data_ready"] else "partial",
            message=conclusion["one_liner"],
        )
        session.add(run)

        existing = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == trading_date,
                ConclusionRecord.part_id == "Step0",
            )
            .first()
        )
        if existing:
            existing.judgment = conclusion["judgment"]
            existing.confidence = conclusion.get("confidence")
            existing.one_liner = conclusion["one_liner"]
        else:
            session.add(
                ConclusionRecord(
                    trading_date=trading_date,
                    part_id="Step0",
                    judgment=conclusion["judgment"],
                    confidence=conclusion.get("confidence"),
                    one_liner=conclusion["one_liner"],
                )
            )
        session.commit()
    finally:
        session.close()


def run_step0(trading_date: date | None = None) -> dict[str, Any]:
    payload = collect_step0(trading_date)
    _persist_run(payload)
    return payload
