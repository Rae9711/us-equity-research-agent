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
from src.utils.data_freshness import freshness_dict
from src.utils.paths import raw_data_path
from src.utils.pit_snapshots import save_snapshot, step_label
from src.utils.trading_calendar import ET, prior_trading_day, prior_close_utc_iso, require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)


# Checklist keys where False is a valid state (not absent data).
_SKIP_MISSING_KEYS: dict[str, frozenset[str]] = {
    "macro": frozenset({"nfp_release_day"}),
    # RSS supplements Polygon; Bloomberg/WSJ feeds 404 or empty often — warn only.
    "news": frozenset({"bloomberg", "wsj"}),
}

# Soft checklist keys: False → ⚠ attention, never blocks data_ready.
_SOFT_CHECKLIST_KEYS: dict[str, frozenset[str]] = {
    "news": frozenset({"bloomberg", "wsj"}),
}


def _flatten_missing(checklist: dict[str, Any], prefix: str) -> list[str]:
    skip = _SKIP_MISSING_KEYS.get(prefix, frozenset())
    missing: list[str] = []
    for key, ok in checklist.items():
        if key in skip:
            continue
        if not ok:
            missing.append(f"{prefix}.{key}")
    return missing


def _soft_checklist_warnings(checklist: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for section, soft_keys in _SOFT_CHECKLIST_KEYS.items():
        items = checklist.get(section) or {}
        for key in soft_keys:
            if items.get(key) is False:
                warnings.append(f"⚠ {section}.{key}: RSS 不可用（可选源，不阻断就绪）")
    return warnings


def collect_step0(trading_date: date | None = None, *, force: bool = False) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    prior_day = prior_trading_day(trading_date)
    collected_at = datetime.now(ET).isoformat()

    logger.info("Step 0 collect starting for %s", trading_date)

    market = collect_market(trading_date)
    macro = collect_macro()
    sector = collect_sectors(trading_date)
    stocks = collect_stocks(trading_date)
    news = collect_news(published_gte=prior_close_utc_iso(trading_date))
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
    missing.extend(_soft_checklist_warnings(checklist))

    quote_session_dates: dict[str, str | None] = {}

    payload: dict[str, Any] = {
        "collected_at": collected_at,
        "data_as_of": collected_at,
        "trading_date": trading_date.isoformat(),
        "prior_trading_day": prior_day.isoformat(),
        "checklist": checklist,
        "market": market,
        "macro": macro,
        "sector": sector,
        "stocks": stocks,
        "news": news,
        "options": options,
    }

    freshness = freshness_dict(payload, trading_date)
    payload["freshness"] = freshness
    payload["quote_session_dates"] = {
        k.split(".", 1)[-1]: v
        for k, v in (freshness.get("field_sessions") or {}).items()
        if k.startswith("market.")
    } or quote_session_dates

    if not freshness["ok"]:
        missing.extend(freshness["reasons"])
    for item in freshness.get("attention") or []:
        missing.append(f"⚠ {item}")

    # Warnings (e.g. lagging FRED macro, optional RSS) must not block data_ready.
    hard_missing = [m for m in missing if not str(m).startswith("⚠")]
    soft_missing = [m for m in missing if str(m).startswith("⚠")]
    data_ready = len(hard_missing) == 0

    if data_ready:
        one_liner = "全部 Raw Data 已更新，可进入 Morning Research"
        if soft_missing:
            shown = soft_missing[:5]
            one_liner += (
                f"；需关注：{', '.join(shown)}"
                + ("…" if len(soft_missing) > 5 else "")
            )
    else:
        shown_hard = hard_missing[:6]
        one_liner = f"缺失项：{', '.join(shown_hard)}" + (
            "…" if len(hard_missing) > 6 else ""
        )
        if soft_missing:
            shown_soft = soft_missing[:4]
            one_liner += (
                f"；需关注：{', '.join(shown_soft)}"
                + ("…" if len(soft_missing) > 4 else "")
            )

    payload.update({
        "data_ready": data_ready,
        "missing": missing,
        "conclusion": {
            "part_id": "Step0",
            "judgment": f"数据就绪：{'YES' if data_ready else 'NO'}",
            "confidence": None,
            "one_liner": one_liner,
        },
    })

    out_path = raw_data_path(trading_date.isoformat())
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote raw data to %s (ready=%s)", out_path, data_ready)

    save_snapshot(trading_date, step_label(0), payload, force=force)
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


def run_step0(trading_date: date | None = None, *, force: bool = False) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step0")
    if d is None:
        return skipped_non_trading_day(trading_date)
    payload = collect_step0(d, force=force)
    _persist_run(payload)
    return payload
