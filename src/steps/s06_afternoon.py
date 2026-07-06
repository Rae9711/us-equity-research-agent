"""Step 6 — Afternoon Review (2:00 PM ET).

Core question: Trade still valid?
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.db.market_case_service import update_case
from src.steps.base import save_step_result
from src.utils.data_freshness import guard_fresh_raw
from src.utils.paths import step_json_path
from src.utils.trading_calendar import require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)


def run_step6_afternoon(trading_date: date | None = None) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step6_afternoon")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    date_str = trading_date.isoformat()
    logger.info("Step 6 Afternoon Review for %s", date_str)

    _, stale = guard_fresh_raw(trading_date, step="run_step6_afternoon")
    if stale:
        return stale

    s4: dict[str, Any] = {}
    if step_json_path(4, date_str).exists():
        s4 = json.loads(step_json_path(4, date_str).read_text(encoding="utf-8"))

    s5: dict[str, Any] = {}
    if step_json_path(5, date_str).exists():
        s5 = json.loads(step_json_path(5, date_str).read_text(encoding="utf-8"))

    should_trade = s4.get("should_trade", False)
    driver_still_valid = s5.get("driver_still_valid", True)

    # Trade validity check
    trade_valid = should_trade and driver_still_valid

    if not should_trade:
        action = "N/A"
        note = "Morning decided NO trade — no position to monitor"
    elif not driver_still_valid:
        action = "减仓"
        note = "Driver shifted at midday — thesis partially broken"
        trade_valid = False
    else:
        action = "Hold"
        note = "Driver intact, thesis still valid"

    body_lines = [
        "## Step 6 — Afternoon Review",
        "",
        f"**Should Trade (S4)**: {'YES' if should_trade else 'NO'}",
        f"**Driver Still Valid (S5)**: {'YES' if driver_still_valid else 'NO'}",
        f"**Trade Valid**: {'YES' if trade_valid else 'NO'}",
        f"**Action**: {action}",
        "",
        f"Note: {note}",
        "",
        "> ADVISORY_ONLY — 仅供参考，非交易指令",
    ]

    judgment = (
        f"Trade valid：{'YES' if trade_valid else 'NO'} · 行动：{action}"
    )
    one_liner = f"交易{'仍有效' if trade_valid else '失效'}，{action}"

    conclusion = {
        "part_id": "S6",
        "judgment": judgment,
        "confidence": 0.65,
        "one_liner": one_liner[:256],
    }

    payload = save_step_result(
        6,
        trading_date,
        step_id="S6",
        job_id="afternoon_review",
        conclusion=conclusion,
        body_md="\n".join(body_lines),
        extra={
            "trade_valid": trade_valid,
            "action": action,
        },
    )

    update_case(date_str, {
        "intraday": {"s6": {"trade_valid": trade_valid, "action": action}},
    })

    return payload
