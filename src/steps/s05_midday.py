"""Step 5 — Midday Review (12:00 PM ET).

Core question: Is today's primary Driver still the Morning Driver?
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.db.market_case_service import update_case
from src.steps.base import save_step_result
from src.utils.data_freshness import guard_fresh_raw
from src.utils.paths import morning_json_path, step_json_path
from src.utils.pit_snapshots import save_snapshot, step_label
from src.utils.trading_calendar import require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)


def run_step5_midday(trading_date: date | None = None) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step5_midday")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    date_str = trading_date.isoformat()
    logger.info("Step 5 Midday Review for %s", date_str)

    _, stale = guard_fresh_raw(trading_date, step="run_step5_midday")
    if stale:
        return stale

    morning: dict[str, Any] = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    s3: dict[str, Any] = {}
    if step_json_path(3, date_str).exists():
        s3 = json.loads(step_json_path(3, date_str).read_text(encoding="utf-8"))

    morning_driver_p10 = ((morning.get("parts") or {}).get("P10") or {}).get("judgment", "Morning Driver")
    s3_driver_changed = "YES" in ((s3.get("conclusion") or {}).get("judgment", ""))

    if s3_driver_changed:
        s3_judgment = (s3.get("conclusion") or {}).get("judgment", "")
        driver_still_valid = False
        action = "Adjust"
        new_driver_note = s3_judgment
    else:
        driver_still_valid = True
        action = "Hold"
        new_driver_note = f"Morning Driver ({morning_driver_p10}) 持续有效"

    body_lines = [
        "## Step 5 — Midday Review",
        "",
        f"**Morning Driver**: {morning_driver_p10}",
        f"**Driver Still Valid**: {'YES' if driver_still_valid else 'NO'}",
        f"**Action**: {action}",
        "",
        f"Note: {new_driver_note}",
    ]

    judgment = (
        f"Driver 仍成立：{'YES' if driver_still_valid else 'NO'} · 行动：{action}"
    )
    one_liner = (
        f"Midday Driver {'仍为' if driver_still_valid else '切换'}，{action}"
    )

    conclusion = {
        "part_id": "S5",
        "judgment": judgment,
        "confidence": 0.65,
        "one_liner": one_liner[:256],
    }

    payload = save_step_result(
        5,
        trading_date,
        step_id="S5",
        job_id="midday_review",
        conclusion=conclusion,
        body_md="\n".join(body_lines),
        extra={
            "driver_still_valid": driver_still_valid,
            "action": action,
        },
    )

    update_case(date_str, {
        "intraday": {"s5": {"driver_still_valid": driver_still_valid, "action": action}},
    })

    save_snapshot(trading_date, step_label(5), {"step5": payload})
    return payload
