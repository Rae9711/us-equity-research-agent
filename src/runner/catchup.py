"""On runner startup, run any today's jobs whose ET time passed but output is missing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from pytz import timezone

from src.utils.trading_calendar import today_et
from src.web.steps_status import step_available

logger = logging.getLogger("runner.catchup")

ET = timezone("America/New_York")


def run_startup_catchup(handlers: dict[str, Callable[[], None]]) -> None:
    """Run missed jobs for today when runner restarts mid-session."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return

    date_str = today_et().isoformat()
    schedule = [
        ("collect_raw", 7, 45, 0),
        ("morning_research", 8, 0, 1),
        ("open_report", 9, 30, 2),
        ("market_update", 10, 0, 3),
        ("trade_decision", 10, 15, 4),
        ("midday_review", 12, 0, 5),
        ("afternoon_review", 14, 0, 6),
        ("evening_review", 16, 10, 7),
        ("learning", 20, 0, 8),
    ]

    for job_id, hour, minute, step_num in schedule:
        handler = handlers.get(job_id)
        if not handler:
            continue
        scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < scheduled:
            continue
        if step_available(step_num, date_str):
            continue
        logger.info("Startup catch-up: %s for %s (scheduled %02d:%02d ET)", job_id, date_str, hour, minute)
        try:
            handler()
        except Exception:
            logger.exception("Startup catch-up failed: %s", job_id)
