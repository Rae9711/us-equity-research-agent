"""On runner startup, run any today's jobs whose ET time passed but output is missing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta

from pytz import timezone

from src.utils.trading_calendar import today_et
from src.web.steps_status import step_available

logger = logging.getLogger("runner.catchup")

ET = timezone("America/New_York")

# End-of-day steps: cron only — never replay on deploy/restart.
CRON_ONLY_JOBS = frozenset({"evening_review", "learning"})

# Catch-up only within this window after scheduled ET time (avoids replay hours later).
CATCHUP_WINDOW = timedelta(minutes=60)

# Hard minimum ET wall-clock; catch-up never runs before these times.
HARD_MIN_ET: dict[str, tuple[int, int]] = {
    "afternoon_review": (14, 0),
    "evening_review": (16, 10),
    "learning": (20, 0),
}

SCHEDULE = [
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


def _now_et() -> datetime:
    return datetime.now(ET)


def _scheduled_et(day: datetime, hour: int, minute: int) -> datetime:
    """Build an ET-localized datetime for today's calendar date."""
    return ET.localize(datetime(day.year, day.month, day.day, hour, minute, 0))


def run_startup_catchup(handlers: dict[str, Callable[[], None]]) -> None:
    """Run missed jobs for today when runner restarts mid-session."""
    now = _now_et()
    if now.weekday() >= 5:
        logger.info("Startup catch-up skipped: weekend (%s ET)", now.strftime("%A"))
        return

    date_str = today_et().isoformat()
    logger.info("Startup catch-up scan for %s at %s ET", date_str, now.strftime("%H:%M:%S"))

    for job_id, hour, minute, step_num in SCHEDULE:
        handler = handlers.get(job_id)
        if not handler:
            continue

        if job_id in CRON_ONLY_JOBS:
            logger.info(
                "Startup catch-up skip %s: cron-only (S7/S8 never replay on restart)",
                job_id,
            )
            continue

        scheduled = _scheduled_et(now, hour, minute)
        hard = HARD_MIN_ET.get(job_id)
        if hard:
            hard_at = _scheduled_et(now, hard[0], hard[1])
            if now < hard_at:
                logger.info(
                    "Startup catch-up skip %s: before hard minimum %02d:%02d ET (now %s ET)",
                    job_id,
                    hard[0],
                    hard[1],
                    now.strftime("%H:%M:%S"),
                )
                continue

        if now < scheduled:
            logger.debug(
                "Startup catch-up skip %s: not yet scheduled (%02d:%02d ET)",
                job_id,
                hour,
                minute,
            )
            continue

        late_by = now - scheduled
        if late_by > CATCHUP_WINDOW:
            logger.info(
                "Startup catch-up skip %s: %.0f min past scheduled %02d:%02d ET (window=%d min)",
                job_id,
                late_by.total_seconds() / 60,
                hour,
                minute,
                int(CATCHUP_WINDOW.total_seconds() // 60),
            )
            continue

        if step_available(step_num, date_str):
            logger.info("Startup catch-up skip %s: step %d output already exists", job_id, step_num)
            continue

        logger.info(
            "Startup catch-up run %s for %s (scheduled %02d:%02d ET, now %s ET)",
            job_id,
            date_str,
            hour,
            minute,
            now.strftime("%H:%M:%S"),
        )
        try:
            handler()
        except Exception:
            logger.exception("Startup catch-up failed: %s", job_id)
