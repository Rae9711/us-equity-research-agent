from __future__ import annotations

import logging
import os
import signal
import sys
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

from src.collectors.step0 import run_step0
from src.db import init_db
from src.research.morning import run_morning_research
from src.steps.s02_open import run_step2_open
from src.steps.s03_update import run_step3_market_update
from src.steps.s04_decision import run_step4_trade_decision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runner")

ET = timezone("America/New_York")


def _placeholder_job(step_id: str) -> None:
    logger.info("Job %s — Phase 0 placeholder (not implemented yet)", step_id)


def _collect_raw_job() -> None:
    logger.info("Job collect_raw — Step 0 data collection starting")
    try:
        payload = run_step0()
        logger.info(
            "Step 0 done: %s — %s",
            payload["conclusion"]["judgment"],
            payload["conclusion"]["one_liner"],
        )
    except Exception:
        logger.exception("Step 0 collect_raw failed")


def _morning_research_job() -> None:
    logger.info("Job morning_research — Step 1 starting")
    try:
        payload = run_morning_research()
        logger.info(
            "Step 1 done: Bias=%s Total=%s LLM=%s",
            payload.get("bias"),
            payload.get("total_score"),
            payload.get("llm_used"),
        )
    except Exception:
        logger.exception("Step 1 morning_research failed")


def _open_report_job() -> None:
    logger.info("Job open_report — Step 2 starting")
    try:
        payload = run_step2_open()
        logger.info("Step 2 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 2 open_report failed")


def _market_update_job() -> None:
    logger.info("Job market_update — Step 3 starting")
    try:
        payload = run_step3_market_update()
        logger.info("Step 3 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 3 market_update failed")


def _trade_decision_job() -> None:
    logger.info("Job trade_decision — Step 4 starting")
    try:
        payload = run_step4_trade_decision()
        logger.info("Step 4 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 4 trade_decision failed")


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=ET)

    # Phase 0: register jobs; Phase 2+ wire real handlers
    schedule = [
        ("7:45", "collect_raw", "mon-fri", 7, 45),
        ("8:00", "morning_research", "mon-fri", 8, 0),
        ("9:30", "open_report", "mon-fri", 9, 30),
        ("10:00", "market_update", "mon-fri", 10, 0),
        ("10:15", "trade_decision", "mon-fri", 10, 15),
        ("12:00", "midday_review", "mon-fri", 12, 0),
        ("14:00", "afternoon_review", "mon-fri", 14, 0),
        ("16:10", "evening_review", "mon-fri", 16, 10),
        ("20:00", "learning", "mon-fri", 20, 0),
    ]

    handlers = {
        "collect_raw": _collect_raw_job,
        "morning_research": _morning_research_job,
        "open_report": _open_report_job,
        "market_update": _market_update_job,
        "trade_decision": _trade_decision_job,
    }

    for _label, step_id, dow, hour, minute in schedule:
        handler = handlers.get(step_id, _placeholder_job)
        kwargs: dict = {"id": step_id, "replace_existing": True}
        if handler is _placeholder_job:
            kwargs["args"] = [step_id]
        scheduler.add_job(
            handler,
            CronTrigger(day_of_week=dow, hour=hour, minute=minute, timezone=ET),
            **kwargs,
        )

    return scheduler


def main() -> None:
    init_db()
    logger.info("Daily Trading OS runner starting (TZ=%s)", os.environ.get("TZ", "UTC"))

    scheduler = build_scheduler()
    for job in scheduler.get_jobs():
        logger.info("Scheduled job: %s", job.id)

    def shutdown(_signum, _frame):
        logger.info("Shutting down scheduler")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
