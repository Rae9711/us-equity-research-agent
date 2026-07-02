from __future__ import annotations

import logging
import os
import signal
import sys
import time

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

from src.collectors.step0 import run_step0
from src.db import init_db
from src.research.morning import run_morning_research
from src.steps.s02_open import run_step2_open
from src.steps.s03_update import run_step3_market_update
from src.steps.s04_decision import run_step4_trade_decision
from src.steps.s05_midday import run_step5_midday
from src.steps.s06_afternoon import run_step6_afternoon
from src.steps.s07_evening import run_step7_evening
from src.runner.catchup import run_startup_catchup

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


def _midday_review_job() -> None:
    logger.info("Job midday_review — Step 5 starting")
    try:
        payload = run_step5_midday()
        logger.info("Step 5 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 5 midday_review failed")


def _afternoon_review_job() -> None:
    logger.info("Job afternoon_review — Step 6 starting")
    try:
        payload = run_step6_afternoon()
        logger.info("Step 6 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 6 afternoon_review failed")


def _evening_review_job() -> None:
    logger.info("Job evening_review — Step 7 starting")
    try:
        payload = run_step7_evening()
        logger.info("Step 7 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 7 evening_review failed")


def _learning_job() -> None:
    logger.info("Job learning — Step 8 starting")
    try:
        payload = run_step8_learning()
        logger.info("Step 8 done: %s", payload["conclusion"]["judgment"])
    except Exception:
        logger.exception("Step 8 learning failed")


def _weekly_ml_job() -> None:
    logger.info("Job weekly_ml starting")
    try:
        from src.engines.ml.train import run_weekly_ml
        result = run_weekly_ml()
        logger.info("Weekly ML done: %s", result.get("status", ""))
    except Exception:
        logger.exception("Weekly ML failed")


def _job_listener(event) -> None:
    if event.code == EVENT_JOB_EXECUTED:
        logger.info("Scheduler executed job: %s", event.job_id)
    elif event.code == EVENT_JOB_ERROR:
        logger.error("Scheduler job failed: %s", event.job_id, exc_info=event.exception)
    elif event.code == EVENT_JOB_MISSED:
        logger.warning("Scheduler missed job: %s (scheduled=%s)", event.job_id, event.scheduled_run_time)


def build_scheduler() -> tuple[BackgroundScheduler, dict[str, object]]:
    scheduler = BackgroundScheduler(
        timezone=ET,
        job_defaults={"coalesce": True, "misfire_grace_time": 3600, "max_instances": 1},
    )
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)

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
        "midday_review": _midday_review_job,
        "afternoon_review": _afternoon_review_job,
        "evening_review": _evening_review_job,
        "learning": _learning_job,
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

    # Weekly ML job (Sunday 20:00 ET)
    scheduler.add_job(
        _weekly_ml_job,
        CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=ET),
        id="weekly_ml",
        replace_existing=True,
    )

    return scheduler, handlers


def main() -> None:
    init_db()
    logger.info("Daily Trading OS runner starting (TZ=%s)", os.environ.get("TZ", "UTC"))

    scheduler, handlers = build_scheduler()
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
        logger.info("Scheduler running — waiting for cron triggers (ET)")
        run_startup_catchup(handlers)
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
