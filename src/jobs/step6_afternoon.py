"""Job: step6_afternoon — invoked by APScheduler at 14:00 ET."""

from __future__ import annotations

import logging

from src.steps.s06_afternoon import run_step6_afternoon

logger = logging.getLogger(__name__)


def main() -> None:
    result = run_step6_afternoon()
    logger.info("Step 6 done: %s", result.get("conclusion", {}).get("one_liner", ""))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
