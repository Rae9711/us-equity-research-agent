"""Job: step5_midday — invoked by APScheduler at 12:00 ET."""

from __future__ import annotations

import logging

from src.steps.s05_midday import run_step5_midday

logger = logging.getLogger(__name__)


def main() -> None:
    result = run_step5_midday()
    logger.info("Step 5 done: %s", result.get("conclusion", {}).get("one_liner", ""))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
