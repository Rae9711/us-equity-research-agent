"""Job: step7_evening — invoked by APScheduler at 16:10 ET."""

from __future__ import annotations

import logging

from src.steps.s07_evening import run_step7_evening

logger = logging.getLogger(__name__)


def main() -> None:
    result = run_step7_evening()
    logger.info(
        "Step 7 done: %s",
        result.get("conclusion", {}).get("one_liner", ""),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
