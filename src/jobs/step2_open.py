from __future__ import annotations

import argparse
import logging

from src.db import init_db
from src.steps.s02_open import run_step2_open
from src.utils.pit_snapshots import parse_as_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 2 Open report")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument(
        "--as-of",
        dest="as_of",
        help="Decision time (HH:MM ET). Default: 09:30. Use 'now' for live debug.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PIT snapshot",
    )
    args = parser.parse_args()
    init_db()
    d = None
    if args.date:
        from datetime import date

        d = date.fromisoformat(args.date)
    payload = run_step2_open(
        d,
        as_of_et=parse_as_of(args.as_of, step_num=2),
        force=args.force,
    )
    print(payload["conclusion"]["judgment"])


if __name__ == "__main__":
    main()
