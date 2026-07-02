from __future__ import annotations

import argparse
import logging

from src.db import init_db
from src.steps.s03_update import run_step3_market_update

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3 market update")
    parser.add_argument("--date", help="YYYY-MM-DD")
    args = parser.parse_args()
    init_db()
    d = None
    if args.date:
        from datetime import date

        d = date.fromisoformat(args.date)
    payload = run_step3_market_update(d)
    print(payload["conclusion"]["judgment"])


if __name__ == "__main__":
    main()
