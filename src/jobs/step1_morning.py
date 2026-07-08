from __future__ import annotations

import argparse
import logging

from src.db import init_db
from src.research.morning import run_morning_research
from src.utils.pit_snapshots import parse_as_of

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 1 Morning Research")
    parser.add_argument("--date", help="Trading date YYYY-MM-DD (default: today ET)")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Rules-only mode (no Anthropic call)",
    )
    parser.add_argument(
        "--resync-only",
        action="store_true",
        help="Re-render report + DB conclusions from existing morning.json",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        help="Decision time (HH:MM ET). Default: 08:00. Use 'now' for live debug.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing morning.json / PIT snapshot",
    )
    args = parser.parse_args()

    init_db()
    trading_date = None
    if args.date:
        from datetime import date

        trading_date = date.fromisoformat(args.date)

    if args.resync_only:
        from src.research.morning import resync_morning_from_disk

        payload = resync_morning_from_disk(trading_date)
        print(f"Morning resync done — {payload.get('trading_date')}")
        return

    payload = run_morning_research(
        trading_date,
        skip_llm=args.skip_llm,
        as_of_et=parse_as_of(args.as_of, step_num=1),
        force=args.force,
    )
    print(f"Morning research done — Bias: {payload.get('bias')} · LLM: {payload.get('llm_used')}")


if __name__ == "__main__":
    main()
