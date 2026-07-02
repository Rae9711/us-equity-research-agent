from __future__ import annotations

import argparse
import logging

from src.db import init_db
from src.research.morning import run_morning_research

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
    args = parser.parse_args()

    init_db()
    trading_date = None
    if args.date:
        from datetime import date

        trading_date = date.fromisoformat(args.date)

    payload = run_morning_research(trading_date, skip_llm=args.skip_llm)
    print(f"Morning research done — Bias: {payload.get('bias')} · LLM: {payload.get('llm_used')}")


if __name__ == "__main__":
    main()
