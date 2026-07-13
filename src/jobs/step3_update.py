from __future__ import annotations

import argparse
import logging

from src.db import init_db
from src.steps.s03_update import run_step3_market_update

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3 market update")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Anthropic call; use rules fallback + session trade re-rank",
    )
    args = parser.parse_args()
    init_db()
    d = None
    if args.date:
        from datetime import date

        d = date.fromisoformat(args.date)
    payload = run_step3_market_update(d, skip_llm=args.skip_llm)
    print(payload["conclusion"]["judgment"])
    su = payload.get("session_update") or payload.get("session_trade_update") or {}
    if su.get("why_changed"):
        print(su["why_changed"])
    primary = su.get("primary") or {}
    if primary.get("symbol"):
        print(
            f"10:00 #1: {primary.get('symbol')} {primary.get('direction') or ''} "
            f"(changed={su.get('changed')})"
        )


if __name__ == "__main__":
    main()
