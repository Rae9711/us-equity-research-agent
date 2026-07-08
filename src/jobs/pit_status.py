"""CLI: list PIT snapshots for a trading day.

Usage::

    python -m src.jobs.pit_status --date 2026-07-07
    python -m src.jobs.pit_status --date 2026-07-07 --prune --keep-days 90
"""

from __future__ import annotations

import argparse
from datetime import date

from src.utils.pit_snapshots import list_snapshots, prune_old_snapshots
from src.utils.trading_calendar import today_et


def main() -> None:
    parser = argparse.ArgumentParser(description="List PIT snapshots for a trading day")
    parser.add_argument("--date", help="Trading date YYYY-MM-DD (default: today ET)")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune snapshot dirs older than --keep-days (default 90)",
    )
    parser.add_argument("--keep-days", type=int, default=90, help="Retention window for --prune")
    parser.add_argument("--dry-run", action="store_true", help="Show prune targets without deleting")
    args = parser.parse_args()

    if args.prune:
        removed = prune_old_snapshots(keep_days=args.keep_days, dry_run=args.dry_run)
        action = "would prune" if args.dry_run else "pruned"
        print(f"{action} {len(removed)} snapshot dir(s) older than {args.keep_days} days")
        for p in removed:
            print(f"  {p}")
        if not args.date:
            return

    trading_date = date.fromisoformat(args.date) if args.date else today_et()
    rows = list_snapshots(trading_date)
    print(f"PIT snapshots for {trading_date.isoformat()}:")
    print(f"{'Step':>4}  {'Label':<14}  {'ET':>5}  {'Status':<8}  Path")
    print("-" * 72)
    for row in rows:
        step = row["step"] if row["step"] is not None else "—"
        sched = row["scheduled_et"] or "—"
        status = "OK" if row["exists"] else "MISSING"
        print(f"{step!s:>4}  {row['label']:<14}  {sched:>5}  {status:<8}  {row['path']}")


if __name__ == "__main__":
    main()
