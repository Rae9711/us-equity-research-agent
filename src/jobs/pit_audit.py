"""Audit immutable PIT partitions before research or backtesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from src.data.pit import (
    PITPartitionStore,
    PartitionNotFoundError,
    audit_rows,
    data_gap_report,
    load_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit point-in-time research data")
    parser.add_argument("--date", required=True, help="partition date (YYYY-MM-DD)")
    parser.add_argument(
        "--decision-at",
        required=True,
        help="timezone-aware ISO timestamp for the decision boundary",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    contract = load_contract()
    store = PITPartitionStore(root=args.root, contract=contract)
    rows = []
    unavailable = []
    for category in contract["required_categories"]:
        try:
            rows.extend(store.load_partition(category, args.date))
        except PartitionNotFoundError:
            unavailable.append(category)
    audit = audit_rows(rows, args.decision_at, contract)
    gaps = data_gap_report(rows, args.decision_at, contract)
    result = {
        "ok": audit.ok and gaps["ok"],
        "partition_date": args.date,
        "unavailable_partitions": unavailable,
        "audit": audit.as_dict(),
        "data_gaps": gaps,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if args.strict and not result["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
