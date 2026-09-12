"""Offline CLI for the daily market-neutral portfolio backtester.

Example:
    python -m src.jobs.market_neutral_backtest data.json --output result.json \
        --report result.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.paper.market_neutral_backtest import (
    BacktestConfig,
    DailyBar,
    run_market_neutral_backtest,
)
from src.strategies.market_neutral_ls import (
    Event,
    MarketNeutralLongShortStrategy,
    StrategyConfig,
)


def _filtered_config(cls: Any, values: Mapping[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "unknown {} keys: {}".format(cls.__name__, ", ".join(unknown))
        )
    return cls(**dict(values))


def _event(row: Mapping[str, Any]) -> Event:
    from src.paper.market_neutral_backtest import _datetime

    return Event(
        symbol=str(row["symbol"]),
        timestamp=_datetime(row["timestamp"]),
        event_type=str(row["event_type"]),
        value=float(row.get("value", 0.0) or 0.0),
        expected=float(row["expected"]) if row.get("expected") is not None else None,
        prior=float(row["prior"]) if row.get("prior") is not None else None,
        sentiment=float(row.get("sentiment", 0.0) or 0.0),
        available_at=_datetime(row["available_at"])
        if row.get("available_at") is not None else None,
        relevance=float(row.get("relevance", 1.0) or 0.0),
        reaction=float(row["reaction"]) if row.get("reaction") is not None else None,
        reaction_known_at=_datetime(row["reaction_known_at"])
        if row.get("reaction_known_at") is not None else None,
    )


def load_dataset(path: Path) -> Dict[str, Any]:
    """Load and validate a self-contained JSON backtest dataset."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = {"bars": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("bars"), list):
        raise ValueError("dataset must be an object with a bars list (or a bars list)")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline point-in-time market-neutral backtest"
    )
    parser.add_argument("dataset", type=Path, help="JSON input dataset")
    parser.add_argument("--output", type=Path, help="write JSON result to this path")
    parser.add_argument("--report", type=Path, help="write a Markdown report")
    parser.add_argument(
        "--record-qualification",
        metavar="EVIDENCE_ID",
        help="explicitly import result metrics into the paper qualification record",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="pretty-print JSON sent to stdout/output"
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        payload = load_dataset(args.dataset)
        bars = [DailyBar.from_mapping(row) for row in payload["bars"]]
        events = [_event(row) for row in payload.get("events", [])]
        strategy_config = _filtered_config(
            StrategyConfig, payload.get("strategy_config", {})
        )
        sector_map = payload.get("sector_map")
        if sector_map is None:
            sector_map = {
                row.symbol: row.sector for row in bars if row.sector is not None
            } or None
        strategy = MarketNeutralLongShortStrategy(strategy_config, sector_map)
        config = _filtered_config(BacktestConfig, payload.get("backtest_config", {}))
        result = run_market_neutral_backtest(
            bars,
            events,
            strategy=strategy,
            config=config,
            borrow_available=payload.get("borrow_available"),
            borrow_rates=payload.get("borrow_rates"),
            ssr_restricted=payload.get("ssr_restricted"),
            forced_cover=payload.get("forced_cover"),
            pit_audit_report=payload.get("pit_audit_report"),
        )
        encoded = json.dumps(
            result.to_dict(),
            indent=2 if args.pretty else None,
            sort_keys=args.pretty,
            allow_nan=False,
        )
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            sys.stdout.write(encoded + "\n")
        if args.report:
            args.report.write_text(result.markdown_report(), encoding="utf-8")
        if args.record_qualification:
            from src.paper.acceptance import record_event_ls_backtest
            from src.paper.account import load_account, save_account

            account = load_account()
            record_event_ls_backtest(
                account,
                result.metrics,
                evidence_id=args.record_qualification,
            )
            save_account(account)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write("market-neutral backtest: {}\n".format(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
