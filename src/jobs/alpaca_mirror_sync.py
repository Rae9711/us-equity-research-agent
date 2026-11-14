"""CLI: reconcile the local paper book into Alpaca paper (mirror).

Reads account.json + circuit_breaker.json, plans with
src.paper.alpaca_mirror, executes cancels / closes / placements through
the paper-only Alpaca client, and writes state/paper/alpaca_mirror.json.

--dry-run: plan only; no orders, no state writes. Safe any day, including
weekends (the placement plan is still computed so Monday's first order can
be reviewed in advance).

Exit codes: 0 = reconciliation completed without action failures
(divergences alone do not fail the run; they are reported in the summary);
1 = Alpaca unreachable, keys missing, or any action failed.

Paper only: the adapter refuses the live Alpaca endpoint. No real money
is ever involved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, time as time_type
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.broker.alpaca_client import (  # noqa: E402
    AlpacaError,
    AlpacaPaperClient,
    sync_status,
)
from src.paper.alpaca_mirror import (  # noqa: E402
    EXITED_STATUS,
    load_mirror_state,
    plan_mirror_actions,
    write_mirror_state,
)
from src.utils.paths import data_root  # noqa: E402
from src.utils.trading_calendar import ET, is_trading_day  # noqa: E402

MARKET_OPEN = time_type(9, 30)
MARKET_CLOSE = time_type(16, 0)

# Execution order: cancel stale legs, close stale positions, then place.
_KIND_ORDER = {"cancel_order": 0, "close_position": 1, "place_bracket": 2}


def market_open_now(now: datetime) -> bool:
    if not is_trading_day(now.date()):
        return False
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def _default_quote(symbol: str, trading_date: str, fallback: float | None) -> tuple[float | None, str]:
    try:
        from src.paper.signals import resolve_quote

        px, source = resolve_quote(symbol, trading_date)
        if px is not None:
            return px, source
    except Exception:
        pass
    return fallback, "local_last_price"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _local_legs(account: dict[str, Any]) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for book, book_legs in (account.get("positions") or {}).items():
        for leg in book_legs or []:
            legs.append(
                {
                    "book": str(book).lower(),
                    "symbol": str(leg.get("symbol") or "").upper(),
                    "direction": str(leg.get("direction") or "LONG").upper(),
                    "shares": leg.get("shares"),
                    "stop": leg.get("stop"),
                    "target": leg.get("target"),
                    "entry": leg.get("avg_entry"),
                    "horizon": leg.get("horizon") or ("Swing" if str(book).lower() == "swing" else "Intraday"),
                    "opened_date": str(leg.get("opened_date") or ""),
                    "last_price": leg.get("last_price"),
                }
            )
    return legs


def run(
    state_dir: str | Path,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
    quote_fn: Any = None,
    client: AlpacaPaperClient | None = None,
) -> tuple[dict[str, Any], int]:
    """Run one mirror reconciliation. Returns (summary, exit_code)."""
    state_dir = Path(state_dir)
    now_et = now or datetime.now(ET)
    today_s = now_et.date().isoformat()

    account = _read_json(state_dir / "paper" / "account.json")
    if account is None:
        return {"date": today_s, "ok": False, "error": "account.json missing or corrupt"}, 1
    legs = _local_legs(account)

    breaker = _read_json(state_dir / "paper" / "circuit_breaker.json") or {}
    day_breaker = breaker.get("day") or {}
    breaker_tripped = bool(day_breaker.get("tripped")) and str(day_breaker.get("date")) == today_s

    mirror_state = load_mirror_state(state_dir)

    if client is None:
        try:
            client = AlpacaPaperClient()
        except ValueError as exc:
            return {"date": today_s, "ok": False, "error": str(exc)}, 1
    try:
        alpaca_positions = client.list_positions()
        alpaca_orders = client.list_orders(status="open")
    except AlpacaError as exc:
        return {"date": today_s, "ok": False, "error": f"Alpaca unreachable: {exc}"}, 1

    quotes: dict[str, float | None] = {}
    quote_sources: dict[str, str] = {}
    for leg in legs:
        symbol = leg["symbol"]
        if not symbol or symbol in quotes:
            continue
        if quote_fn is not None:
            px, source = quote_fn(symbol, today_s, leg.get("last_price"))
        else:
            px, source = _default_quote(symbol, today_s, leg.get("last_price"))
        quotes[symbol] = px
        quote_sources[symbol] = source

    plan = plan_mirror_actions(
        legs,
        alpaca_positions or [],
        alpaca_orders or [],
        mirror_state,
        breaker_tripped_today=breaker_tripped,
        quotes=quotes,
    )
    is_open = market_open_now(now_et)

    summary: dict[str, Any] = {
        "date": today_s,
        "ok": True,
        "dry_run": dry_run,
        "market_open": is_open,
        "breaker_tripped_today": breaker_tripped,
        "local_legs": len(legs),
        "alpaca_positions": len(alpaca_positions or []),
        "alpaca_open_orders": len(alpaca_orders or []),
        "actions": plan["actions"],
        "skipped": plan["skipped"],
        "divergences": plan["divergences"],
        "adopt": plan["adopt"],
        "failures": [],
        "executed": [],
        "deferred": [],
        "quote_sources": quote_sources,
    }

    if dry_run:
        summary["executed"] = []
        summary["deferred"] = [dict(a, would_execute=True) for a in plan["actions"]]
        return summary, 0

    # Apply adopt + mark-exited to the mirror state even when the market
    # is closed (bookkeeping only, no orders).
    for item in plan["adopt"]:
        mirror_state.setdefault("legs", {})[item["leg_key"]] = item["record"]
    for key in plan["mark_exited"]:
        rec = mirror_state.setdefault("legs", {}).setdefault(key, {})
        rec["status"] = EXITED_STATUS

    if not is_open:
        summary["deferred"] = [dict(a, reason="market_closed") for a in plan["actions"]]
        mirror_state["divergences"] = plan["divergences"]
        mirror_state["failures"] = []
        write_mirror_state(state_dir, mirror_state)
        return summary, 0

    failures: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    ordered = sorted(plan["actions"], key=lambda a: _KIND_ORDER.get(a["kind"], 9))
    for action in ordered:
        kind = action["kind"]
        try:
            if kind == "cancel_order":
                client.cancel_order(action["order_id"])
                executed.append({**action, "result": "cancelled"})
            elif kind == "close_position":
                closed = client.close_position(action["symbol"])
                executed.append({**action, "result": "closed" if closed else "already_flat"})
            elif kind == "place_bracket":
                order = client.submit_bracket_order(action["signal"])
                mirror_state.setdefault("legs", {})[action["leg_key"]] = {
                    "status": "placed",
                    "symbol": action["signal"]["symbol"],
                    "direction": action["signal"]["direction"],
                    "shares": action["signal"]["shares"],
                    "stop": action["signal"]["stop"],
                    "target": action["signal"]["target"],
                    "local_entry": action["signal"].get("entry"),
                    "order_id": order.get("id"),
                    "client_order_id": action["signal"]["client_order_id"],
                    "placed_at": now_et.isoformat(),
                }
                executed.append({**action, "result": "placed", "order_id": order.get("id")})
        except (AlpacaError, ValueError) as exc:
            failures.append({**action, "error": str(exc)[:500]})

    mirror_state["divergences"] = plan["divergences"]
    mirror_state["failures"] = failures
    write_mirror_state(state_dir, mirror_state)

    # Best-effort status refresh so the daily report sees fresh Alpaca state.
    try:
        sync_status(client)
    except Exception:
        pass

    summary["executed"] = executed
    summary["failures"] = failures
    summary["ok"] = not failures
    return summary, 0 if not failures else 1


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alpaca_mirror_sync")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; place/cancel/close nothing")
    parser.add_argument("--state-dir", default=None, help="State dir (default: DATA_ROOT)")
    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir) if args.state_dir else data_root()
    summary, code = run(state_dir, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
