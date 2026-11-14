"""Alpaca paper mirror: keep the Alpaca paper account in step with the local paper book.

Why: the local paper sim is the book of record, but it never touches a broker.
The mirror submits the same legs to Alpaca paper as resting bracket orders
(entry market + stop + target), so fills, partial fills and TP/SL exits are
exercised against a real broker API -- a dress rehearsal for live trading,
still with paper money only.

Design:
- Reconciliation, not event-driven: each run compares the local open legs
  (account.json) with Alpaca positions + open orders and the mirror state
  file, then places / cancels / closes only what diverges. A failed run is
  retried by the next run; nothing here can break local paper trading
  (the mirror job is a separate process; the tick path never calls it).
- The local book is the authority. If Alpaca's bracket fills TP/SL while
  the local leg is still open, that is recorded as a divergence and
  reported -- the mirror never force-closes a local leg.
- Safety: paper-only client (live endpoint refused); no placement when the
  day breaker tripped; no placement when the live quote is already beyond
  the stop or the target (a bracket placed there would exit immediately);
  market hours only (Alpaca rejects market orders outside 9:30-16:00 ET).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.trading_calendar import ET

MIRROR_STATE_FILENAME = "alpaca_mirror.json"

# Mirror-record statuses that mean "this leg is live in Alpaca".
PLACED_STATUSES = ("placed", "filled", "adopted")

# A leg whose Alpaca bracket already exited (TP/SL filled or cancelled)
# while the local leg is still open: terminal, never re-placed.
EXITED_STATUS = "alpaca_exited"

# Entry-basis divergence threshold: Alpaca's fill vs the local entry price.
ENTRY_BASIS_TOLERANCE_PCT = 5.0


def leg_key(book: str, symbol: str, opened_date: str) -> str:
    return f"{str(book).lower()}:{str(symbol).upper()}:{opened_date}"


def mirror_client_order_id(book: str, symbol: str, opened_date: str) -> str:
    # Deterministic: re-running the mirror for the same leg reuses the same
    # client order id, so a retry never double-places.
    return f"tos-mirror-{str(symbol).lower()}-{str(book).lower()}-{opened_date}"


def fresh_mirror_state() -> dict[str, Any]:
    return {"legs": {}, "divergences": [], "failures": [], "updated_at": None}


def mirror_state_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "paper" / MIRROR_STATE_FILENAME


def load_mirror_state(state_dir: str | Path) -> dict[str, Any]:
    path = mirror_state_path(state_dir)
    if not path.exists():
        return fresh_mirror_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fresh_mirror_state()
    if not isinstance(data, dict):
        return fresh_mirror_state()
    state = fresh_mirror_state()
    state.update(data)
    if not isinstance(state.get("legs"), dict):
        state["legs"] = {}
    if not isinstance(state.get("divergences"), list):
        state["divergences"] = []
    if not isinstance(state.get("failures"), list):
        state["failures"] = []
    return state


def write_mirror_state(state_dir: str | Path, state: dict[str, Any]) -> Path:
    path = mirror_state_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(ET).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _orders_for_symbol(orders: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    return [o for o in orders if str(o.get("symbol") or "").upper() == symbol]


def _position_for_symbol(
    positions: list[dict[str, Any]], symbol: str
) -> dict[str, Any] | None:
    for p in positions:
        if str(p.get("symbol") or "").upper() == symbol:
            return p
    return None


def _beyond_reason(direction: str, quote: float, stop: float, target: float) -> str | None:
    if direction == "LONG":
        if quote <= stop:
            return "beyond_stop"
        if quote >= target:
            return "beyond_target"
    else:
        if quote >= stop:
            return "beyond_stop"
        if quote <= target:
            return "beyond_target"
    return None


def _as_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def plan_mirror_actions(
    local_legs: list[dict[str, Any]],
    alpaca_positions: list[dict[str, Any]],
    alpaca_open_orders: list[dict[str, Any]],
    mirror_state: dict[str, Any],
    *,
    breaker_tripped_today: bool,
    quotes: dict[str, float | None],
) -> dict[str, Any]:
    """Plan the reconciliation between the local book and Alpaca paper.

    Returns {"actions", "skipped", "divergences", "adopt", "mark_exited"}.
    Pure function: no network, no file writes. The job executes ``actions``
    in the order cancels -> closes -> placements and applies ``adopt`` /
    ``mark_exited`` to the mirror state.
    """
    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    adopt: list[dict[str, Any]] = []
    mark_exited: list[str] = []
    legs_state = mirror_state.get("legs") or {}
    local_symbols: set[str] = set()

    for leg in local_legs:
        book = str(leg.get("book") or "swing").lower()
        symbol = str(leg.get("symbol") or "").upper()
        opened = str(leg.get("opened_date") or "")
        if not symbol or not opened:
            skipped.append({"leg_key": None, "symbol": symbol, "reason": "missing_identity"})
            continue
        key = leg_key(book, symbol, opened)
        local_symbols.add(symbol)
        direction = str(leg.get("direction") or "LONG").upper()
        shares = _as_float(leg.get("shares")) or 0
        stop = _as_float(leg.get("stop"))
        target = _as_float(leg.get("target"))
        entry = _as_float(leg.get("entry"))
        horizon = leg.get("horizon") or ("Swing" if book == "swing" else "Intraday")
        record = legs_state.get(key) or {}
        status = record.get("status")
        pos = _position_for_symbol(alpaca_positions, symbol)
        orders = _orders_for_symbol(alpaca_open_orders, symbol)

        if status == EXITED_STATUS:
            # Terminal: the Alpaca bracket already exited while the local
            # leg is still open. Keep reporting it, never re-place.
            divergences.append(
                {
                    "type": "alpaca_flat_local_open",
                    "leg_key": key,
                    "symbol": symbol,
                    "detail": "Alpaca bracket already exited (stop/target filled or cancelled) while the local leg is still open.",
                }
            )
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "alpaca_exited"})
            continue

        if status in PLACED_STATUSES and (pos is not None or orders):
            # Live in Alpaca: check the position still matches the leg.
            if pos is not None:
                alpaca_qty = _as_float(pos.get("qty")) or 0
                if alpaca_qty != shares:
                    divergences.append(
                        {
                            "type": "qty_mismatch",
                            "leg_key": key,
                            "symbol": symbol,
                            "detail": f"Alpaca qty {alpaca_qty} != local shares {shares}.",
                        }
                    )
                alpaca_entry = _as_float(pos.get("avg_entry_price"))
                if entry and alpaca_entry and abs(alpaca_entry - entry) / entry * 100 > ENTRY_BASIS_TOLERANCE_PCT:
                    divergences.append(
                        {
                            "type": "entry_basis_differs",
                            "leg_key": key,
                            "symbol": symbol,
                            "detail": f"Alpaca entry {alpaca_entry} vs local entry {entry} (>{ENTRY_BASIS_TOLERANCE_PCT:.0f}% apart); P&L bases differ.",
                        }
                    )
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "mirrored"})
            continue

        if status in PLACED_STATUSES:
            # Record says placed, but Alpaca is flat for this symbol while
            # the local leg is still open: the bracket most likely filled
            # its stop or target (or was cancelled). Divergence only.
            divergences.append(
                {
                    "type": "alpaca_flat_local_open",
                    "leg_key": key,
                    "symbol": symbol,
                    "detail": "Alpaca shows no position and no open orders for this symbol while the local leg is still open; the bracket probably filled its stop or target.",
                }
            )
            mark_exited.append(key)
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "alpaca_exited"})
            continue

        # No live record: adopt an already-present Alpaca position/order
        # when it matches the leg (self-heals a lost state file), else
        # diverge rather than double-place.
        if pos is not None:
            pos_dir = str(pos.get("direction") or "").upper()
            pos_qty = _as_float(pos.get("qty")) or 0
            if pos_dir == direction and pos_qty == shares:
                adopt.append(
                    {
                        "leg_key": key,
                        "record": {
                            "status": "adopted",
                            "symbol": symbol,
                            "direction": direction,
                            "shares": shares,
                            "stop": stop,
                            "target": target,
                            "local_entry": entry,
                            "alpaca_entry_price": _as_float(pos.get("avg_entry_price")),
                            "order_id": None,
                            "client_order_id": None,
                            "note": "Adopted an existing Alpaca position that matches the local leg.",
                        },
                    }
                )
                skipped.append({"leg_key": key, "symbol": symbol, "reason": "adopted"})
                continue
            divergences.append(
                {
                    "type": "alpaca_position_mismatch",
                    "leg_key": key,
                    "symbol": symbol,
                    "detail": f"Alpaca holds {pos_dir} {pos_qty} with no matching mirror record; not placing a second bracket.",
                }
            )
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "position_mismatch"})
            continue
        if orders:
            o = orders[0]
            o_dir = str(o.get("direction") or "").upper()
            o_qty = _as_float(o.get("qty")) or 0
            if o_dir == direction and o_qty == shares:
                adopt.append(
                    {
                        "leg_key": key,
                        "record": {
                            "status": "placed",
                            "symbol": symbol,
                            "direction": direction,
                            "shares": shares,
                            "stop": stop,
                            "target": target,
                            "local_entry": entry,
                            "order_id": o.get("id"),
                            "client_order_id": o.get("client_order_id"),
                            "note": "Adopted a resting Alpaca bracket that matches the local leg.",
                        },
                    }
                )
                skipped.append({"leg_key": key, "symbol": symbol, "reason": "adopted"})
                continue
            divergences.append(
                {
                    "type": "alpaca_orders_mismatch",
                    "leg_key": key,
                    "symbol": symbol,
                    "detail": "Alpaca has open orders for this symbol that do not match the local leg; not placing a second bracket.",
                }
            )
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "orders_mismatch"})
            continue

        # Needs a fresh bracket.
        if stop is None or target is None or shares <= 0:
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "bad_plan"})
            continue
        if breaker_tripped_today:
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "breaker_tripped"})
            continue
        quote = quotes.get(symbol)
        if quote is None:
            skipped.append({"leg_key": key, "symbol": symbol, "reason": "no_quote"})
            continue
        beyond = _beyond_reason(direction, quote, stop, target)
        if beyond:
            skipped.append({"leg_key": key, "symbol": symbol, "reason": beyond})
            continue
        actions.append(
            {
                "kind": "place_bracket",
                "leg_key": key,
                "signal": {
                    "symbol": symbol,
                    "direction": direction,
                    "shares": int(shares),
                    "stop": stop,
                    "target": target,
                    "entry": entry,
                    "horizon": horizon,
                    "client_order_id": mirror_client_order_id(book, symbol, opened),
                },
            }
        )

    # Alpaca-side sweep: anything in Alpaca with no open local leg gets
    # closed (positions) / cancelled (resting orders).
    handled_close_symbols: set[str] = set()
    for pos in alpaca_positions:
        symbol = str(pos.get("symbol") or "").upper()
        if not symbol or symbol in local_symbols or symbol in handled_close_symbols:
            continue
        handled_close_symbols.add(symbol)
        for o in _orders_for_symbol(alpaca_open_orders, symbol):
            actions.append(
                {"kind": "cancel_order", "order_id": o.get("id"), "symbol": symbol, "reason": "no_local_leg"}
            )
        actions.append({"kind": "close_position", "symbol": symbol, "reason": "no_local_leg"})
    for o in alpaca_open_orders:
        symbol = str(o.get("symbol") or "").upper()
        if not symbol or symbol in local_symbols or symbol in handled_close_symbols:
            continue
        actions.append(
            {"kind": "cancel_order", "order_id": o.get("id"), "symbol": symbol, "reason": "no_local_leg"}
        )

    return {
        "actions": actions,
        "skipped": skipped,
        "divergences": divergences,
        "adopt": adopt,
        "mark_exited": mark_exited,
    }
