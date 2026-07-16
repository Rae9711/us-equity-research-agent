"""Paper account persistence under DATA_ROOT/paper/.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Dual books:
  - ``positions["intraday"]`` — 短线, flatten at EOD
  - ``positions["swing"]`` — 长线, hold overnight
Legacy ``position`` mirrors the primary open book (intraday preferred) for compat.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from pytz import timezone

from src.utils.paths import data_root

logger = logging.getLogger(__name__)

ET = timezone("America/New_York")
STARTING_CASH = 10_000.0
ACCOUNT_FILENAME = "account.json"

BOOK_INTRADAY = "intraday"
BOOK_SWING = "swing"
BOOKS = (BOOK_INTRADAY, BOOK_SWING)

DEFAULT_PARAMS: dict[str, Any] = {
    "risk_pct": 1.0,
    "max_position_pct": 25.0,
    "prefer_swing_if_no_intraday": True,  # legacy single-book fallback
    "dual_books": True,
    "miss_threshold_pct": 1.0,
    "force_exit_intraday_at_close": True,
    # Reject stop/target/entry fills farther than this from entry/stop/target/last.
    "max_price_deviation_pct": 30.0,
    # Capital allocation (agent-chosen each tick; bounds only)
    "cash_reserve_min_pct": 20.0,
    "cash_reserve_max_pct": 40.0,
    "smart_allocation": True,
}


def paper_dir() -> Path:
    d = data_root() / "paper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def account_path() -> Path:
    return paper_dir() / ACCOUNT_FILENAME


def _now_iso() -> str:
    return datetime.now(ET).isoformat()


def default_account() -> dict[str, Any]:
    return {
        "advisory": True,
        "advisory_zh": "模拟交易 · 不构成投资建议",
        "starting_cash": STARTING_CASH,
        "cash": STARTING_CASH,
        "equity": STARTING_CASH,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "position": None,  # legacy alias
        "positions": {BOOK_INTRADAY: None, BOOK_SWING: None},
        "trades": [],
        "decisions": [],
        "equity_curve": [
            {
                "ts": _now_iso(),
                "equity": STARTING_CASH,
                "cash": STARTING_CASH,
                "label": "seed",
            }
        ],
        "journal": [],
        "params": dict(DEFAULT_PARAMS),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def sync_legacy_position(account: dict[str, Any]) -> None:
    """Keep ``position`` in sync: prefer intraday, else swing."""
    positions = account.get("positions")
    if not isinstance(positions, dict):
        account["position"] = None
        return
    account["position"] = positions.get(BOOK_INTRADAY) or positions.get(BOOK_SWING)


def ensure_positions(account: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy single ``position`` → dual ``positions`` books."""
    positions = account.get("positions")
    if not isinstance(positions, dict):
        positions = {BOOK_INTRADAY: None, BOOK_SWING: None}
        account["positions"] = positions
    for book in BOOKS:
        positions.setdefault(book, None)

    legacy = account.get("position")
    if legacy and not positions.get(BOOK_INTRADAY) and not positions.get(BOOK_SWING):
        horizon = str(legacy.get("horizon") or "Intraday").lower()
        book = BOOK_SWING if "swing" in horizon else BOOK_INTRADAY
        legacy = dict(legacy)
        legacy.setdefault("book", book)
        positions[book] = legacy

    sync_legacy_position(account)
    return positions


def get_position(account: dict[str, Any], book: str) -> dict[str, Any] | None:
    return ensure_positions(account).get(book)


def set_position(
    account: dict[str, Any], book: str, position: dict[str, Any] | None
) -> None:
    positions = ensure_positions(account)
    if book not in BOOKS:
        raise ValueError(f"Unknown book: {book}")
    if position is not None:
        position = dict(position)
        position["book"] = book
    positions[book] = position
    sync_legacy_position(account)


def open_positions(account: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    positions = ensure_positions(account)
    return [(b, p) for b, p in ((book, positions.get(book)) for book in BOOKS) if p]


def load_account() -> dict[str, Any]:
    path = account_path()
    if not path.exists():
        acct = default_account()
        save_account(acct)
        return acct
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Corrupt paper account; reseeding")
        data = default_account()
        save_account(data)
        return data
    # Ensure required keys
    base = default_account()
    for key, val in base.items():
        if key not in data:
            data[key] = deepcopy(val)
    if not isinstance(data.get("params"), dict):
        data["params"] = dict(DEFAULT_PARAMS)
    else:
        for k, v in DEFAULT_PARAMS.items():
            data["params"].setdefault(k, v)
    ensure_positions(data)
    return data


def save_account(account: dict[str, Any]) -> Path:
    account = dict(account)
    ensure_positions(account)
    account["updated_at"] = _now_iso()
    account["advisory"] = True
    account["advisory_zh"] = "模拟交易 · 不构成投资建议"
    path = account_path()
    path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def mark_to_market(
    account: dict[str, Any],
    last_price: float | None = None,
    *,
    price_by_symbol: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Update unrealized PnL + equity from open position marks."""
    ensure_positions(account)
    cash = float(account.get("cash") or 0.0)
    realized = float(account.get("realized_pnl") or 0.0)
    prices = {str(k).upper(): float(v) for k, v in (price_by_symbol or {}).items()}
    open_rows = open_positions(account)
    open_syms = {str(p.get("symbol") or "").upper() for _, p in open_rows}
    unrealized = 0.0

    for _book, pos in open_rows:
        sym = str(pos.get("symbol") or "").upper()
        px = prices.get(sym)
        if px is None and last_price is not None and last_price > 0 and len(open_syms) <= 1:
            px = float(last_price)
        if px is None:
            px = _safe_float(pos.get("last_price"))
        if px is None or px <= 0:
            continue
        shares = float(pos.get("shares") or 0)
        avg = float(pos.get("avg_entry") or 0)
        direction = (pos.get("direction") or "LONG").upper()
        if direction == "SHORT":
            u = (avg - px) * shares
        else:
            u = (px - avg) * shares
        unrealized += u
        pos["last_price"] = round(px, 4)
        pos["unrealized_pnl"] = round(u, 2)
        pos["market_value"] = round(shares * px, 2)

    # Match prior single-position equity: cash + long MV + short unrealized
    equity = cash
    for _book, pos in open_positions(account):
        shares = float(pos.get("shares") or 0)
        px = _safe_float(pos.get("last_price"))
        direction = (pos.get("direction") or "LONG").upper()
        if px is None or px <= 0:
            continue
        if direction == "SHORT":
            equity += float(pos.get("unrealized_pnl") or 0)
        else:
            equity += shares * px

    account["unrealized_pnl"] = round(unrealized, 2)
    account["equity"] = round(equity, 2)
    account["realized_pnl"] = round(realized, 2)
    account["total_pnl"] = round(
        equity - float(account.get("starting_cash") or STARTING_CASH), 2
    )
    account["total_return_pct"] = round(
        (equity / float(account.get("starting_cash") or STARTING_CASH) - 1.0) * 100.0, 2
    )
    sync_legacy_position(account)
    return account


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def append_equity_point(account: dict[str, Any], *, label: str = "tick") -> None:
    curve = account.setdefault("equity_curve", [])
    curve.append(
        {
            "ts": _now_iso(),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "label": label,
        }
    )
    # Cap history
    if len(curve) > 500:
        account["equity_curve"] = curve[-500:]


def append_decision(account: dict[str, Any], decision: dict[str, Any]) -> None:
    rows = account.setdefault("decisions", [])
    row = dict(decision)
    row.setdefault("ts", _now_iso())
    row["advisory"] = True
    rows.append(row)
    if len(rows) > 300:
        account["decisions"] = rows[-300:]


def append_trade(account: dict[str, Any], trade: dict[str, Any]) -> None:
    rows = account.setdefault("trades", [])
    row = dict(trade)
    row.setdefault("ts", _now_iso())
    row["advisory"] = True
    rows.append(row)
    if len(rows) > 300:
        account["trades"] = rows[-300:]


def append_journal(account: dict[str, Any], note: dict[str, Any]) -> None:
    rows = account.setdefault("journal", [])
    row = dict(note)
    row.setdefault("ts", _now_iso())
    row["advisory"] = True
    rows.append(row)
    if len(rows) > 200:
        account["journal"] = rows[-200:]
