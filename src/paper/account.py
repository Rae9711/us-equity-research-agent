"""Paper account persistence under DATA_ROOT/paper/.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
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

DEFAULT_PARAMS: dict[str, Any] = {
    "risk_pct": 1.0,
    "max_position_pct": 25.0,
    "prefer_swing_if_no_intraday": True,
    "miss_threshold_pct": 1.0,
    "force_exit_intraday_at_close": True,
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
        "position": None,
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
    return data


def save_account(account: dict[str, Any]) -> Path:
    account = dict(account)
    account["updated_at"] = _now_iso()
    account["advisory"] = True
    account["advisory_zh"] = "模拟交易 · 不构成投资建议"
    path = account_path()
    path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def mark_to_market(account: dict[str, Any], last_price: float | None) -> dict[str, Any]:
    """Update unrealized PnL + equity from open position mark."""
    pos = account.get("position")
    cash = float(account.get("cash") or 0.0)
    realized = float(account.get("realized_pnl") or 0.0)
    unrealized = 0.0
    if pos and last_price is not None and last_price > 0:
        shares = float(pos.get("shares") or 0)
        avg = float(pos.get("avg_entry") or 0)
        direction = (pos.get("direction") or "LONG").upper()
        if direction == "SHORT":
            unrealized = (avg - last_price) * shares
            equity = cash + unrealized  # short: cash already includes sale proceeds
        else:
            unrealized = (last_price - avg) * shares
            equity = cash + shares * last_price
        pos["last_price"] = round(last_price, 4)
        pos["unrealized_pnl"] = round(unrealized, 2)
        pos["market_value"] = round(shares * last_price, 2)
    else:
        equity = cash
    account["unrealized_pnl"] = round(unrealized, 2)
    account["equity"] = round(equity, 2)
    account["realized_pnl"] = round(realized, 2)
    account["total_pnl"] = round(equity - float(account.get("starting_cash") or STARTING_CASH), 2)
    account["total_return_pct"] = round(
        (equity / float(account.get("starting_cash") or STARTING_CASH) - 1.0) * 100.0, 2
    )
    return account


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
