"""Paper account persistence under DATA_ROOT/paper/.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Dual books (each holds a LIST of positions; 2026-11-12 limits):
  - ``positions["intraday"]`` — 短线, may hold overnight, max 3 trading
    days then flattened; at most 3 positions
  - ``positions["swing"]`` — 长线, hold overnight; at most 2 positions
  - at most 5 positions total across both books
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

# Per-book position limits (Rae, 2026-11-12: intraday 3, swing 2, 5 total;
# was 1 per book). The per-book notional cap (swing 70% / intraday 40% x
# confidence) is shared by the whole book, not per position — enforced in
# _try_entry — and the portfolio 3% planned-risk cap is the total brake.
MAX_POSITIONS_PER_BOOK: dict[str, int] = {BOOK_INTRADAY: 3, BOOK_SWING: 2}
MAX_TOTAL_POSITIONS = 5


def max_positions_for_book(book: str) -> int:
    """Max open positions in one book (intraday 3, swing 2)."""
    return MAX_POSITIONS_PER_BOOK.get(book, 1)

# Pre-2026-09-21 history is unrecoverable: ledger-detail exits sum to
# +$774.54 while the account's realized PnL is +$184.84. The difference is
# a real historical loss from before the surviving records begin — not a
# bookkeeping error — verified 2026-11-05. It is carried as an explicit
# adjustment (fix 4) instead of being silently buried in realized PnL.
HISTORY_ADJUSTMENT_AMOUNT = -589.73
HISTORY_ADJUSTMENT_REASON = (
    "2026-09-21 之前无法恢复的历史亏损（纸盘迁移时核对："
    "分笔明细合计与账户 realized 之差）"
)
# Verification-epoch start: the day the six execution/record fixes landed.
# Statistics from this date on are the validation record for live trading.
VERIFICATION_EPOCH_START = "2026-11-12"


def record_history_adjustment(
    account: dict[str, Any],
    *,
    epoch_start: str = VERIFICATION_EPOCH_START,
) -> bool:
    """Record the pre-2026-09-21 history adjustment + epoch start.

    Idempotent: returns False when already recorded. Does NOT rewrite any
    historical leg or change equity / realized PnL — the adjustment is an
    explicit, visible record, and the epoch marker lets later statistics
    be computed from the fix date onward.
    """
    if account.get("history_adjustment") is not None:
        return False
    account["history_adjustment"] = {
        "amount": HISTORY_ADJUSTMENT_AMOUNT,
        "currency": "USD",
        "reason": HISTORY_ADJUSTMENT_REASON,
        "recorded_date": VERIFICATION_EPOCH_START,
        "recorded_at": _now_iso(),
    }
    account["verification_epoch"] = {
        "started_on": epoch_start,
        "note": (
            "新验证期起点：六项执行/记录修复提交日；"
            "此前数据不计入验证期统计"
        ),
    }
    append_journal(
        account,
        {
            "type": "history_adjustment",
            "trading_date": epoch_start,
            "amount": HISTORY_ADJUSTMENT_AMOUNT,
            "reason": HISTORY_ADJUSTMENT_REASON,
            "verification_epoch_started_on": epoch_start,
        },
    )
    return True


def ensure_history_adjustment(account: dict[str, Any]) -> bool:
    """Backfill the adjustment record for accounts with real trade history.

    Fresh accounts (no trades) carry no pre-history gap and are untouched.
    """
    if not account.get("trades"):
        return False
    return record_history_adjustment(account)


def verification_epoch_summary(account: dict[str, Any]) -> dict[str, Any]:
    """Closed-leg statistics from the verification-epoch start onward."""
    epoch = account.get("verification_epoch") or {}
    start = epoch.get("started_on")
    legs = [
        t
        for t in (account.get("trades") or [])
        if t.get("action") in ("EXIT", "SCALE_OUT")
        and (not start or (t.get("trading_date") or "") >= start)
    ]
    wins = [t for t in legs if float(t.get("pnl") or 0) > 0]
    losses = [t for t in legs if float(t.get("pnl") or 0) < 0]
    realized = round(sum(float(t.get("pnl") or 0) for t in legs), 2)
    r_values = [
        float(t.get("r_multiple"))
        for t in legs
        if t.get("r_multiple") is not None
    ]
    return {
        "started_on": start,
        "closed_legs": len(legs),
        "wins": len(wins),
        "losses": len(losses),
        "realized_pnl": realized,
        "avg_win": round(sum(float(t.get("pnl") or 0) for t in wins) / len(wins), 2)
        if wins
        else None,
        "avg_loss": round(sum(float(t.get("pnl") or 0) for t in losses) / len(losses), 2)
        if losses
        else None,
        "r_multiples": r_values,
        "expectancy_r": round(sum(r_values) / len(r_values), 3) if r_values else None,
    }

DEFAULT_PARAMS: dict[str, Any] = {
    "risk_pct": 1.0,
    "max_position_pct": 25.0,
    "prefer_swing_if_no_intraday": True,  # legacy single-book fallback
    "dual_books": True,
    "miss_threshold_pct": 1.0,
    "force_exit_intraday_at_close": True,
    # Reject stop/target/entry fills farther than this from entry/stop/target/last.
    "max_price_deviation_pct": 10.0,
    # Capital allocation (agent-chosen each tick; bounds only)
    "cash_reserve_min_pct": 20.0,
    "cash_reserve_max_pct": 40.0,
    "smart_allocation": True,
    # Stop execution / risk controls (fail-closed)
    # Stop-market fills never print worse than stop ± this (unless a true gap
    # opened through the stop, which fills at the open — honestly modeled).
    "max_stop_slippage_pct": 0.5,
    # Hard per-trade backstop when the stop is missing/invalid: exit at market
    # once one trade's unrealized loss reaches this % of equity.
    "max_trade_loss_equity_pct": 2.0,
    # Reject entries whose stop is tighter than this % of entry price.
    "min_stop_dist_pct": 0.8,
    # Sizing reserve: added to per-share risk so tight stops size smaller.
    "slippage_reserve_pct": 0.2,
    # After a stop-out, don't re-enter the same symbol in the same book today.
    "stop_cooldown_enabled": True,
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
        "positions": {BOOK_INTRADAY: [], BOOK_SWING: []},
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
    """Keep ``position`` in sync: first intraday leg, else first swing leg."""
    positions = account.get("positions")
    if not isinstance(positions, dict):
        account["position"] = None
        return
    for book in BOOKS:
        legs = positions.get(book) or []
        if isinstance(legs, list) and legs:
            account["position"] = legs[0]
            return
        if isinstance(legs, dict):
            account["position"] = legs
            return
    account["position"] = None


def _coerce_book_legs(value: Any) -> list[dict[str, Any]]:
    """Normalize one book's stored value to a list of position dicts.

    Accepts the current list form, the pre-2026-11-12 single dict (or
    None) — migrated in place by :func:`ensure_positions`.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [p for p in value if isinstance(p, dict)]
    return []


def ensure_positions(account: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy single ``position`` → dual ``positions`` books.

    Each book holds a list of position dicts (max intraday 3 / swing 2).
    """
    positions = account.get("positions")
    if not isinstance(positions, dict):
        positions = {BOOK_INTRADAY: [], BOOK_SWING: []}
        account["positions"] = positions
    for book in BOOKS:
        positions[book] = _coerce_book_legs(positions.get(book))

    legacy = account.get("position")
    if legacy and not positions.get(BOOK_INTRADAY) and not positions.get(BOOK_SWING):
        horizon = str(legacy.get("horizon") or "Intraday").lower()
        book = BOOK_SWING if "swing" in horizon else BOOK_INTRADAY
        legacy = dict(legacy)
        legacy.setdefault("book", book)
        positions[book] = [legacy]

    sync_legacy_position(account)
    return positions


def get_positions(account: dict[str, Any], book: str) -> list[dict[str, Any]]:
    """All open positions in ``book`` (copy of the stored list)."""
    return list(ensure_positions(account).get(book) or [])


def get_position(account: dict[str, Any], book: str) -> dict[str, Any] | None:
    """First open position in ``book`` (compat: was the only position)."""
    legs = get_positions(account, book)
    return legs[0] if legs else None


def set_position(
    account: dict[str, Any], book: str, position: dict[str, Any] | None
) -> None:
    """Replace a book's legs with a single position (or clear it).

    Compat helper for single-position callers; multi-leg writes use
    :func:`add_position` / :func:`remove_position`.
    """
    positions = ensure_positions(account)
    if book not in BOOKS:
        raise ValueError(f"Unknown book: {book}")
    if position is not None:
        position = dict(position)
        position["book"] = book
        positions[book] = [position]
    else:
        positions[book] = []
    sync_legacy_position(account)


def add_position(
    account: dict[str, Any], book: str, position: dict[str, Any]
) -> None:
    """Append one leg to a book (caller enforces the per-book limit)."""
    positions = ensure_positions(account)
    if book not in BOOKS:
        raise ValueError(f"Unknown book: {book}")
    position = dict(position)
    position["book"] = book
    positions[book].append(position)
    sync_legacy_position(account)


def remove_position(
    account: dict[str, Any], book: str, symbol: str
) -> dict[str, Any] | None:
    """Remove one leg (by symbol) from a book; returns it or None."""
    positions = ensure_positions(account)
    if book not in BOOKS:
        raise ValueError(f"Unknown book: {book}")
    sym = str(symbol or "").upper()
    legs = positions[book]
    for i, leg in enumerate(legs):
        if str(leg.get("symbol") or "").upper() == sym:
            removed = legs.pop(i)
            sync_legacy_position(account)
            return removed
    return None


def open_positions(account: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    positions = ensure_positions(account)
    out: list[tuple[str, dict[str, Any]]] = []
    for book in BOOKS:
        for leg in positions.get(book) or []:
            out.append((book, leg))
    return out


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
    ensure_history_adjustment(data)
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
        px = _safe_float(prices.get(sym))
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

    # Equity = cash + signed market value of every open position.
    # For shorts, cash already holds the sale proceeds, so the position
    # contributes its (negative) market value — adding only the unrealized
    # PnL would double-count the proceeds and overstate equity.
    equity = cash
    for _book, pos in open_positions(account):
        shares = float(pos.get("shares") or 0)
        px = _safe_float(pos.get("last_price"))
        direction = (pos.get("direction") or "LONG").upper()
        if px is None or px <= 0:
            continue
        if direction == "SHORT":
            equity -= shares * px
        else:
            equity += shares * px

    if equity != equity or equity in (float("inf"), float("-inf")):
        # Never persist a poisoned equity value: keep the last good one so
        # risk math (sizing, breakers) keeps working off real numbers.
        equity = _finite(account.get("equity"), cash)

    account["unrealized_pnl"] = round(_finite(unrealized), 2)
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
    """Coerce to float; return None for missing/invalid/non-finite values.

    NaN and inf quotes must never flow into PnL/equity math: ``nan or 0``
    is ``nan`` (NaN is truthy) and ``nan <= 0`` is False, so without this
    guard a single bad quote poisons equity for every later tick.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _finite(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a float, or ``default`` when it is not finite."""
    result = _safe_float(value)
    return default if result is None else result


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
