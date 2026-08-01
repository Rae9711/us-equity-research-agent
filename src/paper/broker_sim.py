"""Paper broker — simulate fills at last/current price.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
Models adverse slippage (bps) + optional commission for Mag7/liquid ETFs.

Supports dual books via ``book``: ``intraday`` (短线) | ``swing`` (长线).
"""

from __future__ import annotations

from typing import Any

from src.paper.account import (
    BOOK_INTRADAY,
    BOOKS,
    STARTING_CASH,
    append_trade,
    ensure_positions,
    get_position,
    mark_to_market,
    set_position,
)
from src.paper.price_guard import (
    DEFAULT_MAX_DEVIATION_PCT,
    anchors_from_position,
    anchors_from_signal,
    collect_price_anchors,
    is_sane_fill_price,
)


class NonsensePriceError(ValueError):
    """Fill price is absurd vs entry/stop/target anchors."""


class InsufficientCashError(ValueError):
    """Not enough cash to open the requested size."""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frictions(account: dict[str, Any]) -> tuple[float, float, float]:
    """Return (slippage_bps, commission_per_trade, commission_per_share)."""
    params = account.get("params") or {}
    slip = float(params.get("slippage_bps") or 0.0)
    c_trade = float(params.get("commission_per_trade") or 0.0)
    c_share = float(params.get("commission_per_share") or 0.0)
    return max(0.0, slip), max(0.0, c_trade), max(0.0, c_share)


def apply_slippage(price: float, *, direction: str, is_entry: bool, slippage_bps: float) -> float:
    """Adverse slippage: you always fill a little worse than the quote.

    Entry LONG / exit SHORT → pay up; entry SHORT / exit LONG → receive less.
    """
    if slippage_bps <= 0 or price <= 0:
        return price
    factor = slippage_bps / 10_000.0
    d = (direction or "LONG").upper()
    pay_up = (d == "LONG" and is_entry) or (d == "SHORT" and not is_entry)
    return round(price * (1.0 + factor) if pay_up else price * (1.0 - factor), 6)


def _commission(shares: int, c_trade: float, c_share: float) -> float:
    if shares <= 0:
        return 0.0
    return round(c_trade + shares * c_share, 4)


def _resolve_book(book: str | None, signal: dict[str, Any] | None) -> str:
    if book and book in BOOKS:
        return book
    horizon = str((signal or {}).get("horizon") or "").lower()
    if "swing" in horizon:
        return "swing"
    return BOOK_INTRADAY


def size_shares(
    *,
    cash: float,
    equity: float,
    entry_price: float,
    stop_price: float | None,
    direction: str,
    risk_pct: float = 1.0,
    max_position_pct: float = 25.0,
    max_notional: float | None = None,
) -> int:
    """Risk-based share count; floor to whole shares; 0 if unaffordable.

    ``max_notional`` (optional) caps size by allocation deployable cash.
    """
    if entry_price <= 0 or cash <= 0 or equity <= 0:
        return 0
    risk_budget = equity * (risk_pct / 100.0)
    stop = _safe_float(stop_price)
    if stop is not None and stop > 0:
        per_share_risk = abs(entry_price - stop)
    else:
        per_share_risk = entry_price * 0.02  # 2% fallback
    if per_share_risk <= 0:
        return 0
    by_risk = int(risk_budget // per_share_risk)
    max_cap_notional = equity * (max_position_pct / 100.0)
    if max_notional is not None:
        max_cap_notional = min(max_cap_notional, max(0.0, float(max_notional)))
    by_cap = int(max_cap_notional // entry_price)
    by_cash = int(cash // entry_price)
    shares = max(0, min(by_risk, by_cap, by_cash))
    return shares


def execute_entry(
    account: dict[str, Any],
    *,
    symbol: str,
    direction: str,
    price: float,
    shares: int,
    stop: float | None = None,
    target: float | None = None,
    entry_zone: dict[str, Any] | None = None,
    signal: dict[str, Any] | None = None,
    reason: str = "",
    trading_date: str | None = None,
    book: str | None = None,
) -> dict[str, Any]:
    """Open a position at ``price`` in the given book. Raises InsufficientCashError."""
    ensure_positions(account)
    book_key = _resolve_book(book, signal)
    if get_position(account, book_key):
        raise ValueError(f"Already holding a {book_key} position")
    if shares <= 0:
        raise InsufficientCashError("shares must be > 0")
    if price <= 0:
        raise ValueError("price must be > 0")

    params = account.get("params") or {}
    max_dev = float(
        params.get("max_price_deviation_pct") or DEFAULT_MAX_DEVIATION_PCT
    )
    ok, reject = is_sane_fill_price(
        price,
        anchors=collect_price_anchors(
            entry=_safe_float((signal or {}).get("entry_price")),
            stop=_safe_float(stop),
            target=_safe_float(target),
            extra=anchors_from_signal(signal),
        ),
        max_deviation_pct=max_dev,
    )
    if not ok:
        raise NonsensePriceError(reject or "nonsense fill price")

    direction = (direction or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"Unsupported direction: {direction}")

    slip_bps, c_trade, c_share = _frictions(account)
    fill = apply_slippage(
        price, direction=direction, is_entry=True, slippage_bps=slip_bps
    )
    commission = _commission(shares, c_trade, c_share)
    notional = round(shares * fill, 2)
    cash = float(account.get("cash") or 0.0)

    if direction == "LONG":
        need = notional + commission
        if cash + 1e-9 < need:
            raise InsufficientCashError(
                f"Need ${need:.2f} cash, have ${cash:.2f}"
            )
        account["cash"] = round(cash - need, 2)
    else:  # SHORT: require cash collateral ≥ notional; credit sale proceeds − comm.
        if cash + 1e-9 < notional:
            raise InsufficientCashError(
                f"Need ${notional:.2f} collateral cash for SHORT, have ${cash:.2f}"
            )
        account["cash"] = round(cash + notional - commission, 2)

    sig = signal or {}
    horizon = sig.get("horizon") or (
        "Swing" if book_key == "swing" else "Intraday"
    )
    stop_f = _safe_float(stop)
    risk_per_share = (
        abs(fill - stop_f) if stop_f is not None and stop_f > 0 else fill * 0.02
    )
    position = {
        "symbol": symbol.upper(),
        "direction": direction,
        "shares": shares,
        "initial_shares": shares,
        "avg_entry": round(fill, 4),
        "stop": stop_f,
        "initial_stop": stop_f,
        "risk_per_share": round(risk_per_share, 6) if risk_per_share else None,
        "target": _safe_float(target),
        "target1": _safe_float(target),
        "target2": _safe_float(sig.get("target2") or sig.get("target_price_2")),
        "high_water": round(fill, 4),
        "low_water": round(fill, 4),
        "scaled_out": False,
        "entry_commission": commission,
        "entry_zone": entry_zone,
        "opened_at": None,  # filled below
        "opened_date": trading_date,
        "signal_source": sig.get("source"),
        "horizon": horizon,
        "book": book_key,
        "plan_symbol": sig.get("symbol"),
        "expected_return_pct": sig.get("expected_return_pct"),
        "entry_status_at_entry": (sig.get("entry_status") or {}).get("status")
        if isinstance(sig.get("entry_status"), dict)
        else sig.get("entry_status"),
    }
    from src.paper.account import _now_iso

    position["opened_at"] = _now_iso()
    set_position(account, book_key, position)

    trade = {
        "side": "BUY" if direction == "LONG" else "SELL_SHORT",
        "action": "ENTRY",
        "symbol": symbol.upper(),
        "direction": direction,
        "shares": shares,
        "price": round(fill, 4),
        "requested_price": round(price, 4),
        "slippage_bps": slip_bps,
        "commission": commission,
        "notional": notional,
        "pnl": None,
        "reason": reason,
        "trading_date": trading_date,
        "signal_source": sig.get("source"),
        "horizon": horizon,
        "book": book_key,
    }
    append_trade(account, trade)
    mark_to_market(account, fill, price_by_symbol={symbol.upper(): fill})
    return trade


def execute_exit(
    account: dict[str, Any],
    *,
    price: float,
    reason: str = "",
    trading_date: str | None = None,
    book: str | None = None,
    shares: int | None = None,
) -> dict[str, Any]:
    """Close (or partially reduce) an open position at ``price``.

    ``shares`` — if provided and < held size, this is a partial scale-out; the
    position stays open with the remainder. If omitted, the whole book is closed.
    If ``book`` is omitted, closes the legacy primary book (intraday preferred).
    """
    ensure_positions(account)
    if book:
        pos = get_position(account, book)
        book_key = book
    else:
        # Legacy: close whatever sync_legacy exposes / first open
        pos = account.get("position")
        book_key = None
        if pos:
            book_key = pos.get("book") or _resolve_book(None, pos)
            pos = get_position(account, book_key)
        if not pos:
            for b in BOOKS:
                pos = get_position(account, b)
                if pos:
                    book_key = b
                    break
    if not pos or not book_key:
        raise ValueError("No open position")
    if price <= 0:
        raise ValueError("price must be > 0")

    params = account.get("params") or {}
    max_dev = float(
        params.get("max_price_deviation_pct") or DEFAULT_MAX_DEVIATION_PCT
    )
    ok, reject = is_sane_fill_price(
        price,
        anchors=anchors_from_position(pos),
        max_deviation_pct=max_dev,
    )
    if not ok:
        raise NonsensePriceError(reject or "nonsense fill price")

    held = int(pos["shares"])
    qty = held if shares is None else max(0, min(int(shares), held))
    if qty <= 0:
        raise ValueError("exit shares must be > 0")
    partial = qty < held

    avg = float(pos["avg_entry"])
    direction = (pos.get("direction") or "LONG").upper()
    symbol = pos["symbol"]

    slip_bps, c_trade, c_share = _frictions(account)
    fill = apply_slippage(
        price, direction=direction, is_entry=False, slippage_bps=slip_bps
    )
    exit_commission = _commission(qty, c_trade, c_share)
    # Allocate the position's paid entry commission to the shares being closed.
    entry_comm_total = float(pos.get("entry_commission") or 0.0)
    entry_comm_alloc = round(entry_comm_total * (qty / held), 4) if held else 0.0

    notional = round(qty * fill, 2)
    cash = float(account.get("cash") or 0.0)

    if direction == "LONG":
        gross = (fill - avg) * qty
        account["cash"] = round(cash + notional - exit_commission, 2)
        side = "SELL"
    else:
        gross = (avg - fill) * qty
        account["cash"] = round(cash - notional - exit_commission, 2)
        side = "BUY_COVER"

    pnl = gross - exit_commission - entry_comm_alloc
    realized = float(account.get("realized_pnl") or 0.0) + pnl
    account["realized_pnl"] = round(realized, 2)

    if partial:
        pos["shares"] = held - qty
        pos["entry_commission"] = round(entry_comm_total - entry_comm_alloc, 4)
        pos["scaled_out"] = True
        set_position(account, book_key, pos)
    else:
        set_position(account, book_key, None)

    trade = {
        "side": side,
        "action": "SCALE_OUT" if partial else "EXIT",
        "symbol": symbol,
        "direction": direction,
        "shares": qty,
        "price": round(fill, 4),
        "requested_price": round(price, 4),
        "slippage_bps": slip_bps,
        "commission": round(exit_commission + entry_comm_alloc, 4),
        "notional": notional,
        "pnl": round(pnl, 2),
        "remaining_shares": pos["shares"] if partial else 0,
        "reason": reason,
        "trading_date": trading_date,
        "avg_entry": avg,
        "hold_opened_date": pos.get("opened_date"),
        "signal_source": pos.get("signal_source"),
        "horizon": pos.get("horizon"),
        "book": book_key,
    }
    append_trade(account, trade)
    if partial:
        mark_to_market(account, price_by_symbol={symbol.upper(): fill})
    else:
        mark_to_market(account, None)
    return trade


def can_afford(
    account: dict[str, Any],
    *,
    price: float,
    stop: float | None,
    direction: str,
    risk_pct: float | None = None,
    max_position_pct: float | None = None,
    max_notional: float | None = None,
) -> tuple[int, str | None]:
    """Return (shares, error_message). shares=0 means cannot enter."""
    params = account.get("params") or {}
    shares = size_shares(
        cash=float(account.get("cash") or 0),
        equity=float(account.get("equity") or account.get("cash") or STARTING_CASH),
        entry_price=price,
        stop_price=stop,
        direction=direction,
        risk_pct=float(
            risk_pct if risk_pct is not None else (params.get("risk_pct") or 1.0)
        ),
        max_position_pct=float(
            max_position_pct
            if max_position_pct is not None
            else (
                params.get("max_position_pct")
                if params.get("max_position_pct") is not None
                else 40.0
            )
        ),
        max_notional=max_notional,
    )
    if shares <= 0:
        return 0, "insufficient cash or risk budget for even 1 share"
    return shares, None
