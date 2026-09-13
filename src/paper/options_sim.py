"""Paper options simulation — long Call/Put contracts (incl. 0DTE).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Rules (fixed by product plan):
  - Buy-only (long premium); never naked short options.
  - ATM nearest strike on nearest / same-day expiration.
  - Fill at mid = (bid+ask)/2, else last.
  - Size: risk_pct × equity / (premium × 100).
  - Max loss ≈ premium paid; premium drawdown stop (default 50%).
  - 0DTE forced flat at session close.
  - No chain quote → skip (never fake an equity fill).
"""

from __future__ import annotations

import logging
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
from src.paper.instrument import is_zero_dte, option_right

logger = logging.getLogger(__name__)

MULTIPLIER = 100


class OptionChainUnavailable(ValueError):
    """No usable option chain / premium for paper fill."""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _resolve_book(book: str | None, signal: dict[str, Any] | None) -> str:
    if book and book in BOOKS:
        return book
    horizon = str((signal or {}).get("horizon") or "").lower()
    if "swing" in horizon:
        return "swing"
    return BOOK_INTRADAY


def default_underlying(signal: dict[str, Any] | None = None) -> str:
    """Prefer signal symbol when it is an index ETF; else config options.underlying."""
    sym = str((signal or {}).get("symbol") or "").upper()
    if sym in ("QQQ", "SPY", "IWM", "DIA", "TQQQ"):
        return sym
    try:
        from src.collectors.config import load_symbols

        cfg = load_symbols()
        u = ((cfg.get("options") or {}).get("underlying") or "QQQ")
        return str(u).upper()
    except Exception:
        return "QQQ"


def _mid_from_row(row: dict[str, Any]) -> float | None:
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    last = _safe_float(row.get("lastPrice") or row.get("last") or row.get("regularMarketPrice"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 4)
    if last is not None and last > 0:
        return round(last, 4)
    return None


def _pick_atm_row(chain_df: Any, underlying_px: float) -> dict[str, Any] | None:
    """Pick nearest strike to underlying from a yfinance-like DataFrame."""
    if chain_df is None or getattr(chain_df, "empty", True):
        return None
    try:
        strikes = chain_df["strike"].astype(float)
    except Exception:
        return None
    if strikes.empty:
        return None
    idx = (strikes - float(underlying_px)).abs().idxmin()
    row = chain_df.loc[idx]
    return {k: row[k] for k in chain_df.columns}


def fetch_option_quote(
    *,
    underlying: str,
    right: str,
    underlying_px: float,
    zero_dte: bool,
    trading_date: str | None = None,
    forced: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an ATM contract + mid premium.

    ``forced`` lets tests inject {strike, premium, expiration, bid, ask} without network.
    """
    if forced:
        prem = _safe_float(forced.get("premium") or forced.get("mid") or forced.get("last"))
        strike = _safe_float(forced.get("strike"))
        if prem is None or prem <= 0 or strike is None:
            raise OptionChainUnavailable("forced option quote incomplete")
        return {
            "underlying": underlying.upper(),
            "right": right.lower(),
            "strike": float(strike),
            "expiration": forced.get("expiration") or trading_date,
            "premium": float(prem),
            "bid": _safe_float(forced.get("bid")),
            "ask": _safe_float(forced.get("ask")),
            "source": "forced",
            "zero_dte": bool(zero_dte or forced.get("zero_dte")),
            "contract_symbol": forced.get("contract_symbol"),
        }

    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        raise OptionChainUnavailable(f"yfinance unavailable: {exc}") from exc

    ticker = yf.Ticker(underlying.upper())
    try:
        expirations = list(ticker.options or [])
    except Exception as exc:  # noqa: BLE001
        raise OptionChainUnavailable(f"no expirations for {underlying}: {exc}") from exc
    if not expirations:
        raise OptionChainUnavailable(f"empty option chain for {underlying}")

    target_exp = expirations[0]
    if trading_date:
        # Prefer same-day expiry when present (0DTE); else nearest listed.
        if trading_date in expirations:
            target_exp = trading_date
        elif zero_dte:
            # Still take nearest; label may still be 0DTE intent.
            target_exp = expirations[0]

    try:
        chain = ticker.option_chain(target_exp)
    except Exception as exc:  # noqa: BLE001
        raise OptionChainUnavailable(f"option_chain failed: {exc}") from exc

    frame = chain.calls if right.lower() == "call" else chain.puts
    row = _pick_atm_row(frame, underlying_px)
    if not row:
        raise OptionChainUnavailable("no ATM row")
    prem = _mid_from_row(row)
    strike = _safe_float(row.get("strike"))
    if prem is None or prem <= 0 or strike is None:
        raise OptionChainUnavailable("ATM premium missing")

    return {
        "underlying": underlying.upper(),
        "right": right.lower(),
        "strike": float(strike),
        "expiration": target_exp,
        "premium": float(prem),
        "bid": _safe_float(row.get("bid")),
        "ask": _safe_float(row.get("ask")),
        "source": "yfinance",
        "zero_dte": bool(zero_dte) or (trading_date is not None and target_exp == trading_date),
        "contract_symbol": str(row.get("contractSymbol") or "") or None,
    }


def size_contracts(
    *,
    equity: float,
    cash: float,
    premium: float,
    risk_pct: float = 1.0,
    max_position_pct: float = 25.0,
    max_notional: float | None = None,
) -> int:
    """Long-option contract count from risk budget and cash."""
    if equity <= 0 or cash <= 0 or premium <= 0:
        return 0
    per_contract = premium * MULTIPLIER
    if per_contract <= 0:
        return 0
    risk_budget = equity * (risk_pct / 100.0)
    by_risk = int(risk_budget // per_contract)
    max_cap = equity * (max_position_pct / 100.0)
    if max_notional is not None:
        max_cap = min(max_cap, max(0.0, float(max_notional)))
    by_cap = int(max_cap // per_contract)
    by_cash = int(cash // per_contract)
    return max(0, min(by_risk, by_cap, by_cash))


def can_afford_option(
    account: dict[str, Any],
    *,
    premium: float,
    risk_pct: float | None = None,
    max_position_pct: float | None = None,
    max_notional: float | None = None,
) -> tuple[int, str | None]:
    params = account.get("params") or {}
    contracts = size_contracts(
        equity=float(account.get("equity") or account.get("cash") or STARTING_CASH),
        cash=float(account.get("cash") or 0),
        premium=premium,
        risk_pct=float(
            risk_pct if risk_pct is not None else (params.get("risk_pct") or 1.0)
        ),
        max_position_pct=float(
            max_position_pct
            if max_position_pct is not None
            else (params.get("max_position_pct") or 40.0)
        ),
        max_notional=max_notional,
    )
    if contracts <= 0:
        return 0, "insufficient cash or risk budget for even 1 option contract"
    return contracts, None


def execute_option_entry(
    account: dict[str, Any],
    *,
    quote: dict[str, Any],
    contracts: int,
    signal: dict[str, Any] | None = None,
    reason: str = "",
    trading_date: str | None = None,
    book: str | None = None,
    underlying_stop: float | None = None,
    underlying_target: float | None = None,
) -> dict[str, Any]:
    """Open a long option position; cash pays premium × 100 × contracts."""
    ensure_positions(account)
    book_key = _resolve_book(book, signal)
    if get_position(account, book_key):
        raise ValueError(f"Already holding a {book_key} position")
    if contracts <= 0:
        raise ValueError("contracts must be > 0")

    premium = float(quote["premium"])
    if premium <= 0:
        raise ValueError("premium must be > 0")

    notional = round(contracts * premium * MULTIPLIER, 2)
    cash = float(account.get("cash") or 0.0)
    if cash + 1e-9 < notional:
        from src.paper.broker_sim import InsufficientCashError

        raise InsufficientCashError(f"Need ${notional:.2f} cash for options, have ${cash:.2f}")

    account["cash"] = round(cash - notional, 2)
    sig = signal or {}
    right = str(quote.get("right") or "call").lower()
    underlying = str(quote.get("underlying") or "").upper()
    strike = float(quote["strike"])
    zero = bool(quote.get("zero_dte"))
    horizon = sig.get("horizon") or ("0DTE" if zero else "Intraday")
    # Premium stop: lose option_premium_stop_pct of entry premium → exit.
    params = account.get("params") or {}
    stop_pct = float(params.get("option_premium_stop_pct") or 50.0) / 100.0
    premium_stop = round(premium * (1.0 - stop_pct), 4)
    # Soft target: +100% premium as default T1 when no better signal.
    premium_target = round(premium * float(params.get("option_premium_target_mult") or 2.0), 4)

    from src.paper.account import _now_iso

    position = {
        "symbol": underlying,
        "asset_class": "option",
        "instrument": sig.get("instrument")
        or f"{underlying} {'0DTE ' if zero else ''}{'Call' if right == 'call' else 'Put'}",
        "option_right": right,
        "strike": strike,
        "expiration": quote.get("expiration"),
        "contract_symbol": quote.get("contract_symbol"),
        "multiplier": MULTIPLIER,
        "zero_dte": zero,
        "direction": "LONG",  # long premium
        "shares": contracts,  # contracts stored in shares slot for dual-book reuse
        "contracts": contracts,
        "initial_shares": contracts,
        "avg_entry": round(premium, 4),
        "stop": premium_stop,
        "initial_stop": premium_stop,
        "risk_per_share": round(premium - premium_stop, 6),
        "target": premium_target,
        "target1": premium_target,
        "target2": None,
        "underlying_stop": _safe_float(underlying_stop),
        "underlying_target": _safe_float(underlying_target),
        "high_water": round(premium, 4),
        "low_water": round(premium, 4),
        "scaled_out": False,
        "entry_commission": 0.0,
        "opened_at": _now_iso(),
        "opened_date": trading_date,
        "signal_source": sig.get("source"),
        "horizon": horizon,
        "book": book_key,
        "plan_symbol": sig.get("symbol"),
        "expected_return_pct": sig.get("expected_return_pct"),
        "entry_status_at_entry": (sig.get("entry_status") or {}).get("status")
        if isinstance(sig.get("entry_status"), dict)
        else sig.get("entry_status"),
        "last_price": round(premium, 4),
    }
    # Fix typo if I accidentally left opened andate - let me check... yes I have a typo
    set_position(account, book_key, position)

    trade = {
        "side": "BUY_CALL" if right == "call" else "BUY_PUT",
        "action": "ENTRY",
        "symbol": underlying,
        "direction": "LONG",
        "asset_class": "option",
        "instrument": position["instrument"],
        "option_right": right,
        "strike": strike,
        "expiration": quote.get("expiration"),
        "shares": contracts,
        "contracts": contracts,
        "multiplier": MULTIPLIER,
        "price": round(premium, 4),
        "requested_price": round(premium, 4),
        "slippage_bps": 0.0,
        "commission": 0.0,
        "notional": notional,
        "pnl": None,
        "reason": reason,
        "trading_date": trading_date,
        "signal_source": sig.get("source"),
        "horizon": horizon,
        "book": book_key,
    }
    from src.paper.trade_report import enrich_trade_report

    enrich_trade_report(trade, signal=sig, position=position)
    append_trade(account, trade)
    mark_to_market(
        account,
        price_by_symbol={underlying: round(premium, 4)},
    )
    return trade


def execute_option_exit(
    account: dict[str, Any],
    *,
    premium: float,
    reason: str = "",
    trading_date: str | None = None,
    book: str | None = None,
    contracts: int | None = None,
) -> dict[str, Any]:
    """Close (or partially reduce) a long option position at ``premium``."""
    ensure_positions(account)
    if book:
        pos = get_position(account, book)
        book_key = book
    else:
        pos = account.get("position")
        book_key = None
        if pos:
            book_key = pos.get("book") or BOOK_INTRADAY
            pos = get_position(account, book_key)
        if not pos:
            for b in BOOKS:
                pos = get_position(account, b)
                if pos:
                    book_key = b
                    break
    if not pos or not book_key:
        raise ValueError("No open option position")
    if (pos.get("asset_class") or "equity") != "option":
        raise ValueError("Position is not an option")
    if premium < 0:
        raise ValueError("premium must be >= 0")

    held = int(pos.get("contracts") or pos.get("shares") or 0)
    qty = held if contracts is None else max(0, min(int(contracts), held))
    if qty <= 0:
        raise ValueError("exit contracts must be > 0")
    partial = qty < held

    avg = float(pos["avg_entry"])
    underlying = str(pos.get("symbol") or "").upper()
    notional = round(qty * premium * MULTIPLIER, 2)
    cash = float(account.get("cash") or 0.0)
    account["cash"] = round(cash + notional, 2)
    gross = (premium - avg) * qty * MULTIPLIER
    pnl = gross
    account["realized_pnl"] = round(float(account.get("realized_pnl") or 0.0) + pnl, 2)

    if partial:
        pos["shares"] = held - qty
        pos["contracts"] = held - qty
        pos["scaled_out"] = True
        set_position(account, book_key, pos)
    else:
        set_position(account, book_key, None)

    trade = {
        "side": "SELL_CALL" if (pos.get("option_right") or "").lower() == "call" else "SELL_PUT",
        "action": "SCALE_OUT" if partial else "EXIT",
        "symbol": underlying,
        "direction": "LONG",
        "asset_class": "option",
        "instrument": pos.get("instrument"),
        "option_right": pos.get("option_right"),
        "strike": pos.get("strike"),
        "expiration": pos.get("expiration"),
        "shares": qty,
        "contracts": qty,
        "multiplier": MULTIPLIER,
        "price": round(premium, 4),
        "requested_price": round(premium, 4),
        "slippage_bps": 0.0,
        "commission": 0.0,
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
    from src.paper.account import STARTING_CASH
    from src.paper.trade_report import enrich_trade_report

    enrich_trade_report(
        trade,
        position=pos,
        equity=float(account.get("equity") or account.get("starting_cash") or STARTING_CASH),
    )
    # After full close pos may be None — use stored fields from before clear
    if not partial:
        trade["side"] = (
            "SELL_CALL" if str(trade.get("option_right") or "").lower() == "call" else "SELL_PUT"
        )
    append_trade(account, trade)
    if partial:
        mark_to_market(account, price_by_symbol={underlying: round(premium, 4)})
    else:
        mark_to_market(account, None)
    return trade


def resolve_option_entry_quote(
    signal: dict[str, Any],
    *,
    underlying_px: float,
    trading_date: str | None = None,
) -> dict[str, Any]:
    """Build option quote for a signal or raise OptionChainUnavailable."""
    instrument = str(signal.get("instrument") or "")
    right = option_right(instrument)
    if right is None:
        # Infer from direction when instrument says Call/Put poorly
        direction = str(signal.get("direction") or "LONG").upper()
        right = "call" if direction == "LONG" else "put"
    underlying = default_underlying(signal)
    zero = is_zero_dte(instrument, signal.get("horizon"))
    forced = signal.get("forced_option_quote")
    return fetch_option_quote(
        underlying=underlying,
        right=right,
        underlying_px=underlying_px,
        zero_dte=zero,
        trading_date=trading_date,
        forced=forced if isinstance(forced, dict) else None,
    )


def resolve_open_option_premium(
    pos: dict[str, Any],
    *,
    underlying_px: float | None = None,
    trading_date: str | None = None,
    forced_premium: float | None = None,
) -> float | None:
    """Mark an open option; prefer forced/test premium, else re-fetch ATM mid."""
    if forced_premium is not None:
        return max(0.0, float(forced_premium))
    forced = pos.get("forced_mark_premium")
    if forced is not None:
        try:
            return max(0.0, float(forced))
        except (TypeError, ValueError):
            pass
    u_px = underlying_px if underlying_px is not None else _safe_float(pos.get("underlying_last"))
    if u_px is None or u_px <= 0:
        return _safe_float(pos.get("last_price"))
    try:
        q = fetch_option_quote(
            underlying=str(pos.get("symbol") or ""),
            right=str(pos.get("option_right") or "call"),
            underlying_px=float(u_px),
            zero_dte=bool(pos.get("zero_dte")),
            trading_date=trading_date or str(pos.get("expiration") or "") or None,
        )
        return float(q["premium"])
    except OptionChainUnavailable:
        return _safe_float(pos.get("last_price"))


def plan_option_exit(
    pos: dict[str, Any],
    premium: float,
    underlying_px: float | None,
    params: dict[str, Any],
    *,
    phase: str = "open",
) -> dict[str, Any]:
    """Exit plan for long option using premium stop/target + underlying levels + 0DTE EOD."""
    held = int(pos.get("contracts") or pos.get("shares") or 0)
    plan: dict[str, Any] = {
        "action": "hold",
        "shares": None,
        "reason": "",
    }
    if held <= 0:
        return plan

    stop = _safe_float(pos.get("stop"))
    target = _safe_float(pos.get("target"))
    if stop is not None and premium <= stop:
        plan["action"] = "exit"
        plan["reason"] = f"期权权利金止损 @ {premium} (stop={stop})"
        return plan
    if target is not None and premium >= target:
        plan["action"] = "exit"
        plan["reason"] = f"期权权利金止盈 @ {premium} (target={target})"
        return plan

    # Underlying geometry as secondary trigger.
    u_stop = _safe_float(pos.get("underlying_stop"))
    u_target = _safe_float(pos.get("underlying_target"))
    right = str(pos.get("option_right") or "call").lower()
    if underlying_px is not None:
        if right == "call":
            if u_stop is not None and underlying_px <= u_stop:
                plan["action"] = "exit"
                plan["reason"] = f"标的触及止损 {underlying_px} ≤ {u_stop}，平仓期权"
                return plan
            if u_target is not None and underlying_px >= u_target:
                plan["action"] = "exit"
                plan["reason"] = f"标的触及止盈 {underlying_px} ≥ {u_target}，平仓期权"
                return plan
        else:
            if u_stop is not None and underlying_px >= u_stop:
                plan["action"] = "exit"
                plan["reason"] = f"标的触及止损 {underlying_px} ≥ {u_stop}，平仓期权"
                return plan
            if u_target is not None and underlying_px <= u_target:
                plan["action"] = "exit"
                plan["reason"] = f"标的触及止盈 {underlying_px} ≤ {u_target}，平仓期权"
                return plan

    # Max premium loss vs entry (extra cap).
    avg = _safe_float(pos.get("avg_entry"))
    max_loss_pct = float(params.get("option_premium_stop_pct") or 50.0) / 100.0
    if avg and avg > 0 and premium <= avg * (1.0 - max_loss_pct):
        plan["action"] = "exit"
        plan["reason"] = f"期权亏损达权利金 {max_loss_pct:.0%} @ {premium}"
        return plan

    zero = bool(pos.get("zero_dte")) or is_zero_dte(
        str(pos.get("instrument") or ""), str(pos.get("horizon") or "")
    )
    if zero and phase == "closed":
        plan["action"] = "exit"
        plan["reason"] = f"0DTE 收盘强制平仓 @ {premium}"
        return plan

    return plan
