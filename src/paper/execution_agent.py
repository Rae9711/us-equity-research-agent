"""Rules-based paper execution agent: ENTRY / HOLD / EXIT.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
Uses Ideal Entry zone + Entry Status + stop/target vs last price.
"""

from __future__ import annotations

from typing import Any

from src.paper.account import mark_to_market
from src.paper.broker_sim import (
    InsufficientCashError,
    can_afford,
    execute_entry,
    execute_exit,
)
from src.paper.signals import load_candidate_signals, normalize_slot, pick_signal, resolve_quote
from src.research.entry_status import infer_session_phase


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stop_hit(direction: str, price: float, stop: float | None) -> bool:
    if stop is None:
        return False
    if direction == "LONG":
        return price <= stop
    if direction == "SHORT":
        return price >= stop
    return False


def _target_hit(direction: str, price: float, target: float | None) -> bool:
    if target is None:
        return False
    if direction == "LONG":
        return price >= target
    if direction == "SHORT":
        return price <= target
    return False


def _first_signal_symbol(trading_date: str) -> str | None:
    ctx = load_candidate_signals(trading_date)
    for raw_slot, source, horizon in (
        (ctx.get("session_primary"), "session_primary", "Intraday"),
        (ctx.get("morning_primary"), "morning_primary", "Intraday"),
        (ctx.get("swing"), "swing", "Swing"),
    ):
        norm = normalize_slot(raw_slot or {}, source=source, horizon=horizon)
        if norm:
            return norm["symbol"]
    return None


def decide_and_act(
    account: dict[str, Any],
    trading_date: str,
    *,
    session_phase: str | None = None,
    raw: dict[str, Any] | None = None,
    force_price: float | None = None,
) -> dict[str, Any]:
    """One paper-trading tick. Mutates ``account``. Returns decision summary."""
    phase = session_phase or infer_session_phase(trading_date)
    params = account.get("params") or {}
    result: dict[str, Any] = {
        "trading_date": trading_date,
        "session_phase": phase,
        "action": "HOLD",
        "reason": "",
        "trade": None,
        "signal": None,
        "quote": None,
        "quote_source": None,
        "advisory": True,
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }

    price_map: dict[str, float] | None = None
    if force_price is not None:
        sym = None
        pos = account.get("position")
        if pos:
            sym = pos.get("symbol")
        if not sym:
            sym = _first_signal_symbol(trading_date)
        if sym:
            price_map = {str(sym).upper(): float(force_price)}

    pos = account.get("position")

    # --- Manage open position ---
    if pos:
        sym = pos["symbol"]
        direction = (pos.get("direction") or "LONG").upper()
        if price_map and sym in price_map:
            px, qsrc = price_map[sym], "forced"
        else:
            px, qsrc = resolve_quote(sym, trading_date, raw=raw, prefer_live=True)
        result["quote"] = px
        result["quote_source"] = qsrc
        result["signal"] = {
            "symbol": sym,
            "direction": direction,
            "source": pos.get("signal_source"),
            "horizon": pos.get("horizon"),
        }

        if px is None:
            result["action"] = "HOLD"
            result["reason"] = f"持仓 {sym}：无报价，继续持有"
            return result

        stop = _safe_float(pos.get("stop"))
        target = _safe_float(pos.get("target"))
        horizon = (pos.get("horizon") or "Intraday").lower()

        if _stop_hit(direction, px, stop):
            trade = execute_exit(
                account,
                price=px,
                reason=f"止损触发 @ {px} (stop={stop})",
                trading_date=trading_date,
            )
            result["action"] = "EXIT"
            result["reason"] = trade["reason"]
            result["trade"] = trade
            return result

        if _target_hit(direction, px, target):
            trade = execute_exit(
                account,
                price=px,
                reason=f"止盈触发 @ {px} (target={target})",
                trading_date=trading_date,
            )
            result["action"] = "EXIT"
            result["reason"] = trade["reason"]
            result["trade"] = trade
            return result

        force_eod = bool(params.get("force_exit_intraday_at_close", True))
        if force_eod and "swing" not in horizon and phase == "closed":
            trade = execute_exit(
                account,
                price=px,
                reason=f"日内仓位收盘平仓 @ {px}",
                trading_date=trading_date,
            )
            result["action"] = "EXIT"
            result["reason"] = trade["reason"]
            result["trade"] = trade
            return result

        mark_to_market(account, px)
        result["action"] = "HOLD"
        result["reason"] = (
            f"持仓 {sym} {direction} {pos.get('shares')}股 @ {pos.get('avg_entry')}；"
            f"现价 {px}，未触止损/止盈"
        )
        return result

    # --- No position: look for entry ---
    if phase == "premarket":
        result["action"] = "SKIP"
        result["reason"] = "盘前观望，不提前成交"
        return result

    picked = pick_signal(
        trading_date,
        session_phase=phase,
        prefer_swing_if_no_intraday=bool(
            params.get("prefer_swing_if_no_intraday", True)
        ),
        miss_threshold_pct=float(params.get("miss_threshold_pct") or 1.0),
        raw=raw,
        current_price_by_symbol=price_map,
    )

    result["signal"] = picked.get("signal")
    result["quote"] = picked.get("quote")
    result["quote_source"] = picked.get("quote_source")
    result["entry_status"] = picked.get("entry_status")

    action = picked.get("action")
    if action == "skip":
        result["action"] = "SKIP"
        result["reason"] = picked.get("reason") or "跳过"
        return result

    if action == "wait":
        result["action"] = "WAIT"
        result["reason"] = picked.get("reason") or "等待 Ideal Entry"
        return result

    sig = picked.get("signal") or {}
    px = picked.get("quote")
    if px is None or px <= 0:
        result["action"] = "SKIP"
        result["reason"] = f"有信号但无报价：{sig.get('symbol')}"
        return result

    shares, err = can_afford(
        account,
        price=px,
        stop=sig.get("stop_price"),
        direction=sig.get("direction") or "LONG",
    )
    if shares <= 0:
        result["action"] = "SKIP"
        result["reason"] = f"资金不足无法开仓：{err}"
        return result

    try:
        trade = execute_entry(
            account,
            symbol=str(sig["symbol"]),
            direction=str(sig["direction"]),
            price=float(px),
            shares=shares,
            stop=sig.get("stop_price"),
            target=sig.get("target_price"),
            entry_zone=sig.get("entry_zone"),
            signal=sig,
            reason=picked.get("reason") or "模拟买入",
            trading_date=trading_date,
        )
    except InsufficientCashError as exc:
        result["action"] = "SKIP"
        result["reason"] = f"资金不足：{exc}"
        return result

    result["action"] = "ENTRY"
    result["reason"] = trade.get("reason") or picked.get("reason")
    result["trade"] = trade
    return result
