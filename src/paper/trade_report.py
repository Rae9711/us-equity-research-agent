"""Per-trade narrative cards: 做了什么 / 怎么做 / 收益多少.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


def build_method_snapshot(
    *,
    signal: dict[str, Any] | None = None,
    allocation: dict[str, Any] | None = None,
    trade: dict[str, Any] | None = None,
    position: dict[str, Any] | None = None,
    book: str | None = None,
    entry_status: str | None = None,
) -> dict[str, Any]:
    """Structured method snapshot attached to ENTRY (and carried on EXIT)."""
    sig = signal or {}
    alloc = allocation or {}
    tr = trade or {}
    pos = position or {}

    inst_text = str(tr.get("instrument") or sig.get("instrument") or pos.get("instrument") or "").upper()
    asset_class = str(
        tr.get("asset_class")
        or pos.get("asset_class")
        or ("option" if ("CALL" in inst_text or "PUT" in inst_text) else "equity")
    )
    instrument = (
        tr.get("instrument")
        or pos.get("instrument")
        or sig.get("instrument")
        or ("期权" if asset_class == "option" else "股票/ETF")
    )
    symbol = str(tr.get("symbol") or pos.get("symbol") or sig.get("symbol") or "").upper()
    direction = str(tr.get("direction") or pos.get("direction") or sig.get("direction") or "LONG").upper()
    book_key = book or tr.get("book") or pos.get("book") or alloc.get("book")
    book_zh = "长线" if book_key == "swing" else "短线"
    horizon = tr.get("horizon") or pos.get("horizon") or sig.get("horizon") or ""

    status = entry_status
    if status is None:
        status = tr.get("entry_status") or pos.get("entry_status_at_entry")
        if isinstance(sig.get("entry_status"), dict):
            status = sig["entry_status"].get("status")
        elif sig.get("entry_status"):
            status = sig.get("entry_status")

    win_prob = _safe_float(alloc.get("win_prob") if alloc.get("win_prob") is not None else sig.get("win_prob"))
    rr = _safe_float(alloc.get("rr") if alloc.get("rr") is not None else sig.get("risk_reward"))
    ev = _safe_float(alloc.get("expected_r"))
    risk_pct = _safe_float(alloc.get("risk_pct"))
    reserve = _safe_float(alloc.get("cash_reserve_pct"))
    conf = _safe_float(alloc.get("confidence"))

    stop = _safe_float(pos.get("stop") or sig.get("stop_price") or tr.get("stop"))
    target = _safe_float(pos.get("target") or sig.get("target_price") or tr.get("target"))
    qty = int(tr.get("contracts") or tr.get("shares") or pos.get("contracts") or pos.get("shares") or 0)
    price = _safe_float(tr.get("price") or pos.get("avg_entry"))

    if asset_class == "option":
        right = str(tr.get("option_right") or pos.get("option_right") or "").lower()
        strike = tr.get("strike") or pos.get("strike")
        what = (
            f"买入 {symbol} {instrument}"
            + (f" · 行权价 {strike}" if strike is not None else "")
            + (f" · {qty} 张" if qty else "")
        )
        how = (
            f"权利金多头 · {book_zh}/{horizon or '—'} · 状态 {status or '—'}"
            f" · 胜率 {_fmt_pct(win_prob)} · R:R {rr if rr is not None else '—'}"
            f" · 期望R {ev if ev is not None else '—'}"
            f" · 权利金止损 {stop if stop is not None else '—'}"
            f" / 止盈 {target if target is not None else '—'}"
        )
    else:
        side_zh = "做多" if direction == "LONG" else "做空"
        unit = "股"
        what = f"{side_zh} {symbol}（{instrument}）· {qty}{unit}" + (
            f" @ {price}" if price is not None else ""
        )
        how = (
            f"{direction} · {book_zh}/{horizon or '—'} · 状态 {status or '—'}"
            f" · 胜率 {_fmt_pct(win_prob)} · R:R {rr if rr is not None else '—'}"
            f" · 期望R {ev if ev is not None else '—'}"
            f" · 止损 {stop if stop is not None else '—'}"
            f" / 止盈 {target if target is not None else '—'}"
        )

    why_size = (
        f"risk {risk_pct if risk_pct is not None else '—'}%"
        f" · 预留现金 {reserve if reserve is not None else '—'}%"
        f" · 信心 {_fmt_pct((conf or 0) * 100 if conf is not None and conf <= 1 else conf)}"
    )
    if alloc.get("reason_zh"):
        why_size = f"{why_size} · {alloc['reason_zh']}"

    return {
        "what": what,
        "how": how,
        "why_size": why_size,
        "instrument": instrument,
        "asset_class": asset_class,
        "symbol": symbol,
        "direction": direction,
        "expected_r_at_entry": ev,
        "win_prob": win_prob,
        "rr": rr,
        "entry_status": status,
        "stop": stop,
        "target": target,
        "book": book_key,
        "horizon": horizon,
        "risk_pct": risk_pct,
        "cash_reserve_pct": reserve,
        "confidence": conf,
        "option_right": tr.get("option_right") or pos.get("option_right"),
        "strike": tr.get("strike") or pos.get("strike"),
        "expiration": tr.get("expiration") or pos.get("expiration"),
    }


def enrich_trade_report(
    trade: dict[str, Any],
    *,
    signal: dict[str, Any] | None = None,
    allocation: dict[str, Any] | None = None,
    position: dict[str, Any] | None = None,
    equity: float | None = None,
    entry_status: str | None = None,
) -> dict[str, Any]:
    """Attach method_snapshot / narrative_zh / pnl metrics onto a trade row."""
    snap = trade.get("method_snapshot")
    if not isinstance(snap, dict):
        snap = build_method_snapshot(
            signal=signal,
            allocation=allocation or trade.get("allocation"),
            trade=trade,
            position=position,
            book=trade.get("book"),
            entry_status=entry_status,
        )
        trade["method_snapshot"] = snap

    pnl = _safe_float(trade.get("pnl"))
    eq = equity or _safe_float((allocation or {}).get("equity"))
    if pnl is not None and eq and eq > 0:
        trade["pnl_pct_equity"] = round(pnl / eq * 100.0, 3)
    rps = None
    if position:
        rps = _safe_float(position.get("risk_per_share"))
    if rps is None:
        rps = _safe_float((trade.get("method_snapshot") or {}).get("risk_per_share"))
    avg = _safe_float(trade.get("avg_entry"))
    px = _safe_float(trade.get("price"))
    qty = float(trade.get("contracts") or trade.get("shares") or 0)
    if pnl is not None and rps and rps > 0 and qty > 0:
        mult = float(trade.get("multiplier") or (100 if trade.get("asset_class") == "option" else 1))
        risk_dollars = rps * qty * mult
        if risk_dollars > 0:
            trade["r_multiple"] = round(pnl / risk_dollars, 2)
    elif pnl is not None and avg and px is not None and avg > 0:
        # Fallback R from stop distance not available — skip
        pass

    action = str(trade.get("action") or "").upper()
    what = snap.get("what") or trade.get("symbol") or ""
    if action == "ENTRY":
        trade["narrative_zh"] = f"开仓：{what}。{snap.get('how') or ''}"
    elif action in ("EXIT", "SCALE_OUT"):
        pnl_txt = f"${pnl:+.2f}" if pnl is not None else "—"
        pct = trade.get("pnl_pct_equity")
        r_m = trade.get("r_multiple")
        extra = []
        if pct is not None:
            extra.append(f"{pct:+.2f}%权益")
        if r_m is not None:
            extra.append(f"{r_m:+.2f}R")
        extra_s = (" · " + " · ".join(extra)) if extra else ""
        verb = "减仓" if action == "SCALE_OUT" else "平仓"
        trade["narrative_zh"] = (
            f"{verb}：{what} · 收益 {pnl_txt}{extra_s}。理由：{trade.get('reason') or '—'}"
        )
    else:
        trade["narrative_zh"] = trade.get("reason") or what
    return trade


def build_skip_card(
    *,
    trading_date: str | None,
    symbol: str | None,
    action: str,
    reason: str,
    allocation: dict[str, Any] | None = None,
    book: str | None = None,
) -> dict[str, Any]:
    """Card for SKIP/WAIT so UI explains 0% size without looking like a fill."""
    alloc = allocation or {}
    reserve = alloc.get("cash_reserve_pct")
    return {
        "trading_date": trading_date,
        "action": action,
        "symbol": symbol,
        "book": book or alloc.get("book"),
        "pnl": None,
        "pnl_pct_equity": None,
        "r_multiple": None,
        "method_snapshot": {
            "what": f"{action} · {symbol or '—'}",
            "how": reason,
            "why_size": (
                f"仓位 0% · 预留现金 {reserve}%" if reserve is not None else "仓位 0%"
            ),
            "asset_class": "none",
            "instrument": None,
        },
        "narrative_zh": f"{action}：{reason}",
        "is_skip": True,
    }


def build_trade_cards(
    account: dict[str, Any],
    *,
    trading_date: str | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """UI cards from trades + recent SKIP decisions."""
    cards: list[dict[str, Any]] = []
    for t in reversed(account.get("trades") or []):
        if t.get("voided"):
            continue
        if trading_date and t.get("trading_date") and t.get("trading_date") != trading_date:
            # Still include all recent for blotter unless filtering
            pass
        row = dict(t)
        if not row.get("method_snapshot"):
            enrich_trade_report(row, equity=_safe_float(account.get("equity")))
        row.setdefault("pnl_pct_equity", None)
        row.setdefault("r_multiple", None)
        row.setdefault("is_skip", False)
        cards.append(row)
        if len(cards) >= limit:
            break

    # Append today's unique SKIP narratives (dedupe identical reason+symbol).
    seen: set[tuple[Any, ...]] = set()
    for d in reversed(account.get("decisions") or []):
        if d.get("voided"):
            continue
        if trading_date and d.get("trading_date") != trading_date:
            continue
        act = str(d.get("action") or "").upper()
        if act not in ("SKIP", "WAIT"):
            continue
        key = (act, d.get("symbol"), d.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            build_skip_card(
                trading_date=d.get("trading_date"),
                symbol=d.get("symbol"),
                action=act,
                reason=str(d.get("reason") or ""),
                allocation=d.get("allocation") if isinstance(d.get("allocation"), dict) else None,
                book=d.get("book"),
            )
        )
        if len(cards) >= limit:
            break
    return cards
