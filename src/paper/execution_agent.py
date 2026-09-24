"""Rules-based paper execution agent: ENTRY / HOLD / EXIT.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
Uses Ideal Entry zone + Entry Status + stop/target vs last price.

Dual books (default):
  - intraday / 短线 — Primary #1, force flat at EOD
  - swing / 长线 — swing_trade, hold overnight
"""

from __future__ import annotations

from typing import Any

from src.paper.account import (
    BOOK_INTRADAY,
    BOOK_SWING,
    ensure_positions,
    get_position,
    mark_to_market,
    open_positions,
)
from src.paper.allocation import allocate_for_entry
from src.paper.broker_sim import (
    InsufficientCashError,
    NonsensePriceError,
    can_afford,
    execute_entry,
    execute_exit,
)
from src.paper.price_guard import (
    DEFAULT_MAX_DEVIATION_PCT,
    anchors_from_position,
    anchors_from_signal,
    is_sane_fill_price,
)
from src.paper.signals import (
    load_candidate_signals,
    normalize_slot,
    pick_intraday_signal,
    pick_signal,
    pick_swing_signal,
    resolve_quote,
    resolve_quote_bar,
)
from src.research.entry_status import infer_session_phase

BOOK_LABEL_ZH = {
    BOOK_INTRADAY: "短线",
    BOOK_SWING: "长线",
}


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


def _param(params: dict[str, Any], key: str, default: float) -> float:
    """params lookup that tolerates missing keys; explicit 0 stays 0."""
    v = (params or {}).get(key)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _r_multiple(
    direction: str, entry: float | None, stop: float | None, price: float | None
) -> float | None:
    """Signed R multiple of ``price`` vs the entry/stop plan."""
    if entry is None or stop is None or price is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    d = (direction or "LONG").upper()
    if d == "SHORT":
        return (entry - price) / risk
    return (price - entry) / risk


def _stop_fill_price(
    direction: str,
    stop: float,
    quote: float,
    bar: dict[str, float],
    max_slippage_pct: float,
) -> tuple[float, bool, bool]:
    """Stop-market fill with bounded slippage. Returns (fill, capped, gap).

    A real stop-market fills at first touch of the stop. When the tick's quote
    is far beyond the stop (late tick after the breach), fill at stop ± cap
    instead of the stale far quote — that far quote was never tradable at the
    trigger moment. A true gap that *opened* through the stop fills at the
    open (flagged ``gap=True``): that loss is real and must not be hidden.
    With no bar evidence, assume the first-touch fill (stop ± cap).
    """
    cap = abs(stop) * max(0.0, float(max_slippage_pct or 0.0)) / 100.0
    open_px = bar.get("open")
    prev_close = bar.get("prev_close")
    d = (direction or "LONG").upper()
    if d == "SHORT":
        # True gap up through the stop: safe at prev close, opened beyond it.
        if (
            prev_close is not None
            and open_px is not None
            and prev_close < stop
            and open_px >= stop
        ):
            return open_px, False, True
        worst = stop + cap
        if quote > worst:
            return worst, True, False
        return quote, False, False
    # LONG
    if (
        prev_close is not None
        and open_px is not None
        and prev_close > stop
        and open_px <= stop
    ):
        return open_px, False, True
    worst = stop - cap
    if quote < worst:
        return worst, True, False
    return quote, False, False


def _validate_stop_geometry(
    direction: str,
    entry: float | None,
    stop: float | None,
    min_dist_pct: float,
) -> str | None:
    """Return a reject reason when the stop is unusable, else None."""
    if entry is None or entry <= 0 or stop is None or stop <= 0:
        return None  # nothing to validate; the loss breaker covers the tail
    d = (direction or "LONG").upper()
    if d == "LONG" and stop >= entry:
        return f"止损 ${stop:.4g} 不在多头入场价 ${entry:.4g} 下方，拒绝开仓"
    if d == "SHORT" and stop <= entry:
        return f"止损 ${stop:.4g} 不在空头入场价 ${entry:.4g} 上方，拒绝开仓"
    dist_pct = abs(entry - stop) / entry * 100.0
    if dist_pct < min_dist_pct:
        return (
            f"止损距离 {dist_pct:.2f}% < 下限 {min_dist_pct:g}%"
            "（止损过紧，正常波动即穿），拒绝开仓"
        )
    return None


def _cooldown_key(trading_date: str, book: str, symbol: str) -> str:
    return f"{trading_date}:{book}:{(symbol or '').upper()}"


def _set_stop_cooldown(
    account: dict[str, Any], trading_date: str, book: str, symbol: str
) -> None:
    params = account.get("params") or {}
    if not params.get("stop_cooldown_enabled", True):
        return
    rs = account.setdefault("risk_state", {})
    if not isinstance(rs, dict):
        return
    cd = rs.setdefault("stop_cooldown", {})
    if isinstance(cd, dict):
        cd[_cooldown_key(trading_date, book, symbol)] = True


def _is_stop_cooldown(
    account: dict[str, Any], trading_date: str, book: str, symbol: str
) -> bool:
    params = account.get("params") or {}
    if not params.get("stop_cooldown_enabled", True):
        return False
    rs = account.get("risk_state") or {}
    cd = rs.get("stop_cooldown") or {}
    return bool(cd.get(_cooldown_key(trading_date, book, symbol)))


def _unrealized_loss_equity_pct(
    direction: str,
    shares: float,
    avg_entry: float,
    quote: float,
    equity: float,
) -> float | None:
    """Current unrealized loss as % of equity (None when not computable)."""
    if equity <= 0 or shares <= 0 or avg_entry <= 0 or quote <= 0:
        return None
    d = (direction or "LONG").upper()
    if d == "SHORT":
        unreal = (avg_entry - quote) * shares
    else:
        unreal = (quote - avg_entry) * shares
    if unreal >= 0:
        return 0.0
    return -unreal / equity * 100.0


def _collect_symbols(trading_date: str, account: dict[str, Any]) -> list[str]:
    syms: list[str] = []
    for _book, pos in open_positions(account):
        s = (pos.get("symbol") or "").upper()
        if s and s not in syms:
            syms.append(s)
    ctx = load_candidate_signals(trading_date)
    for raw_slot, source, horizon in (
        (ctx.get("session_primary"), "session_primary", "Intraday"),
        (ctx.get("morning_primary"), "morning_primary", "Intraday"),
        (ctx.get("swing"), "swing", "Swing"),
    ):
        norm = normalize_slot(raw_slot or {}, source=source, horizon=horizon)
        if norm:
            s = norm["symbol"]
            if s not in syms:
                syms.append(s)
    return syms


def _manage_position(
    account: dict[str, Any],
    book: str,
    trading_date: str,
    phase: str,
    price_map: dict[str, float] | None,
    raw: dict[str, Any] | None,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Manage one open book. Returns a book-result dict or None if empty."""
    pos = get_position(account, book)
    if not pos:
        return None

    label = BOOK_LABEL_ZH.get(book, book)
    sym = pos["symbol"]
    direction = (pos.get("direction") or "LONG").upper()
    if price_map and sym in price_map:
        px, qsrc = price_map[sym], "forced"
    else:
        px, qsrc = resolve_quote(sym, trading_date, raw=raw, prefer_live=True)

    base: dict[str, Any] = {
        "book": book,
        "book_zh": label,
        "action": "HOLD",
        "reason": "",
        "trade": None,
        "signal": {
            "symbol": sym,
            "direction": direction,
            "source": pos.get("signal_source"),
            "horizon": pos.get("horizon"),
        },
        "quote": px,
        "quote_source": qsrc,
    }

    if px is None:
        base["reason"] = f"{label}持仓 {sym}：无报价，继续持有"
        return base

    max_dev = float(
        params.get("max_price_deviation_pct") or DEFAULT_MAX_DEVIATION_PCT
    )
    sane, reject = is_sane_fill_price(
        px, anchors=anchors_from_position(pos), max_deviation_pct=max_dev
    )
    if not sane:
        base["reason"] = (
            f"{label}持仓 {sym}：{reject or '报价异常'}，拒绝止损/止盈/平仓，继续持有"
        )
        base["quote_rejected"] = True
        return base

    stop = _safe_float(pos.get("stop"))
    target = _safe_float(pos.get("target"))
    entry_px = _safe_float(pos.get("avg_entry"))
    horizon = (pos.get("horizon") or ("Swing" if book == BOOK_SWING else "Intraday")).lower()

    if _stop_hit(direction, px, stop):
        max_slip = _param(params, "max_stop_slippage_pct", 0.5)
        bar = resolve_quote_bar(sym, trading_date, raw=raw)
        fill, capped, gap = _stop_fill_price(direction, stop, px, bar, max_slip)
        r_mult = _r_multiple(direction, entry_px, stop, fill)
        reason = f"{label}止损触发 @ {fill:.4g} (stop={stop})"
        if gap:
            reason += "·跳空穿止损按开盘价成交"
        elif capped:
            reason += f"·成交价触滑点上限 {max_slip:g}%"
        if r_mult is not None:
            reason += f"·R={r_mult:.2f}"
            if r_mult >= 0:
                reason += "（盈利/保本离场，非亏损止损）"
        trade = execute_exit(
            account,
            price=fill,
            reason=reason,
            trading_date=trading_date,
            book=book,
        )
        trade["stop_trigger"] = stop
        trade["fill_price"] = round(fill, 4)
        trade["slippage_capped"] = capped
        trade["gap_through_stop"] = gap
        trade["r_multiple"] = round(r_mult, 2) if r_mult is not None else None
        _set_stop_cooldown(account, trading_date, book, sym)
        base["action"] = "EXIT"
        base["reason"] = trade["reason"]
        base["trade"] = trade
        return base

    # Backstop: no (valid) stop, but the trade is deep underwater — hard exit.
    # With a valid stop the branch above always fires first, so this only
    # catches missing/mis-sided stops.
    bad_stop = stop is None or (
        _validate_stop_geometry(direction, entry_px, stop, min_dist_pct=0.0)
        is not None
    )
    if bad_stop:
        max_loss_pct = _param(params, "max_trade_loss_equity_pct", 2.0)
        equity = float(account.get("equity") or 0.0)
        loss_pct = _unrealized_loss_equity_pct(
            direction,
            float(pos.get("shares") or 0),
            entry_px or 0.0,
            px,
            equity,
        )
        if loss_pct is not None and loss_pct >= max_loss_pct:
            trade = execute_exit(
                account,
                price=px,
                reason=(
                    f"{label}单笔亏损熔断 @ {px}"
                    f"（浮亏 {loss_pct:.2f}% ≥ {max_loss_pct:g}%，无有效止损）"
                ),
                trading_date=trading_date,
                book=book,
            )
            trade["r_multiple"] = _r_multiple(direction, entry_px, stop, px)
            _set_stop_cooldown(account, trading_date, book, sym)
            base["action"] = "EXIT"
            base["reason"] = trade["reason"]
            base["trade"] = trade
            return base

    if _target_hit(direction, px, target):
        trade = execute_exit(
            account,
            price=px,
            reason=f"{label}止盈触发 @ {px} (target={target})",
            trading_date=trading_date,
            book=book,
        )
        base["action"] = "EXIT"
        base["reason"] = trade["reason"]
        base["trade"] = trade
        return base

    force_eod = bool(params.get("force_exit_intraday_at_close", True))
    is_swing = book == BOOK_SWING or "swing" in horizon
    if force_eod and not is_swing and phase == "closed":
        trade = execute_exit(
            account,
            price=px,
            reason=f"短线仓位收盘平仓 @ {px}",
            trading_date=trading_date,
            book=book,
        )
        base["action"] = "EXIT"
        base["reason"] = trade["reason"]
        base["trade"] = trade
        return base

    mark_to_market(account, price_by_symbol={sym: px})
    base["reason"] = (
        f"{label}持仓 {sym} {direction} {pos.get('shares')}股 @ {pos.get('avg_entry')}；"
        f"现价 {px}，未触止损/止盈"
    )
    return base


def _try_entry(
    account: dict[str, Any],
    book: str,
    picked: dict[str, Any],
    trading_date: str,
    *,
    peer_entering: bool = False,
) -> dict[str, Any]:
    label = BOOK_LABEL_ZH.get(book, book)
    action = picked.get("action")
    result: dict[str, Any] = {
        "book": book,
        "book_zh": label,
        "action": "SKIP",
        "reason": picked.get("reason") or "跳过",
        "trade": None,
        "signal": picked.get("signal"),
        "quote": picked.get("quote"),
        "quote_source": picked.get("quote_source"),
        "entry_status": picked.get("entry_status"),
        "allocation": None,
    }

    if action == "skip":
        result["action"] = "SKIP"
        return result
    if action == "wait":
        result["action"] = "WAIT"
        result["reason"] = picked.get("reason") or f"等待{label} Ideal Entry"
        return result

    sig = picked.get("signal") or {}
    px = picked.get("quote")
    if px is None or px <= 0:
        result["action"] = "SKIP"
        result["reason"] = f"{label}有信号但无报价：{sig.get('symbol')}"
        return result

    params = account.get("params") or {}
    max_dev = float(
        params.get("max_price_deviation_pct") or DEFAULT_MAX_DEVIATION_PCT
    )
    sane, reject = is_sane_fill_price(
        float(px),
        anchors=anchors_from_signal(sig),
        max_deviation_pct=max_dev,
    )
    if not sane:
        result["action"] = "SKIP"
        result["reason"] = f"{label}拒绝开仓：{reject or '报价异常'}"
        result["quote_rejected"] = True
        return result

    sym = str(sig.get("symbol") or "").upper()
    if sym and _is_stop_cooldown(account, trading_date, book, sym):
        result["action"] = "SKIP"
        result["reason"] = f"{label}跳过：{sym} 当日已止损/熔断离场，冷却中不再重入"
        return result

    direction = str(sig.get("direction") or "LONG").upper()
    entry_ref = _safe_float(sig.get("entry_price") or sig.get("entry")) or float(px)
    stop_ref = _safe_float(sig.get("stop_price") or sig.get("stop"))
    geo_reject = _validate_stop_geometry(
        direction, entry_ref, stop_ref, _param(params, "min_stop_dist_pct", 0.8)
    )
    if geo_reject:
        result["action"] = "SKIP"
        result["reason"] = f"{label}拒绝开仓：{geo_reject}"
        return result

    smart = bool(params.get("smart_allocation", True))
    allocation = None
    if smart:
        allocation = allocate_for_entry(
            account,
            book=book,
            signal=sig,
            quote=float(px),
            entry_status=picked.get("entry_status"),
            peer_entering=peer_entering,
        )
        result["allocation"] = allocation
        if allocation.get("hold_cash"):
            result["action"] = "SKIP"
            result["reason"] = allocation.get("reason_zh") or "保留现金"
            return result

    shares, err = can_afford(
        account,
        price=px,
        stop=sig.get("stop_price"),
        direction=sig.get("direction") or "LONG",
        risk_pct=(allocation or {}).get("risk_pct"),
        max_position_pct=(allocation or {}).get("max_position_pct"),
        max_notional=(allocation or {}).get("deployable_cash"),
    )
    if shares <= 0:
        result["action"] = "SKIP"
        if allocation and float(allocation.get("deployable_cash") or 0) <= 0:
            result["reason"] = (
                f"{label}保留现金：现金储备下限 "
                f"{allocation.get('cash_reserve_pct')}%，无可部署资金"
            )
        else:
            result["reason"] = f"{label}资金不足无法开仓：{err}"
        return result

    alloc_reason = (allocation or {}).get("reason_zh") or ""
    entry_reason = picked.get("reason") or f"模拟{label}买入"
    if alloc_reason:
        entry_reason = f"{entry_reason} · {alloc_reason}"

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
            reason=entry_reason,
            trading_date=trading_date,
            book=book,
        )
    except InsufficientCashError as exc:
        result["action"] = "SKIP"
        result["reason"] = f"{label}资金不足：{exc}"
        return result
    except NonsensePriceError as exc:
        result["action"] = "SKIP"
        result["reason"] = f"{label}拒绝开仓：{exc}"
        result["quote_rejected"] = True
        return result

    if allocation:
        trade["allocation"] = {
            "cash_reserve_pct": allocation.get("cash_reserve_pct"),
            "risk_pct": allocation.get("risk_pct"),
            "max_position_pct": allocation.get("max_position_pct"),
            "budget_share": allocation.get("budget_share"),
            "confidence": allocation.get("confidence"),
            "reason_zh": allocation.get("reason_zh"),
        }
        equity = float(account.get("equity") or 1)
        notional = float(trade.get("notional") or 0)
        trade["position_pct"] = round(notional / equity * 100.0, 1) if equity else 0

    result["action"] = "ENTRY"
    result["reason"] = trade.get("reason") or entry_reason
    result["trade"] = trade
    return result


def _aggregate(books: list[dict[str, Any]], trading_date: str, phase: str) -> dict[str, Any]:
    priority = {"EXIT": 0, "ENTRY": 1, "WAIT": 2, "HOLD": 3, "SKIP": 4}
    best = None
    for row in books:
        if best is None or priority.get(row["action"], 9) < priority.get(best["action"], 9):
            best = row
    reasons = " · ".join(
        f"[{r.get('book_zh') or r.get('book')}] {r.get('action')}: {r.get('reason')}"
        for r in books
        if r.get("reason")
    )
    quotes = {r["book"]: r.get("quote") for r in books if r.get("quote") is not None}
    allocations = [r.get("allocation") for r in books if r.get("allocation")]
    return {
        "trading_date": trading_date,
        "session_phase": phase,
        "action": (best or {}).get("action") or "HOLD",
        "reason": reasons or (best or {}).get("reason") or "",
        "trade": (best or {}).get("trade"),
        "signal": (best or {}).get("signal"),
        "quote": (best or {}).get("quote"),
        "quote_source": (best or {}).get("quote_source"),
        "entry_status": (best or {}).get("entry_status"),
        "allocation": (best or {}).get("allocation")
        or (allocations[0] if allocations else None),
        "allocations": allocations,
        "books": books,
        "quotes": quotes,
        "advisory": True,
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }



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
    ensure_positions(account)
    dual = bool(params.get("dual_books", True))

    price_map: dict[str, float] | None = None
    if force_price is not None:
        price_map = {sym: float(force_price) for sym in _collect_symbols(trading_date, account)}
        # If no symbols yet, still allow empty map; entry pick injects via force later
        if not price_map:
            price_map = {}

    if not dual:
        return _decide_single_book(
            account,
            trading_date,
            phase=phase,
            raw=raw,
            price_map=price_map,
            force_price=force_price,
            params=params,
        )

    book_results: list[dict[str, Any]] = []

    # 1) Manage open positions (exits free cash before new entries)
    for book in (BOOK_INTRADAY, BOOK_SWING):
        managed = _manage_position(
            account, book, trading_date, phase, price_map, raw, params
        )
        if managed:
            book_results.append(managed)

    # 2) Entries for empty books (skip new 短线 when session closed)
    if phase == "premarket":
        if not book_results:
            return {
                "trading_date": trading_date,
                "session_phase": phase,
                "action": "SKIP",
                "reason": "盘前观望，不提前成交",
                "trade": None,
                "signal": None,
                "quote": None,
                "quote_source": None,
                "books": [],
                "advisory": True,
                "advisory_zh": "模拟交易 · 不构成投资建议",
            }
        return _aggregate(book_results, trading_date, phase)

    miss = float(params.get("miss_threshold_pct") or 1.0)

    def _inject_force(picked: dict[str, Any], inj: dict[str, float]) -> dict[str, Any]:
        if force_price is not None and picked.get("signal"):
            sym = str(picked["signal"].get("symbol") or "").upper()
            if sym:
                picked = dict(picked)
                picked["quote"] = float(force_price)
                picked["quote_source"] = "forced"
                inj[sym] = float(force_price)
        return picked

    inj = dict(price_map or {})
    if force_price is not None:
        for s in _collect_symbols(trading_date, account):
            inj[s] = float(force_price)

    need_intraday = not get_position(account, BOOK_INTRADAY) and phase != "closed"
    need_swing = not get_position(account, BOOK_SWING)

    picked_intra = None
    picked_swing = None
    if need_intraday:
        picked_intra = pick_intraday_signal(
            trading_date,
            session_phase=phase,
            miss_threshold_pct=miss,
            raw=raw,
            current_price_by_symbol=inj or None,
        )
        picked_intra = _inject_force(picked_intra, inj)
    elif not get_position(account, BOOK_INTRADAY) and phase == "closed":
        book_results.append(
            {
                "book": BOOK_INTRADAY,
                "book_zh": "短线",
                "action": "SKIP",
                "reason": "收盘后不再新开短线仓",
                "trade": None,
                "signal": None,
                "quote": None,
                "quote_source": None,
            }
        )

    if need_swing:
        picked_swing = pick_swing_signal(
            trading_date,
            session_phase=phase,
            miss_threshold_pct=miss,
            raw=raw,
            current_price_by_symbol=inj or None,
        )
        picked_swing = _inject_force(picked_swing, inj)

    both_enter = (
        bool(picked_intra and picked_intra.get("action") == "enter")
        and bool(picked_swing and picked_swing.get("action") == "enter")
    )

    # Swing core first so cash reserve + budget share apply cleanly, then satellite
    if picked_swing is not None:
        book_results.append(
            _try_entry(
                account,
                BOOK_SWING,
                picked_swing,
                trading_date,
                peer_entering=both_enter,
            )
        )
    if picked_intra is not None:
        book_results.append(
            _try_entry(
                account,
                BOOK_INTRADAY,
                picked_intra,
                trading_date,
                peer_entering=both_enter
                or bool(get_position(account, BOOK_SWING)),
            )
        )

    # Final mark with all known quotes
    px_map: dict[str, float] = {}
    for row in book_results:
        sig = row.get("signal") or {}
        sym = (sig.get("symbol") or (row.get("trade") or {}).get("symbol") or "").upper()
        if sym and row.get("quote") is not None:
            px_map[sym] = float(row["quote"])
    if px_map:
        mark_to_market(account, price_by_symbol=px_map)
    elif not open_positions(account):
        mark_to_market(account, None)

    return _aggregate(book_results, trading_date, phase)


def _decide_single_book(
    account: dict[str, Any],
    trading_date: str,
    *,
    phase: str,
    raw: dict[str, Any] | None,
    price_map: dict[str, float] | None,
    force_price: float | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Legacy single-position path (dual_books=False)."""
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

    # Prefer managing intraday book slot used as legacy position
    ensure_positions(account)
    pos = account.get("position")
    book = BOOK_INTRADAY
    if pos:
        book = pos.get("book") or (
            BOOK_SWING if "swing" in str(pos.get("horizon") or "").lower() else BOOK_INTRADAY
        )
        managed = _manage_position(
            account, book, trading_date, phase, price_map, raw, params
        )
        if managed:
            result.update(
                {
                    "action": managed["action"],
                    "reason": managed["reason"],
                    "trade": managed["trade"],
                    "signal": managed["signal"],
                    "quote": managed["quote"],
                    "quote_source": managed["quote_source"],
                    "books": [managed],
                }
            )
            return result

    if phase == "premarket":
        result["action"] = "SKIP"
        result["reason"] = "盘前观望，不提前成交"
        return result

    inj = dict(price_map or {})
    if force_price is not None:
        for s in _collect_symbols(trading_date, account):
            inj[s] = float(force_price)

    picked = pick_signal(
        trading_date,
        session_phase=phase,
        prefer_swing_if_no_intraday=bool(
            params.get("prefer_swing_if_no_intraday", True)
        ),
        miss_threshold_pct=float(params.get("miss_threshold_pct") or 1.0),
        raw=raw,
        current_price_by_symbol=inj or None,
    )
    if force_price is not None and picked.get("signal"):
        picked["quote"] = float(force_price)
        picked["quote_source"] = "forced"

    sig = picked.get("signal") or {}
    entry_book = (
        BOOK_SWING
        if "swing" in str(sig.get("horizon") or "").lower() or sig.get("source") == "swing"
        else BOOK_INTRADAY
    )
    entered = _try_entry(account, entry_book, picked, trading_date)
    result.update(
        {
            "action": entered["action"],
            "reason": entered["reason"],
            "trade": entered["trade"],
            "signal": entered["signal"],
            "quote": entered["quote"],
            "quote_source": entered["quote_source"],
            "entry_status": entered.get("entry_status"),
            "allocation": entered.get("allocation"),
            "books": [entered],
        }
    )
    return result
