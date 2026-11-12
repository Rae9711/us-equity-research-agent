"""Rules-based paper execution agent: ENTRY / HOLD / EXIT.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
Uses Ideal Entry zone + Entry Status + stop/target vs last price.

Dual books (default):
  - intraday / 短线 — Primary #1, may hold overnight, at most 3 trading
    days then flattened at the close window; at most 3 legs
  - swing / 长线 — swing_trade, hold overnight; at most 2 legs
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.paper.account import (
    BOOK_INTRADAY,
    BOOK_SWING,
    ensure_positions,
    get_position,
    get_positions,
    mark_to_market,
    max_positions_for_book,
    open_positions,
    append_decision,
)
from src.paper.allocation import (
    allocate_for_entry,
    apply_portfolio_risk_cap,
    book_notional_cap_pct,
)
from src.paper.circuit_breaker import breaker_decision_row, check_circuit_breaker
from src.paper.news_gate import news_block_reason, news_gate_for_date
from src.paper.broker_sim import (
    InsufficientCashError,
    NonsensePriceError,
    can_afford,
    execute_entry,
    execute_exit,
    r_multiple,
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
    resolve_minute_bars,
    resolve_quote,
    resolve_quote_bar,
)
from src.research.entry_status import infer_session_phase
from src.utils.trading_calendar import ET, holding_trading_days, session_in_close_window

logger = logging.getLogger(__name__)

# Fix 2 (revised 2026-11-12): an intraday leg may hold overnight, at most
# 3 trading days; on the 3rd trading day it is flattened at the close.
INTRADAY_MAX_HOLDING_DAYS = 3
# Swing legs also have a max hold (2026-11-12): 20 trading days, matching
# the shadow ledger's assumption. A sideways swing leg must not jam one of
# the two swing slots indefinitely.
SWING_MAX_HOLDING_DAYS = 20

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
    """Signed R multiple of ``price`` vs the entry/stop plan.

    Thin wrapper around :func:`src.paper.broker_sim.r_multiple` (kept so
    existing imports keep working); ``execute_exit`` now records R on every
    exit centrally, so call sites only need R for reason strings.
    """
    return r_multiple(direction, entry, stop, price)


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


def _parse_ts_et(value: Any) -> datetime | None:
    """Parse a bar/entry timestamp to an aware ET datetime (None if bad).

    Bar ``ts`` values are ISO strings from :func:`resolve_minute_bars`
    (tz-aware ET); fixtures may use naive ``"YYYY-MM-DD HH:MM"`` strings,
    which are read as ET. Unparseable values return None — callers treat
    that as "unknown time", never as a reason to drop a bar.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = ET.localize(dt)
    return dt.astimezone(ET)


def _bar_stop_trigger_price(direction: str, bar: dict[str, Any]) -> float | None:
    """Adverse extreme of a bar (the price that would have touched the stop)."""
    d = (direction or "LONG").upper()
    if d == "SHORT":
        return _safe_float(bar.get("high"))
    return _safe_float(bar.get("low"))


def _bar_target_trigger_price(direction: str, bar: dict[str, Any]) -> float | None:
    """Favourable extreme of a bar (the price that would have touched the target)."""
    d = (direction or "LONG").upper()
    if d == "SHORT":
        return _safe_float(bar.get("low"))
    return _safe_float(bar.get("high"))


def _scan_bars_for_exit(
    direction: str,
    stop: float | None,
    target: float | None,
    bars: list[dict[str, Any]],
    prev_close: float | None,
    entry_ts: datetime | None = None,
) -> tuple[str, dict[str, Any], float | None] | None:
    """First stop/target touch across 1-minute bars, in time order.

    Returns ``(kind, bar, prev_close_of_bar)`` with kind ``"stop"`` or
    ``"target"``, or None when no bar touched either level. Within one bar
    the stop is checked first: order inside a minute is unknowable, and the
    risk control takes precedence over the profit target.

    D1: bars starting before ``entry_ts`` (the position's entry time) can
    never trigger an exit — a leg that exited and re-entered the same day
    must not exit instantly on a stale pre-entry touch. Such bars are
    skipped for touch purposes, but their closes still feed the
    previous-close chain used for gap detection on later bars. When
    ``entry_ts`` is None the full day is scanned (previous behavior).
    """
    prev = prev_close
    for bar in bars:
        bar_dt = _parse_ts_et(bar.get("ts"))
        if entry_ts is not None and bar_dt is not None and bar_dt < entry_ts:
            close = _safe_float(bar.get("close"))
            if close is not None:
                prev = close
            continue
        trig_stop = _bar_stop_trigger_price(direction, bar)
        if stop is not None and trig_stop is not None and _stop_hit(direction, trig_stop, stop):
            return ("stop", bar, prev)
        trig_target = _bar_target_trigger_price(direction, bar)
        if target is not None and trig_target is not None and _target_hit(
            direction, trig_target, target
        ):
            return ("target", bar, prev)
        close = _safe_float(bar.get("close"))
        if close is not None:
            prev = close
    return None


def _entry_ts_for_scan(
    pos: dict[str, Any], trading_date: str
) -> datetime | None:
    """Entry time used to skip pre-entry bars, or None (no filtering).

    Only a leg opened *today* needs filtering: today's bars all postdate
    an earlier day's entry, so no filter applies then. When the leg opened
    today but ``opened_at`` is missing or unparseable, fall back to
    scanning the full day (previous behavior) and log it — a missing
    timestamp must never silently drop a genuine touch. An entry whose
    wall-clock date disagrees with the trading date (e.g. a fixture whose
    entry was stamped "now" for a past session) also skips the filter:
    applying it would drop every bar.
    """
    if (pos.get("opened_date") or "") != trading_date:
        return None
    raw = pos.get("opened_at")
    if not raw:
        logger.warning(
            "position %s opened %s has no opened_at; scanning full-day bars",
            pos.get("symbol"),
            trading_date,
        )
        return None
    dt = _parse_ts_et(raw)
    if dt is None:
        logger.warning(
            "position %s opened_at %r unparseable; scanning full-day bars",
            pos.get("symbol"),
            raw,
        )
        return None
    if dt.date().isoformat() != trading_date:
        logger.debug(
            "position %s opened_at %s is not on trading date %s; "
            "skipping pre-entry bar filter",
            pos.get("symbol"),
            raw,
            trading_date,
        )
        return None
    return dt


def _last_bar_close(bars: list[dict[str, Any]] | None) -> float | None:
    """Close of the last 1-minute bar with a usable close (None if none)."""
    if not bars:
        return None
    for bar in reversed(bars):
        close = _safe_float(bar.get("close"))
        if close is not None and close > 0:
            return close
    return None


def _resolve_bars_for_symbol(
    sym: str,
    trading_date: str,
    raw: dict[str, Any] | None,
    bars_map: dict[str, list[dict[str, Any]]] | None,
    forced: bool,
) -> tuple[list[dict[str, Any]] | None, bool]:
    """Today's 1-minute bars for one symbol, preferring already-fetched bars.

    Returns ``(bars, resolved)``: ``resolved`` is True once the source has
    been consulted, so a tick never fetches the same bars twice (the D3
    fallback and the stop/target scan share the one resolution).
    """
    if bars_map is not None:
        return bars_map.get(sym), True
    if forced:
        return None, True
    try:
        return resolve_minute_bars(sym, trading_date, raw=raw), True
    except Exception:
        logger.debug("minute bar scan failed for %s", sym, exc_info=True)
        return None, True


def _target_fill_price(
    direction: str,
    target: float,
    bar: dict[str, Any],
    prev_close: float | None,
) -> tuple[float, bool]:
    """Fill for a target touch inside one 1-minute bar.

    A bar that *opened* already beyond the target is a true gap: fill at the
    open (the gap gain is real, mirroring the gap-through-stop rule). An
    intraday touch inside the bar fills at the target itself — the
    first-touch price — never at a later tick's farther price.
    """
    open_px = _safe_float(bar.get("open"))
    d = (direction or "LONG").upper()
    if open_px is not None and prev_close is not None:
        if d == "SHORT" and prev_close > target and open_px <= target:
            return open_px, True
        if d == "LONG" and prev_close < target and open_px >= target:
            return open_px, True
    return target, False


def _manage_single_position(
    account: dict[str, Any],
    book: str,
    pos: dict[str, Any],
    trading_date: str,
    phase: str,
    price_map: dict[str, float] | None,
    raw: dict[str, Any] | None,
    params: dict[str, Any],
    bars_map: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Manage one open leg. Returns a book-result dict or None if skipped."""

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
    anchors = anchors_from_position(pos)
    sane, reject = is_sane_fill_price(
        px, anchors=anchors, max_deviation_pct=max_dev
    )
    forced = bool(price_map and sym in price_map)
    bars: list[dict[str, Any]] | None = None
    bars_resolved = False
    if not sane:
        # D3: when the live quote is rejected, fall back to the last
        # 1-minute bar close already fetched for this symbol — but only
        # when THAT price also passes the sane check against the same
        # anchors. When the fallback is also insane or unavailable, keep
        # the previous refuse-and-hold behavior.
        bars, bars_resolved = _resolve_bars_for_symbol(
            sym, trading_date, raw, bars_map, forced
        )
        fb_close = _last_bar_close(bars)
        fb_sane = False
        if fb_close is not None:
            fb_sane, _fb_reject = is_sane_fill_price(
                fb_close, anchors=anchors, max_deviation_pct=max_dev
            )
        if fb_close is not None and fb_sane:
            logger.info(
                "%s %s: live quote $%s rejected (%s); using last 1m bar "
                "close $%s as quote",
                label,
                sym,
                px,
                reject,
                fb_close,
            )
            px = fb_close
            qsrc = "minute_bar_close_fallback"
            base["quote"] = px
            base["quote_source"] = qsrc
            base["quote_fallback"] = True
        else:
            logger.warning(
                "%s %s: live quote $%s rejected (%s); bar-close fallback %s "
                "— refusing stop/target/close exit, holding",
                label,
                sym,
                px,
                reject,
                "also rejected" if fb_close is not None else "unavailable",
            )
            base["reason"] = (
                f"{label}持仓 {sym}：{reject or '报价异常'}，拒绝止损/止盈/平仓，继续持有"
            )
            base["quote_rejected"] = True
            return base

    stop = _safe_float(pos.get("stop"))
    target = _safe_float(pos.get("target"))
    entry_px = _safe_float(pos.get("avg_entry"))
    horizon = (pos.get("horizon") or ("Swing" if book == BOOK_SWING else "Intraday")).lower()

    # --- Fix 1: sequential 1-minute bars ---------------------------------
    # A 30-minute tick only sees the last price: a stop touched mid-interval
    # and recovered, or a target touched and run past, was invisible or
    # filled at the stale far quote. Scan today's 1m bars in order; the
    # first touch wins and fills off that bar, not the late tick quote.
    # When no bars are available (or the quote was force-injected for a
    # demo/test), fall through to the single-price checks below.
    # Bars already resolved for the D3 fallback above are reused here.
    if not bars_resolved:
        bars, bars_resolved = _resolve_bars_for_symbol(
            sym, trading_date, raw, bars_map, forced
        )
    if bars:
        bar_prev_close = resolve_quote_bar(sym, trading_date, raw=raw).get(
            "prev_close"
        )
        hit = _scan_bars_for_exit(
            direction, stop, target, bars, bar_prev_close,
            entry_ts=_entry_ts_for_scan(pos, trading_date),
        )
        if hit is not None:
            kind, bar, bar_prev = hit
            if kind == "stop":
                max_slip = _param(params, "max_stop_slippage_pct", 0.5)
                trig = _bar_stop_trigger_price(direction, bar)
                bar_evidence = {"open": bar.get("open"), "prev_close": bar_prev}
                fill, capped, gap = _stop_fill_price(
                    direction, stop, trig, bar_evidence, max_slip
                )
                r_mult = _r_multiple(direction, entry_px, stop, fill)
                reason = f"{label}止损触发 @ {fill:.4g} (stop={stop})"
                if gap:
                    reason += "·跳空穿止损按开盘价成交"
                elif capped:
                    reason += f"·成交价触滑点上限 {max_slip:g}%"
                reason += f"·1分钟K线 {bar.get('ts')} 先触及"
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
                    symbol=sym,
                )
                trade["stop_trigger"] = stop
                trade["fill_price"] = round(fill, 4)
                trade["slippage_capped"] = capped
                trade["gap_through_stop"] = gap
                trade["bar_ts"] = bar.get("ts")
                _set_stop_cooldown(account, trading_date, book, sym)
                base["action"] = "EXIT"
                base["reason"] = trade["reason"]
                base["trade"] = trade
                return base
            fill, gap = _target_fill_price(direction, target, bar, bar_prev)
            r_mult = _r_multiple(direction, entry_px, stop, fill)
            target_reason = f"{label}止盈触发 @ {fill:.4g} (target={target})"
            if gap:
                target_reason += "·跳空穿目标按开盘价成交"
            target_reason += f"·1分钟K线 {bar.get('ts')} 先触及"
            if r_mult is not None:
                target_reason += f"·R={r_mult:.2f}"
            trade = execute_exit(
                account,
                price=fill,
                reason=target_reason,
                trading_date=trading_date,
                book=book,
                symbol=sym,
            )
            trade["fill_price"] = round(fill, 4)
            trade["gap_through_target"] = gap
            trade["bar_ts"] = bar.get("ts")
            base["action"] = "EXIT"
            base["reason"] = trade["reason"]
            base["trade"] = trade
            return base

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
            symbol=sym,
        )
        trade["stop_trigger"] = stop
        trade["fill_price"] = round(fill, 4)
        trade["slippage_capped"] = capped
        trade["gap_through_stop"] = gap
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
                symbol=sym,
            )
            _set_stop_cooldown(account, trading_date, book, sym)
            base["action"] = "EXIT"
            base["reason"] = trade["reason"]
            base["trade"] = trade
            return base

    if _target_hit(direction, px, target):
        r_mult = _r_multiple(direction, entry_px, stop, px)
        target_reason = f"{label}止盈触发 @ {px} (target={target})"
        if r_mult is not None:
            target_reason += f"·R={r_mult:.2f}"
        trade = execute_exit(
            account,
            price=px,
            reason=target_reason,
            trading_date=trading_date,
            book=book,
            symbol=sym,
        )
        base["action"] = "EXIT"
        base["reason"] = trade["reason"]
        base["trade"] = trade
        return base

    # Fix 2 (revised 2026-11-12): an intraday leg may hold overnight, at
    # most 3 trading days. On the 3rd trading day it is deterministically
    # flattened in the close window (the last tick of the day); if that
    # window was missed (no tick ran), the first later tick flattens it
    # instead (overdue). Swing legs are untouched. The check lives in the
    # tick flow — there is no EOD job — and reuses flatten_intraday_at_close.
    force_eod = bool(params.get("force_exit_intraday_at_close", True))
    is_swing = book == BOOK_SWING or "swing" in horizon
    if force_eod and not is_swing:
        holding_days = holding_trading_days(pos.get("opened_date"), trading_date)
        due = (
            holding_days is not None
            and holding_days >= INTRADAY_MAX_HOLDING_DAYS
            and (
                session_in_close_window(trading_date)
                or (
                    holding_days > INTRADAY_MAX_HOLDING_DAYS
                    and phase != "premarket"
                )
            )
        )
        if due:
            flattened = flatten_intraday_at_close(
                account,
                trading_date,
                price=px,
                raw=raw,
                symbol=sym,
                min_holding_days=INTRADAY_MAX_HOLDING_DAYS,
            )
            if flattened is not None and flattened.get("action") in (
                "EXIT",
                "HOLD",
            ):
                return flattened
            # SKIP (not due — defensive) falls through to the HOLD tail.

    # Fix 2 (extended 2026-11-12): a swing leg also has a max hold — 20
    # trading days, matching the shadow ledger. On the 20th trading day it
    # is flattened in the close window; if that window was missed, the
    # first later tick flattens it instead (overdue). A time exit is
    # neutral, like a target exit: no stop cooldown is set.
    force_swing_eod = bool(params.get("force_exit_swing_at_close", True))
    if force_swing_eod and is_swing:
        holding_days = holding_trading_days(pos.get("opened_date"), trading_date)
        due = (
            holding_days is not None
            and holding_days >= SWING_MAX_HOLDING_DAYS
            and (
                session_in_close_window(trading_date)
                or (
                    holding_days > SWING_MAX_HOLDING_DAYS
                    and phase != "premarket"
                )
            )
        )
        if due:
            r_mult = _r_multiple(direction, entry_px, stop, px)
            reason = f"{label}持仓 {sym} 满 {holding_days} 个交易日收盘平仓 @ {px}"
            if r_mult is not None:
                reason += f"·R={r_mult:.2f}"
            trade = execute_exit(
                account,
                price=px,
                reason=reason,
                trading_date=trading_date,
                book=book,
                symbol=sym,
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


def _manage_position(
    account: dict[str, Any],
    book: str,
    trading_date: str,
    phase: str,
    price_map: dict[str, float] | None,
    raw: dict[str, Any] | None,
    params: dict[str, Any],
    bars_map: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Manage every open leg in a book. Returns one result per leg."""
    results: list[dict[str, Any]] = []
    for leg in get_positions(account, book):
        managed = _manage_single_position(
            account, book, leg, trading_date, phase, price_map, raw, params,
            bars_map=bars_map,
        )
        if managed:
            results.append(managed)
    return results


def flatten_intraday_at_close(
    account: dict[str, Any],
    trading_date: str,
    *,
    price: float | None = None,
    bars: list[dict[str, Any]] | None = None,
    raw: dict[str, Any] | None = None,
    symbol: str | None = None,
    min_holding_days: int | None = INTRADAY_MAX_HOLDING_DAYS,
) -> dict[str, Any] | None:
    """Deterministically flatten one intraday leg at/after the session close.

    Replaces the old opportunistic ``phase == "closed"`` branch (which only
    fired if a tick happened to run with that phase). Called from
    :func:`_manage_single_position` when the leg's holding limit expires —
    there is no EOD job, the check lives in the tick flow. Swing legs are
    untouched.

    ``min_holding_days`` is a guard: legs held fewer trading days are left
    alone (returned as SKIP). Exit price preference: injected price > last
    1-minute bar close (the true close) > live/last quote > the leg's last
    mark. Returns a book-result dict, or None when no matching leg is open.
    """
    if symbol:
        legs = get_positions(account, BOOK_INTRADAY)
        pos = next(
            (
                p
                for p in legs
                if str(p.get("symbol") or "").upper() == str(symbol).upper()
            ),
            None,
        )
    else:
        pos = get_position(account, BOOK_INTRADAY)
    if not pos:
        return None

    label = BOOK_LABEL_ZH.get(BOOK_INTRADAY, BOOK_INTRADAY)
    sym = pos["symbol"]
    direction = (pos.get("direction") or "LONG").upper()
    entry_px = _safe_float(pos.get("avg_entry"))
    stop = _safe_float(pos.get("stop"))
    holding_days = holding_trading_days(pos.get("opened_date"), trading_date)

    base: dict[str, Any] = {
        "book": BOOK_INTRADAY,
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
        "quote": None,
        "quote_source": "none",
    }
    if min_holding_days is not None and (
        holding_days is None or holding_days < min_holding_days
    ):
        held = "未知" if holding_days is None else f"{holding_days} 天"
        base["action"] = "SKIP"
        base["reason"] = (
            f"{label}持仓 {sym}：已持有 {held}，未到 {min_holding_days} 个交易日，"
            "收盘不平仓"
        )
        return base

    px: float | None = None
    qsrc = "none"
    if price is not None and price > 0:
        px, qsrc = float(price), "injected"
    if px is None:
        use_bars = bars
        if use_bars is None:
            try:
                use_bars = resolve_minute_bars(sym, trading_date, raw=raw)
            except Exception:
                logger.debug("EOD bar fetch failed for %s", sym, exc_info=True)
                use_bars = None
        if use_bars:
            last_close = _safe_float(use_bars[-1].get("close"))
            if last_close is not None and last_close > 0:
                px, qsrc = last_close, "minute_bar_close"
    if px is None:
        px, qsrc = resolve_quote(sym, trading_date, raw=raw, prefer_live=True)
    if px is None:
        px = _safe_float(pos.get("last_price"))
        qsrc = "last_mark" if px else "none"
    base["quote"] = px
    base["quote_source"] = qsrc

    if px is None or px <= 0:
        base["reason"] = f"{label}持仓 {sym}：收盘平仓无可用报价，继续持有"
        return base

    params = account.get("params") or {}
    max_dev = float(
        params.get("max_price_deviation_pct") or DEFAULT_MAX_DEVIATION_PCT
    )
    sane, reject = is_sane_fill_price(
        px, anchors=anchors_from_position(pos), max_deviation_pct=max_dev
    )
    if not sane:
        base["reason"] = (
            f"{label}持仓 {sym}：{reject or '报价异常'}，拒绝收盘平仓，继续持有"
        )
        base["quote_rejected"] = True
        return base

    r_mult = _r_multiple(direction, entry_px, stop, px)
    reason = (
        f"{label}持仓 {sym} 满 {holding_days} 个交易日收盘平仓 @ {px}"
    )
    if r_mult is not None:
        reason += f"·R={r_mult:.2f}"
    trade = execute_exit(
        account,
        price=px,
        reason=reason,
        trading_date=trading_date,
        book=BOOK_INTRADAY,
        symbol=sym,
    )
    base["action"] = "EXIT"
    base["reason"] = trade["reason"]
    base["trade"] = trade
    return base


def _try_entry(
    account: dict[str, Any],
    book: str,
    picked: dict[str, Any],
    trading_date: str,
    *,
    peer_entering: bool = False,
    news_gate: dict[str, Any] | None = None,
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

    # Fix 7: one symbol at most once per book — a second leg must be a
    # different symbol; the signal picker may re-pick a held symbol.
    if sym and any(
        str(p.get("symbol") or "").upper() == sym
        for p in get_positions(account, book)
    ):
        result["action"] = "SKIP"
        result["reason"] = f"{label}跳过：{sym} 已在{label}持仓，不重复开仓"
        return result

    # Fix 7: book-full guard (decide_and_act also gates the pick, but
    # _try_entry must stay safe when called directly).
    limit = max_positions_for_book(book)
    if len(get_positions(account, book)) >= limit:
        result["action"] = "SKIP"
        result["reason"] = f"{label}跳过：{label}账本已满（{limit} 个仓位）"
        return result

    # Fix 6: news-availability gate — when Polygon's intraday news fetch
    # failed/429'd for this symbol (or globally), the research behind the
    # signal was built without its news, so no new entry. Existing
    # positions are managed elsewhere and are unaffected.
    block_reason = news_block_reason(news_gate, sym)
    if block_reason:
        result["action"] = "SKIP"
        result["reason"] = f"{label}跳过：{block_reason}"
        result["news_blocked"] = True
        result["news_gate"] = news_gate
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

    # Fix 7: the book's notional cap (40% intraday / 70% swing x
    # confidence) is shared by all its legs, not per leg. This entry must
    # fit BOTH its own per-entry cap and the headroom left under the book
    # cap; when the book already uses the whole cap, skip instead of opening.
    effective_max_pos_pct = (allocation or {}).get("max_position_pct")
    book_notional_used_pct = 0.0
    book_cap_pct = None
    if allocation is not None:
        equity = float(account.get("equity") or 0.0)
        used = 0.0
        for leg in get_positions(account, book):
            mv = _safe_float(leg.get("market_value"))
            if mv is None:
                px_leg = _safe_float(leg.get("last_price")) or _safe_float(
                    leg.get("avg_entry")
                )
                mv = float(leg.get("shares") or 0) * (px_leg or 0.0)
            used += mv or 0.0
        book_notional_used_pct = (used / equity * 100.0) if equity else 0.0
        book_cap_pct = book_notional_cap_pct(book, allocation.get("confidence"))
        headroom_pct = book_cap_pct - book_notional_used_pct
        effective_max_pos_pct = max(
            0.0,
            min(float(effective_max_pos_pct or 0.0), headroom_pct),
        )
        allocation = dict(allocation)
        allocation["max_position_pct"] = effective_max_pos_pct
        allocation["book_notional_used_pct"] = round(book_notional_used_pct, 2)
        allocation["book_cap_pct"] = round(book_cap_pct, 1)
        result["allocation"] = allocation
        if book_notional_used_pct > 0 and headroom_pct <= 0:
            result["action"] = "SKIP"
            result["reason"] = (
                f"{label}跳过：账本名义上限已满，已用 "
                f"{book_notional_used_pct:.1f}% ≥ 账本上限 {book_cap_pct:.1f}%"
            )
            return result

    shares, err = can_afford(
        account,
        price=px,
        stop=sig.get("stop_price"),
        direction=sig.get("direction") or "LONG",
        risk_pct=(allocation or {}).get("risk_pct"),
        max_position_pct=effective_max_pos_pct,
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

    # Fix 5: portfolio-level planned-risk cap — existing open positions'
    # entry→stop dollars plus this entry's planned risk must stay within
    # 3% of equity. Reduce shares to fit, or skip when even 1 share
    # breaches the cap. Per-trade 3% clamp and notional caps are unchanged.
    shares, cap_detail = apply_portfolio_risk_cap(
        account,
        direction=direction,
        entry_price=float(px),
        stop_price=sig.get("stop_price"),
        shares=shares,
    )
    result["portfolio_risk_cap"] = cap_detail
    if shares <= 0:
        result["action"] = "SKIP"
        result["reason"] = f"{label}跳过：{cap_detail.get('reason_zh') or '组合计划风险已满'}"
        return result

    alloc_reason = (allocation or {}).get("reason_zh") or ""
    entry_reason = picked.get("reason") or f"模拟{label}买入"
    if alloc_reason:
        entry_reason = f"{entry_reason} · {alloc_reason}"
    if cap_detail.get("capped"):
        entry_reason = f"{entry_reason} · {cap_detail.get('reason_zh')}"
    if book_notional_used_pct > 0:
        entry_reason = (
            f"{entry_reason} · 账本名义已用 {book_notional_used_pct:.1f}%"
            f"（上限 {book_cap_pct:.1f}%），本笔上限 {effective_max_pos_pct:.1f}%"
        )

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

    if cap_detail.get("capped"):
        trade["portfolio_risk_capped"] = True
        trade["portfolio_planned_risk"] = cap_detail

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
    bars_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    tick_dt: datetime | None = None,
) -> dict[str, Any]:
    """One paper-trading tick. Mutates ``account``. Returns decision summary.

    ``bars_by_symbol`` injects today's 1-minute bars per symbol (tests /
    replay); when omitted the tick fetches live bars for stop/target scans.

    ``tick_dt`` is the time of this tick (ET); it is threaded to the
    circuit breaker so trip/reset events record the tick time, not the
    wall-clock run time. When omitted the breaker falls back to now,
    which is the tick time for a live run.
    """
    phase = session_phase or infer_session_phase(trading_date)
    params = account.get("params") or {}
    ensure_positions(account)
    dual = bool(params.get("dual_books", True))
    # Fix 6: news gate is resolved once per tick from the freshest
    # collection evidence (Step 3 intraday errors, else Step 0 raw errors).
    news_gate = news_gate_for_date(trading_date, raw=raw)

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
            bars_map=bars_by_symbol,
            tick_dt=tick_dt,
        )

    book_results: list[dict[str, Any]] = []

    # 1) Manage open positions (exits free cash before new entries)
    for book in (BOOK_INTRADAY, BOOK_SWING):
        managed = _manage_position(
            account, book, trading_date, phase, price_map, raw, params,
            bars_map=bars_by_symbol,
        )
        book_results.extend(managed or [])

    # Item 4 (circuit breaker): day -1% / week -2% blocks ALL new entries.
    # Position management above already ran and is unaffected. Trips latch
    # for the rest of the day/week; check_circuit_breaker persists state
    # and returns any trip/reset events for the decision record.
    breaker_status, breaker_events = check_circuit_breaker(
        account, trading_date, tick_dt=tick_dt
    )
    for _ev in breaker_events:
        append_decision(account, breaker_decision_row(_ev, trading_date))

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
                "news_gate": news_gate,
                "circuit_breaker": breaker_status,
                "advisory": True,
                "advisory_zh": "模拟交易 · 不构成投资建议",
            }
        _pre = _aggregate(book_results, trading_date, phase)
        _pre["circuit_breaker"] = breaker_status
        return _pre

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

    # Fix 7: a book with a free slot may take a new leg (intraday ≤3,
    # swing ≤2). No new intraday leg after the close.
    need_intraday = (
        len(get_positions(account, BOOK_INTRADAY))
        < max_positions_for_book(BOOK_INTRADAY)
        and phase != "closed"
    )
    need_swing = len(get_positions(account, BOOK_SWING)) < max_positions_for_book(
        BOOK_SWING
    )

    # Item 4 (circuit breaker): a tripped day/week breaker blocks ALL new
    # entries. Position management above is unaffected.
    if breaker_status.get("blocked"):
        need_intraday = False
        need_swing = False
        book_results.append(
            {
                "book": "circuit_breaker",
                "book_zh": "熔断",
                "action": "SKIP",
                "reason": breaker_status.get("reason") or "熔断封锁新开仓",
                "trade": None,
                "signal": None,
                "quote": None,
                "quote_source": None,
            }
        )

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
                news_gate=news_gate,
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
                news_gate=news_gate,
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

    result = _aggregate(book_results, trading_date, phase)
    result["news_gate"] = news_gate
    result["circuit_breaker"] = breaker_status
    return result


def _decide_single_book(
    account: dict[str, Any],
    trading_date: str,
    *,
    phase: str,
    raw: dict[str, Any] | None,
    price_map: dict[str, float] | None,
    force_price: float | None,
    params: dict[str, Any],
    bars_map: dict[str, list[dict[str, Any]]] | None = None,
    tick_dt: datetime | None = None,
) -> dict[str, Any]:
    """Legacy single-position path (dual_books=False)."""
    news_gate = news_gate_for_date(trading_date, raw=raw)
    # Item 4 (circuit breaker): evaluated on every tick so trip/reset
    # events are recorded and the status is exposed; it gates entries
    # only, never the position management below.
    _breaker_status, _breaker_events = check_circuit_breaker(
        account, trading_date, tick_dt=tick_dt
    )
    for _ev in _breaker_events:
        append_decision(account, breaker_decision_row(_ev, trading_date))
    result: dict[str, Any] = {
        "trading_date": trading_date,
        "session_phase": phase,
        "action": "HOLD",
        "reason": "",
        "trade": None,
        "signal": None,
        "quote": None,
        "quote_source": None,
        "news_gate": news_gate,
        "circuit_breaker": _breaker_status,
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
            account, book, trading_date, phase, price_map, raw, params,
            bars_map=bars_map,
        )
        if managed:
            first = managed[0]
            result.update(
                {
                    "action": first["action"],
                    "reason": first["reason"],
                    "trade": first["trade"],
                    "signal": first["signal"],
                    "quote": first["quote"],
                    "quote_source": first["quote_source"],
                    "books": managed,
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

    if _breaker_status.get("blocked"):
        result["action"] = "SKIP"
        result["reason"] = _breaker_status.get("reason") or "熔断封锁新开仓"
        return result

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
    entered = _try_entry(account, entry_book, picked, trading_date, news_gate=news_gate)
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
