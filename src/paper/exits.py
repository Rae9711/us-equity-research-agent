"""Active exit management for paper positions.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Pure, side-effect-free helpers that turn an open position + current price into
an exit plan. The three profitability levers, in priority order:

  1. Hard stop / target (respect the plan).
  2. Partial scale-out at the first target, then let a runner ride.
  3. Ratchet the stop: breakeven after +breakeven_trigger_r, then a chandelier
     trail (high-water − trail_distance_r × R) once past +trail_trigger_r.

``R`` is the initial per-share risk = |entry − initial_stop|. Everything is
expressed in R multiples so the same rules work across symbols and horizons.
"""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def risk_per_share(pos: dict[str, Any]) -> float | None:
    """Initial per-share risk (R). Prefer stored value, else derive from levels."""
    r = _safe_float(pos.get("risk_per_share"))
    if r is not None and r > 0:
        return r
    entry = _safe_float(pos.get("avg_entry"))
    stop = _safe_float(pos.get("initial_stop")) or _safe_float(pos.get("stop"))
    if entry is not None and stop is not None and abs(entry - stop) > 0:
        return abs(entry - stop)
    if entry is not None and entry > 0:
        return entry * 0.02
    return None


def update_water_marks(pos: dict[str, Any], price: float) -> None:
    """Track the max-favorable-excursion needed by trailing stops."""
    hi = _safe_float(pos.get("high_water"))
    lo = _safe_float(pos.get("low_water"))
    pos["high_water"] = price if hi is None else max(hi, price)
    pos["low_water"] = price if lo is None else min(lo, price)


def open_profit_r(pos: dict[str, Any], price: float) -> float | None:
    """Current open profit expressed in R multiples (negative if underwater)."""
    r = risk_per_share(pos)
    entry = _safe_float(pos.get("avg_entry"))
    if r is None or r <= 0 or entry is None:
        return None
    direction = (pos.get("direction") or "LONG").upper()
    move = (price - entry) if direction == "LONG" else (entry - price)
    return move / r


def max_loss_hit(pos: dict[str, Any], price: float, params: dict[str, Any]) -> bool:
    """True when open loss reaches/exceeds ``max_loss_per_trade_r`` (e.g. −1R)."""
    cap = _safe_float(params.get("max_loss_per_trade_r"))
    if cap is None or cap <= 0:
        return False
    pnl_r = open_profit_r(pos, price)
    if pnl_r is None:
        return False
    return pnl_r <= -abs(cap)


def ratchet_stop(pos: dict[str, Any], params: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return (new_stop, note) if the stop should tighten; else (current, None).

    Never loosens a stop. Uses realised high/low water, so it is monotonic and
    cannot manufacture a false stop-out on a flat tick.
    """
    direction = (pos.get("direction") or "LONG").upper()
    entry = _safe_float(pos.get("avg_entry"))
    r = risk_per_share(pos)
    cur_stop = _safe_float(pos.get("stop"))
    if entry is None or r is None or r <= 0:
        return cur_stop, None

    be_trigger = float(params.get("breakeven_trigger_r", 1.0))
    be_buffer = float(params.get("breakeven_buffer_r", 0.05))
    trail_trigger = float(params.get("trail_trigger_r", 1.5))
    trail_dist = float(params.get("trail_distance_r", 1.0))

    hi = _safe_float(pos.get("high_water")) or entry
    lo = _safe_float(pos.get("low_water")) or entry
    best_r = ((hi - entry) if direction == "LONG" else (entry - lo)) / r

    candidate = cur_stop
    note = None

    # 1) Breakeven once the trade has proven itself.
    if best_r >= be_trigger:
        if direction == "LONG":
            be = entry + be_buffer * r
            if candidate is None or be > candidate:
                candidate, note = be, "移动止损至保本"
        else:
            be = entry - be_buffer * r
            if candidate is None or be < candidate:
                candidate, note = be, "移动止损至保本"

    # 2) Chandelier trail once well in profit (overrides BE when tighter).
    if best_r >= trail_trigger:
        if direction == "LONG":
            trail = hi - trail_dist * r
            if candidate is None or trail > candidate:
                candidate, note = trail, "跟踪止损上移"
        else:
            trail = lo + trail_dist * r
            if candidate is None or trail < candidate:
                candidate, note = trail, "跟踪止损下移"

    if candidate is not None and (cur_stop is None or candidate != cur_stop):
        return round(candidate, 4), note
    return cur_stop, None


def runner_target(pos: dict[str, Any], params: dict[str, Any]) -> float | None:
    """Extended target for the runner after the first scale-out."""
    t2 = _safe_float(pos.get("target2"))
    if t2 is not None and t2 > 0:
        return t2
    entry = _safe_float(pos.get("avg_entry"))
    r = risk_per_share(pos)
    if entry is None or r is None:
        return None
    mult = float(params.get("runner_target_r", 3.0))
    direction = (pos.get("direction") or "LONG").upper()
    return round(entry + mult * r if direction == "LONG" else entry - mult * r, 4)


def plan_exit(
    pos: dict[str, Any],
    price: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Decide what to do with an open position at ``price``.

    Returns a plan dict:
      { action: 'hold'|'exit'|'scale_out',
        shares: int|None,           # for scale_out
        new_stop: float|None,       # tighten if set
        stop_note, reason }
    """
    direction = (pos.get("direction") or "LONG").upper()
    held = int(pos.get("shares") or 0)
    manage = bool(params.get("exit_management", True))

    plan: dict[str, Any] = {
        "action": "hold",
        "shares": None,
        "new_stop": None,
        "stop_note": None,
        "reason": "",
    }
    if held <= 0:
        return plan

    update_water_marks(pos, price)

    if not manage:
        return plan

    # Hard R loss circuit breaker (slippage / gap beyond plan stop).
    if max_loss_hit(pos, price, params):
        plan["action"] = "exit"
        pnl_r = open_profit_r(pos, price)
        cap = float(params.get("max_loss_per_trade_r") or 1.0)
        plan["reason"] = (
            f"单笔亏损熔断 {pnl_r:.2f}R ≤ −{cap:.2f}R @ {price}"
            if pnl_r is not None
            else f"单笔亏损熔断 @ {price}"
        )
        return plan

    first_target = _safe_float(pos.get("target1")) or _safe_float(pos.get("target"))
    already_scaled = bool(pos.get("scaled_out"))
    scale_enabled = bool(params.get("scale_out_enabled", True))
    scale_pct = float(params.get("scale_out_pct", 0.5))

    def _target_hit(t: float | None) -> bool:
        if t is None:
            return False
        return price >= t if direction == "LONG" else price <= t

    # Scale out at the first target (once), then run the remainder.
    if (
        scale_enabled
        and not already_scaled
        and first_target is not None
        and _target_hit(first_target)
        and held >= 2
        and 0.0 < scale_pct < 1.0
    ):
        qty = max(1, min(held - 1, int(round(held * scale_pct))))
        plan["action"] = "scale_out"
        plan["shares"] = qty
        # Lock the runner to (at least) breakeven after taking money off.
        entry = _safe_float(pos.get("avg_entry"))
        r = risk_per_share(pos)
        if entry is not None and r is not None:
            buf = float(params.get("breakeven_buffer_r", 0.05)) * r
            be = entry + buf if direction == "LONG" else entry - buf
            cur = _safe_float(pos.get("stop"))
            if direction == "LONG":
                plan["new_stop"] = round(be if cur is None else max(cur, be), 4)
            else:
                plan["new_stop"] = round(be if cur is None else min(cur, be), 4)
        plan["reason"] = f"首个目标 {first_target} 触及，减仓 {qty} 股锁定利润，剩余跟踪"
        return plan

    # After scaling, only the ratcheted stop or the runner target closes it.
    new_stop, note = ratchet_stop(pos, params)
    if new_stop is not None and note is not None:
        plan["new_stop"] = new_stop
        plan["stop_note"] = note

    return plan


def plan_eod_exit(
    pos: dict[str, Any],
    price: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """End-of-day policy for the *intraday* book.

    Modes (``eod_exit_mode``):
      - force: always flatten (legacy)
      - soft: only force-close if flat/losing beyond threshold; winners may
        scale-out then keep a runner overnight (payoff-asymmetry fix)
      - off: never flatten solely because the session closed

    Returns same shape as ``plan_exit`` plus optional ``promote_overnight``.
    """
    held = int(pos.get("shares") or 0)
    out: dict[str, Any] = {
        "action": "hold",
        "shares": None,
        "new_stop": None,
        "stop_note": None,
        "reason": "",
        "promote_overnight": False,
    }
    if held <= 0:
        return out

    mode = str(params.get("eod_exit_mode") or "soft").lower()
    # Compat: force_exit_intraday_at_close=False → off
    if not bool(params.get("force_exit_intraday_at_close", True)):
        mode = "off"
    if mode == "off":
        out["reason"] = "收盘不平仓（eod_exit_mode=off）"
        return out
    if mode == "force":
        out["action"] = "exit"
        out["reason"] = f"短线仓位收盘强制平仓 @ {price}"
        return out

    # soft mode
    pnl_r = open_profit_r(pos, price)
    threshold = float(params.get("eod_force_close_if_pnl_r_below", 0.25))
    allow_runner = bool(params.get("eod_allow_runner_overnight", True))
    scale_winners = bool(params.get("eod_scale_out_winners", True))
    already_scaled = bool(pos.get("scaled_out"))

    # Already running a scaled-out winner → keep overnight with BE/trail stop.
    if allow_runner and already_scaled and (pnl_r is None or pnl_r >= threshold):
        new_stop, note = ratchet_stop(pos, params)
        if new_stop is not None:
            out["new_stop"] = new_stop
            out["stop_note"] = note
        out["promote_overnight"] = True
        out["reason"] = (
            f"短线已减仓，收盘保留 runner 过夜"
            f"（开仓盈亏 {pnl_r:.2f}R ≥ {threshold:.2f}R）" if pnl_r is not None
            else "短线已减仓，收盘保留 runner 过夜"
        )
        return out

    # Strong open winner, not yet scaled: lock half, run the rest overnight.
    if (
        scale_winners
        and allow_runner
        and not already_scaled
        and pnl_r is not None
        and pnl_r >= max(threshold, 0.75)
        and held >= 2
        and bool(params.get("scale_out_enabled", True))
    ):
        scale_pct = float(params.get("scale_out_pct", 0.5))
        qty = max(1, min(held - 1, int(round(held * scale_pct))))
        out["action"] = "scale_out"
        out["shares"] = qty
        out["promote_overnight"] = True
        entry = _safe_float(pos.get("avg_entry"))
        r = risk_per_share(pos)
        direction = (pos.get("direction") or "LONG").upper()
        if entry is not None and r is not None:
            buf = float(params.get("breakeven_buffer_r", 0.05)) * r
            be = entry + buf if direction == "LONG" else entry - buf
            cur = _safe_float(pos.get("stop"))
            if direction == "LONG":
                out["new_stop"] = round(be if cur is None else max(cur, be), 4)
            else:
                out["new_stop"] = round(be if cur is None else min(cur, be), 4)
        out["reason"] = (
            f"收盘锁定部分利润（{pnl_r:.2f}R），减仓 {qty} 股，runner 过夜"
        )
        return out

    # Flat / losing / weak → cut. This is the main fix for cutting winners at EOD.
    if pnl_r is None or pnl_r < threshold:
        out["action"] = "exit"
        detail = f"{pnl_r:.2f}R" if pnl_r is not None else "未知"
        out["reason"] = (
            f"短线收盘平仓：开仓盈亏 {detail} < {threshold:.2f}R（避免过夜弱势仓）"
        )
        return out

    # Mildly profitable but below scale threshold — hold overnight with BE stop.
    new_stop, note = ratchet_stop(pos, params)
    if new_stop is not None:
        out["new_stop"] = new_stop
        out["stop_note"] = note
    out["promote_overnight"] = True
    out["reason"] = (
        f"短线小幅盈利 {pnl_r:.2f}R，收盘保留并抬止损过夜（非强制砍盈利单）"
    )
    return out
