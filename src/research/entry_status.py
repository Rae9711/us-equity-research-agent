"""Entry Status / live execution layer vs Morning Ideal Entry.

ADVISORY ONLY — 不构成投资建议.

Ideal Entry (entry_zone / entry_price / entry_source) remains the premarket plan.
This module classifies whether that plan is still fillable given current price,
and when MISSED, proposes a light alternate-entry stub (not a full replan).
"""

from __future__ import annotations

from typing import Any

ADVISORY_TAG = "ADVISORY — 不构成投资建议"

# Gap past Ideal Entry zone (beyond high for LONG / below low for SHORT) → MISSED
DEFAULT_MISS_THRESHOLD_PCT = 1.0
# Alternate entry band around current / VWAP (±pct of current)
ALT_ZONE_BAND_PCT = 0.4

STATUS_PREMARKET = "PREMARKET"
STATUS_PLANNED = "PLANNED"
STATUS_ACTIVE = "ACTIVE"
STATUS_READY = "READY"
STATUS_TRIGGERED = "TRIGGERED"
STATUS_MISSED = "MISSED"
STATUS_INVALIDATED = "INVALIDATED"
STATUS_EXPIRED = "EXPIRED"

ACTION_WAIT_VWAP = "Wait VWAP Pullback"
ACTION_DO_NOT_CHASE = "Do Not Chase"
ACTION_WATCH_ALT = "Watch alt entry"
ACTION_NEW_SETUP = ACTION_WATCH_ALT  # alias
ACTION_ABANDON = "Abandon today"
ACTION_VALID = "Entry still valid"

ALT_ACTION_WATCH = "Watch only"
ALT_ACTION_SECONDARY = "Secondary setup"

# Min R:R (reward/risk) for Secondary setup (else Watch only)
MIN_ALT_RR = 1.5

_ALT_REASON_LABELS = {
    "wait_pullback": "VWAP retest",
    "vwap_reject": "VWAP reject",
    "failed_breakdown_retest": "Failed breakdown retest",
    "do_not_chase": "Do not chase Ideal Entry",
    "abandon": "Abandon today",
}

_CHASE_HINTS = frozenset({STATUS_MISSED, STATUS_INVALIDATED, STATUS_EXPIRED})


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _zone_bounds(
    entry_zone: dict[str, Any] | None,
    entry_price: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Return (low, mid/ideal, high)."""
    low = mid = high = None
    if entry_zone:
        low = _safe_float(entry_zone.get("low"))
        high = _safe_float(entry_zone.get("high"))
        mid = _safe_float(entry_zone.get("mid"))
    if mid is None:
        mid = entry_price
    if mid is not None:
        if low is None:
            low = mid
        if high is None:
            high = mid
    return low, mid, high


def _distance_pct(current: float, ideal: float, direction: str) -> float:
    """Signed distance: positive = adverse to fill (above ideal for LONG, below for SHORT)."""
    raw = (current - ideal) / ideal * 100.0
    if direction == "SHORT":
        return round(-raw, 2)
    return round(raw, 2)


def remaining_er_pct(
    *,
    direction: str,
    current_price: float | None,
    target_price: float | None,
) -> float | None:
    """Dynamic ER from **current → ideal target** (not frozen Morning entry→target)."""
    current = _safe_float(current_price)
    target = _safe_float(target_price)
    if current is None or target is None or current <= 0:
        return None
    direction = (direction or "").upper()
    if direction == "LONG":
        return round((target - current) / current * 100.0, 2)
    if direction == "SHORT":
        return round((current - target) / current * 100.0, 2)
    return None


# Public alias used in UI / Step3 extras
live_expected_return_pct = remaining_er_pct


def _make_zone(mid: float, band_pct: float = ALT_ZONE_BAND_PCT) -> dict[str, Any]:
    half = mid * (band_pct / 100.0)
    low = round(mid - half, 2)
    high = round(mid + half, 2)
    mid_r = round(mid, 2)
    return {
        "low": low,
        "mid": mid_r,
        "high": high,
        "display": f"Alt Entry {low:.2f}–{high:.2f}",
    }


def _geometry_ok(
    direction: str,
    entry: float,
    stop: float | None,
    target: float | None,
) -> bool:
    if stop is None or target is None:
        return False
    if direction == "LONG":
        return stop < entry <= target
    if direction == "SHORT":
        return target <= entry < stop
    return False


def _alt_rr(
    direction: str,
    entry: float,
    stop: float | None,
    target: float | None,
) -> float | None:
    if stop is None or target is None or entry <= 0:
        return None
    risk = abs(entry - stop) / entry * 100.0
    if risk <= 0:
        return None
    reward = remaining_er_pct(direction=direction, current_price=entry, target_price=target)
    if reward is None or reward <= 0:
        return None
    return round(reward / risk, 2)


def _atr_proxy(current: float, ideal: float | None) -> float:
    """Rough ATR proxy (~0.8% of price, floored by distance from ideal)."""
    base = current * 0.008
    if ideal is not None and ideal > 0:
        base = max(base, abs(current - ideal) * 0.25)
    return max(base, current * 0.004)


def _invalidated_has_reentry_room(
    *,
    direction: str,
    current: float,
    stop: float | None,
    rem_er: float | None,
) -> bool:
    """INVALIDATED but not deep past stop + still some ER → soft re-entry watch."""
    if stop is None or rem_er is None or rem_er < 0.5:
        return False
    if direction == "LONG":
        # Stopped out below; room if not more than ~1.2% below stop
        return current >= stop * (1.0 - 0.012)
    if direction == "SHORT":
        return current <= stop * (1.0 + 0.012)
    return False


def _build_alternate_entry(
    *,
    status: str,
    direction: str,
    current: float,
    ideal: float | None,
    target_price: float | None,
    stop_price: float | None,
    vwap: float | None = None,
    orb_high: float | None = None,
    orb_low: float | None = None,
) -> dict[str, Any] | None:
    """Light MISSED (and INVALIDATED-with-room) alternate plan stub.

    Does not invent a full Execution Agent replan — provisional levels only.
    """
    if status not in (STATUS_MISSED, STATUS_INVALIDATED):
        return None

    rem_er = remaining_er_pct(
        direction=direction, current_price=current, target_price=target_price
    )
    reeval_note = "re-evaluate at next session update"

    if status == STATUS_INVALIDATED and not _invalidated_has_reentry_room(
        direction=direction, current=current, stop=stop_price, rem_er=rem_er
    ):
        return {
            "suggested_action": ACTION_DO_NOT_CHASE,
            "suggestion": "do_not_chase",
            "alt_entry_zone": None,
            "alt_entry": None,
            "alt_stop": None,
            "alt_target": None,
            "alt_er_pct": None,
            "alt_reason": "Ideal Entry invalidated",
            "alt_action": ALT_ACTION_WATCH,
            "remaining_er_pct": rem_er,
            "live_expected_return_pct": rem_er,
            "alt_rr": None,
            "anchor": None,
            "anchor_price": None,
            "note": f"Ideal Entry invalidated — {reeval_note}",
            "advisory": True,
        }

    # --- MISSED (or INVALIDATED with re-entry room) ---
    atr = _atr_proxy(current, ideal)
    anchor_name = "current_retest"
    anchor_px = current
    suggestion = "failed_breakdown_retest" if direction == "SHORT" else "wait_pullback"

    if direction == "LONG":
        # Prefer VWAP pullback zone below current
        if vwap is not None and vwap < current:
            anchor_name = "vwap"
            anchor_px = vwap
            suggestion = "wait_pullback"
        elif orb_low is not None and orb_low < current:
            anchor_name = "orb_low"
            anchor_px = orb_low
            suggestion = "wait_pullback"
        else:
            # Soft pullback band ~0.5% below current
            anchor_px = current * (1.0 - 0.005)
            suggestion = "wait_pullback"
    else:
        # SHORT: VWAP reject / failed-breakdown retest near current
        if vwap is not None and abs(vwap - current) / current * 100.0 <= 1.5:
            anchor_name = "vwap"
            anchor_px = vwap
            suggestion = "vwap_reject"
        elif orb_high is not None and abs(orb_high - current) / current * 100.0 <= 2.0:
            anchor_name = "orb_high"
            anchor_px = orb_high
            suggestion = "failed_breakdown_retest"
        else:
            anchor_name = "current_retest"
            anchor_px = current
            suggestion = "failed_breakdown_retest"

    alt_zone = _make_zone(anchor_px)
    alt_entry = float(alt_zone["mid"])
    alt_stop: float | None = None
    alt_target: float | None = None

    if direction == "LONG":
        cand_stop = stop_price
        if cand_stop is None or cand_stop >= alt_entry:
            cand_stop = round(alt_entry - atr, 2)
        if cand_stop >= alt_entry:
            cand_stop = round(alt_entry * (1.0 - 0.008), 2)
        cand_target = target_price
        if cand_target is None or cand_target < alt_entry:
            # Midway stub: half the gap from alt_entry toward a 1.5% upside
            cand_target = round(alt_entry * 1.0075, 2)
        if target_price is not None and target_price > alt_entry:
            # Prefer original target; else midway between alt_entry and original
            if target_price >= alt_entry:
                cand_target = round(target_price, 2)
        alt_stop = round(cand_stop, 2)
        alt_target = round(cand_target, 2) if cand_target is not None else None
    else:
        # Stop above recent high / current + ATR proxy
        cand_stop = stop_price
        floor_stop = round(alt_entry + atr, 2)
        if orb_high is not None and orb_high > alt_entry:
            floor_stop = max(floor_stop, round(orb_high * 1.002, 2))
        if cand_stop is None or cand_stop <= alt_entry:
            cand_stop = floor_stop
        else:
            cand_stop = max(cand_stop, floor_stop)
        cand_target = target_price
        if cand_target is None or cand_target >= alt_entry:
            # Midway toward a soft downside if original target unusable
            if ideal is not None and ideal < alt_entry:
                cand_target = round((alt_entry + ideal) / 2.0, 2)
            else:
                cand_target = round(alt_entry * (1.0 - 0.0075), 2)
        elif target_price is not None and target_price < alt_entry:
            # Original target still valid; optionally use midway if very far
            gap = alt_entry - target_price
            if gap / alt_entry > 0.04:
                cand_target = round(alt_entry - gap * 0.5, 2)
            else:
                cand_target = round(target_price, 2)
        alt_stop = round(cand_stop, 2)
        alt_target = round(cand_target, 2) if cand_target is not None else None

    # Prefer ER from **current → alt_target** for alt_er_pct (chase context)
    alt_er = remaining_er_pct(
        direction=direction, current_price=current, target_price=alt_target
    )
    rr = _alt_rr(direction, alt_entry, alt_stop, alt_target)

    if not _geometry_ok(direction, alt_entry, alt_stop, alt_target):
        suggested = ACTION_DO_NOT_CHASE
        alt_action = ALT_ACTION_WATCH
        levels_note = f"Alt levels not geometrically valid — {reeval_note}"
    elif rr is None or rr < MIN_ALT_RR:
        suggested = ACTION_DO_NOT_CHASE
        alt_action = ALT_ACTION_WATCH
        levels_note = (
            f"Provisional alt near {anchor_name} but R:R weak"
            + (f" ({rr})" if rr is not None else "")
            + " — Do Not Chase Ideal Entry"
        )
    elif direction == "LONG" and suggestion == "wait_pullback" and anchor_name in (
        "vwap",
        "orb_low",
    ):
        suggested = ACTION_WAIT_VWAP
        alt_action = ALT_ACTION_SECONDARY
        levels_note = (
            f"Wait VWAP/pullback alt LONG near {anchor_name} — do not chase Ideal Entry"
        )
    else:
        suggested = ACTION_WATCH_ALT
        alt_action = ALT_ACTION_SECONDARY
        levels_note = (
            f"Provisional alt {direction} near {anchor_name} "
            f"(Ideal Entry missed) — Watch alt entry"
        )

    alt_reason = _ALT_REASON_LABELS.get(
        suggestion, suggestion.replace("_", " ").title()
    )

    return {
        "suggested_action": suggested,
        "suggestion": suggestion,
        "alt_entry_zone": alt_zone,
        "alt_entry": alt_entry,
        "alt_stop": alt_stop,
        "alt_target": alt_target,
        "alt_er_pct": alt_er,
        "alt_reason": alt_reason,
        "alt_action": alt_action,
        "remaining_er_pct": rem_er,
        "live_expected_return_pct": rem_er,
        "alt_rr": rr,
        "anchor": anchor_name,
        "anchor_price": round(anchor_px, 2),
        "note": levels_note,
        "advisory": True,
    }


def planned_entry_status(
    *,
    entry_zone: dict[str, Any] | None = None,
    entry_price: float | None = None,
    current_price: float | None = None,
    direction: str = "",
    target_price: float | None = None,
) -> dict[str, Any]:
    """Premarket / Morning status — Ideal Entry is a plan, not live execution."""
    _, ideal, _ = _zone_bounds(entry_zone, entry_price)
    dist = None
    if current_price is not None and ideal is not None and ideal > 0 and direction in ("LONG", "SHORT"):
        dist = _distance_pct(current_price, ideal, direction)
    rem = remaining_er_pct(
        direction=direction, current_price=current_price, target_price=target_price
    )
    return {
        "status": STATUS_PLANNED,
        "status_reason": "Premarket Ideal Entry — plan not yet live",
        "action_hint": ACTION_VALID,
        "current_price": round(current_price, 2) if current_price is not None else None,
        "ideal_entry": round(ideal, 2) if ideal is not None else None,
        "distance_pct": dist,
        "remaining_er_pct": rem,
        "live_expected_return_pct": rem,
        "new_plan": None,
        "alternate_entry": None,
        "advisory": True,
        "advisory_tag": ADVISORY_TAG,
    }


def classify_entry_status(
    *,
    direction: str,
    current_price: float | None,
    entry_zone: dict[str, Any] | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    miss_threshold_pct: float = DEFAULT_MISS_THRESHOLD_PCT,
    session_phase: str | None = None,
    vwap: float | None = None,
    orb_high: float | None = None,
    orb_low: float | None = None,
) -> dict[str, Any]:
    """Compare current price to Ideal Entry zone → execution status.

    Status machine (LONG; SHORT mirrors + extended-above miss):
    - PREMARKET/PLANNED: before open
    - EXPIRED: session closed / plan stale
    - INVALIDATED: current through stop before a valid fill
    - MISSED: gapped/moved past zone away from fill by > miss_threshold
      (SHORT also MISSED when extended **above** Ideal — failed-breakdown case)
    - READY: current inside Ideal Entry zone
    - TRIGGERED: current through entry mid in trade direction, not yet MISSED
    - ACTIVE: waiting for pullback into zone
    """
    direction = (direction or "").upper()
    low, ideal, high = _zone_bounds(entry_zone, entry_price)
    stop = _safe_float(stop_price)
    target = _safe_float(target_price)
    current = _safe_float(current_price)
    phase = (session_phase or "").lower().strip()
    vwap_f = _safe_float(vwap)
    orb_hi = _safe_float(orb_high)
    orb_lo = _safe_float(orb_low)

    rem = remaining_er_pct(
        direction=direction, current_price=current, target_price=target
    )

    base: dict[str, Any] = {
        "current_price": round(current, 2) if current is not None else None,
        "ideal_entry": round(ideal, 2) if ideal is not None else None,
        "distance_pct": None,
        "remaining_er_pct": rem,
        "live_expected_return_pct": rem,
        "new_plan": None,
        "alternate_entry": None,
        "advisory": True,
        "advisory_tag": ADVISORY_TAG,
        "zone_low": round(low, 2) if low is not None else None,
        "zone_high": round(high, 2) if high is not None else None,
        "miss_threshold_pct": miss_threshold_pct,
    }

    def _with_plan(status: str, reason: str, hint: str) -> dict[str, Any]:
        plan = None
        if current is not None:
            plan = _build_alternate_entry(
                status=status,
                direction=direction,
                current=current,
                ideal=ideal,
                target_price=target,
                stop_price=stop,
                vwap=vwap_f,
                orb_high=orb_hi,
                orb_low=orb_lo,
            )
        if plan and plan.get("suggested_action"):
            if status in (STATUS_MISSED, STATUS_INVALIDATED):
                hint = plan["suggested_action"]
        live_er = rem
        if plan and plan.get("live_expected_return_pct") is not None:
            live_er = plan["live_expected_return_pct"]
        elif plan and plan.get("remaining_er_pct") is not None:
            live_er = plan["remaining_er_pct"]
        return {
            **base,
            "status": status,
            "status_reason": reason,
            "action_hint": hint,
            "new_plan": plan,
            "alternate_entry": plan,
            "remaining_er_pct": live_er,
            "live_expected_return_pct": live_er,
        }

    if phase in ("premarket", "planned", "morning"):
        out = planned_entry_status(
            entry_zone=entry_zone,
            entry_price=entry_price,
            current_price=current,
            direction=direction,
            target_price=target,
        )
        out["status"] = STATUS_PREMARKET if phase == "premarket" else STATUS_PLANNED
        return {**base, **out}

    if phase in ("closed", "expired", "after_hours"):
        if current is not None and ideal is not None and ideal > 0 and direction in ("LONG", "SHORT"):
            base["distance_pct"] = _distance_pct(current, ideal, direction)
        return {
            **base,
            "status": STATUS_EXPIRED,
            "status_reason": "Session ended — Ideal Entry plan stale",
            "action_hint": ACTION_ABANDON,
        }

    if direction not in ("LONG", "SHORT") or ideal is None or ideal <= 0:
        return {
            **base,
            "status": STATUS_PLANNED,
            "status_reason": "No Ideal Entry levels to evaluate",
            "action_hint": ACTION_VALID,
        }

    if current is None:
        return {
            **base,
            "status": STATUS_PLANNED,
            "status_reason": "No current price — Ideal Entry unchanged",
            "action_hint": ACTION_VALID,
        }

    dist = _distance_pct(current, ideal, direction)
    base["distance_pct"] = dist
    miss_frac = miss_threshold_pct / 100.0
    zone_low = low if low is not None else ideal
    zone_high = high if high is not None else ideal

    if direction == "LONG":
        if stop is not None and current < stop:
            return _with_plan(
                STATUS_INVALIDATED,
                f"Stop hit — current ${current:.2f} below stop ${stop:.2f}",
                ACTION_ABANDON,
            )
        miss_line = zone_high * (1.0 + miss_frac)
        if current > miss_line:
            gap_pct = round((current - zone_high) / zone_high * 100.0, 2)
            return _with_plan(
                STATUS_MISSED,
                f"Gapped above entry by {gap_pct}%",
                ACTION_DO_NOT_CHASE,
            )
        if zone_low <= current <= zone_high:
            return {
                **base,
                "status": STATUS_READY,
                "status_reason": "Current inside Ideal Entry zone",
                "action_hint": ACTION_VALID,
            }
        if current > zone_high:
            return {
                **base,
                "status": STATUS_TRIGGERED,
                "status_reason": (
                    f"Price through Ideal Entry mid ${ideal:.2f} "
                    f"(still within {miss_threshold_pct:g}% miss band)"
                ),
                "action_hint": ACTION_VALID,
            }
        return {
            **base,
            "status": STATUS_ACTIVE,
            "status_reason": "Below Ideal Entry — waiting for pullback into zone or breakout",
            "action_hint": ACTION_VALID,
        }

    # SHORT
    if stop is not None and current > stop:
        return _with_plan(
            STATUS_INVALIDATED,
            f"Stop hit — current ${current:.2f} above stop ${stop:.2f}",
            ACTION_ABANDON,
        )
    # Traditional miss: gapped below Ideal (already moved through fill)
    miss_line_below = zone_low * (1.0 - miss_frac)
    if current < miss_line_below:
        gap_pct = round((zone_low - current) / zone_low * 100.0, 2)
        return _with_plan(
            STATUS_MISSED,
            f"Gapped below entry by {gap_pct}%",
            ACTION_DO_NOT_CHASE,
        )
    # Extended above Ideal — Ideal fill unlikely; failed-breakdown / retest watch
    miss_line_above = zone_high * (1.0 + miss_frac)
    if current > miss_line_above:
        gap_pct = round((current - zone_high) / zone_high * 100.0, 2)
        return _with_plan(
            STATUS_MISSED,
            f"Extended above Ideal Entry by {gap_pct}% — Ideal fill unlikely without chase",
            ACTION_NEW_SETUP,
        )
    if zone_low <= current <= zone_high:
        return {
            **base,
            "status": STATUS_READY,
            "status_reason": "Current inside Ideal Entry zone",
            "action_hint": ACTION_VALID,
        }
    if current < zone_low:
        return {
            **base,
            "status": STATUS_TRIGGERED,
            "status_reason": (
                f"Price through Ideal Entry mid ${ideal:.2f} "
                f"(still within {miss_threshold_pct:g}% miss band)"
            ),
            "action_hint": ACTION_VALID,
        }
    # current slightly above zone_high but within miss band
    return {
        **base,
        "status": STATUS_ACTIVE,
        "status_reason": "Above Ideal Entry — waiting for pullback into zone or breakdown",
        "action_hint": ACTION_VALID,
    }


def attach_entry_status(
    slot: dict[str, Any],
    *,
    current_price: float | None = None,
    session_phase: str | None = None,
    miss_threshold_pct: float = DEFAULT_MISS_THRESHOLD_PCT,
    vwap: float | None = None,
) -> dict[str, Any]:
    """Compute and attach ``entry_status`` onto a trade slot (mutates and returns)."""
    price = current_price
    if price is None:
        price = _safe_float(slot.get("current_price"))

    anchors = slot.get("level_anchors") or {}
    if not isinstance(anchors, dict):
        anchors = {
            "vwap": getattr(anchors, "vwap", None),
            "orb_high": getattr(anchors, "orb_high", None),
            "orb_low": getattr(anchors, "orb_low", None),
        }
    if vwap is None:
        vwap = _safe_float(anchors.get("vwap"))
    orb_high = _safe_float(anchors.get("orb_high"))
    orb_low = _safe_float(anchors.get("orb_low"))

    status = classify_entry_status(
        direction=str(slot.get("direction") or ""),
        current_price=price,
        entry_zone=slot.get("entry_zone"),
        entry_price=_safe_float(slot.get("entry_price")),
        stop_price=_safe_float(slot.get("stop_price")),
        target_price=_safe_float(slot.get("target_price")),
        miss_threshold_pct=miss_threshold_pct,
        session_phase=session_phase,
        vwap=vwap,
        orb_high=orb_high,
        orb_low=orb_low,
    )
    slot["entry_status"] = status
    return slot


def is_chase_status(status: str | None) -> bool:
    return (status or "") in _CHASE_HINTS


def resolve_symbol_last(
    symbol: str,
    trading_date: str,
    *,
    raw: dict[str, Any] | None = None,
) -> float | None:
    """Best-effort last price from raw / session_observation (no live websocket)."""
    from datetime import date as date_type

    from src.utils.paths import raw_data_path
    from src.utils.quote_resolve import session_observation
    from src.utils.trading_calendar import prior_trading_day

    sym = (symbol or "").upper()
    if not sym:
        return None

    day = date_type.fromisoformat(trading_date)
    if raw is None:
        path = raw_data_path(trading_date)
        if not path.exists():
            return None
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    prior_raw: dict[str, Any] = {}
    prior_path = raw_data_path(prior_trading_day(day).isoformat())
    if prior_path.exists():
        try:
            import json

            prior_raw = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    section = "market"
    if sym in ("SMH", "XLK", "XLF", "XLE"):
        section = "sector"
    elif sym not in ("QQQ", "SPY", "DIA", "TQQQ", "^VIX", "DX-Y.NYB", "ES=F"):
        section = "stocks"

    try:
        obs = session_observation(
            sym, raw or {}, prior_raw, day, section=section, prefer_raw=True
        )
    except Exception:
        return None
    if "error" in obs:
        return None
    return _safe_float(obs.get("last") or obs.get("close") or obs.get("open"))


def infer_session_phase(trading_date: str) -> str:
    """Rough session phase for homepage: premarket / open / closed."""
    from datetime import datetime

    try:
        from pytz import timezone

        et = timezone("America/New_York")
        now = datetime.now(et)
        today = now.date().isoformat()
        if trading_date < today:
            return "closed"
        if trading_date > today:
            return "premarket"
        t = now.time()
        from datetime import time as time_type

        if t < time_type(9, 30):
            return "premarket"
        if t >= time_type(16, 0):
            return "closed"
        return "open"
    except Exception:
        return "open"


def enrich_primary_from_morning(
    morning: dict[str, Any],
    trading_date: str,
    *,
    session_phase: str = "open",
    current_price: float | None = None,
) -> dict[str, Any] | None:
    """Attach live entry_status to morning primary trade (for Step 2/3 extras)."""
    primary = (morning.get("best_trades") or {}).get("primary")
    if not primary or primary.get("direction") not in ("LONG", "SHORT"):
        return None
    slot = dict(primary)
    px = current_price
    if px is None:
        px = resolve_symbol_last(str(slot.get("symbol") or ""), trading_date)
    attach_entry_status(slot, current_price=px, session_phase=session_phase)
    es = slot.get("entry_status") or {}
    return {
        "symbol": slot.get("symbol"),
        "direction": slot.get("direction"),
        "entry_zone": slot.get("entry_zone"),
        "entry_price": slot.get("entry_price"),
        "target_price": slot.get("target_price"),
        "stop_price": slot.get("stop_price"),
        "entry_status": es,
        "current_price": es.get("current_price"),
        "remaining_er_pct": es.get("remaining_er_pct"),
        "live_expected_return_pct": es.get("live_expected_return_pct")
        or es.get("remaining_er_pct"),
        "alternate_entry": es.get("alternate_entry") or es.get("new_plan"),
        "advisory": True,
    }


def build_session_trade_update(
    morning: dict[str, Any],
    trading_date: str,
    *,
    raw: dict[str, Any] | None = None,
    session_phase: str = "open",
) -> dict[str, Any] | None:
    """10:00 mid-session trade re-rank for Step 3 ``session_trade_update``.

    Rules-engine only (no LLM). Returns None if morning has no usable trade context.
    Shape: ``{top_trades, primary, changed, why_changed, compared_to_morning_primary}``.
    """
    if not morning:
        return None

    from src.utils.paths import raw_data_path

    if raw is None:
        path = raw_data_path(trading_date)
        if path.exists():
            try:
                import json

                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
        else:
            raw = {}

    morning_primary = (morning.get("best_trades") or {}).get("primary") or {}
    morning_sym = morning_primary.get("symbol")
    morning_dir = morning_primary.get("direction")
    morning_entry_status = None
    if morning_primary.get("direction") in ("LONG", "SHORT"):
        morning_entry_status = (
            enrich_primary_from_morning(
                morning, trading_date, session_phase=session_phase
            )
            or {}
        ).get("entry_status")

    compared = {
        "symbol": morning_sym,
        "direction": morning_dir,
        "entry_price": morning_primary.get("entry_price"),
        "expected_return_pct": morning_primary.get("expected_return_pct"),
        "entry_status": (morning_entry_status or {}).get("status")
        if isinstance(morning_entry_status, dict)
        else None,
        "live_expected_return_pct": (
            (morning_entry_status or {}).get("live_expected_return_pct")
            if isinstance(morning_entry_status, dict)
            else None
        ),
    }

    # Refresh morning top_trades entry_status with live quotes (fallback board)
    top_src = list(
        (morning.get("transparency") or {}).get("top_trades")
        or morning.get("top_trades")
        or []
    )
    refreshed_top: list[dict[str, Any]] = []
    for t in top_src[:8]:
        row = dict(t)
        sym = str(row.get("symbol") or "")
        px = resolve_symbol_last(sym, trading_date, raw=raw) if sym else None
        attach_entry_status(row, current_price=px, session_phase=session_phase)
        es = row.get("entry_status") or {}
        refreshed_top.append(
            {
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "rank": row.get("rank"),
                "final_score": row.get("final_score"),
                "expected_return_pct": row.get("expected_return_pct"),
                "live_expected_return_pct": es.get("live_expected_return_pct")
                or es.get("remaining_er_pct"),
                "entry_price": row.get("entry_price"),
                "target_price": row.get("target_price"),
                "stop_price": row.get("stop_price"),
                "entry_zone": row.get("entry_zone"),
                "entry_status": es,
                "remaining_er_pct": es.get("remaining_er_pct"),
                "trade_action": row.get("trade_action"),
            }
        )

    re_ranked: list[dict[str, Any]] = []
    new_primary: dict[str, Any] | None = None
    p16_gate = None
    best_opportunity = None
    re_rank_error: str | None = None
    todays_opps: list[dict[str, Any]] = []

    try:
        from src.research.trade_candidates import compute_trade_decision

        parts = morning.get("parts") or {}
        rule_bundle = {
            "bias": morning.get("bias") or "Neutral",
            "total": int(morning.get("total_score") or 0),
            "driver_type": morning.get("driver_type") or "",
            "daily_driver": morning.get("daily_driver") or "",
            "macro_calendar": morning.get("macro_calendar") or {},
            "driver_tree": morning.get("driver_tree") or {},
            "catalysts_today": morning.get("catalysts_today")
            or (morning.get("context") or {}).get("catalysts_today")
            or [],
            "edges": morning.get("edges"),
            "trade_plan": morning.get("trade_plan"),
        }
        if raw:
            raw = dict(raw)
            raw.setdefault("trading_date", trading_date)
            decision = compute_trade_decision(
                raw,
                rule_bundle=rule_bundle,
                parts=parts,
                edges=rule_bundle.get("edges"),
                as_of_et="now",
            )
            best_trades = decision.get("best_trades") or {}
            new_primary = best_trades.get("primary")
            if new_primary:
                px = resolve_symbol_last(
                    str(new_primary.get("symbol") or ""), trading_date, raw=raw
                )
                attach_entry_status(
                    new_primary, current_price=px, session_phase=session_phase
                )
            for row in (decision.get("top_trades") or decision.get("trade_candidates") or [])[
                :5
            ]:
                r = dict(row)
                px = resolve_symbol_last(
                    str(r.get("symbol") or ""), trading_date, raw=raw
                )
                attach_entry_status(r, current_price=px, session_phase=session_phase)
                es = r.get("entry_status") or {}
                re_ranked.append(
                    {
                        "symbol": r.get("symbol"),
                        "direction": r.get("direction"),
                        "rank": r.get("rank"),
                        "final_score": r.get("final_score"),
                        "win_prob": r.get("win_prob"),
                        "expected_return_pct": r.get("expected_return_pct"),
                        "live_expected_return_pct": es.get("live_expected_return_pct")
                        or es.get("remaining_er_pct"),
                        "entry_price": r.get("entry_price"),
                        "target_price": r.get("target_price"),
                        "stop_price": r.get("stop_price"),
                        "entry_zone": r.get("entry_zone"),
                        "entry_status": es,
                        "remaining_er_pct": es.get("remaining_er_pct"),
                        "trade_action": r.get("trade_action"),
                        "why_factors": (r.get("why_factors") or [])[:3],
                    }
                )
            best_opportunity = decision.get("best_opportunity")
            p16_gate = (best_opportunity or {}).get("p16_gate") or best_trades.get(
                "p16_gate"
            )
            todays_opps = (decision.get("transparency") or {}).get(
                "todays_opportunities"
            ) or []
        else:
            re_rank_error = "no raw data for re-score"
    except Exception as exc:
        re_rank_error = f"{type(exc).__name__}: {exc}"
        new_primary = None

    new_sym = (new_primary or {}).get("symbol")
    changed = bool(new_sym and morning_sym and new_sym != morning_sym)
    morning_missed = (
        isinstance(morning_entry_status, dict)
        and morning_entry_status.get("status") == STATUS_MISSED
    )
    if changed:
        why = (
            f"#1 changed vs Morning: {morning_sym} {morning_dir or ''} → "
            f"{new_sym} {(new_primary or {}).get('direction') or ''}".strip()
        )
        if morning_missed:
            why = (
                f"Morning Ideal Entry MISSED on {morning_sym}; "
                f"better symbol now ranks #1: {new_sym}"
            )
    elif new_sym and morning_sym and new_sym == morning_sym:
        why = f"#1 unchanged vs Morning: {morning_sym}"
        if morning_missed:
            why += " (Ideal Entry still MISSED — see alt stub)"
    elif not morning_sym and new_sym:
        why = f"#1 set at session re-eval: {new_sym}"
        changed = True
    else:
        why = "No actionable primary at re-eval (or Morning had none)"

    primary_out: dict[str, Any] | None
    if new_primary:
        es = new_primary.get("entry_status") or {}
        primary_out = {
            "symbol": new_primary.get("symbol"),
            "direction": new_primary.get("direction"),
            "entry_price": new_primary.get("entry_price"),
            "entry_zone": new_primary.get("entry_zone"),
            "target_price": new_primary.get("target_price"),
            "stop_price": new_primary.get("stop_price"),
            "expected_return_pct": new_primary.get("expected_return_pct"),
            "live_expected_return_pct": es.get("live_expected_return_pct")
            or es.get("remaining_er_pct"),
            "final_score": new_primary.get("final_score"),
            "entry_status": es,
            "remaining_er_pct": es.get("remaining_er_pct"),
            "alternate_entry": es.get("alternate_entry") or es.get("new_plan"),
            "trade_action": new_primary.get("trade_action"),
        }
    else:
        primary_out = enrich_primary_from_morning(
            morning, trading_date, session_phase=session_phase
        )

    return {
        "as_of": "10:00",
        "session_phase": session_phase,
        "top_trades": re_ranked or refreshed_top,
        "best_opportunity_asof": best_opportunity,
        "primary": primary_out,
        "changed": changed,
        "why_changed": why,
        "compared_to_morning_primary": compared,
        # extras (UI / body_md)
        "morning_primary_live": enrich_primary_from_morning(
            morning, trading_date, session_phase=session_phase
        ),
        "morning_top_refreshed": refreshed_top,
        "todays_opportunities": todays_opps[:6],
        "p16_gate": p16_gate,
        "best_opportunity": best_opportunity,
        "re_rank_error": re_rank_error,
        "note": why,
        "primary_changed": changed,
        "advisory": True,
        "advisory_tag": ADVISORY_TAG,
    }


# Backward-compatible alias
build_trade_reeval = build_session_trade_update
