"""Entry Status / live execution layer vs Morning Ideal Entry.

ADVISORY ONLY — 不构成投资建议.

Ideal Entry (entry_zone / entry_price / entry_source) remains the premarket plan.
This module classifies whether that plan is still fillable given current price.
"""

from __future__ import annotations

from typing import Any

ADVISORY_TAG = "ADVISORY — 不构成投资建议"

# Gap past Ideal Entry zone (beyond high for LONG / below low for SHORT) → MISSED
DEFAULT_MISS_THRESHOLD_PCT = 1.0

STATUS_PREMARKET = "PREMARKET"
STATUS_PLANNED = "PLANNED"
STATUS_ACTIVE = "ACTIVE"
STATUS_READY = "READY"
STATUS_TRIGGERED = "TRIGGERED"
STATUS_MISSED = "MISSED"
STATUS_INVALIDATED = "INVALIDATED"
STATUS_EXPIRED = "EXPIRED"

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


def _action_hint(status: str, *, direction: str) -> str:
    if status in (STATUS_PREMARKET, STATUS_PLANNED):
        return "Entry still valid"
    if status == STATUS_READY:
        return "Entry still valid"
    if status == STATUS_ACTIVE:
        return "Entry still valid"
    if status == STATUS_TRIGGERED:
        return "Entry still valid"
    if status == STATUS_MISSED:
        return "Do Not Chase"
    if status == STATUS_INVALIDATED:
        return "Abandon today"
    if status == STATUS_EXPIRED:
        return "Abandon today"
    return "Entry still valid"


def _new_plan_stub(
    status: str,
    *,
    vwap: float | None,
) -> dict[str, Any] | None:
    if status != STATUS_MISSED:
        return None
    if vwap is not None:
        return {
            "suggestion": "wait_pullback",
            "anchor": "vwap",
            "anchor_price": round(vwap, 2),
            "note": "Wait VWAP Pullback — do not chase Ideal Entry",
            "advisory": True,
        }
    return {
        "suggestion": "abandon",
        "anchor": None,
        "anchor_price": None,
        "note": "Abandon today — Ideal Entry missed, no pullback anchor",
        "advisory": True,
    }


def planned_entry_status(
    *,
    entry_zone: dict[str, Any] | None = None,
    entry_price: float | None = None,
    current_price: float | None = None,
    direction: str = "",
) -> dict[str, Any]:
    """Premarket / Morning status — Ideal Entry is a plan, not live execution."""
    _, ideal, _ = _zone_bounds(entry_zone, entry_price)
    dist = None
    if current_price is not None and ideal is not None and ideal > 0 and direction in ("LONG", "SHORT"):
        dist = _distance_pct(current_price, ideal, direction)
    return {
        "status": STATUS_PLANNED,
        "status_reason": "Premarket Ideal Entry — plan not yet live",
        "action_hint": "Entry still valid",
        "current_price": round(current_price, 2) if current_price is not None else None,
        "ideal_entry": round(ideal, 2) if ideal is not None else None,
        "distance_pct": dist,
        "new_plan": None,
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
    miss_threshold_pct: float = DEFAULT_MISS_THRESHOLD_PCT,
    session_phase: str | None = None,
    vwap: float | None = None,
) -> dict[str, Any]:
    """Compare current price to Ideal Entry zone → execution status.

    Status machine (LONG; SHORT mirrors):
    - PREMARKET/PLANNED: before open
    - EXPIRED: session closed / plan stale
    - INVALIDATED: current through stop before a valid fill
    - MISSED: gapped/moved past zone away from fill by > miss_threshold
    - READY: current inside Ideal Entry zone
    - TRIGGERED: current through entry mid in trade direction, not yet MISSED
    - ACTIVE: current at/below zone (LONG) waiting for pullback or breakout
    """
    direction = (direction or "").upper()
    low, ideal, high = _zone_bounds(entry_zone, entry_price)
    stop = _safe_float(stop_price)
    current = _safe_float(current_price)
    phase = (session_phase or "").lower().strip()

    base: dict[str, Any] = {
        "current_price": round(current, 2) if current is not None else None,
        "ideal_entry": round(ideal, 2) if ideal is not None else None,
        "distance_pct": None,
        "new_plan": None,
        "advisory": True,
        "advisory_tag": ADVISORY_TAG,
        "zone_low": round(low, 2) if low is not None else None,
        "zone_high": round(high, 2) if high is not None else None,
        "miss_threshold_pct": miss_threshold_pct,
    }

    if phase in ("premarket", "planned", "morning"):
        out = planned_entry_status(
            entry_zone=entry_zone,
            entry_price=entry_price,
            current_price=current,
            direction=direction,
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
            "action_hint": "Abandon today",
        }

    if direction not in ("LONG", "SHORT") or ideal is None or ideal <= 0:
        return {
            **base,
            "status": STATUS_PLANNED,
            "status_reason": "No Ideal Entry levels to evaluate",
            "action_hint": "Entry still valid",
        }

    if current is None:
        return {
            **base,
            "status": STATUS_PLANNED,
            "status_reason": "No current price — Ideal Entry unchanged",
            "action_hint": "Entry still valid",
        }

    dist = _distance_pct(current, ideal, direction)
    base["distance_pct"] = dist
    miss_frac = miss_threshold_pct / 100.0
    zone_low = low if low is not None else ideal
    zone_high = high if high is not None else ideal

    if direction == "LONG":
        if stop is not None and current < stop:
            return {
                **base,
                "status": STATUS_INVALIDATED,
                "status_reason": f"Stop hit — current ${current:.2f} below stop ${stop:.2f}",
                "action_hint": "Abandon today",
            }
        miss_line = zone_high * (1.0 + miss_frac)
        if current > miss_line:
            gap_pct = round((current - zone_high) / zone_high * 100.0, 2)
            status = STATUS_MISSED
            reason = f"Gapped above entry by {gap_pct}%"
            hint = "Do Not Chase"
            new_plan = _new_plan_stub(status, vwap=vwap)
            if new_plan and new_plan.get("suggestion") == "wait_pullback":
                hint = "Wait VWAP Pullback"
            return {
                **base,
                "status": status,
                "status_reason": reason,
                "action_hint": hint,
                "new_plan": new_plan,
            }
        if zone_low <= current <= zone_high:
            return {
                **base,
                "status": STATUS_READY,
                "status_reason": "Current inside Ideal Entry zone",
                "action_hint": "Entry still valid",
            }
        if current > zone_high:
            return {
                **base,
                "status": STATUS_TRIGGERED,
                "status_reason": (
                    f"Price through Ideal Entry mid ${ideal:.2f} "
                    f"(still within {miss_threshold_pct:g}% miss band)"
                ),
                "action_hint": "Entry still valid",
            }
        # current < zone_low
        return {
            **base,
            "status": STATUS_ACTIVE,
            "status_reason": "Below Ideal Entry — waiting for pullback into zone or breakout",
            "action_hint": "Entry still valid",
        }

    # SHORT
    if stop is not None and current > stop:
        return {
            **base,
            "status": STATUS_INVALIDATED,
            "status_reason": f"Stop hit — current ${current:.2f} above stop ${stop:.2f}",
            "action_hint": "Abandon today",
        }
    miss_line = zone_low * (1.0 - miss_frac)
    if current < miss_line:
        gap_pct = round((zone_low - current) / zone_low * 100.0, 2)
        status = STATUS_MISSED
        reason = f"Gapped below entry by {gap_pct}%"
        hint = "Do Not Chase"
        new_plan = _new_plan_stub(status, vwap=vwap)
        if new_plan and new_plan.get("suggestion") == "wait_pullback":
            hint = "Wait VWAP Pullback"
        return {
            **base,
            "status": status,
            "status_reason": reason,
            "action_hint": hint,
            "new_plan": new_plan,
        }
    if zone_low <= current <= zone_high:
        return {
            **base,
            "status": STATUS_READY,
            "status_reason": "Current inside Ideal Entry zone",
            "action_hint": "Entry still valid",
        }
    if current < zone_low:
        return {
            **base,
            "status": STATUS_TRIGGERED,
            "status_reason": (
                f"Price through Ideal Entry mid ${ideal:.2f} "
                f"(still within {miss_threshold_pct:g}% miss band)"
            ),
            "action_hint": "Entry still valid",
        }
    # current > zone_high
    return {
        **base,
        "status": STATUS_ACTIVE,
        "status_reason": "Above Ideal Entry — waiting for pullback into zone or breakdown",
        "action_hint": "Entry still valid",
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
    if vwap is None:
        vwap = _safe_float(anchors.get("vwap"))

    status = classify_entry_status(
        direction=str(slot.get("direction") or ""),
        current_price=price,
        entry_zone=slot.get("entry_zone"),
        entry_price=_safe_float(slot.get("entry_price")),
        stop_price=_safe_float(slot.get("stop_price")),
        miss_threshold_pct=miss_threshold_pct,
        session_phase=session_phase,
        vwap=vwap,
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
    return {
        "symbol": slot.get("symbol"),
        "direction": slot.get("direction"),
        "entry_zone": slot.get("entry_zone"),
        "entry_price": slot.get("entry_price"),
        "entry_status": slot.get("entry_status"),
        "current_price": (slot.get("entry_status") or {}).get("current_price"),
        "advisory": True,
    }
