"""Day/week circuit breaker for paper trading (item 4, approved 2026-11-12).

Rules (owner-approved):
- Day: equity down >= 1.0% vs the prior close blocks ALL new entries
  until the next trading day.
- Week: equity down >= 2.0% vs the last close before this week (Monday)
  blocks ALL new entries until next week.
- A trip latches for the rest of the day/week even if equity recovers.
- Existing positions are never blocked: stops/targets keep managing.
- State lives in its own file (DATA_ROOT/paper/circuit_breaker.json),
  never in account.json.
- Every trip/reset is returned as an event; the caller appends it to the
  account decision record.

ADVISORY ONLY -- 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.utils.paths import data_root
from src.utils.trading_calendar import ET

logger = logging.getLogger(__name__)

DAY_THRESHOLD_PCT = 1.0
WEEK_THRESHOLD_PCT = 2.0
STATE_FILENAME = "circuit_breaker.json"
MAX_EVENTS_KEPT = 50


def circuit_breaker_path(state_path: str | Path | None = None) -> Path:
    if state_path is not None:
        return Path(state_path)
    d = data_root() / "paper"
    d.mkdir(parents=True, exist_ok=True)
    return d / STATE_FILENAME


def _default_state() -> dict[str, Any]:
    return {
        "day": {
            "date": None,
            "ref_equity": None,
            "tripped": False,
            "trip_pct": None,
            "tripped_at": None,
        },
        "week": {
            "week": None,
            "ref_equity": None,
            "tripped": False,
            "trip_pct": None,
            "tripped_at": None,
        },
        "events": [],
    }


def _load_state(state_path: str | Path | None) -> dict[str, Any]:
    path = circuit_breaker_path(state_path)
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("breaker state is not a JSON object")
        base = _default_state()
        for key in ("day", "week"):
            if isinstance(data.get(key), dict):
                base[key].update(data[key])
        if isinstance(data.get("events"), list):
            base["events"] = data["events"][-MAX_EVENTS_KEPT:]
        return base
    except Exception:
        # Fail CLOSED (2026-11-12): a corrupt safety file must never
        # silently reseed to "untripped". The state is marked corrupt;
        # check_circuit_breaker blocks new entries and refuses to
        # overwrite the file, so the block persists until a human repairs
        # or deletes it.
        logger.error(
            "Corrupt circuit breaker state at %s; failing CLOSED "
            "(new entries blocked until repaired)",
            path,
        )
        state = _default_state()
        state["corrupt"] = True
        return state


def _save_state(state: dict[str, Any], state_path: str | Path | None) -> Path:
    path = circuit_breaker_path(state_path)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _parse_point_date(ts: Any) -> date | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).date()
    except Exception:
        try:
            return date.fromisoformat(str(ts)[:10])
        except Exception:
            return None


def week_monday(d: date) -> date:
    """Monday of the ISO week containing ``d``."""
    from datetime import timedelta

    return d - timedelta(days=d.weekday())


def reference_equities(
    account: dict[str, Any], trading_date: date | str
) -> tuple[float, float]:
    """(day_ref, week_ref) from the equity curve.

    day_ref: equity at the last point dated before today (prior close).
    week_ref: equity at the last point dated before this Monday.
    Falls back to starting_cash when there is no earlier point.
    """
    if isinstance(trading_date, str):
        trading_date = date.fromisoformat(trading_date)
    monday = week_monday(trading_date)
    fallback = float(account.get("starting_cash") or 0.0) or float(
        account.get("equity") or 0.0
    )
    day_ref: float | None = None
    week_ref: float | None = None
    day_ts: datetime | None = None
    week_ts: datetime | None = None
    for point in account.get("equity_curve") or []:
        try:
            eq = float(point.get("equity"))
        except (TypeError, ValueError):
            continue
        pdate = _parse_point_date(point.get("ts"))
        if pdate is None:
            continue
        try:
            pts = datetime.fromisoformat(str(point.get("ts")))
        except Exception:
            continue
        if pdate < trading_date and (day_ts is None or pts > day_ts):
            day_ts, day_ref = pts, eq
        if pdate < monday and (week_ts is None or pts > week_ts):
            week_ts, week_ref = pts, eq
    return (day_ref if day_ref is not None else fallback), (
        week_ref if week_ref is not None else fallback
    )


def _pct(now: float, ref: float) -> float:
    if not ref:
        return 0.0
    return (now - ref) / ref * 100.0


def _tick_iso(tick_dt: datetime | str | None) -> str:
    """ISO timestamp for trip/reset events: the tick time, not run time.

    ``tick_dt`` is the time of the tick being evaluated (ET). When omitted
    (live tick without an explicit tick time) it falls back to now, which
    is the tick time for a live run. Naive datetimes are read as ET.
    """
    if tick_dt is None:
        return datetime.now(ET).isoformat()
    if isinstance(tick_dt, str):
        tick_dt = datetime.fromisoformat(tick_dt)
    if tick_dt.tzinfo is None:
        tick_dt = ET.localize(tick_dt)
    return tick_dt.isoformat()


def _reason(status: dict[str, Any]) -> str | None:
    parts: list[str] = []
    if status.get("tripped_day"):
        parts.append(
            f"当日权益较上收 {status['day_pct']:+.2f}%（≥{status['day_threshold_pct']:.0f}%），新开仓已封锁至明日"
        )
    if status.get("tripped_week"):
        parts.append(
            f"当周权益较上周末 {status['week_pct']:+.2f}%（≥{status['week_threshold_pct']:.0f}%），新开仓已封锁至下周"
        )
    return ("熔断：" + "；".join(parts)) if parts else None


def evaluate(
    account: dict[str, Any],
    trading_date: date | str,
    state: dict[str, Any],
    *,
    tick_dt: datetime | str | None = None,
    day_threshold_pct: float = DAY_THRESHOLD_PCT,
    week_threshold_pct: float = WEEK_THRESHOLD_PCT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the breaker for ``trading_date`` against ``state``.

    Mutates and returns ``state`` (caller persists it). Returns
    (status, new_events). The status never gates exits; callers must only
    use ``status["blocked"]`` to skip new entries.

    ``tick_dt`` is the time of the tick being evaluated; trip/reset events
    and ``tripped_at`` record it (never the wall-clock run time). When
    omitted it defaults to now, which is the tick time for a live run.
    """
    if isinstance(trading_date, str):
        trading_date = date.fromisoformat(trading_date)
    equity = float(account.get("equity") or 0.0)
    day_ref, week_ref = reference_equities(account, trading_date)
    day_pct = _pct(equity, day_ref)
    week_pct = _pct(equity, week_ref)
    tick_iso = _tick_iso(tick_dt)
    week_key = f"{trading_date.isocalendar()[0]}-W{trading_date.isocalendar()[1]:02d}"
    events: list[dict[str, Any]] = []

    day = state.setdefault("day", {})
    week = state.setdefault("week", {})

    # Day rollover: a new trading day resets the day latch. A reset event
    # is recorded only if the previous day was actually tripped.
    if day.get("date") != trading_date.isoformat():
        if day.get("tripped"):
            events.append(
                {
                    "type": "reset",
                    "scope": "day",
                    "trading_date": trading_date.isoformat(),
                    "ref_equity": day_ref,
                    "equity": equity,
                    "pct": day_pct,
                    "ts": tick_iso,
                }
            )
        day.update(
            {
                "date": trading_date.isoformat(),
                "ref_equity": day_ref,
                "tripped": False,
                "trip_pct": None,
                "tripped_at": None,
            }
        )

    # Week rollover: same rule, keyed by ISO week.
    if week.get("week") != week_key:
        if week.get("tripped"):
            events.append(
                {
                    "type": "reset",
                    "scope": "week",
                    "trading_date": trading_date.isoformat(),
                    "ref_equity": week_ref,
                    "equity": equity,
                    "pct": week_pct,
                    "ts": tick_iso,
                }
            )
        week.update(
            {
                "week": week_key,
                "ref_equity": week_ref,
                "tripped": False,
                "trip_pct": None,
                "tripped_at": None,
            }
        )

    # Trips latch for the rest of the day/week.
    if not day.get("tripped") and day_pct <= -abs(day_threshold_pct):
        day.update({"tripped": True, "trip_pct": day_pct, "tripped_at": tick_iso})
        events.append(
            {
                "type": "trip",
                "scope": "day",
                "trading_date": trading_date.isoformat(),
                "ref_equity": day_ref,
                "equity": equity,
                "pct": day_pct,
                "ts": tick_iso,
            }
        )
    if not week.get("tripped") and week_pct <= -abs(week_threshold_pct):
        week.update({"tripped": True, "trip_pct": week_pct, "tripped_at": tick_iso})
        events.append(
            {
                "type": "trip",
                "scope": "week",
                "trading_date": trading_date.isoformat(),
                "ref_equity": week_ref,
                "equity": equity,
                "pct": week_pct,
                "ts": tick_iso,
            }
        )

    status: dict[str, Any] = {
        "blocked": bool(day.get("tripped") or week.get("tripped")),
        "tripped_day": bool(day.get("tripped")),
        "tripped_week": bool(week.get("tripped")),
        "day_pct": round(day_pct, 2),
        "week_pct": round(week_pct, 2),
        "day_ref_equity": day_ref,
        "week_ref_equity": week_ref,
        "equity": equity,
        "day_threshold_pct": day_threshold_pct,
        "week_threshold_pct": week_threshold_pct,
        "reason": None,
    }
    status["reason"] = _reason(status)

    state.setdefault("events", []).extend(events)
    state["events"] = state["events"][-MAX_EVENTS_KEPT:]
    return status, events


def check_circuit_breaker(
    account: dict[str, Any],
    trading_date: date | str,
    *,
    tick_dt: datetime | str | None = None,
    state_path: str | Path | None = None,
    day_threshold_pct: float = DAY_THRESHOLD_PCT,
    week_threshold_pct: float = WEEK_THRESHOLD_PCT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load state, evaluate, persist, return (status, new_events).

    Reads ``account`` only; never mutates it and never touches
    account.json. Safe to call on every tick.

    ``tick_dt`` is the time of the tick being evaluated; trip/reset events
    and ``tripped_at`` record it (never the wall-clock run time).
    """
    state = _load_state(state_path)
    status, events = evaluate(
        account,
        trading_date,
        state,
        tick_dt=tick_dt,
        day_threshold_pct=day_threshold_pct,
        week_threshold_pct=week_threshold_pct,
    )
    if state.get("corrupt"):
        # Fail closed: block new entries and do NOT overwrite the corrupt
        # file with a fresh state (that would silently heal the safety).
        # The block persists until a human repairs or deletes the file.
        tick_iso = _tick_iso(tick_dt)
        day_s = trading_date.isoformat() if hasattr(trading_date, "isoformat") else str(trading_date)
        corrupt_event = {
            "type": "corrupt",
            "scope": "day",
            "trading_date": day_s,
            "ref_equity": None,
            "equity": float(account.get("equity") or 0.0),
            "pct": None,
            "ts": tick_iso,
        }
        events = [*events, corrupt_event]
        status = {
            **status,
            "blocked": True,
            "tripped_day": True,
            "corrupt": True,
            "reason": "熔断状态文件损坏，默认封锁新开仓（需人工修复或删除状态文件）",
        }
        return status, events
    _save_state(state, state_path)
    return status, events


def entries_allowed(status: dict[str, Any] | None) -> bool:
    return not bool((status or {}).get("blocked"))


def breaker_decision_row(event: dict[str, Any], trading_date: str) -> dict[str, Any]:
    """Convert a trip/reset event into an account decision-record row."""
    scope_zh = "当日" if event.get("scope") == "day" else "当周"
    if event.get("type") == "corrupt":
        return {
            "trading_date": trading_date,
            "action": "CIRCUIT_BREAKER_CORRUPT",
            "reason": (
                "熔断状态文件损坏，默认封锁新开仓，需人工修复或删除状态文件；"
                "已持仓位照常管理"
            ),
            "book": None,
            "book_zh": "熔断",
            "symbol": None,
            "quote": None,
            "pnl": None,
            "advisory": True,
        }
    if event.get("type") == "trip":
        action = "CIRCUIT_BREAKER_TRIP"
        reason = (
            f"熔断触发：{scope_zh}权益 {event.get('pct', 0.0):+.2f}%"
            f"（参考权益 ${event.get('ref_equity', 0.0):,.2f}，当前 ${event.get('equity', 0.0):,.2f}），"
            "新开仓已封锁，已持仓位照常管理"
        )
    else:
        action = "CIRCUIT_BREAKER_RESET"
        reason = f"熔断复位：{scope_zh}封锁解除，新开仓恢复（{event.get('pct', 0.0):+.2f}%）"
    return {
        "trading_date": trading_date,
        "action": action,
        "reason": reason,
        "book": None,
        "book_zh": "熔断",
        "symbol": None,
        "quote": None,
        "pnl": None,
        "advisory": True,
    }
