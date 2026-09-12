"""Report-driven signal provider for the paper-engine backtest.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Once the daily pipeline has persisted real research signals to
``data/reports/<date>/morning.json`` (Step 1) and ``step3.json`` (Step 3), this
module replays those *actual* signals through the backtest engine instead of the
momentum-breakout PROXY. This is what turns the backtest from "does the
execution engine work" into "is the research signal actually profitable".

It reuses ``load_candidate_signals`` / ``normalize_slot`` from ``paper.signals``
so the exact same report parsing the live trader uses is what gets backtested —
no divergence between backtest and production signal interpretation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, time
from typing import Any

from src.paper.signals import load_candidate_signals, normalize_slot
from src.utils.paths import morning_json_path, reports_dir


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if not math.isfinite(v):
        return None
    return v


def _extract_target2(slot: dict[str, Any] | None) -> float | None:
    """Pull the T2 price from a report slot's ``targets`` list if present."""
    if not slot:
        return None
    targets = slot.get("targets")
    if isinstance(targets, list):
        for t in targets:
            if isinstance(t, dict) and str(t.get("label") or "").upper() == "T2":
                px = _safe_float(t.get("price"))
                if px:
                    return px
        # Fall back to the last listed target if no explicit T2 label.
        prices = [_safe_float(t.get("price")) for t in targets if isinstance(t, dict)]
        prices = [p for p in prices if p]
        if len(prices) >= 2:
            return prices[-1]
    return _safe_float(slot.get("target2") or slot.get("target_price_2"))


def _to_engine_signal(
    norm: dict[str, Any] | None, *, horizon: str
) -> dict[str, Any] | None:
    """Map a normalized report slot into the backtest engine signal schema."""
    if not norm:
        return None
    entry = _safe_float(norm.get("entry_price"))
    stop = _safe_float(norm.get("stop_price"))
    target = _safe_float(norm.get("target_price"))
    direction = (norm.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None
    if entry is None or stop is None or target is None or entry <= 0:
        return None
    # Respect the research plan's own gate: never backtest a "Pass".
    if str(norm.get("trade_action") or "").strip().lower() == "pass":
        return None
    # Geometry sanity (mirrors the live level-invariant checks).
    if direction == "LONG" and not (stop < entry <= target):
        return None
    if direction == "SHORT" and not (target <= entry < stop):
        return None

    raw_slot = norm.get("raw_slot") if isinstance(norm.get("raw_slot"), dict) else {}
    win_prob = _safe_float(norm.get("win_prob")) or 55.0
    er = _safe_float(norm.get("expected_return_pct")) or 0.0
    rr = _safe_float(norm.get("risk_reward"))
    if rr is None and entry and stop and abs(entry - stop) > 0:
        rr = abs(target - entry) / abs(entry - stop)
    from src.paper.allocation import expected_r

    ev = expected_r(win_prob=win_prob, rr=rr)
    return {
        "symbol": norm["symbol"],
        "direction": direction,
        "entry_price": round(entry, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "target2": _extract_target2(raw_slot),
        "entry_zone": norm.get("entry_zone"),
        "win_prob": round(win_prob, 1),
        "expected_return_pct": round(er, 2),
        "expected_r": ev,
        "risk_reward": rr,
        "horizon": horizon or norm.get("horizon") or "Intraday",
        "source": f"report:{norm.get('source')}",
        "entry_status": {"status": "READY"},
        # Rank by calibrated EV, not bare target-distance ER.
        "_edge": round((ev or 0.0) * 100.0 + win_prob * 0.1, 2),
    }


def report_signals_for_date(trading_date: str) -> list[dict[str, Any]]:
    """Return ranked engine-schema signals from the persisted reports for a day.

    Prefers the Step 3 live re-rank (``session_primary``) over the morning plan,
    then morning primary, then ``top_trades``; the swing slot is emitted with a
    Swing horizon so the engine routes it to the swing book.
    """
    ctx = load_candidate_signals(trading_date)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(slot: dict[str, Any] | None, *, source: str, horizon: str) -> None:
        norm = normalize_slot(slot or {}, source=source, horizon=horizon)
        sig = _to_engine_signal(norm, horizon=horizon)
        if not sig:
            return
        sym = sig["symbol"]
        if sym in seen:
            return
        seen.add(sym)
        out.append(sig)

    _add(ctx.get("swing"), source="swing", horizon="Swing")
    _add(ctx.get("session_primary"), source="session_primary", horizon="Intraday")
    _add(ctx.get("morning_primary"), source="morning_primary", horizon="Intraday")
    for row in ctx.get("top_trades") or []:
        _add(row, source="top_trade", horizon="Intraday")

    out.sort(key=lambda s: s.get("_edge", 0.0), reverse=True)
    return out


def available_report_dates() -> list[str]:
    """Sorted ISO dates that have a persisted ``morning.json`` with a signal."""
    base = reports_dir()
    if not base.exists():
        return []
    dates: list[str] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if (child / "morning.json").exists():
            dates.append(child.name)
    return sorted(dates)


def report_symbols(dates: list[str]) -> list[str]:
    """Union of symbols named across the given report dates."""
    syms: list[str] = []
    seen: set[str] = set()
    for d in dates:
        for sig in report_signals_for_date(d):
            s = sig["symbol"]
            if s not in seen:
                seen.add(s)
                syms.append(s)
    return syms


def event_ls_signals_for_date(trading_date: str) -> list[dict[str, Any]]:
    """Read the persisted multi-leg event-LS target without collapsing it.

    This is intentionally separate from ``report_signals_for_date``: the legacy
    paper engine has two single-position books, while event-LS needs an
    N-position rebalance. A failed research/cost gate always returns no signal.
    """
    path = morning_json_path(trading_date)
    if not path.exists():
        return []
    try:
        morning = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    portfolio = morning.get("event_ls_portfolio") or {}
    if not isinstance(portfolio, dict):
        return []
    costs_gate = portfolio.get("costs_gate") or {}
    if (
        not isinstance(costs_gate, dict)
        or portfolio.get("deploy") is not True
        or costs_gate.get("pass") is not True
    ):
        return []

    as_of = portfolio.get("as_of") or portfolio.get("known_at")
    if not as_of:
        return []
    try:
        known_at = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
        decision = datetime.combine(
            datetime.fromisoformat(trading_date).date(), time(9, 30)
        )
        if known_at.tzinfo is not None:
            from zoneinfo import ZoneInfo

            known_at = known_at.astimezone(
                ZoneInfo("America/New_York")
            ).replace(tzinfo=None)
        if known_at > decision:
            return []
    except ValueError:
        return []
    hold = portfolio.get("hold_days") or [2, 10]
    if (
        not isinstance(hold, (list, tuple))
        or len(hold) != 2
        or not all(isinstance(value, int) for value in hold)
        or not (2 <= hold[0] <= hold[1] <= 10)
    ):
        return []
    rows: list[dict[str, Any]] = []
    legs = portfolio.get("legs") or []
    if not isinstance(legs, list) or not all(isinstance(leg, dict) for leg in legs):
        return []
    raw_legs = [{**leg, "_validated_hedge": False} for leg in legs]
    hedge = portfolio.get("hedge")
    if hedge is not None:
        if not isinstance(hedge, dict):
            return []
        raw_legs.append({**hedge, "_validated_hedge": True})
    seen: set[str] = set()
    for leg in raw_legs:
        symbol = str(leg.get("symbol") or "").upper()
        direction = str(leg.get("direction") or "").upper()
        if "weight_pct" in leg:
            raw_weight = _safe_float(leg.get("weight_pct"))
            weight = raw_weight / 100.0 if raw_weight is not None else None
        else:
            weight = _safe_float(leg.get("weight"))
        leg_hold = leg.get("hold_days") or hold
        cost = _safe_float(leg.get("estimated_cost_bps"))
        alpha = _safe_float(leg.get("predicted_alpha_bps"))
        beta_value = _safe_float(leg.get("beta"))
        borrow_fee = _safe_float(leg.get("borrow_fee_bps"))
        total_cost = (
            cost + (borrow_fee or 0.0)
            if cost is not None and direction == "SHORT"
            else cost
        )
        if (
            not symbol
            or symbol in seen
            or direction not in ("LONG", "SHORT")
            or weight is None
            or not 0 < abs(weight) <= 0.10
            or not isinstance(leg_hold, (list, tuple))
            or list(leg_hold) != list(hold)
            or cost is None
            or cost < 0
            or alpha is None
            or total_cost is None
            or alpha <= 2.0 * total_cost
            or beta_value is None
            or (
                direction == "SHORT"
                and (
                    leg.get("borrow_available") is not True
                    or borrow_fee is None
                    or borrow_fee < 0
                )
            )
        ):
            return []
        seen.add(symbol)
        signed = abs(weight)
        if direction == "SHORT":
            signed = -signed
        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "target_weight": round(signed, 8),
                "sector": leg.get("sector") or leg.get("industry"),
                "beta": beta_value,
                "predicted_alpha_bps": alpha,
                "estimated_cost_bps": total_cost,
                "hold_days": list(leg_hold),
                "known_at": as_of,
                "source": "report:event_ls",
                "is_hedge": leg.get("_validated_hedge") is True,
            }
        )
    if not rows:
        return []
    gross = sum(abs(row["target_weight"]) for row in rows)
    net = sum(row["target_weight"] for row in rows)
    sectors: dict[str, float] = {}
    beta = 0.0
    for row in rows:
        if not row["is_hedge"]:
            sector = str(row.get("sector") or "UNKNOWN")
            sectors[sector] = sectors.get(sector, 0.0) + abs(row["target_weight"])
        beta += row["target_weight"] * (
            row["beta"] if row["beta"] is not None else 1.0
        )
    if (
        gross > 1.0 + 1e-9
        or abs(net) > 0.10 + 1e-9
        or any(value > 0.15 + 1e-9 for value in sectors.values())
        or abs(beta) > 0.05 + 1e-9
    ):
        return []
    return rows


def available_event_ls_report_dates() -> list[str]:
    """Report dates whose persisted event-LS portfolio passes its cost gate."""
    return [d for d in available_report_dates() if event_ls_signals_for_date(d)]
