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

from typing import Any

from src.paper.signals import load_candidate_signals, normalize_slot
from src.utils.paths import reports_dir


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
