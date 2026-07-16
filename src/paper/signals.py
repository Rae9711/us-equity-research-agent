"""Resolve Morning / session #1 trade signal for paper trading.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Default: Intraday #1 if actionable, else Swing if present, else skip.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.research.entry_status import (
    STATUS_ACTIVE,
    STATUS_READY,
    STATUS_TRIGGERED,
    attach_entry_status,
    infer_session_phase,
    resolve_symbol_last,
)
from src.utils.paths import morning_json_path, step_json_path

logger = logging.getLogger(__name__)

# Enter immediately
ENTER_STATUSES = frozenset({STATUS_READY, STATUS_TRIGGERED})
# Wait for pullback into zone (no chase)
WAIT_STATUSES = frozenset({STATUS_ACTIVE})


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load %s", path)
        return {}


def _slot_actionable(status: str | None) -> str:
    """Return 'enter' | 'wait' | 'skip'."""
    s = (status or "").upper()
    if s in ENTER_STATUSES:
        return "enter"
    if s in WAIT_STATUSES:
        return "wait"
    return "skip"


def normalize_slot(slot: dict[str, Any], *, source: str, horizon: str) -> dict[str, Any] | None:
    if not slot:
        return None
    direction = (slot.get("direction") or "").upper()
    symbol = (slot.get("symbol") or "").upper()
    if direction not in ("LONG", "SHORT") or not symbol:
        return None
    entry = _safe_float(slot.get("entry_price") or slot.get("entry"))
    stop = _safe_float(slot.get("stop_price") or slot.get("stop"))
    target = _safe_float(slot.get("target_price") or slot.get("target"))
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry,
        "entry_zone": slot.get("entry_zone"),
        "stop_price": stop,
        "target_price": target,
        "expected_return_pct": slot.get("expected_return_pct"),
        "win_prob": slot.get("win_prob"),
        "horizon": horizon or slot.get("horizon") or "Intraday",
        "source": source,
        "trade_action": slot.get("trade_action"),
        "level_anchors": slot.get("level_anchors"),
        "raw_slot": slot,
    }


def resolve_quote(
    symbol: str,
    trading_date: str,
    *,
    raw: dict[str, Any] | None = None,
    prefer_live: bool = True,
) -> tuple[float | None, str]:
    """Best-effort last price. Prefer live yfinance when market may be open; else raw."""
    from src.research.entry_status import infer_session_phase

    phase = infer_session_phase(trading_date)
    px = None
    source = "none"

    if prefer_live and phase == "open":
        try:
            from src.collectors.yfinance_client import fetch_quote

            q = fetch_quote(symbol)
            px = _safe_float(
                (q or {}).get("last")
                or (q or {}).get("price")
                or (q or {}).get("close")
            )
            if px:
                source = "yfinance_live"
        except Exception:
            logger.debug("yfinance live quote failed for %s", symbol, exc_info=True)

    if px is None:
        px = resolve_symbol_last(symbol, trading_date, raw=raw)
        if px is not None:
            source = "raw_snapshot"

    if px is None and prefer_live:
        # Last resort even when closed (demo / after hours)
        try:
            from src.collectors.yfinance_client import fetch_quote

            q = fetch_quote(symbol)
            px = _safe_float(
                (q or {}).get("last")
                or (q or {}).get("price")
                or (q or {}).get("close")
            )
            if px:
                source = "yfinance_last"
        except Exception:
            pass

    return px, source


def load_candidate_signals(trading_date: str) -> dict[str, Any]:
    """Load morning + step3 session primary + swing."""
    morning = _load_json(morning_json_path(trading_date))
    step3 = _load_json(step_json_path(3, trading_date))
    session = (
        step3.get("session_trade_update")
        or step3.get("trade_reeval")
        or step3.get("session_update")
        or {}
    )
    primary = (morning.get("best_trades") or {}).get("primary") or morning.get(
        "best_opportunity"
    )
    # When primary is NO TRADE / null, fall back to top_trades[0]
    if not primary or (primary.get("direction") or "").upper() not in ("LONG", "SHORT"):
        top = (morning.get("best_trades") or {}).get("top_trades") or morning.get(
            "top_trades"
        ) or []
        if top and isinstance(top, list):
            candidate = top[0]
            if (candidate.get("direction") or "").upper() in ("LONG", "SHORT"):
                primary = candidate
    swing = morning.get("swing_trade") or (morning.get("best_trades") or {}).get("swing")
    session_primary = session.get("primary") if isinstance(session, dict) else None
    if session_primary and (session_primary.get("direction") or "").upper() not in (
        "LONG",
        "SHORT",
    ):
        session_primary = None
    return {
        "morning": morning,
        "step3": step3,
        "session_primary": session_primary,
        "morning_primary": primary,
        "swing": swing,
    }


def pick_signal(
    trading_date: str,
    *,
    current_price_by_symbol: dict[str, float] | None = None,
    session_phase: str | None = None,
    prefer_swing_if_no_intraday: bool = True,
    miss_threshold_pct: float = 1.0,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick #1 trade for paper tick.

    Returns dict with keys: action ('enter'|'wait'|'skip'|'hold_signal'),
    signal (normalized), entry_status, reason, quote, quote_source.
    """
    phase = session_phase or infer_session_phase(trading_date)
    ctx = load_candidate_signals(trading_date)
    prices = current_price_by_symbol or {}

    candidates: list[tuple[str, dict[str, Any] | None, str]] = [
        ("session_primary", normalize_slot(ctx.get("session_primary") or {}, source="session_primary", horizon="Intraday"), "Intraday #1 (Step3)"),
        ("morning_primary", normalize_slot(ctx.get("morning_primary") or {}, source="morning_primary", horizon="Intraday"), "Morning Primary"),
    ]
    if prefer_swing_if_no_intraday:
        candidates.append(
            ("swing", normalize_slot(ctx.get("swing") or {}, source="swing", horizon="Swing"), "Swing")
        )

    evaluated: list[dict[str, Any]] = []
    for key, norm, label in candidates:
        if not norm:
            continue
        sym = norm["symbol"]
        px = prices.get(sym)
        quote_source = "injected"
        if px is None:
            px, quote_source = resolve_quote(sym, trading_date, raw=raw, prefer_live=True)
        slot = dict(norm.get("raw_slot") or {})
        slot.update(
            {
                "symbol": sym,
                "direction": norm["direction"],
                "entry_price": norm["entry_price"],
                "entry_zone": norm["entry_zone"],
                "stop_price": norm["stop_price"],
                "target_price": norm["target_price"],
                "level_anchors": norm.get("level_anchors"),
            }
        )
        attach_entry_status(
            slot,
            current_price=px,
            session_phase=phase,
            miss_threshold_pct=miss_threshold_pct,
        )
        es = slot.get("entry_status") or {}
        act = _slot_actionable(es.get("status"))
        evaluated.append(
            {
                "key": key,
                "label": label,
                "signal": {**norm, "entry_status": es, "current_price": px},
                "action": act,
                "entry_status": es,
                "quote": px,
                "quote_source": quote_source,
            }
        )

    # Prefer first enterable intraday, else wait on first wait, else swing enter, else skip
    for row in evaluated:
        if row["key"] != "swing" and row["action"] == "enter":
            return {
                "action": "enter",
                "signal": row["signal"],
                "entry_status": row["entry_status"],
                "reason": f"模拟买入：{row['label']} Entry Status={row['entry_status'].get('status')}（价位贴近 Ideal Entry）",
                "quote": row["quote"],
                "quote_source": row["quote_source"],
                "candidates": evaluated,
            }
    for row in evaluated:
        if row["key"] != "swing" and row["action"] == "wait":
            return {
                "action": "wait",
                "signal": row["signal"],
                "entry_status": row["entry_status"],
                "reason": f"等待回踩：{row['label']} Entry Status=ACTIVE，未追高",
                "quote": row["quote"],
                "quote_source": row["quote_source"],
                "candidates": evaluated,
            }
    for row in evaluated:
        if row["key"] == "swing" and row["action"] == "enter":
            return {
                "action": "enter",
                "signal": row["signal"],
                "entry_status": row["entry_status"],
                "reason": f"Intraday 不可执行，改用 Swing：Entry Status={row['entry_status'].get('status')}",
                "quote": row["quote"],
                "quote_source": row["quote_source"],
                "candidates": evaluated,
            }
    for row in evaluated:
        if row["key"] == "swing" and row["action"] == "wait":
            return {
                "action": "wait",
                "signal": row["signal"],
                "entry_status": row["entry_status"],
                "reason": "等待 Swing Ideal Entry 回踩",
                "quote": row["quote"],
                "quote_source": row["quote_source"],
                "candidates": evaluated,
            }

    why = "无可用 Primary / Swing 信号"
    if evaluated:
        statuses = ", ".join(
            f"{r['label']}={((r.get('entry_status') or {}).get('status') or 'N/A')}"
            for r in evaluated
        )
        why = f"信号均不可执行（{statuses}）— MISSED/INVALIDATED/EXPIRED 不追"
    return {
        "action": "skip",
        "signal": None,
        "entry_status": None,
        "reason": why,
        "quote": None,
        "quote_source": None,
        "candidates": evaluated,
    }
