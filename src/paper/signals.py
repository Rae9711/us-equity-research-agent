"""Resolve Morning / session #1 trade signal for paper trading.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Dual-book mode (default): Intraday book uses Primary #1; Swing book uses
``swing_trade`` independently.

Legacy single-book: Intraday #1 if actionable, else Swing fallback.
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
        "risk_reward": slot.get("risk_reward") or slot.get("rr"),
        "horizon": horizon or slot.get("horizon") or "Intraday",
        "source": source,
        "trade_action": slot.get("trade_action"),
        "level_anchors": slot.get("level_anchors"),
        "rank": slot.get("rank"),
        "raw_slot": slot,
    }


def resolve_quote(
    symbol: str,
    trading_date: str,
    *,
    raw: dict[str, Any] | None = None,
    prefer_live: bool = True,
) -> tuple[float | None, str]:
    """Best-effort last price.

    Prefer DATA_ROOT raw/snapshot (same feed as live research) first so paper
    mirrors real collected prices; fall back to live yfinance when open/stale.
    """
    from src.research.entry_status import infer_session_phase

    phase = infer_session_phase(trading_date)
    px = None
    source = "none"

    # 1) Point-in-time raw / snapshot from DATA_ROOT (preferred for paper↔live parity)
    px = resolve_symbol_last(symbol, trading_date, raw=raw)
    if px is not None:
        source = "raw_snapshot"

    # 2) Live quote when session is open and raw is missing/stale
    if (px is None or (prefer_live and phase == "open")) and prefer_live:
        try:
            from src.collectors.yfinance_client import fetch_quote

            q = fetch_quote(symbol)
            live = _safe_float(
                (q or {}).get("last")
                or (q or {}).get("price")
                or (q or {}).get("close")
            )
            if live:
                # Prefer live only when raw missing, or when open (fresher tick).
                if px is None or phase == "open":
                    px, source = live, "yfinance_live"
        except Exception:
            logger.debug("yfinance live quote failed for %s", symbol, exc_info=True)

    if px is None and prefer_live:
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


def resolve_session_bar(
    symbol: str,
    trading_date: str,
    *,
    raw: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """OHLC-ish session bar from DATA_ROOT raw quotes when available.

    Used so stop/target can fire on the high/low path between ticks, not only
    on the last print. Missing fields are None (caller falls back to last).
    """
    from src.utils.paths import raw_data_path

    sym = (symbol or "").upper()
    out: dict[str, float | None] = {
        "open": None, "high": None, "low": None, "last": None, "close": None,
    }
    if not sym:
        return out

    payload = raw
    if payload is None:
        path = raw_data_path(trading_date)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        last, _ = resolve_quote(sym, trading_date, raw=raw, prefer_live=False)
        out["last"] = last
        return out

    section = "market"
    if sym in ("SMH", "XLK", "XLF", "XLE"):
        section = "sector"
    elif sym not in ("QQQ", "SPY", "DIA", "TQQQ"):
        section = "stocks"
    quotes = ((payload.get(section) or {}).get("quotes") or {})
    q = quotes.get(sym) or {}
    if not q:
        for ticker, row in quotes.items():
            if str(ticker).upper() == sym:
                q = row
                break
    out["open"] = _safe_float(q.get("open"))
    out["high"] = _safe_float(q.get("high") or q.get("day_high") or q.get("h"))
    out["low"] = _safe_float(q.get("low") or q.get("day_low") or q.get("l"))
    out["close"] = _safe_float(q.get("close"))
    out["last"] = (
        _safe_float(q.get("last") or q.get("current_price") or q.get("close"))
        or resolve_symbol_last(sym, trading_date, raw=payload)
    )
    # Sanity: if high/low missing but last known, treat last as both extremes.
    if out["last"] is not None:
        if out["high"] is None:
            out["high"] = out["last"]
        if out["low"] is None:
            out["low"] = out["last"]
    return out


def bar_path_prices(
    *,
    direction: str,
    last: float,
    high: float | None,
    low: float | None,
) -> list[float]:
    """Conservative intra-bar path: adverse extreme before favourable, then last."""
    d = (direction or "LONG").upper()
    hi = high if high is not None else last
    lo = low if low is not None else last
    if d == "LONG":
        # Stop checked via low first; target via high.
        path = [lo, hi, last]
    else:
        path = [hi, lo, last]
    # Dedupe while preserving order
    out: list[float] = []
    for p in path:
        if p is None:
            continue
        if not out or abs(out[-1] - float(p)) > 1e-9:
            out.append(float(p))
    return out or [float(last)]


def _morning_top_trades(morning: dict[str, Any]) -> list[dict[str, Any]]:
    top = (morning.get("best_trades") or {}).get("top_trades") or morning.get(
        "top_trades"
    ) or []
    if not isinstance(top, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in top:
        if not isinstance(row, dict):
            continue
        sym = (row.get("symbol") or "").upper()
        direction = (row.get("direction") or "").upper()
        if direction not in ("LONG", "SHORT") or not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(row)
    return out


def load_candidate_signals(trading_date: str) -> dict[str, Any]:
    """Load morning + step3 session primary + swing + top_trades list."""
    morning = _load_json(morning_json_path(trading_date))
    step3 = _load_json(step_json_path(3, trading_date))
    session = (
        step3.get("session_trade_update")
        or step3.get("trade_reeval")
        or step3.get("session_update")
        or {}
    )
    top_trades = _morning_top_trades(morning)
    primary = (morning.get("best_trades") or {}).get("primary") or morning.get(
        "best_opportunity"
    )
    # When primary is NO TRADE / null, fall back to top_trades[0]
    if not primary or (primary.get("direction") or "").upper() not in ("LONG", "SHORT"):
        if top_trades:
            primary = top_trades[0]
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
        "top_trades": top_trades,
        "swing": swing,
    }


def _evaluate_slot(
    norm: dict[str, Any],
    *,
    key: str,
    label: str,
    trading_date: str,
    phase: str,
    prices: dict[str, float],
    miss_threshold_pct: float,
    raw: dict[str, Any] | None,
) -> dict[str, Any]:
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
    return {
        "key": key,
        "label": label,
        "signal": {**norm, "entry_status": es, "current_price": px},
        "action": act,
        "entry_status": es,
        "quote": px,
        "quote_source": quote_source,
    }


def _candidate_edge_score(row: dict[str, Any]) -> float:
    """Higher = better executable setup (calibrated EV / expected R, not ER alone)."""
    from src.paper.allocation import MIN_EXPECTED_R, MIN_GEOMETRIC_UPSIDE_PCT, expected_r

    sig = row.get("signal") or {}
    wp = _safe_float(sig.get("win_prob")) or 50.0
    er = _safe_float(sig.get("expected_return_pct")) or 0.0
    rr = _safe_float(sig.get("risk_reward"))
    if rr is None:
        entry = _safe_float(sig.get("entry_price"))
        stop = _safe_float(sig.get("stop_price"))
        target = _safe_float(sig.get("target_price"))
        px = _safe_float(row.get("quote")) or entry
        if entry and stop and target and abs(entry - stop) > 0:
            reward = abs(target - (px or entry))
            rr = reward / abs(entry - stop)
        else:
            rr = 1.0
    ev = expected_r(win_prob=wp, rr=rr)
    # Sub-threshold EV or tiny geometric upside → crush score so pickers skip it.
    if ev is None or ev < MIN_EXPECTED_R:
        return 0.0
    if er < MIN_GEOMETRIC_UPSIDE_PCT:
        return 0.0
    status = ((row.get("entry_status") or {}).get("status") or "").upper()
    status_boost = {"TRIGGERED": 1.15, "READY": 1.1, "ACTIVE": 0.55}.get(status, 0.0)
    if row.get("action") != "enter":
        status_boost *= 0.4
    rr_w = max(0.5, min(1.5, float(rr) / 2.0))
    # Rank by expected R × win_prob soft weight × status (not bare ER distance).
    return float(max(ev, 0.01)) * 100.0 * rr_w * max(status_boost, 0.15) * (wp / 50.0)


def _pick_from_evaluated(
    evaluated: list[dict[str, Any]],
    *,
    allow_swing_fallback: bool,
) -> dict[str, Any]:
    """Pick best *executable* setup — not always board #1 if MISSED / weak R:R."""
    enterable = [
        r for r in evaluated if r["key"] != "swing" and r["action"] == "enter"
    ]
    if enterable:
        best = max(enterable, key=_candidate_edge_score)
        rank_note = ""
        if best.get("key", "").startswith("top_trade") or best.get("label", "").startswith(
            "Top"
        ):
            rank_note = f"（非盲目跟 #1，按可执行度+胜率/R:R 优选 {best.get('label')}）"
        return {
            "action": "enter",
            "signal": best["signal"],
            "entry_status": best["entry_status"],
            "reason": (
                f"模拟买入：{best['label']} Entry Status="
                f"{best['entry_status'].get('status')}（价位贴近 Ideal Entry）{rank_note}"
            ),
            "quote": best["quote"],
            "quote_source": best["quote_source"],
            "candidates": evaluated,
            "pick_score": round(_candidate_edge_score(best), 2),
        }

    waitable = [
        r for r in evaluated if r["key"] != "swing" and r["action"] == "wait"
    ]
    if waitable:
        best = max(waitable, key=_candidate_edge_score)
        return {
            "action": "wait",
            "signal": best["signal"],
            "entry_status": best["entry_status"],
            "reason": f"等待回踩：{best['label']} Entry Status=ACTIVE，未追高",
            "quote": best["quote"],
            "quote_source": best["quote_source"],
            "candidates": evaluated,
        }

    if allow_swing_fallback:
        for row in evaluated:
            if row["key"] == "swing" and row["action"] == "enter":
                return {
                    "action": "enter",
                    "signal": row["signal"],
                    "entry_status": row["entry_status"],
                    "reason": (
                        f"Intraday 不可执行，改用 Swing：Entry Status="
                        f"{row['entry_status'].get('status')}"
                    ),
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
        why = f"信号均不可执行（{statuses}）— MISSED/INVALIDATED/EXPIRED 不追；保留现金"
    return {
        "action": "skip",
        "signal": None,
        "entry_status": None,
        "reason": why,
        "quote": None,
        "quote_source": None,
        "candidates": evaluated,
    }


def pick_intraday_signal(
    trading_date: str,
    *,
    current_price_by_symbol: dict[str, float] | None = None,
    session_phase: str | None = None,
    miss_threshold_pct: float = 1.0,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick best executable Intraday among session #1, morning primary, top_trades."""
    phase = session_phase or infer_session_phase(trading_date)
    ctx = load_candidate_signals(trading_date)
    prices = current_price_by_symbol or {}
    evaluated: list[dict[str, Any]] = []
    seen_syms: set[str] = set()

    def _add(key: str, raw_slot: dict[str, Any] | None, label: str, horizon: str) -> None:
        norm = normalize_slot(raw_slot or {}, source=key, horizon=horizon)
        if not norm:
            return
        sym = norm["symbol"]
        if sym in seen_syms and key.startswith("top_trade"):
            return
        seen_syms.add(sym)
        evaluated.append(
            _evaluate_slot(
                norm,
                key=key,
                label=label,
                trading_date=trading_date,
                phase=phase,
                prices=prices,
                miss_threshold_pct=miss_threshold_pct,
                raw=raw,
            )
        )

    _add(
        "session_primary",
        ctx.get("session_primary"),
        "Intraday #1 (Step3)",
        "Intraday",
    )
    _add("morning_primary", ctx.get("morning_primary"), "Morning Primary", "Intraday")
    for i, row in enumerate(ctx.get("top_trades") or []):
        rank = row.get("rank") or (i + 1)
        _add(f"top_trade_{rank}", row, f"Top#{rank} {row.get('symbol')}", "Intraday")

    return _pick_from_evaluated(evaluated, allow_swing_fallback=False)


def pick_swing_signal(
    trading_date: str,
    *,
    current_price_by_symbol: dict[str, float] | None = None,
    session_phase: str | None = None,
    miss_threshold_pct: float = 1.0,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick Swing / 长线 signal only."""
    phase = session_phase or infer_session_phase(trading_date)
    ctx = load_candidate_signals(trading_date)
    prices = current_price_by_symbol or {}
    norm = normalize_slot(ctx.get("swing") or {}, source="swing", horizon="Swing")
    evaluated: list[dict[str, Any]] = []
    if norm:
        evaluated.append(
            _evaluate_slot(
                norm,
                key="swing",
                label="Swing · 长线",
                trading_date=trading_date,
                phase=phase,
                prices=prices,
                miss_threshold_pct=miss_threshold_pct,
                raw=raw,
            )
        )
    # Treat swing rows as the only candidates; reuse enter/wait paths via fallback flag
    if not evaluated:
        return {
            "action": "skip",
            "signal": None,
            "entry_status": None,
            "reason": "无 Swing / 长线信号",
            "quote": None,
            "quote_source": None,
            "candidates": [],
        }
    row = evaluated[0]
    if row["action"] == "enter":
        return {
            "action": "enter",
            "signal": row["signal"],
            "entry_status": row["entry_status"],
            "reason": f"长线买入：Swing Entry Status={row['entry_status'].get('status')}",
            "quote": row["quote"],
            "quote_source": row["quote_source"],
            "candidates": evaluated,
        }
    if row["action"] == "wait":
        return {
            "action": "wait",
            "signal": row["signal"],
            "entry_status": row["entry_status"],
            "reason": "等待长线 Ideal Entry 回踩",
            "quote": row["quote"],
            "quote_source": row["quote_source"],
            "candidates": evaluated,
        }
    return {
        "action": "skip",
        "signal": row["signal"],
        "entry_status": row["entry_status"],
        "reason": f"长线信号不可执行（{(row.get('entry_status') or {}).get('status') or 'N/A'}）",
        "quote": row["quote"],
        "quote_source": row["quote_source"],
        "candidates": evaluated,
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
    """Pick best executable trade for legacy single-book paper tick.

    Returns dict with keys: action ('enter'|'wait'|'skip'|'hold_signal'),
    signal (normalized), entry_status, reason, quote, quote_source.
    """
    phase = session_phase or infer_session_phase(trading_date)
    ctx = load_candidate_signals(trading_date)
    prices = current_price_by_symbol or {}

    evaluated: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(key: str, raw_slot: dict[str, Any] | None, label: str, horizon: str) -> None:
        norm = normalize_slot(raw_slot or {}, source=key, horizon=horizon)
        if not norm:
            return
        if norm["symbol"] in seen and key.startswith("top_trade"):
            return
        seen.add(norm["symbol"])
        evaluated.append(
            _evaluate_slot(
                norm,
                key=key,
                label=label,
                trading_date=trading_date,
                phase=phase,
                prices=prices,
                miss_threshold_pct=miss_threshold_pct,
                raw=raw,
            )
        )

    _add(
        "session_primary",
        ctx.get("session_primary"),
        "Intraday #1 (Step3)",
        "Intraday",
    )
    _add("morning_primary", ctx.get("morning_primary"), "Morning Primary", "Intraday")
    for i, row in enumerate(ctx.get("top_trades") or []):
        rank = row.get("rank") or (i + 1)
        _add(f"top_trade_{rank}", row, f"Top#{rank} {row.get('symbol')}", "Intraday")
    if prefer_swing_if_no_intraday:
        _add("swing", ctx.get("swing"), "Swing", "Swing")

    return _pick_from_evaluated(
        evaluated, allow_swing_fallback=prefer_swing_if_no_intraday
    )
