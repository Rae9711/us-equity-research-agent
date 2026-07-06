"""Validate Step 0 quote dates against the trading calendar."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.utils.trading_calendar import ET, prior_trading_day


def parse_quote_session_date(q: dict[str, Any]) -> date | None:
    raw = q.get("quote_session_date") or q.get("date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def quote_fresh_for_checklist(
    q: dict[str, Any],
    trading_date: date,
    prior_day: date,
) -> bool:
    """True when close exists and session date is trading_date or prior_day (pre-market)."""
    if "error" in q or q.get("close") is None:
        return False
    qsd = parse_quote_session_date(q)
    if qsd is None:
        return False
    if qsd < prior_day:
        return False
    if qsd == trading_date:
        return True
    if qsd == prior_day:
        collected = str(q.get("data_as_of") or "")[:10]
        return q.get("session_type") == "premarket" and collected == trading_date.isoformat()
    return False


def _collected_date_et(payload: dict[str, Any]) -> date | None:
    raw = payload.get("collected_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).astimezone(ET).date()
    except ValueError:
        return None


def assess_raw_freshness(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level freshness for UI and step availability."""
    trading_date = date.fromisoformat(payload["trading_date"])
    prior_raw = payload.get("prior_trading_day")
    prior_day = (
        date.fromisoformat(prior_raw)
        if prior_raw
        else prior_trading_day(trading_date)
    )
    reasons: list[str] = []

    coll_date = _collected_date_et(payload)
    if coll_date is not None and coll_date != trading_date:
        reasons.append(f"collected_at 不在交易日 {trading_date.isoformat()}")

    cfg_market = (payload.get("market") or {}).get("quotes") or {}
    for label in ("SPY", "QQQ"):
        q = cfg_market.get(label) or {}
        if not quote_fresh_for_checklist(q, trading_date, prior_day):
            qsd = parse_quote_session_date(q)
            reasons.append(
                f"market.{label} quote_session_date={qsd} (需要 {trading_date} 或盘前 {prior_day})"
            )

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "collected_date": coll_date.isoformat() if coll_date else None,
        "trading_date": trading_date.isoformat(),
        "prior_trading_day": prior_day.isoformat(),
    }
