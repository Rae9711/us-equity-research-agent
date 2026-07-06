"""Quote session date helpers (no collector dependencies)."""

from __future__ import annotations

from datetime import date
from typing import Any

_PRIOR_SESSION_TYPES = frozenset({"premarket", "prior_close", "prior_session"})


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
        if q.get("prior_session"):
            return True
        session_type = q.get("session_type")
        if session_type in _PRIOR_SESSION_TYPES:
            collected = str(q.get("data_as_of") or "")[:10]
            return collected == trading_date.isoformat()
    return False
