"""Validate Step 0 quote dates against the trading calendar."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.utils.quote_session import parse_quote_session_date, quote_fresh_for_checklist
from src.utils.trading_calendar import ET

__all__ = ["parse_quote_session_date", "quote_fresh_for_checklist", "assess_raw_freshness"]


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
    from src.utils.data_freshness import freshness_dict

    trading_date = date.fromisoformat(payload["trading_date"])
    return freshness_dict(payload, trading_date)
