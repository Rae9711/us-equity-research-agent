from __future__ import annotations

from datetime import date, datetime, timedelta

from pytz import timezone

ET = timezone("America/New_York")


def today_et() -> date:
    return datetime.now(ET).date()


def prior_trading_day(d: date | None = None) -> date:
    """Previous weekday (MVP; does not skip US market holidays)."""
    d = d or today_et()
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d
