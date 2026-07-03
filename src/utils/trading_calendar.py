from __future__ import annotations

from datetime import date, datetime, timedelta

from pytz import timezone

ET = timezone("America/New_York")
UTC = timezone("UTC")


def today_et() -> date:
    return datetime.now(ET).date()


def prior_trading_day(d: date | None = None) -> date:
    """Previous weekday (MVP; does not skip US market holidays)."""
    d = d or today_et()
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def market_open_et(d: date | None = None) -> datetime:
    """US cash equity session open — 9:30 AM ET."""
    d = d or today_et()
    return ET.localize(datetime(d.year, d.month, d.day, 9, 30, 0))


def prior_close_utc_iso(trading_date: date | None = None) -> str:
    """Prior session close (4:00 PM ET on prior trading day) as UTC ISO for Polygon filters."""
    trading_date = trading_date or today_et()
    prior = prior_trading_day(trading_date)
    close_et = ET.localize(datetime(prior.year, prior.month, prior.day, 16, 0, 0))
    return close_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
