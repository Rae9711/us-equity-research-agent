from __future__ import annotations

from datetime import date, datetime, timedelta

from pytz import timezone

ET = timezone("America/New_York")
UTC = timezone("UTC")

# NYSE full-day closures (date → label). Extend annually.
_NYSE_HOLIDAYS: dict[date, str] = {
    # 2025
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "MLK Day",
    date(2025, 2, 17): "Presidents Day",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 6, 19): "Juneteenth",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving",
    date(2025, 12, 25): "Christmas",
    # 2026 — Independence Day observed Fri 7/3 (7/4 is Saturday); include both
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "MLK Day",
    date(2026, 2, 16): "Presidents Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (observed)",
    date(2026, 7, 4): "Independence Day",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving",
    date(2026, 12, 25): "Christmas",
    # 2027
    date(2027, 1, 1): "New Year's Day",
    date(2027, 1, 18): "MLK Day",
    date(2027, 2, 15): "Presidents Day",
    date(2027, 3, 26): "Good Friday",
    date(2027, 5, 31): "Memorial Day",
    date(2027, 6, 18): "Juneteenth (observed)",
    date(2027, 7, 5): "Independence Day (observed)",
    date(2027, 9, 6): "Labor Day",
    date(2027, 11, 25): "Thanksgiving",
    date(2027, 12, 24): "Christmas (observed)",
}


def nyse_holiday(d: date | None = None) -> str | None:
    """Return holiday label if NYSE is closed, else None."""
    d = d or today_et()
    return _NYSE_HOLIDAYS.get(d)


def is_nyse_holiday(d: date | None = None) -> bool:
    return nyse_holiday(d) is not None


def is_trading_day(d: date | None = None) -> bool:
    """True on NYSE open days (weekday and not a full closure)."""
    d = d or today_et()
    if d.weekday() >= 5:
        return False
    return not is_nyse_holiday(d)


def today_et() -> date:
    return datetime.now(ET).date()


def prior_trading_day(d: date | None = None) -> date:
    """Previous NYSE session (skips weekends and NYSE holidays)."""
    d = d or today_et()
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date | None = None) -> date:
    d = d or today_et()
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def count_trading_days(start: date, end: date) -> int:
    """Inclusive count of NYSE sessions from start through end."""
    if end < start:
        return 0
    n = 0
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            n += 1
        cur += timedelta(days=1)
    return n


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def employment_situation_date(year: int, month: int) -> date:
    """BLS NFP release date — normally first Friday; moves to Thursday if Friday is NYSE holiday."""
    friday = first_friday(year, month)
    if is_nyse_holiday(friday) or not is_trading_day(friday):
        thursday = friday - timedelta(days=1)
        if is_trading_day(thursday):
            return thursday
    return friday


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
