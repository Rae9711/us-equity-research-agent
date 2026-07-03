from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.trading_calendar import count_trading_days, is_trading_day, today_et

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "launch.yaml"


@lru_cache(maxsize=1)
def launch_date() -> date:
    if _CONFIG.exists():
        data = yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {}
        raw = data.get("launch_date")
        if raw:
            return date.fromisoformat(str(raw))
    return today_et()


def trading_day_number(trading_date: str | date) -> int | None:
    """1-based NYSE session count from launch_date. None if before launch or holiday."""
    d = trading_date if isinstance(trading_date, date) else date.fromisoformat(trading_date)
    start = launch_date()
    if d < start:
        return None
    if not is_trading_day(d):
        return None
    return count_trading_days(start, d)


def launch_label(trading_date: str | date) -> str:
    d = trading_date if isinstance(trading_date, date) else date.fromisoformat(trading_date)
    from src.utils.trading_calendar import nyse_holiday

    holiday = nyse_holiday(d)
    if holiday:
        return "NYSE 休市"
    n = trading_day_number(d)
    if n is None:
        return "预热"
    return f"Day {n}"
