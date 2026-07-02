from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.trading_calendar import today_et

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "launch.yaml"


@lru_cache(maxsize=1)
def launch_date() -> date:
    if _CONFIG.exists():
        data = yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {}
        raw = data.get("launch_date")
        if raw:
            return date.fromisoformat(str(raw))
    return today_et()


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def trading_day_number(trading_date: str | date) -> int | None:
    """1-based NYSE weekday count from launch_date. None if before launch."""
    d = trading_date if isinstance(trading_date, date) else date.fromisoformat(trading_date)
    start = launch_date()
    if d < start:
        return None
    n = 0
    cur = start
    while cur <= d:
        if _is_weekday(cur):
            n += 1
        cur += timedelta(days=1)
    return n


def launch_label(trading_date: str | date) -> str:
    n = trading_day_number(trading_date)
    if n is None:
        return "预热"
    return f"Day {n}"
