from __future__ import annotations

from datetime import date

from src.steps.meta import STEPS
from src.utils.trading_calendar import is_trading_day, nyse_holiday, today_et
from src.web.launch import launch_date, launch_label, trading_day_number
from src.web.steps_status import steps_status


def build_timeline(trading_date: str) -> list[dict]:
    status = steps_status(trading_date)
    items = []
    for s in STEPS:
        items.append(
            {
                "num": s.num,
                "time": s.time_et,
                "title": s.title,
                "subtitle": s.subtitle,
                "step_id": s.step_id,
                "available": status.get(s.num, False),
                "url": f"/step/{s.num}?date={trading_date}",
            }
        )
    return items


def day_summary(trading_date: str) -> dict:
    d = date.fromisoformat(trading_date)
    holiday = nyse_holiday(d)
    is_holiday = holiday is not None or not is_trading_day(d)
    status = steps_status(trading_date)
    done = sum(1 for v in status.values() if v)
    total = len(STEPS)
    return {
        "trading_date": trading_date,
        "launch_date": launch_date().isoformat(),
        "is_today": trading_date == today_et().isoformat(),
        "is_holiday": is_holiday,
        "holiday_label": holiday,
        "day_label": launch_label(trading_date),
        "day_number": trading_day_number(trading_date),
        "steps_done": done if not is_holiday else 0,
        "steps_total": total,
        "progress_pct": 0 if is_holiday else (round(100 * done / total) if total else 0),
        "steps": status,
        "timeline": build_timeline(trading_date),
    }
