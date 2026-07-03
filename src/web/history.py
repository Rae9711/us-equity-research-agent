from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.utils.paths import morning_json_path, morning_report_path, raw_data_path, raw_dir, reports_dir
from src.utils.trading_calendar import today_et
from src.web.launch import launch_date, trading_day_number
from src.web.steps_status import steps_status


def _is_trading_date_str(s: str) -> bool:
    try:
        date.fromisoformat(s)
    except ValueError:
        return False
    return len(s) == 10


def _day_entry(date_str: str) -> dict:
    meta: dict = {}
    json_path = morning_json_path(date_str)
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    return {
        "date": date_str,
        "day_number": trading_day_number(date_str),
        "has_morning": morning_report_path(date_str).exists(),
        "has_raw": raw_data_path(date_str).exists(),
        "bias": meta.get("bias"),
        "total_score": meta.get("total_score"),
        "llm_used": meta.get("llm_used"),
        "data_ready": meta.get("data_ready"),
        "steps": steps_status(date_str),
    }


def list_trading_days() -> list[dict]:
    """All dates that have at least a raw file or morning report, newest first."""
    seen: set[str] = set()

    for p in reports_dir().iterdir():
        if p.is_dir() and _is_trading_date_str(p.name):
            seen.add(p.name)

    for p in raw_dir().glob("*.json"):
        if _is_trading_date_str(p.stem):
            seen.add(p.stem)

    days = [_day_entry(date_str) for date_str in sorted(seen, reverse=True)]

    # Include today in the picker once we're on/after launch (even before step0 runs).
    today_str = today_et().isoformat()
    if not any(d["date"] == today_str for d in days):
        t = today_et()
        if t >= launch_date() and t.weekday() < 5:
            days.insert(0, _day_entry(today_str))

    return days
