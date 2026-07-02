from __future__ import annotations

import json
from pathlib import Path

from src.utils.paths import morning_json_path, morning_report_path, raw_data_path, raw_dir, reports_dir
from src.web.launch import trading_day_number
from src.web.steps_status import steps_status


def list_trading_days() -> list[dict]:
    """All dates that have at least a raw file or morning report, newest first."""
    seen: set[str] = set()

    for p in reports_dir().iterdir():
        if p.is_dir():
            seen.add(p.name)

    for p in raw_dir().glob("*.json"):
        seen.add(p.stem)

    days: list[dict] = []
    for date_str in sorted(seen, reverse=True):
        meta: dict = {}
        json_path = morning_json_path(date_str)
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}

        days.append(
            {
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
        )
    return days
