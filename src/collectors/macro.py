from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.collectors.config import load_symbols
from src.collectors.fred_client import FredClient


def _days_until(text: str, today: date) -> int | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not m:
        return None
    target = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (target - today).days


def collect_macro() -> dict[str, Any]:
    cfg = load_symbols()
    fred = FredClient()
    series: dict[str, Any] = {}
    errors: list[str] = []

    for key, series_id in cfg["fred"].items():
        try:
            obs = fred.latest_observation(series_id)
            if obs:
                series[key] = {
                    "series_id": series_id,
                    "date": obs.get("date"),
                    "value": obs.get("value"),
                }
            else:
                errors.append(series_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{series_id}:{exc}")

    checklist: dict[str, bool] = {}
    for item in cfg.get("macro_checklist", []):
        sid = item["fred_series"]
        checklist[item["id"]] = sid in series or any(
            v.get("series_id") == sid for v in series.values()
        )

    # Economic calendar + nonfarm countdown from FRED release dates
    calendar: list[dict[str, Any]] = []
    nonfarm_days: int | None = None
    try:
        releases = fred.release_dates(limit=30)
        for row in releases.get("release_dates") or []:
            name = (row.get("release_name") or "").lower()
            entry = {
                "release_name": row.get("release_name"),
                "date": row.get("date"),
            }
            # Preserve consensus fields if a future calendar source provides them
            for field in ("consensus", "forecast", "estimate", "expected"):
                if row.get(field) is not None:
                    entry[field] = row.get(field)
            calendar.append(entry)
            if "employment situation" in name or "nonfarm" in name:
                d = _days_until(str(row.get("date", "")), date.today())
                if d is not None and d >= 0 and (nonfarm_days is None or d < nonfarm_days):
                    nonfarm_days = d
    except Exception as exc:  # noqa: BLE001
        errors.append(f"release_dates:{exc}")

    checklist["economic_calendar"] = len(calendar) > 0
    checklist["nonfarm_countdown"] = nonfarm_days is not None

    return {
        "series": series,
        "checklist": checklist,
        "economic_calendar": calendar[:10],
        "nonfarm_days_until": nonfarm_days,
        "errors": errors,
        "ok": len(errors) == 0 and checklist.get("fed", False),
    }
