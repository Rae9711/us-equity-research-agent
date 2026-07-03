from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.collectors.config import load_symbols
from src.collectors.fred_client import FredClient
from src.utils.trading_calendar import employment_situation_date, today_et


def _days_until(text: str, today: date) -> int | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not m:
        return None
    target = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (target - today).days


def _series_present(series: dict[str, Any], series_id: str) -> bool:
    if series_id in series:
        return True
    return any(v.get("series_id") == series_id for v in series.values())


def _inject_nfp_holiday_move(calendar: list[dict[str, Any]], today: date) -> None:
    """When BLS moves NFP off a holiday Friday, ensure calendar reflects Thursday release."""
    nfp_date = employment_situation_date(today.year, today.month)
    nfp_str = nfp_date.isoformat()
    has_nfp = any(
        "employment situation" in str(e.get("release_name") or "").lower()
        or "nonfarm" in str(e.get("release_name") or "").lower()
        for e in calendar
        if str(e.get("date") or "") == nfp_str
    )
    if not has_nfp and nfp_date == today:
        calendar.insert(
            0,
            {
                "release_name": "Employment Situation (NFP — holiday-adjusted)",
                "date": nfp_str,
                "source": "bls_schedule",
            },
        )


def collect_macro() -> dict[str, Any]:
    cfg = load_symbols()
    fred = FredClient()
    today = today_et()
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
        checklist[item["id"]] = _series_present(series, sid)

    calendar: list[dict[str, Any]] = []
    nonfarm_days: int | None = None
    nfp_release_today = employment_situation_date(today.year, today.month) == today

    try:
        releases = fred.release_dates(limit=40)
        for row in releases.get("release_dates") or []:
            name = (row.get("release_name") or "").lower()
            entry = {
                "release_name": row.get("release_name"),
                "date": row.get("date"),
            }
            for field in ("consensus", "forecast", "estimate", "expected"):
                if row.get(field) is not None:
                    entry[field] = row.get(field)
            calendar.append(entry)
            if "employment situation" in name or "nonfarm" in name:
                d = _days_until(str(row.get("date", "")), today)
                if d is not None and d >= 0 and (nonfarm_days is None or d < nonfarm_days):
                    nonfarm_days = d
    except Exception as exc:  # noqa: BLE001
        errors.append(f"release_dates:{exc}")

    _inject_nfp_holiday_move(calendar, today)

    if nonfarm_days is None:
        nfp_target = employment_situation_date(today.year, today.month)
        delta = (nfp_target - today).days
        if delta >= 0:
            nonfarm_days = delta

    checklist["economic_calendar"] = len(calendar) > 0
    checklist["nonfarm_countdown"] = nonfarm_days is not None
    checklist["nfp_release_day"] = nfp_release_today
    checklist["unemployment_rate"] = _series_present(series, "UNRATE")
    checklist["avg_hourly_earnings"] = _series_present(series, "AHETPI")
    checklist["initial_claims"] = _series_present(series, "ICSA")

    return {
        "series": series,
        "checklist": checklist,
        "economic_calendar": calendar[:15],
        "nonfarm_days_until": nonfarm_days,
        "nfp_release_today": nfp_release_today,
        "employment_situation_date": employment_situation_date(today.year, today.month).isoformat(),
        "errors": errors,
        "ok": len(errors) == 0 and checklist.get("fed", False),
    }
