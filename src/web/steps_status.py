from __future__ import annotations

import json

from src.steps.meta import STEPS
from src.utils.paths import morning_report_path, raw_data_path, step_json_path


def _load_raw(trading_date: str) -> dict | None:
    path = raw_data_path(trading_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def step0_available(trading_date: str) -> bool:
    payload = _load_raw(trading_date)
    if not payload:
        return False
    freshness = payload.get("freshness") or {}
    if freshness:
        return bool(payload.get("data_ready")) and bool(freshness.get("ok"))
    # Legacy files without freshness metadata: require data_ready only.
    return bool(payload.get("data_ready"))


def step_available(step_num: int, trading_date: str) -> bool:
    if step_num == 0:
        return step0_available(trading_date)
    if step_num == 1:
        return morning_report_path(trading_date).exists()
    return step_json_path(step_num, trading_date).exists()


def steps_status(trading_date: str) -> dict[int, bool]:
    return {s.num: step_available(s.num, trading_date) for s in STEPS}


def steps_status_summary(trading_date: str) -> dict:
    status = steps_status(trading_date)
    done = sum(1 for v in status.values() if v)
    return {"date": trading_date, "steps": status, "done_count": done, "total": len(STEPS)}


def step0_freshness(trading_date: str) -> dict | None:
    payload = _load_raw(trading_date)
    if not payload:
        return None
    return payload.get("freshness")
