from __future__ import annotations

from src.steps.meta import STEPS
from src.utils.paths import morning_report_path, raw_data_path, step_json_path


def step_available(step_num: int, trading_date: str) -> bool:
    if step_num == 0:
        return raw_data_path(trading_date).exists()
    if step_num == 1:
        return morning_report_path(trading_date).exists()
    return step_json_path(step_num, trading_date).exists()


def steps_status(trading_date: str) -> dict[int, bool]:
    return {s.num: step_available(s.num, trading_date) for s in STEPS}


def steps_status_summary(trading_date: str) -> dict:
    status = steps_status(trading_date)
    done = sum(1 for v in status.values() if v)
    return {"date": trading_date, "steps": status, "done_count": done, "total": len(STEPS)}
