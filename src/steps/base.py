from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from pytz import timezone

from src.db import ConclusionRecord, DailyRun
from src.db.session import get_session
from src.utils.paths import step_json_path, step_report_path

ET = timezone("America/New_York")


def save_step_result(
    step_num: int,
    trading_date: date,
    *,
    step_id: str,
    job_id: str,
    conclusion: dict[str, Any],
    body_md: str,
    extra: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    date_str = trading_date.isoformat()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(ET).isoformat(),
        "trading_date": date_str,
        "step_num": step_num,
        "step_id": step_id,
        "conclusion": conclusion,
        "body_md": body_md,
        **(extra or {}),
    }

    step_json_path(step_num, date_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from src.steps.report import render_step_markdown

    step_report_path(step_num, date_str).write_text(
        render_step_markdown(payload), encoding="utf-8"
    )

    session = get_session()
    try:
        session.add(
            DailyRun(
                trading_date=trading_date,
                step_id=job_id,
                status=status,
                message=conclusion.get("one_liner", ""),
            )
        )
        row = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == trading_date,
                ConclusionRecord.part_id == step_id,
            )
            .first()
        )
        if row:
            row.judgment = conclusion.get("judgment", "")
            row.confidence = conclusion.get("confidence")
            row.one_liner = conclusion.get("one_liner", "")
        else:
            session.add(
                ConclusionRecord(
                    trading_date=trading_date,
                    part_id=step_id,
                    judgment=conclusion.get("judgment", ""),
                    confidence=conclusion.get("confidence"),
                    one_liner=conclusion.get("one_liner", ""),
                )
            )
        session.commit()
    finally:
        session.close()

    return payload
