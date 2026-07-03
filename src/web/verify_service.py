from __future__ import annotations

from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel, Field


class VerifyItem(BaseModel):
    part_id: str
    verification: Optional[str] = Field(None, pattern=r"^(对|错|部分对)?$")
    user_judgment: Optional[str] = None
    notes: Optional[str] = None


class VerifyPayload(BaseModel):
    trading_date: str
    items: list[VerifyItem]


class SaveVerificationsResult(BaseModel):
    updated: int = 0
    learning_refreshed: bool = False
    learning_summary: str | None = None


def save_verifications(
    trading_date: date_type, items: list[VerifyItem]
) -> SaveVerificationsResult:
    from src.db import ConclusionRecord
    from src.db.session import get_session

    session = get_session()
    updated = 0
    try:
        for item in items:
            row = (
                session.query(ConclusionRecord)
                .filter(
                    ConclusionRecord.trading_date == trading_date,
                    ConclusionRecord.part_id == item.part_id,
                )
                .first()
            )
            if not row:
                continue
            row.verification = item.verification
            row.user_judgment = item.user_judgment or None
            row.notes = item.notes or None
            updated += 1
        session.commit()
    finally:
        session.close()

    learning_refreshed = False
    learning_summary: str | None = None
    if updated > 0:
        from src.web.verify_learning import refresh_learning_from_verify

        try:
            result = refresh_learning_from_verify(trading_date)
            learning_refreshed = bool(result.get("refreshed"))
            learning_summary = result.get("step8_one_liner")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).error(
                "Verify learning refresh failed for %s: %s",
                trading_date.isoformat(),
                exc,
            )

    return SaveVerificationsResult(
        updated=updated,
        learning_refreshed=learning_refreshed,
        learning_summary=learning_summary,
    )
