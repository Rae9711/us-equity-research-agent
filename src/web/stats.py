from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func

from src.db import ConclusionRecord
from src.db.session import get_session


@dataclass
class AccuracySummary:
    total: int
    verified: int
    correct: int
    wrong: int
    partial: int

    @property
    def accuracy_pct(self) -> float | None:
        if self.verified == 0:
            return None
        return round(self.correct / self.verified * 100, 1)


def global_accuracy() -> AccuracySummary:
    session = get_session()
    try:
        verified = (
            session.query(ConclusionRecord)
            .filter(ConclusionRecord.verification.isnot(None))
            .filter(ConclusionRecord.verification != "")
            .all()
        )
    finally:
        session.close()

    correct = sum(1 for r in verified if r.verification == "对")
    wrong = sum(1 for r in verified if r.verification == "错")
    partial = sum(1 for r in verified if r.verification == "部分对")
    return AccuracySummary(
        total=len(verified),
        verified=len(verified),
        correct=correct,
        wrong=wrong,
        partial=partial,
    )


def accuracy_by_date() -> list[dict]:
    session = get_session()
    try:
        rows = (
            session.query(
                ConclusionRecord.trading_date,
                func.count(ConclusionRecord.id).label("verified"),
                func.sum(
                    case((ConclusionRecord.verification == "对", 1), else_=0)
                ).label("correct"),
            )
            .filter(ConclusionRecord.verification.isnot(None))
            .filter(ConclusionRecord.verification != "")
            .group_by(ConclusionRecord.trading_date)
            .order_by(ConclusionRecord.trading_date.desc())
            .all()
        )
    finally:
        session.close()

    out: list[dict] = []
    for row in rows:
        verified = int(row.verified or 0)
        correct = int(row.correct or 0)
        out.append(
            {
                "date": row.trading_date.isoformat(),
                "verified": verified,
                "correct": correct,
                "accuracy_pct": round(correct / verified * 100, 1) if verified else None,
            }
        )
    return out


def accuracy_for_date(trading_date) -> AccuracySummary:
    session = get_session()
    try:
        verified = (
            session.query(ConclusionRecord)
            .filter(ConclusionRecord.trading_date == trading_date)
            .filter(ConclusionRecord.verification.isnot(None))
            .filter(ConclusionRecord.verification != "")
            .all()
        )
    finally:
        session.close()

    correct = sum(1 for r in verified if r.verification == "对")
    wrong = sum(1 for r in verified if r.verification == "错")
    partial = sum(1 for r in verified if r.verification == "部分对")
    return AccuracySummary(
        total=len(verified),
        verified=len(verified),
        correct=correct,
        wrong=wrong,
        partial=partial,
    )
