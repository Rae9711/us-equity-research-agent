from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DailyRun(Base):
    __tablename__ = "daily_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trading_date: Mapped[date] = mapped_column(Date, index=True)
    step_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ConclusionRecord(Base):
    __tablename__ = "conclusions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trading_date: Mapped[date] = mapped_column(Date, index=True)
    part_id: Mapped[str] = mapped_column(String(16), index=True)
    judgment: Mapped[str] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    one_liner: Mapped[str] = mapped_column(String(256))
    verification: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    user_judgment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


def init_db() -> None:
    from src.db.session import get_engine

    Base.metadata.create_all(get_engine())
