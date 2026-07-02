from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.schemas.conclusion import Conclusion


class DailyState(BaseModel):
    """In-memory / persisted daily workflow state."""

    trading_date: date
    prior_trading_day: Optional[date] = None
    raw_data: dict[str, Any] = Field(default_factory=dict)
    parts: dict[str, Conclusion] = Field(default_factory=dict)
    steps: dict[str, Conclusion] = Field(default_factory=dict)
    morning_summary: list[Conclusion] = Field(default_factory=list)
