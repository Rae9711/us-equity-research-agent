from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    CORRECT = "对"
    WRONG = "错"
    PARTIAL = "部分对"


class Conclusion(BaseModel):
    """Agent conclusion block — see WORKFLOW.md 结论输出规范."""

    part_id: str = Field(..., description="e.g. P1, S4, Step0")
    judgment: str
    confidence: Optional[float] = Field(None, ge=0, le=1)
    one_liner: str = Field(..., max_length=120)
    verification: Optional[VerificationStatus] = None
    user_judgment: Optional[str] = None
    notes: Optional[str] = None
