from __future__ import annotations

from src.research.parts_meta import part_about as part_about_p
from src.research.parts_meta import part_label as part_label_p
from src.steps.meta import step_about, step_label


def unified_label(part_id: str) -> str:
    if part_id.startswith("S"):
        return step_label(part_id)
    return part_label_p(part_id)


def unified_about(part_id: str) -> str:
    if part_id.startswith("S"):
        return step_about(part_id)
    return part_about_p(part_id)
