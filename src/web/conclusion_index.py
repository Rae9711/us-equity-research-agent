from __future__ import annotations

from typing import Any

from src.db import ConclusionRecord
from src.research.parts_meta import MORNING_REPORT_ORDER, part_label
from src.web.verify_order import conclusion_sort_key


def _conf_display(conf: Any) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.0%}"
    return "N/A"


def morning_index_from_parts(parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build R0 + P1–P17 rows for the Step 1 conclusion index table."""
    rows: list[dict[str, Any]] = []
    for pid in MORNING_REPORT_ORDER:
        p = parts.get(pid) or {}
        rows.append(
            {
                "part_id": pid,
                "label": part_label(pid),
                "judgment": p.get("judgment") or "—",
                "confidence": _conf_display(p.get("confidence")),
                "one_liner": p.get("one_liner") or "—",
            }
        )
    return rows


def morning_index_from_records(records: list[ConclusionRecord]) -> list[dict[str, Any]]:
    by_id = {r.part_id: r for r in records}
    rows: list[dict[str, Any]] = []
    for pid in MORNING_REPORT_ORDER:
        r = by_id.get(pid)
        if r is None:
            rows.append(
                {
                    "part_id": pid,
                    "label": part_label(pid),
                    "judgment": "—",
                    "confidence": "N/A",
                    "one_liner": "—",
                }
            )
            continue
        rows.append(
            {
                "part_id": pid,
                "label": part_label(pid),
                "judgment": r.judgment or "—",
                "confidence": _conf_display(r.confidence),
                "one_liner": r.one_liner or "—",
            }
        )
    return rows


def verify_index_from_records(records: list[ConclusionRecord]) -> list[dict[str, Any]]:
    """Compact rows for verify page summary (Step0 + R0 + P* + S*)."""
    return [
        {
            "part_id": r.part_id,
            "label": r.part_id,
            "judgment": r.judgment or "—",
            "confidence": _conf_display(r.confidence),
            "one_liner": r.one_liner or "—",
            "verification": r.verification,
        }
        for r in sorted(records, key=lambda r: conclusion_sort_key(r.part_id))
    ]
