from __future__ import annotations

PART_SORT_ORDER = (
    ["Step0", "R0"]
    + [f"P{i}" for i in range(1, 17)]
    + ["P17"]
    + [f"S{i}" for i in range(2, 9)]
)


def conclusion_sort_key(part_id: str) -> tuple[int, int, str]:
    if part_id in PART_SORT_ORDER:
        return (0, PART_SORT_ORDER.index(part_id), part_id)
    if part_id.startswith("P"):
        try:
            return (1, int(part_id[1:]), part_id)
        except ValueError:
            pass
    if part_id.startswith("S") or part_id.startswith("Step"):
        return (2, 0, part_id)
    return (3, 0, part_id)


def sort_conclusions(rows: list) -> list:
    return sorted(rows, key=lambda r: conclusion_sort_key(r.part_id))
