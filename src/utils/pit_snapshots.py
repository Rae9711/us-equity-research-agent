"""Point-in-time (PIT) snapshot storage for scheduled trading steps.

Each scheduled step saves an immutable snapshot at its decision boundary so
afternoon reruns replay the same prices as the original run — not live data.

Paths: ``data/snapshots/YYYY-MM-DD/{label}.json`` (e.g. step0_0745.json).
First successful write wins; later reruns are skipped unless ``--force``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

from pytz import timezone

from src.utils.paths import data_root

logger = logging.getLogger(__name__)

ET = timezone("America/New_York")

# step_num -> (label, scheduled ET time)
STEP_PIT: dict[int, tuple[str, time]] = {
    0: ("step0_0745", time(7, 45)),
    1: ("step1_0800", time(8, 0)),
    2: ("step2_0930", time(9, 30)),
    3: ("step3_1000", time(10, 0)),
    4: ("step4_1015", time(10, 15)),
    5: ("step5_1200", time(12, 0)),
    6: ("step6_1400", time(14, 0)),
    7: ("step7_1610", time(16, 10)),
    8: ("step8_2000", time(20, 0)),
}

LABEL_TO_STEP: dict[str, int] = {label: num for num, (label, _) in STEP_PIT.items()}

PRIOR_SNAPSHOT: dict[int, str] = {
    1: "step0_0745",
    2: "step1_0800",
    3: "step2_0930",
    4: "step3_1000",
    5: "step4_1015",
    6: "step5_1200",
    7: "step6_1400",
    8: "step7_1610",
}


class PITSnapshotMissingError(FileNotFoundError):
    """Raised when a required PIT snapshot is absent for replay."""


def step_label(step_num: int) -> str:
    return STEP_PIT[step_num][0]


def step_scheduled_time(step_num: int) -> time:
    return STEP_PIT[step_num][1]


def _hhmm(t: time) -> str:
    return t.strftime("%H%M")


def snapshots_dir(trading_date: date | str) -> str:
    d = trading_date.isoformat() if isinstance(trading_date, date) else trading_date
    return str(data_root() / "snapshots" / d)


def snapshot_path(trading_date: date | str, as_of_et: time) -> str:
    """Return legacy HHMM path (for backward-compat reads only)."""
    d = trading_date.isoformat() if isinstance(trading_date, date) else trading_date
    return str(data_root() / "snapshots" / d / f"{_hhmm(as_of_et)}.json")


def snapshot_path_for_label(trading_date: date | str, label: str) -> str:
    """Return path: data/snapshots/YYYY-MM-DD/{label}.json"""
    d = trading_date.isoformat() if isinstance(trading_date, date) else trading_date
    return str(data_root() / "snapshots" / d / f"{label}.json")


def _legacy_hhmm_path(trading_date: date | str, label: str) -> Path | None:
    step_num = LABEL_TO_STEP.get(label)
    if step_num is None:
        return None
    d = trading_date.isoformat() if isinstance(trading_date, date) else trading_date
    return data_root() / "snapshots" / d / f"{_hhmm(step_scheduled_time(step_num))}.json"


def _resolve_snapshot_path(trading_date: date | str, label: str) -> Path:
    primary = Path(snapshot_path_for_label(trading_date, label))
    if primary.exists():
        return primary
    legacy = _legacy_hhmm_path(trading_date, label)
    if legacy and legacy.exists():
        return legacy
    return primary


def parse_as_of(
    as_of: str | None,
    *,
    step_num: int | None = None,
    default_time: time | None = None,
) -> time | Literal["now"]:
    """Parse CLI --as-of value. Default: scheduled time for step_num."""
    if as_of is None:
        if step_num is not None:
            return step_scheduled_time(step_num)
        if default_time is not None:
            return default_time
        raise ValueError("as_of required when step_num and default_time are None")
    text = as_of.strip().lower()
    if text == "now":
        return "now"
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text.upper().replace("ET", "").strip(), fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Invalid --as-of value: {as_of!r} (use HH:MM, HH:MM:SS, or 'now')")


def as_of_et_iso(trading_date: date, as_of: time | Literal["now"]) -> str:
    if as_of == "now":
        return datetime.now(ET).isoformat()
    dt = ET.localize(datetime.combine(trading_date, as_of))
    return dt.isoformat()


def format_as_of_display(as_of: time | Literal["now"], *, step_num: int | None = None) -> str:
    if as_of == "now":
        return "实时"
    label = f"{as_of.strftime('%H:%M')} ET"
    if step_num == 1:
        return f"{label}（早盘决策）"
    if step_num == 2:
        return f"{label}（开盘观察）"
    return label


def snapshot_exists(trading_date: date | str, label: str) -> bool:
    return _resolve_snapshot_path(trading_date, label).exists()


def load_snapshot(trading_date: date | str, label: str) -> dict[str, Any] | None:
    path = _resolve_snapshot_path(trading_date, label)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load snapshot %s", path)
        return None


def load_snapshot_raw(trading_date: date | str, label: str) -> dict[str, Any] | None:
    """Return embedded raw/quotes payload from a snapshot."""
    snap = load_snapshot(trading_date, label)
    if not snap:
        return None
    data = snap.get("data") or snap.get("raw")
    if isinstance(data, dict) and "raw" in data and isinstance(data["raw"], dict):
        return data["raw"]
    return data if isinstance(data, dict) else None


def save_snapshot(
    trading_date: date | str,
    label: str,
    data: dict[str, Any],
    *,
    force: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist PIT snapshot. Refuses overwrite unless force=True."""
    path = Path(snapshot_path_for_label(trading_date, label))
    if path.exists() and not force:
        logger.info("PIT snapshot skipped: already exists — %s (use --force to overwrite)", label)
        return json.loads(path.read_text(encoding="utf-8"))

    d = date.fromisoformat(trading_date) if isinstance(trading_date, str) else trading_date
    step_num = LABEL_TO_STEP.get(label)
    as_of_t = step_scheduled_time(step_num) if step_num is not None else None
    if as_of_t is None:
        suffix = label.rsplit("_", 1)[-1]
        if len(suffix) == 4 and suffix.isdigit():
            as_of_t = time(int(suffix[:2]), int(suffix[2:]))

    payload: dict[str, Any] = {
        "label": label,
        "trading_date": d.isoformat(),
        "decision_as_of": as_of_et_iso(d, as_of_t) if as_of_t else None,
        "saved_at": datetime.now(ET).isoformat(),
        "data": data,
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("PIT snapshot saved: %s (immutable)", label)
    return payload


# Alias required by runner wiring spec
save_pit_snapshot = save_snapshot


def require_pit_raw(
    trading_date: date,
    step_num: int,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Load prior-step raw for replay. Raises if snapshot missing."""
    label = label or PRIOR_SNAPSHOT.get(step_num, "")
    if not label:
        raise PITSnapshotMissingError(f"No prior snapshot configured for step {step_num}")
    snap_raw = load_snapshot_raw(trading_date, label)
    if snap_raw:
        return snap_raw
    path = snapshot_path_for_label(trading_date, label)
    logger.error(
        "PIT SNAPSHOT MISSING: %s for %s — rerun will NOT use live prices. "
        "Run the scheduled job first or pass --force after manual collect.",
        label,
        trading_date.isoformat(),
    )
    raise PITSnapshotMissingError(
        f"Required PIT snapshot missing: {path} (needed for step {step_num} replay)"
    )


def pit_raw_for_step(
    trading_date: date,
    step_num: int,
    *,
    fallback_raw: dict[str, Any] | None = None,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Load the raw data snapshot appropriate for running step_num."""
    if step_num == 0:
        return fallback_raw or {}
    label = PRIOR_SNAPSHOT.get(step_num)
    if label:
        snap_raw = load_snapshot_raw(trading_date, label)
        if snap_raw:
            return snap_raw
        if not allow_fallback:
            return {}
    if allow_fallback and fallback_raw:
        return fallback_raw
    return fallback_raw or {}


def list_snapshots(trading_date: date | str) -> list[dict[str, Any]]:
    """List all PIT snapshots for a trading day (scheduled + any extras)."""
    d = trading_date.isoformat() if isinstance(trading_date, date) else trading_date
    snap_dir = data_root() / "snapshots" / d
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for step_num in sorted(STEP_PIT):
        label, sched = STEP_PIT[step_num]
        path = _resolve_snapshot_path(d, label)
        exists = path.exists()
        seen.add(path.name)
        rows.append({
            "step": step_num,
            "label": label,
            "scheduled_et": sched.strftime("%H:%M"),
            "exists": exists,
            "path": str(path),
            "size_bytes": path.stat().st_size if exists else 0,
        })

    if snap_dir.is_dir():
        for p in sorted(snap_dir.glob("*.json")):
            if p.name in seen:
                continue
            rows.append({
                "step": None,
                "label": p.stem,
                "scheduled_et": None,
                "exists": True,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            })
    return rows


def prune_old_snapshots(*, keep_days: int = 90, dry_run: bool = False) -> list[str]:
    """Remove snapshot directories older than keep_days. Returns removed paths."""
    root = data_root() / "snapshots"
    if not root.is_dir():
        return []
    cutoff = today_et() - timedelta(days=keep_days)
    removed: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            d = date.fromisoformat(child.name)
        except ValueError:
            continue
        if d >= cutoff:
            continue
        removed.append(str(child))
        if not dry_run:
            import shutil
            shutil.rmtree(child)
            logger.info("Pruned old PIT snapshots: %s", child)
    return removed


def today_et() -> date:
    from src.utils.trading_calendar import today_et as _today_et
    return _today_et()
