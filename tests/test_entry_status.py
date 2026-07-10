"""Tests for Ideal Entry vs live Entry Status classification.

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

from src.research.entry_status import (
    STATUS_ACTIVE,
    STATUS_INVALIDATED,
    STATUS_MISSED,
    STATUS_PLANNED,
    STATUS_PREMARKET,
    STATUS_READY,
    STATUS_TRIGGERED,
    attach_entry_status,
    classify_entry_status,
    planned_entry_status,
)


ZONE_LONG = {"low": 340.0, "mid": 341.0, "high": 342.0, "display": "Entry Zone 340–342"}
ZONE_SHORT = {"low": 410.0, "mid": 411.0, "high": 412.0, "display": "Entry Zone 410–412"}


def test_long_gap_above_missed():
    """LONG ARM 340–342, current 347 → MISSED (Do Not Chase)."""
    out = classify_entry_status(
        direction="LONG",
        current_price=347.0,
        entry_zone=ZONE_LONG,
        entry_price=341.0,
        stop_price=335.0,
        miss_threshold_pct=1.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_MISSED
    assert "Gapped above" in out["status_reason"]
    assert out["action_hint"] in ("Do Not Chase", "Wait VWAP Pullback")
    assert out["distance_pct"] is not None
    assert out["distance_pct"] > 0
    assert out["ideal_entry"] == 341.0
    assert out["advisory"] is True


def test_long_at_stop_invalidated():
    out = classify_entry_status(
        direction="LONG",
        current_price=334.0,
        entry_zone=ZONE_LONG,
        stop_price=335.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_INVALIDATED
    assert "Stop hit" in out["status_reason"]
    assert out["action_hint"] == "Abandon today"


def test_long_in_zone_ready():
    out = classify_entry_status(
        direction="LONG",
        current_price=341.0,
        entry_zone=ZONE_LONG,
        stop_price=335.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_READY
    assert out["action_hint"] == "Entry still valid"


def test_long_below_zone_active():
    out = classify_entry_status(
        direction="LONG",
        current_price=338.0,
        entry_zone=ZONE_LONG,
        stop_price=335.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_ACTIVE
    assert out["action_hint"] == "Entry still valid"


def test_long_slightly_above_triggered():
    # 342 * 1.01 = 345.42 miss line; 343 is through zone but not missed
    out = classify_entry_status(
        direction="LONG",
        current_price=343.0,
        entry_zone=ZONE_LONG,
        stop_price=335.0,
        miss_threshold_pct=1.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_TRIGGERED


def test_short_gap_below_missed():
    out = classify_entry_status(
        direction="SHORT",
        current_price=400.0,
        entry_zone=ZONE_SHORT,
        entry_price=411.0,
        stop_price=420.0,
        miss_threshold_pct=1.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_MISSED
    assert "Gapped below" in out["status_reason"]
    assert out["action_hint"] in ("Do Not Chase", "Wait VWAP Pullback")


def test_short_at_stop_invalidated():
    out = classify_entry_status(
        direction="SHORT",
        current_price=421.0,
        entry_zone=ZONE_SHORT,
        stop_price=420.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_INVALIDATED
    assert out["action_hint"] == "Abandon today"


def test_short_in_zone_ready():
    out = classify_entry_status(
        direction="SHORT",
        current_price=411.0,
        entry_zone=ZONE_SHORT,
        stop_price=420.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_READY


def test_short_above_zone_active():
    out = classify_entry_status(
        direction="SHORT",
        current_price=415.0,
        entry_zone=ZONE_SHORT,
        stop_price=420.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_ACTIVE


def test_premarket_planned():
    out = classify_entry_status(
        direction="LONG",
        current_price=341.0,
        entry_zone=ZONE_LONG,
        session_phase="premarket",
    )
    assert out["status"] == STATUS_PREMARKET
    assert "Ideal Entry" in out["status_reason"] or "plan" in out["status_reason"].lower()


def test_planned_helper():
    out = planned_entry_status(entry_zone=ZONE_LONG, entry_price=341.0, current_price=340.5, direction="LONG")
    assert out["status"] == STATUS_PLANNED
    assert out["ideal_entry"] == 341.0


def test_missed_new_plan_vwap_stub():
    out = classify_entry_status(
        direction="LONG",
        current_price=350.0,
        entry_zone=ZONE_LONG,
        miss_threshold_pct=1.0,
        session_phase="open",
        vwap=345.0,
    )
    assert out["status"] == STATUS_MISSED
    assert out["new_plan"] is not None
    assert out["new_plan"]["suggestion"] == "wait_pullback"
    assert out["action_hint"] == "Wait VWAP Pullback"


def test_attach_entry_status_on_slot():
    slot = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_zone": ZONE_LONG,
        "entry_price": 341.0,
        "stop_price": 335.0,
        "current_price": 347.0,
    }
    attach_entry_status(slot, session_phase="open")
    assert slot["entry_status"]["status"] == STATUS_MISSED
    assert slot["entry_status"]["current_price"] == 347.0
