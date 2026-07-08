"""Tests for driver tree ranking."""

from __future__ import annotations

from src.research.driver_tree import build_driver_tree, primary_from_tree


def test_geo_primary_over_chip_rotation():
    macro_calendar = {
        "breaking": [
            {
                "name": "Geopolitical Risk (Iran)",
                "impact_score": 90,
                "evidence": "Iran headline",
            }
        ],
        "scheduled": [
            {"name": "FOMC Minutes", "impact_score": 82, "evidence": "2pm ET"},
        ],
        "commodity": [],
    }
    tree = build_driver_tree(
        macro_calendar=macro_calendar,
        chip_selloff=True,
        smh_pct=-3.5,
        strongest_sector="SMH",
        weakest_sector="XLK",
        qqq_pct=-0.8,
        news_signals={},
    )
    assert tree["primary"]["type"] == "Political"
    assert "Iran" in tree["primary"]["label"]
    assert tree["secondary"] is not None
    labels = {tree["secondary"]["label"], (tree.get("tertiary") or {}).get("label")}
    assert "AI Chip Rotation" in labels or "FOMC Minutes" in labels


def test_primary_from_tree_backward_compat():
    tree = {
        "primary": {"type": "Fed", "label": "FOMC Minutes", "evidence": "x", "score": 80},
    }
    dtype, driver = primary_from_tree(tree)
    assert dtype == "Fed"
    assert driver == "FOMC Minutes"
