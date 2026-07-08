"""Tests for macro calendar — impact sorting and catalyst categories."""

from __future__ import annotations

from src.research.macro_calendar import build_macro_calendar, has_geopolitical_risk


def _raw_with_calendar_and_news() -> dict:
    return {
        "trading_date": "2026-03-18",
        "macro": {
            "economic_calendar": [
                {
                    "release_name": "FOMC Minutes",
                    "date": "2026-03-18",
                },
                {
                    "release_name": "10-Year Note Auction",
                    "date": "2026-03-18",
                },
            ]
        },
        "sector": {
            "quotes": {
                "XLE": {"change_pct": 3.2},
            }
        },
        "news": {
            "polygon": [
                {
                    "title": "Iran tensions escalate as oil prices surge",
                    "description": "Middle East conflict",
                },
                {"title": "Oil rally continues on supply fears", "description": "crude energy"},
            ],
            "rss": [],
        },
    }


def test_macro_calendar_includes_fomc_minutes_and_geo():
    cal = build_macro_calendar(_raw_with_calendar_and_news(), trading_date="2026-03-18")
    names = [c["name"] for c in cal["catalysts"]]
    assert "FOMC Minutes" in names
    assert cal["has_material_catalyst"] is True
    assert has_geopolitical_risk(cal) is True
    # Highest impact first — geo or FOMC should lead
    assert cal["catalysts"][0]["impact_score"] >= 70


def test_macro_calendar_sorts_by_impact():
    cal = build_macro_calendar(_raw_with_calendar_and_news(), trading_date="2026-03-18")
    scores = [c["impact_score"] for c in cal["catalysts"]]
    assert scores == sorted(scores, reverse=True)


def test_empty_calendar_not_material():
    raw = {"trading_date": "2026-01-10", "macro": {"economic_calendar": []}, "news": {}}
    cal = build_macro_calendar(raw, trading_date="2026-01-10")
    assert cal["has_material_catalyst"] is False
    assert cal["catalysts"] == []
