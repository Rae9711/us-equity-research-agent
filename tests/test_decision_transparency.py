"""Tests for decision transparency module."""

from __future__ import annotations

from src.research.decision_transparency import (
    build_todays_opportunities,
    compute_trade_economics,
    day_risks,
    finalize_win_prob_breakdown,
    index_rejection_reasons,
    init_win_prob_breakdown,
    why_not_alternatives,
    why_wins_today,
)
from src.research.level_sources import LevelAnchors, build_level_reasons


def test_win_prob_breakdown_sums():
    wp = init_win_prob_breakdown()
    wp["rs"] = 12
    wp["volume"] = 4
    out = finalize_win_prob_breakdown(wp, clamped=66.0)
    assert out["total"] == 66.0
    assert len(out["components"]) >= 2
    assert any(c["factor"] == "RS" for c in out["components"])


def test_trade_economics_dollar_pnl():
    econ = compute_trade_economics(
        entry_price=412.0,
        stop_price=419.8,
        target_price=406.45,
        direction="SHORT",
        trade_action="Small",
    )
    assert econ["win_usd"] > 0
    assert econ["loss_usd"] > 0
    assert econ["shares"] >= 1
    assert "Win" in econ["display"]


def test_level_reasons_short_entry():
    anchors = LevelAnchors(
        vwap=413.0,
        orb_high=419.8,
        orb_low=412.0,
        prev_high=419.5,
        prev_low=401.0,
        prior_close=419.77,
        orb_from_minute=True,
    )
    reasons = build_level_reasons(
        "SHORT",
        anchors,
        entry_px=412.0,
        entry_src="orb_low",
        stop_px=419.8,
        stop_src="orb_high",
        target_px=406.45,
        target_src="expected_low",
        current=416.0,
    )
    assert reasons["entry"]["price"] == 412.0
    assert reasons["entry"]["source"] == "orb_low"
    assert any(r["label"] == "ORB Low" for r in reasons["entry"]["reasons"])
    assert reasons["stop"]["source"] == "orb_high"


def test_why_wins_today_primary_beats_others():
    primary = {
        "symbol": "TSLA",
        "relative_strength": -1.2,
        "win_prob": 81,
        "risk_reward": 2.1,
        "upside_pct": 2.0,
        "downside_risk_pct": 1.5,
        "factor_breakdown": {"news_count": 2, "volume_signal": True},
        "edge_type": "Stock Edge",
    }
    ranked = [
        primary,
        {
            "symbol": "NVDA",
            "relative_strength": 0.5,
            "win_prob": 68,
            "risk_reward": 1.5,
            "upside_pct": 1.0,
            "downside_risk_pct": 1.0,
            "factor_breakdown": {"news_count": 1},
        },
    ]
    wins = why_wins_today(primary, ranked)
    assert len(wins) >= 1
    assert any(w["dimension"] == "win_prob" for w in wins)


def test_why_not_alternatives_multiple():
    primary = {
        "symbol": "TSLA",
        "rank": 1,
        "win_prob": 81,
        "expected_return_pct": 2.3,
        "final_score": 5.2,
        "relative_strength": -1.0,
        "risk_reward": 2.0,
        "trade_action": "Small",
    }
    ranked = [
        primary,
        {
            "symbol": "NVDA",
            "rank": 2,
            "win_prob": 68,
            "expected_return_pct": 1.8,
            "final_score": 3.1,
            "relative_strength": 0.5,
            "risk_reward": 1.4,
            "trade_action": "Pass",
        },
        {
            "symbol": "QQQ",
            "rank": 3,
            "win_prob": 55,
            "expected_return_pct": 0.8,
            "final_score": 1.2,
            "relative_strength": 0.1,
            "risk_reward": 1.0,
            "trade_action": "Pass",
        },
    ]
    alts = why_not_alternatives(primary, ranked)
    assert len(alts) == 2
    assert alts[0]["symbol"] == "NVDA"


def test_index_rejection_reasons():
    reasons = index_rejection_reasons(
        index_trade="NO TRADE",
        edges={"index_edge": {"edge": "NO", "why": "指数方向不明"}},
        p16_gate="Wait",
        ranked=[{"symbol": "QQQ", "trade_action": "Pass", "final_score": 1.0}],
        qqq_pct=-0.3,
        smh_pct=-0.5,
        total_score=-1,
    )
    assert len(reasons) >= 2
    assert any(r["factor"] == "P16 Gate" for r in reasons)


def test_todays_opportunities_direction_from_slots():
    """Null RS must not default to SHORT — use slot/bias direction."""
    opps = build_todays_opportunities(
        [
            {"symbol": "TSLA", "win_prob": 55, "trade_action": "Small", "relative_strength": None},
            {"symbol": "NVDA", "win_prob": 50, "trade_action": "Pass", "relative_strength": None},
        ],
        direction="SHORT",
        index_trade="NO TRADE",
        primary_symbol="TSLA",
        slot_directions={"TSLA": "SHORT", "NVDA": "SHORT"},
    )
    assert opps[0]["direction"] == "SHORT"
    assert "SHORT" in opps[0]["label"]
    assert opps[1]["direction"] == "Avoid"

    opps_long = build_todays_opportunities(
        [{"symbol": "TSLA", "win_prob": 70, "trade_action": "Small", "relative_strength": None}],
        direction="LONG",
        slot_directions={"TSLA": "LONG"},
    )
    assert opps_long[0]["direction"] == "LONG"


def test_day_risks_and_opportunities():
    risks = day_risks(
        primary={"symbol": "TSLA", "factor_breakdown": {"gap_pct": 4.5}},
        ranked=[],
        catalysts=[{"name": "GDP"}],
        vix_chg=6.0,
        total_score=-2,
        gap_pct_market=3.5,
    )
    assert risks["risks"]
    assert risks["donts"]
    assert "Don't Chase" in risks["donts"]

    opps = build_todays_opportunities(
        [
            {"symbol": "TSLA", "win_prob": 81, "trade_action": "Small", "relative_strength": -1},
            {"symbol": "QQQ", "win_prob": 50, "trade_action": "Pass", "relative_strength": 0},
        ],
        direction="SHORT",
        index_trade="NO TRADE",
        primary_symbol="TSLA",
        slot_directions={"TSLA": "SHORT"},
    )
    assert opps[0]["is_primary"]
    assert "★" in opps[0]["stars"]
