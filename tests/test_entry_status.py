"""Tests for Ideal Entry vs live Entry Status classification.

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

from src.paper.account import default_account
from src.paper.execution_agent import _try_entry
from src.paper.signals import normalize_slot
from src.research.entry_status import (
    ACTION_NEW_SETUP,
    ACTION_WAIT_VWAP,
    ACTION_WATCH_ALT,
    ALT_ACTION_SECONDARY,
    ALT_ACTION_WATCH,
    STATUS_ACTIVE,
    STATUS_INVALIDATED,
    STATUS_MISSED,
    STATUS_PLANNED,
    STATUS_PREMARKET,
    STATUS_READY,
    STATUS_TRIGGERED,
    attach_entry_status,
    build_session_trade_update,
    build_trade_reeval,
    classify_entry_status,
    planned_entry_status,
    remaining_er_pct,
)


ZONE_LONG = {"low": 340.0, "mid": 341.0, "high": 342.0, "display": "Entry Zone 340–342"}
ZONE_SHORT = {"low": 410.0, "mid": 411.0, "high": 412.0, "display": "Entry Zone 410–412"}
ZONE_MU_SHORT = {"low": 910.0, "mid": 913.0, "high": 916.0, "display": "Entry Zone 910–916"}


def test_long_gap_above_missed():
    """LONG ARM 340–342, current 347 → MISSED (Do Not Chase)."""
    out = classify_entry_status(
        direction="LONG",
        current_price=347.0,
        entry_zone=ZONE_LONG,
        entry_price=341.0,
        stop_price=335.0,
        target_price=354.0,
        miss_threshold_pct=1.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_MISSED
    assert "Gapped above" in out["status_reason"]
    assert out["action_hint"] in ("Do Not Chase", "Wait VWAP Pullback", ACTION_WAIT_VWAP)
    assert out["distance_pct"] is not None
    assert out["distance_pct"] > 0
    assert out["ideal_entry"] == 341.0
    assert out["advisory"] is True
    assert out["remaining_er_pct"] is not None


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
    assert out["action_hint"] in ("Abandon today", "Do Not Chase")


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
    assert out["action_hint"] in ("Do Not Chase", "Wait VWAP Pullback", ACTION_NEW_SETUP)


def test_short_at_stop_invalidated():
    out = classify_entry_status(
        direction="SHORT",
        current_price=421.0,
        entry_zone=ZONE_SHORT,
        stop_price=420.0,
        session_phase="open",
    )
    assert out["status"] == STATUS_INVALIDATED
    assert out["action_hint"] in ("Abandon today", "Do Not Chase")


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
    """Slightly above zone but within miss band → still ACTIVE."""
    out = classify_entry_status(
        direction="SHORT",
        current_price=415.0,
        entry_zone=ZONE_SHORT,
        stop_price=420.0,
        miss_threshold_pct=1.0,
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
    out = planned_entry_status(
        entry_zone=ZONE_LONG, entry_price=341.0, current_price=340.5, direction="LONG"
    )
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
        target_price=360.0,
    )
    assert out["status"] == STATUS_MISSED
    assert out["new_plan"] is not None
    assert out["alternate_entry"] is not None
    assert out["new_plan"]["suggestion"] in ("wait_pullback", "do_not_chase")
    assert out["action_hint"] in ("Wait VWAP Pullback", ACTION_WAIT_VWAP, "Do Not Chase")
    assert out["new_plan"]["alt_entry_zone"] is not None


def test_attach_entry_status_on_slot():
    slot = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_zone": ZONE_LONG,
        "entry_price": 341.0,
        "stop_price": 335.0,
        "target_price": 354.0,
        "current_price": 347.0,
    }
    attach_entry_status(slot, session_phase="open")
    assert slot["entry_status"]["status"] == STATUS_MISSED
    assert slot["entry_status"]["current_price"] == 347.0


def test_remaining_er_pct_short_current_to_target():
    """933 → 899 ≈ 3.64% remaining ER for SHORT (dynamic, not Morning entry ER)."""
    er = remaining_er_pct(direction="SHORT", current_price=933.0, target_price=899.29)
    assert er is not None
    assert 3.5 <= er <= 3.8


def test_mu_short_extended_above_missed_alt_stub():
    """MU SHORT Ideal 910–916, current 933 → MISSED + alt retest + remaining ER."""
    out = classify_entry_status(
        direction="SHORT",
        current_price=933.0,
        entry_zone=ZONE_MU_SHORT,
        entry_price=913.0,
        stop_price=954.0,
        target_price=899.29,
        miss_threshold_pct=1.0,
        session_phase="open",
        vwap=928.0,
        orb_high=920.0,
    )
    assert out["status"] == STATUS_MISSED
    assert "Extended above" in out["status_reason"]
    assert out["remaining_er_pct"] is not None
    assert out["live_expected_return_pct"] == out["remaining_er_pct"]
    assert 3.5 <= out["remaining_er_pct"] <= 3.8
    plan = out["alternate_entry"] or out["new_plan"]
    assert plan is not None
    assert plan["suggested_action"] in (
        ACTION_WATCH_ALT,
        ACTION_NEW_SETUP,
        "Wait VWAP Pullback",
        "Do Not Chase",
    )
    assert plan["alt_entry_zone"] is not None
    assert plan["alt_entry"] is not None
    assert plan["alt_stop"] is not None and plan["alt_stop"] > plan["alt_entry"]
    assert plan["alt_target"] is not None and plan["alt_target"] < plan["alt_entry"]
    assert plan["alt_er_pct"] is not None
    assert plan["remaining_er_pct"] is not None
    assert plan["live_expected_return_pct"] == plan["remaining_er_pct"]
    assert plan.get("alt_reason") in (
        "VWAP reject",
        "Failed breakdown retest",
        "Do not chase Ideal Entry",
        "VWAP retest",
    )
    assert plan.get("alt_action") in (ALT_ACTION_WATCH, ALT_ACTION_SECONDARY)
    # Weak R:R (< 1.5) → Watch only / Do Not Chase; stronger → Secondary setup
    assert out["action_hint"] in (ACTION_WATCH_ALT, "Do Not Chase", ACTION_WAIT_VWAP)


def test_build_trade_reeval_with_morning_mock(monkeypatch):
    morning = {
        "bias": "Bearish",
        "total_score": -1,
        "driver_type": "Sector",
        "daily_driver": "Semi",
        "parts": {"P16": {"judgment": "No Trade — wait", "one_liner": "no trade"}},
        "best_trades": {
            "primary": {
                "symbol": "MU",
                "direction": "SHORT",
                "entry_zone": ZONE_MU_SHORT,
                "entry_price": 913.0,
                "stop_price": 954.0,
                "target_price": 899.29,
                "expected_return_pct": 1.5,
                "level_anchors": {"vwap": 928.0, "orb_high": 920.0},
            }
        },
        "top_trades": [
            {
                "symbol": "MU",
                "direction": "SHORT",
                "rank": 1,
                "entry_zone": ZONE_MU_SHORT,
                "entry_price": 913.0,
                "stop_price": 954.0,
                "target_price": 899.29,
                "final_score": 5.0,
                "level_anchors": {"vwap": 928.0},
            }
        ],
        "transparency": {},
        "edges": {},
        "macro_calendar": {},
        "driver_tree": {},
    }

    monkeypatch.setattr(
        "src.research.entry_status.resolve_symbol_last",
        lambda symbol, trading_date, raw=None: 933.0 if symbol == "MU" else 100.0,
    )

    def _fake_compute(raw, *, rule_bundle, parts, edges=None, as_of_et=None):
        primary = {
            "symbol": "NVDA",
            "direction": "SHORT",
            "entry_price": 120.0,
            "target_price": 115.0,
            "stop_price": 125.0,
            "final_score": 9.0,
            "expected_return_pct": 4.0,
            "trade_action": "Small",
            "entry_zone": {"low": 119.0, "mid": 120.0, "high": 121.0, "display": "119–121"},
        }
        return {
            "best_trades": {"primary": primary, "p16_gate": "No Trade"},
            "best_opportunity": {"p16_gate": "No Trade", "symbol": "NVDA"},
            "top_trades": [
                {**primary, "rank": 1, "win_prob": 60, "why_factors": ["rs"]},
                {
                    "symbol": "MU",
                    "direction": "SHORT",
                    "rank": 2,
                    "final_score": 5.0,
                    "entry_price": 913.0,
                    "target_price": 899.29,
                    "stop_price": 954.0,
                    "win_prob": 55,
                    "why_factors": [],
                    "entry_zone": ZONE_MU_SHORT,
                },
            ],
            "trade_candidates": [],
            "transparency": {"todays_opportunities": [{"symbol": "NVDA", "label": "NVDA"}]},
        }

    monkeypatch.setattr(
        "src.research.trade_candidates.compute_trade_decision",
        _fake_compute,
    )

    out = build_session_trade_update(
        morning,
        "2026-07-13",
        raw={"trading_date": "2026-07-13", "stocks": {"quotes": {}}},
        session_phase="open",
    )
    assert out is not None
    assert out["changed"] is True
    assert out["primary"]["symbol"] == "NVDA"
    assert "why_changed" in out
    assert "compared_to_morning_primary" in out
    assert out["compared_to_morning_primary"]["symbol"] == "MU"
    note = (out.get("why_changed") or out.get("note") or "").lower()
    assert "nvda" in note or "missed" in note or "changed" in note
    assert out["morning_primary_live"] is not None
    assert out["morning_primary_live"]["entry_status"]["status"] == STATUS_MISSED
    assert out["top_trades"]
    assert out["p16_gate"] == "No Trade"
    # alias still works
    assert build_trade_reeval is build_session_trade_update


# ---------------------------------------------------------------------------
# Regression (2026-11-17): session primary must carry win_prob / risk_reward
# ---------------------------------------------------------------------------
# Real case: the 2026-11-17 09:56 MU SHORT session primary carried
# win_prob 36 in the morning slot, but build_session_trade_update rebuilt
# it through whitelists that dropped win_prob (and risk_reward), so the
# tick's allocation saw win_prob=None -> "胜率 0% 不足 48%" and held cash
# for the wrong reason. The floor itself (48%) is by design; only the
# pass-through was the bug.


def _mu_short_session_update(monkeypatch):
    """Morning MU SHORT (win_prob 36) -> session re-rank, #1 unchanged."""
    morning = {
        "bias": "Bearish",
        "total_score": -1,
        "driver_type": "Sector",
        "daily_driver": "Semi",
        "parts": {"P16": {"judgment": "No Trade — wait", "one_liner": "no trade"}},
        "best_trades": {
            "primary": {
                "symbol": "MU",
                "direction": "SHORT",
                "entry_zone": ZONE_MU_SHORT,
                "entry_price": 913.0,
                "stop_price": 954.0,
                "target_price": 899.29,
                "expected_return_pct": 1.5,
                "win_prob": 36,
                "risk_reward": 1.34,
                "level_anchors": {"vwap": 928.0, "orb_high": 920.0},
            }
        },
        "top_trades": [
            {
                "symbol": "MU",
                "direction": "SHORT",
                "rank": 1,
                "entry_zone": ZONE_MU_SHORT,
                "entry_price": 913.0,
                "stop_price": 954.0,
                "target_price": 899.29,
                "final_score": 5.0,
                "win_prob": 36,
                "risk_reward": 1.34,
                "level_anchors": {"vwap": 928.0},
            }
        ],
        "transparency": {},
        "edges": {},
        "macro_calendar": {},
        "driver_tree": {},
    }

    monkeypatch.setattr(
        "src.research.entry_status.resolve_symbol_last",
        lambda symbol, trading_date, raw=None: 913.0,
    )

    def _fake_compute(raw, *, rule_bundle, parts, edges=None, as_of_et=None):
        primary = {
            "symbol": "MU",
            "direction": "SHORT",
            "entry_price": 913.0,
            "target_price": 899.29,
            "stop_price": 954.0,
            "final_score": 5.0,
            "expected_return_pct": 1.5,
            "trade_action": "Small",
            "entry_zone": ZONE_MU_SHORT,
            "win_prob": 36,
            "risk_reward": 1.34,
        }
        return {
            "best_trades": {"primary": primary, "p16_gate": "No Trade"},
            "best_opportunity": {"p16_gate": "No Trade", "symbol": "MU"},
            "top_trades": [{**primary, "rank": 1, "why_factors": ["rs"]}],
            "trade_candidates": [],
            "transparency": {"todays_opportunities": [{"symbol": "MU", "label": "MU"}]},
        }

    monkeypatch.setattr(
        "src.research.trade_candidates.compute_trade_decision",
        _fake_compute,
    )

    out = build_session_trade_update(
        morning,
        "2026-11-17",
        raw={"trading_date": "2026-11-17", "stocks": {"quotes": {}}},
        session_phase="open",
    )
    assert out is not None
    return out


def test_session_trade_update_carries_win_prob(monkeypatch):
    out = _mu_short_session_update(monkeypatch)
    assert out["changed"] is False  # same #1, only the values matter here
    # primary_out whitelist (the live primary the tick trades on).
    assert out["primary"]["symbol"] == "MU"
    assert out["primary"]["win_prob"] == 36
    assert out["primary"]["risk_reward"] == 1.34
    # re_ranked top_trades whitelist.
    assert out["top_trades"][0]["win_prob"] == 36
    assert out["top_trades"][0]["risk_reward"] == 1.34
    # enrich_primary_from_morning fallback (morning primary, live status).
    assert out["morning_primary_live"]["win_prob"] == 36
    assert out["morning_primary_live"]["risk_reward"] == 1.34
    # refreshed_top whitelist (morning top_trades, live status).
    assert out["morning_top_refreshed"][0]["win_prob"] == 36
    assert out["morning_top_refreshed"][0]["risk_reward"] == 1.34


def test_try_entry_sees_session_primary_win_prob(monkeypatch, tmp_path):
    # Empty DATA_ROOT: no shadow summary on disk -> calibration filter
    # has no map and passes, so the entry reaches allocation.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    out = _mu_short_session_update(monkeypatch)
    sig = normalize_slot(
        out["primary"], source="session_trade_update", horizon="Intraday"
    )
    assert sig is not None
    assert sig["win_prob"] == 36
    assert sig["risk_reward"] == 1.34
    picked = {
        "action": "enter",
        "signal": sig,
        "quote": 913.0,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }
    result = _try_entry(default_account(), "intraday", picked, "2026-11-17")
    assert result["allocation"] is not None
    # Allocation sees the signal's real win_prob (36), not None/0.
    assert result["allocation"]["win_prob"] == 36
    # 36 < 48 floor holds cash -- by design -- but for the right reason.
    assert result["action"] == "SKIP"
    assert "胜率 36%" in (result["reason"] or "")
    assert "胜率 0%" not in (result["reason"] or "")
