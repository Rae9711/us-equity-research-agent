"""Tests for win rate calibration module."""

from __future__ import annotations

from src.research.win_rate_calibration import (
    _bucket_gap,
    _bucket_rs,
    _signature_match_score,
    _trade_outcome_from_case,
    _wilson_ci,
    build_feature_signature,
    calibrate_win_prob,
    find_similar_days,
    return_distribution_bins,
)


def test_bucket_rs_short_weak():
    assert _bucket_rs(-1.5, is_short=True) == "weak"
    assert _bucket_rs(1.0, is_short=True) == "against"


def test_bucket_gap_extended():
    assert _bucket_gap(4.5) == "extended"
    assert _bucket_gap(0.8) == "tight"


def test_signature_match_score():
    a = build_feature_signature(direction="SHORT", rs_vs_qqq=-1.2, gap_pct=0.5)
    b = dict(a)
    assert _signature_match_score(a, b) == 5
    b["gap_bucket"] = "extended"
    assert _signature_match_score(a, b) == 4


def test_trade_outcome_from_case_long_win():
    case = {
        "labels": {"trade_direction": "LONG", "primary_symbol": "TSLA"},
        "features": {"tsla_chg": 2.5},
    }
    won, ret = _trade_outcome_from_case(case)
    assert won is True
    assert ret == 2.5


def test_trade_outcome_from_case_short_loss():
    case = {
        "labels": {"trade_direction": "SHORT", "primary_symbol": "NVDA"},
        "features": {"nvda_chg": 1.2},
    }
    won, _ = _trade_outcome_from_case(case)
    assert won is False


def test_calibrate_win_prob_rules_fallback():
    sig = build_feature_signature(direction="LONG", rs_vs_qqq=1.5, gap_pct=0.5)
    out = calibrate_win_prob(sig, rules_win_prob=65.0)
    assert out["source"] == "rules_fallback"
    assert out["win_prob_source"] == "rules"
    assert out["calibrated_win_prob"] == 52.0
    assert out["disclaimer"]


def test_wilson_ci_bounds():
    low, high = _wilson_ci(10, 20)
    assert 0 <= low < 50 < high <= 100


def test_find_similar_days_empty_without_history():
    sig = build_feature_signature(direction="LONG", rs_vs_qqq=0.5)
    assert find_similar_days(sig) == []


def test_return_distribution_bins():
    bins = return_distribution_bins([-2.0, -1.0, 0.0, 1.0, 2.0], bins=5)
    assert len(bins) == 5
    assert sum(b["count"] for b in bins) == 5
