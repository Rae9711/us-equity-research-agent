"""Tests for the calibrated-expectancy entry filter (2026-11-17).

Filter only: a candidate whose calibrated expectancy is negative is
skipped at entry time, with a recorded reason. It never touches sizing
— the allocation for a passing candidate must be identical with the
filter on or off.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.paper.account import default_account
from src.paper.calibration_filter import (
    calibration_filter,
    calibration_filter_for_signal,
    calibrated_expectancy,
    planned_risk_reward,
)
from src.paper.execution_agent import _try_entry
from src.paper.shadow_ledger import extract_morning_candidates, make_record
from src.research.edges import compute_edges, format_p13_from_edges
from src.research.trade_candidates import compute_trade_decision

MAPPING = {
    "pooled": {
        "<50": {"n": 13, "wins": 3, "calibrated_win_prob": 0.319},
        "50-60": {"n": 3, "wins": 2, "calibrated_win_prob": 0.572},
        "60-70": {"n": 5, "wins": 3, "calibrated_win_prob": 0.627},
        ">70": {"n": 13, "wins": 5, "calibrated_win_prob": 0.530},
    },
    "by_horizon": {
        "intraday": {
            "<50": {"n": 2, "wins": 1, "calibrated_win_prob": 0.45},
            "50-60": {"n": 0, "wins": 0, "calibrated_win_prob": None},
            "60-70": {"n": 1, "wins": 1, "calibrated_win_prob": 0.80},
            ">70": {"n": 7, "wins": 3, "calibrated_win_prob": 0.55},
        },
        "swing": {
            "<50": {"n": 11, "wins": 2, "calibrated_win_prob": 0.28},
            "50-60": {"n": 3, "wins": 2, "calibrated_win_prob": 0.57},
            "60-70": {"n": 4, "wins": 2, "calibrated_win_prob": 0.60},
            ">70": {"n": 3, "wins": 0, "calibrated_win_prob": 0.35},
        },
        "unknown": {},
    },
}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _write_summary(root, mapping=None):
    paper = root / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "shadow_summary.json").write_text(
        json.dumps(
            {
                "recalibration": (mapping or MAPPING)["pooled"],
                "recalibration_by_horizon": (mapping or MAPPING)["by_horizon"],
            }
        ),
        encoding="utf-8",
    )


def _picked(symbol, *, price=200.0, stop=195.0, target=210.0,
            direction="LONG", win_prob=60, risk_reward=None, book="intraday"):
    sig = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": price,
        "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
        "stop_price": stop,
        "target_price": target,
        "horizon": "Swing" if book == "swing" else "Intraday",
        "source": "morning_primary",
        "win_prob": win_prob,
    }
    if risk_reward is not None:
        sig["risk_reward"] = risk_reward
    return {
        "action": "enter",
        "signal": sig,
        "quote": price,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }


# --- pure math -----------------------------------------------------------

def test_negative_expectancy_skips():
    # <50 bucket calibrates to 0.319; at R:R 1.33 expectancy is -0.26R.
    out = calibration_filter(43.0, 1.33, mapping=MAPPING)
    assert out["filter"] == "skip"
    assert out["calibrated_win_prob"] == 0.319
    assert out["calibrated_expectancy_r"] == pytest.approx(0.319 * 1.33 - 0.681, abs=0.002)
    assert "校准期望值" in (out["reason"] or "")


def test_positive_expectancy_passes():
    out = calibration_filter(75.0, 1.33, mapping=MAPPING)
    assert out["filter"] == "pass"
    assert out["calibrated_expectancy_r"] == pytest.approx(0.53 * 1.33 - 0.47, abs=0.002)


def test_no_data_passes():
    assert calibration_filter(43.0, 1.33, mapping={})["filter"] == "no-data"
    assert calibration_filter(None, 1.33, mapping=MAPPING)["filter"] == "no-data"
    assert calibration_filter(43.0, 0.0, mapping=MAPPING)["filter"] == "no-data"
    assert calibration_filter(43.0, None, mapping=MAPPING)["filter"] == "no-data"


def test_by_horizon_preferred_and_fallback():
    # Intraday <50 has n=2 evidence (0.45) -> used over pooled 0.319.
    out = calibrated_expectancy(43.0, 1.33, horizon="intraday", mapping=MAPPING)
    assert out["source"] == "by-horizon"
    assert out["calibrated_win_prob"] == 0.45
    # Intraday 50-60 has n=0 -> falls back to pooled 0.572.
    out = calibrated_expectancy(55.0, 1.33, horizon="intraday", mapping=MAPPING)
    assert out["source"] == "pooled"
    assert out["calibrated_win_prob"] == 0.572
    # No horizon -> pooled.
    out = calibrated_expectancy(43.0, 1.33, mapping=MAPPING)
    assert out["source"] == "pooled"


def test_rr_from_levels():
    assert planned_risk_reward(100.0, 95.0, 110.0, "LONG") == pytest.approx(2.0)
    assert planned_risk_reward(100.0, 105.0, 90.0, "SHORT") == pytest.approx(2.0)
    assert planned_risk_reward(100.0, 100.0, 110.0, "LONG") is None
    sig = {"win_prob": 43.0, "entry_price": 200.0, "stop_price": 195.0,
           "target_price": 210.0, "direction": "LONG"}
    out = calibration_filter_for_signal(sig, mapping=MAPPING)
    assert out["filter"] == "skip"  # RR 2.0, p 0.319 -> -0.04R


# --- entry path ----------------------------------------------------------

def test_try_entry_skips_negative_calibrated(data_root):
    # Swing <50 bucket: n=11, calibrated 0.28 -> E = -0.35R at R:R 1.33.
    _write_summary(data_root)
    out = _try_entry(
        default_account(), "swing",
        _picked("AAPL", win_prob=43, risk_reward=1.33, book="swing"),
        "2026-11-12",
    )
    assert out["action"] == "SKIP"
    assert out.get("calibration_filtered") is True
    assert "校准期望值" in (out["reason"] or "")
    assert out["calibration"]["calibrated_expectancy_r"] < 0
    assert out["calibration"]["source"] == "by-horizon"


def test_try_entry_skips_negative_calibrated_intraday(data_root):
    # Intraday <50 bucket: n=2, calibrated 0.45. Levels give R:R 1.0
    # (entry 200 / stop 195 / target 205), so expectancy is -0.10R.
    _write_summary(data_root)
    out = _try_entry(
        default_account(), "intraday",
        _picked("AAPL", price=200.0, stop=195.0, target=205.0, win_prob=49),
        "2026-11-12",
    )
    assert out["action"] == "SKIP"
    assert out.get("calibration_filtered") is True


def test_try_entry_positive_unaffected_and_allocation_unchanged(data_root):
    picked = _picked("MSFT", price=500.0, stop=490.0, target=520.0,
                     win_prob=75, risk_reward=2.0)
    out_with_map = _try_entry(default_account(), "intraday", picked, "2026-11-12")
    assert out_with_map["action"] == "ENTRY"
    _write_summary(data_root)
    out_with_filter = _try_entry(default_account(), "intraday", picked, "2026-11-12")
    assert out_with_filter["action"] == "ENTRY"
    # Allocation identical with the filter on vs off: sizing untouched.
    assert out_with_filter["allocation"] == out_with_map["allocation"]


def test_try_entry_no_map_passes(data_root):
    # No shadow summary on disk -> no-data -> entry proceeds.
    # (win_prob 60: above the allocator's 48% hold-cash floor.)
    out = _try_entry(
        default_account(), "intraday",
        _picked("AAPL", win_prob=60, risk_reward=1.33),
        "2026-11-12",
    )
    assert out["action"] == "ENTRY"


# --- research-time attach + shadow ledger --------------------------------

def _bullish_raw():
    return {
        "trading_date": "2026-07-06",
        "market": {"quotes": {
            "QQQ": {"close": 520.0, "prev_close": 518.0, "high": 521.0, "low": 517.0, "change_pct": 0.4, "session_type": "premarket"},
            "SPY": {"close": 600.0, "prev_close": 599.0, "change_pct": 0.2},
            "TQQQ": {"close": 80.0, "prev_close": 79.0, "change_pct": 1.2},
            "^VIX": {"close": 14.0, "prev_close": 15.0, "change_pct": -6.0},
        }},
        "sector": {"quotes": {
            "SMH": {"close": 280.0, "prev_close": 275.0, "high": 282.0, "low": 274.0, "change_pct": 1.8},
        }},
        "stocks": {"quotes": {
            "NVDA": {"close": 140.0, "prev_close": 138.0, "high": 141.0, "low": 137.0, "change_pct": 1.4, "session_type": "premarket", "volume": 1000},
            "TSLA": {"close": 255.0, "prev_close": 240.0, "open": 246.0, "change_pct": 2.5, "volume": 5000, "session_type": "premarket", "high": 258, "low": 238},
        }},
        "news": {"polygon": [{"ticker": "TSLA", "title": "TSLA delivery beat"}], "rss": []},
        "options": {"avg_implied_volatility": 0.20},
        "macro": {"economic_calendar": []},
    }


def _mock_obs(symbol, raw, prior_raw, trading_day, **kwargs):
    section = "market" if symbol in ("QQQ", "SPY", "TQQQ") else (
        "sector" if symbol == "SMH" else "stocks"
    )
    q = ((raw.get(section) or {}).get("quotes") or {}).get(symbol) or {}
    if not q:
        return {"ticker": symbol, "error": "no quote"}
    close = float(q.get("close") or 100)
    prev = float(q.get("prev_close") or close * 0.99)
    open_px = float(q.get("open") or prev * 1.01)
    return {
        "ticker": symbol, "last": close, "close": close, "open": open_px,
        "prev_close": prev, "change_pct": q.get("change_pct"),
        "gap_pct": round((open_px - prev) / prev * 100, 2) if prev else 0,
    }


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
@patch("src.research.trade_candidates._fetch_swing_context",
       return_value={"atr": 3.0, "atr_source": "proxy_test", "week_high": None, "week_low": None, "prior_5d_chg_pct": 3.0})
def test_compute_trade_decision_attaches_calibrated_fields(_swing, _prior, _obs, data_root):
    _write_summary(data_root)
    _prior.return_value = {"stocks": {"quotes": {"TSLA": {"change_pct": 6.0, "close": 240.0}}}}
    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias", "total": 3, "driver_type": "Momentum",
        "daily_driver": "AI Momentum", "catalysts_today": [],
        "edges": compute_edges(raw, catalysts_today=[], qqq_pct=0.4, smh_pct=1.8, spy_pct=0.2, sym_pcts={"TSLA": 2.5}, driver_type="Momentum"),
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "Yes", "buy_call": "Yes", "buy_put": "No"},
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": format_p13_from_edges(rule_bundle["edges"]),
        "P16": {"judgment": "计划 Trade", "one_liner": "买 TSLA"},
    }
    result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    for row in result["trade_candidates"]:
        assert "calibrated_win_prob" in row
        assert "calibrated_expectancy_r" in row
        assert row["calibration_filter"] in ("pass", "skip", "no-data")
    # The top slot carries the fields too (shadow-ledger visibility).
    primary = result["best_trades"]["primary"]
    assert primary is not None
    assert "calibrated_expectancy_r" in primary
    assert "calibration_filter" in primary


def test_shadow_ledger_carries_calibrated_fields():
    morning = {
        "trade_candidates": [
            {"symbol": "AAPL", "direction": "LONG", "win_prob": 43.0,
             "current_price": 200.0, "expected_high": 210.0, "expected_low": 195.0,
             "risk_reward": 1.33, "trade_action": "Small",
             "calibrated_win_prob": 0.319, "calibrated_expectancy_r": -0.257,
             "calibration_filter": "skip"},
        ],
    }
    cands = extract_morning_candidates(morning)
    assert len(cands) == 1
    rec = make_record("2026-11-17", "morning", cands[0])
    assert rec["calibrated_expectancy_r"] == pytest.approx(-0.257)
    assert rec["calibration_filter"] == "skip"


# --- verdict durability (2026-11-18) -------------------------------------
# Live finding: after two live days (11-17, 11-18) the ledger held 300
# decisions and ZERO carried any calibration key — a pass verdict was
# discarded in _try_entry, and record_tick_decision never persisted the
# field even on the skip path. The watch could not tell "filter passed"
# from "filter never ran". The verdict is now recorded on every
# entry-path decision and persisted into the decision rows.

def test_try_entry_records_pass_verdict(data_root):
    # Intraday >70 bucket: n=7, calibrated 0.55 (by-horizon). Levels give
    # R:R 2.0 (500/490/520) -> expectancy +0.65R, passes.
    _write_summary(data_root)
    out = _try_entry(
        default_account(), "intraday",
        _picked("MSFT", price=500.0, stop=490.0, target=520.0, win_prob=75),
        "2026-11-18",
    )
    assert out["action"] == "ENTRY"
    cal = out["calibration"]
    assert cal["filter"] == "pass"
    assert cal["bucket"] == ">70"
    assert cal["calibrated_win_prob"] == 0.55
    assert cal["calibrated_expectancy_r"] == pytest.approx(0.65)
    assert cal["source"] == "by-horizon"
    assert cal["n"] == 7
    assert out.get("calibration_filtered") is None


def test_try_entry_records_no_data_verdict(data_root):
    # No shadow summary on disk -> no-data verdict, still recorded.
    out = _try_entry(
        default_account(), "intraday",
        _picked("AAPL", win_prob=60, risk_reward=1.33),
        "2026-11-18",
    )
    assert out["action"] == "ENTRY"
    assert out["calibration"]["filter"] == "no-data"
    assert out["calibration"]["calibrated_win_prob"] is None


def test_record_tick_decision_persists_calibration():
    from src.paper.journal import record_tick_decision

    account = default_account()
    record_tick_decision(account, {
        "trading_date": "2026-11-18",
        "books": [
            {
                "action": "ENTRY", "reason": "test entry", "book": "intraday",
                "book_zh": "短线",
                "signal": {"symbol": "AVGO", "horizon": "Intraday"},
                "quote": 350.0, "quote_source": "test",
                "entry_status": "READY", "trade": {"pnl": None},
                "allocation": None,
                "calibration": {"filter": "pass", "bucket": ">70",
                                "calibrated_win_prob": 0.55,
                                "calibrated_expectancy_r": 1.2,
                                "source": "by-horizon", "n": 7,
                                "horizon_used": "intraday", "reason": None},
                "calibration_filtered": None,
            },
            {
                "action": "SKIP", "reason": "test skip", "book": "swing",
                "book_zh": "长线",
                "signal": {"symbol": "AAPL", "horizon": "Swing"},
                "quote": 200.0, "quote_source": "test",
                "entry_status": "READY", "trade": None,
                "allocation": None,
                "calibration": {"filter": "skip", "bucket": "<50",
                                "calibrated_win_prob": 0.28,
                                "calibrated_expectancy_r": -0.35,
                                "source": "by-horizon", "n": 11,
                                "horizon_used": "swing",
                                "reason": "校准期望值 -0.35R 为负"},
                "calibration_filtered": True,
            },
        ],
    })
    rows = account["decisions"]
    assert len(rows) == 2
    assert rows[0]["calibration"]["filter"] == "pass"
    assert rows[0]["calibration"]["calibrated_expectancy_r"] == pytest.approx(1.2)
    assert rows[0].get("calibration_filtered") is None
    assert rows[1]["calibration"]["filter"] == "skip"
    assert rows[1]["calibration_filtered"] is True
    # No-books fallback: keys present, None when no entry was attempted.
    record_tick_decision(account, {
        "trading_date": "2026-11-18", "action": "SKIP", "reason": "blocked",
        "signal": None, "trade": None, "quote": None, "quote_source": None,
        "entry_status": None, "allocation": None,
    })
    assert account["decisions"][2]["calibration"] is None
