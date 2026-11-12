"""Tests for the shadow ledger (src/paper/shadow_ledger.py).

All fixtures are synthetic -- no live quotes, no state files.
"""

from __future__ import annotations

from src.paper.shadow_ledger import (
    MAX_HOLD_DAYS,
    compute_summary,
    derive_levels,
    evaluate_day,
    evaluate_entry,
    extract_morning_candidates,
    extract_step3_candidates,
    make_record,
    r_multiple,
    update_record,
    win_prob_bucket,
)


def _bar(date, o, h, l, c):
    return {"date": date, "open": o, "high": h, "low": l, "close": c}


def _morning_doc():
    return {
        "trading_date": "2026-11-12",
        "best_opportunity": {
            "symbol": "AVGO", "direction": "LONG", "horizon": "Intraday",
            "win_prob": 54.0, "entry_price": 347.8, "stop_price": 347.6,
            "target_price": 354.3, "trade_action": "Small",
            "entry_zone": {"low": 347.6, "high": 348.0, "mid": 347.8},
        },
        "swing_trade": {
            "symbol": "AMD", "direction": "LONG", "horizon": "Swing",
            "win_prob": 65.0, "entry_price": 557.63, "stop_price": 493.4,
            "target_price": 643.27, "trade_action": None,
            "entry_zone": {"low": 536.22, "high": 559.31, "mid": 557.63},
        },
        "best_trades": {
            "primary": {
                "symbol": "AVGO", "direction": "LONG", "horizon": "Intraday",
                "win_prob": 54.0, "entry_price": 347.8, "stop_price": 347.6,
                "target_price": 354.3, "trade_action": "Small",
            },
            "swing": {
                "symbol": "AMD", "direction": "LONG", "horizon": "Swing",
                "win_prob": 65.0, "entry_price": 557.63, "stop_price": 493.4,
                "target_price": 643.27,
            },
        },
        "top_trades": [
            {"symbol": "AVGO", "direction": "LONG", "horizon": "Intraday",
             "win_prob": 54.0, "entry_price": 347.8, "stop_price": 347.6,
             "target_price": 354.3, "trade_action": "Small"},
        ],
        "trade_candidates": [
            {"symbol": "AMD", "direction": "LONG", "win_prob": 65.0,
             "current_price": 559.31, "expected_high": 578.64,
             "expected_low": 539.98, "expected_return_pct": 3.46,
             "risk_reward": 1.0, "trade_action": "Pass"},
            {"symbol": "TQQQ", "direction": "LONG", "win_prob": 66.0,
             "current_price": 100.0, "expected_high": 103.0,
             "expected_low": 97.0, "expected_return_pct": 3.0,
             "risk_reward": 2.0, "trade_action": "Pass"},
            {"symbol": "META", "direction": "SHORT", "win_prob": 44.0,
             "current_price": 200.0, "expected_high": 206.0,
             "expected_low": 194.0, "expected_return_pct": 3.0,
             "risk_reward": 1.0, "trade_action": "Pass"},
        ],
    }


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

def test_derive_levels_long():
    entry, stop, target = derive_levels("LONG", 100.0, 103.0, 97.0, 2.0)
    assert entry == 100.0 and target == 103.0
    assert stop == 98.5  # 3.0 move / rr 2.0


def test_derive_levels_short():
    entry, stop, target = derive_levels("SHORT", 200.0, 206.0, 194.0, 1.0)
    assert entry == 200.0 and target == 194.0 and stop == 206.0


def test_derive_levels_invalid():
    assert derive_levels("LONG", 100.0, 99.0, 97.0, 1.0) is None  # target below entry
    assert derive_levels("LONG", None, 103.0, 97.0, 1.0) is None


def test_r_multiple():
    assert r_multiple("LONG", 100.0, 90.0, 120.0) == 2.0
    assert r_multiple("LONG", 100.0, 90.0, 90.0) == -1.0
    assert r_multiple("SHORT", 100.0, 110.0, 80.0) == 2.0
    assert r_multiple("SHORT", 100.0, 110.0, 110.0) == -1.0
    assert r_multiple("LONG", 100.0, 100.0, 120.0) is None  # degenerate risk


# ---------------------------------------------------------------------------
# Extraction / merging
# ---------------------------------------------------------------------------

def test_extract_morning_merges_plans_and_candidates():
    cands = extract_morning_candidates(_morning_doc())
    by_sym = {(c["symbol"], c["direction"]): c for c in cands}
    # AVGO: best_opportunity == best_trades.primary == top_trades[0] -> one row
    assert ("AVGO", "LONG") in by_sym
    avgo = by_sym[("AVGO", "LONG")]
    assert avgo["levels_source"] == "plan"
    assert avgo["entry"] == 347.8 and avgo["stop"] == 347.6 and avgo["target"] == 354.3
    assert avgo["horizon"] == "intraday"
    assert len(avgo["sources"]) >= 3  # merged plan slots
    # AMD: swing plan wins over the plain candidate (same win_prob)
    amd = by_sym[("AMD", "LONG")]
    assert amd["levels_source"] == "plan"
    assert amd["entry"] == 557.63 and amd["stop"] == 493.4 and amd["target"] == 643.27
    assert amd["horizon"] == "swing"
    assert "trade_candidates" in amd["sources"]
    # TQQQ: no plan -> derived row, unknown horizon
    tqqq = by_sym[("TQQQ", "LONG")]
    assert tqqq["levels_source"] == "derived"
    assert tqqq["horizon"] == "unknown"
    assert tqqq["entry"] == 100.0 and tqqq["target"] == 103.0 and tqqq["stop"] == 98.5
    # META SHORT derived
    meta = by_sym[("META", "SHORT")]
    assert meta["entry"] == 200.0 and meta["target"] == 194.0 and meta["stop"] == 206.0
    assert len(cands) == 4  # no double counting


def test_extract_step3_dedupes_and_marks_intraday():
    doc = {"trade_reeval": {"top_trades": [
        {"symbol": "AMD", "direction": "LONG", "win_prob": 77.0,
         "entry_price": 566.83, "stop_price": 554.9, "target_price": 595.68},
        {"symbol": "AMD", "direction": "LONG", "win_prob": 77.0,
         "entry_price": 566.83, "stop_price": 554.9, "target_price": 595.68},
    ]}}
    cands = extract_step3_candidates(doc)
    assert len(cands) == 1
    assert cands[0]["horizon"] == "intraday"
    assert cands[0]["horizon_source"] == "inferred-step3-intraday"
    assert extract_step3_candidates({}) == []
    assert extract_step3_candidates({"primary_entry_status": {"symbol": "X"}}) == []


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _rec(**kw):
    base = {"symbol": "AAA", "direction": "LONG", "horizon": "intraday",
            "win_prob": 60.0, "entry": 100.0, "stop": 90.0, "target": 120.0,
            "entry_zone": None, "levels_source": "plan", "sources": ["t"],
            "trade_action": None}
    base.update(kw)
    return make_record("2026-11-12", "morning", base)


def test_entry_long_open_below_entry_fills_at_open():
    rec = _rec()
    outcome, bar, _ = evaluate_entry(rec, [_bar("2026-11-13", 99.0, 101.0, 98.0, 100.5)])
    assert outcome == "entered" and bar["open"] == 99.0


def test_entry_long_gap_up_skips():
    rec = _rec()
    outcome, _, reason = evaluate_entry(rec, [_bar("2026-11-13", 105.0, 106.0, 104.0, 105.0)])
    assert outcome == "skipped" and reason == "open-missed"


def test_entry_short_gap_down_skips_and_zone_entry():
    rec = _rec(direction="SHORT", entry=100.0, stop=110.0, target=80.0)
    outcome, _, reason = evaluate_entry(rec, [_bar("2026-11-13", 95.0, 96.0, 94.0, 95.0)])
    assert outcome == "skipped" and reason == "open-missed"
    rec2 = _rec(entry_zone={"low": 99.0, "high": 101.0})
    outcome, bar, _ = evaluate_entry(rec2, [_bar("2026-11-13", 100.5, 101.0, 100.0, 100.5)])
    assert outcome == "entered" and bar["open"] == 100.5


def test_entry_pending_without_future_bar():
    rec = _rec()
    outcome, bar, _ = evaluate_entry(rec, [_bar("2026-11-12", 99.0, 101.0, 98.0, 100.5)])
    assert outcome == "pending" and bar is None


def test_entry_open_below_stop_is_skipped_not_instant_stop():
    # LONG, open gaps below the stop: no trade, not an instant stop-out.
    rec = _rec()
    outcome, _, reason = evaluate_entry(rec, [_bar("2026-11-13", 85.0, 86.0, 84.0, 85.0)])
    assert outcome == "skipped" and reason == "open-below-stop"
    update_record(rec, [_bar("2026-11-13", 85.0, 86.0, 84.0, 85.0)], "2026-11-13")
    assert rec["status"] == "skipped" and rec["skip_reason"] == "open-below-stop"
    assert rec["r_multiple"] is None
    # SHORT mirror: open gaps above the stop.
    rec = _rec(direction="SHORT", entry=100.0, stop=110.0, target=80.0)
    outcome, _, reason = evaluate_entry(rec, [_bar("2026-11-13", 115.0, 116.0, 114.0, 115.0)])
    assert outcome == "skipped" and reason == "open-above-stop"
    # LONG open gaps above the target: also skipped.
    rec = _rec()
    outcome, _, reason = evaluate_entry(rec, [_bar("2026-11-13", 125.0, 126.0, 124.0, 125.0)])
    assert outcome == "skipped" and reason == "open-above-target"


def test_entry_stop_too_tight_skipped_live_parity():
    # Live refuses entries whose stop is < 0.8% from the entry; the shadow
    # ledger must skip them too instead of letting noise stop them out.
    # LONG 100 / stop 99.5 = 0.5% -> too tight.
    rec = _rec(entry=100.0, stop=99.5, target=104.0)
    outcome, _, reason = evaluate_entry(
        rec, [_bar("2026-11-13", 99.8, 100.5, 99.6, 100.0)])
    assert outcome == "skipped" and reason == "stop-too-tight"
    update_record(rec, [_bar("2026-11-13", 99.8, 100.5, 99.6, 100.0)], "2026-11-13")
    assert rec["status"] == "skipped" and rec["skip_reason"] == "stop-too-tight"
    assert rec["r_multiple"] is None
    # SHORT mirror: 200 / stop 201 = 0.5% -> too tight.
    rec = _rec(direction="SHORT", entry=200.0, stop=201.0, target=190.0)
    outcome, _, reason = evaluate_entry(
        rec, [_bar("2026-11-13", 199.5, 200.5, 199.0, 199.8)])
    assert outcome == "skipped" and reason == "stop-too-tight"


def test_entry_stop_at_floor_is_tradable():
    # A stop clearly above the 0.8% floor (1.0%): live allows it, so does
    # the ledger. (Exactly 0.8% nominal is float-fuzzy — 100/99.2 computes
    # to 0.7999...% — and the live check rejects it too, so parity holds.)
    rec = _rec(entry=100.0, stop=99.0, target=104.0)
    outcome, bar, _ = evaluate_entry(
        rec, [_bar("2026-11-13", 99.5, 100.5, 99.3, 100.0)])
    assert outcome == "entered" and bar is not None


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

def _open_rec(**kw):
    rec = _rec(**kw)
    rec["status"] = "open"
    rec["entry_date"] = "2026-11-13"
    rec["entry_price"] = 100.0
    rec["hold_days"] = 1
    return rec


def test_exit_target_hit_records_positive_r():
    rec = _open_rec()
    out = evaluate_day(rec, _bar("2026-11-13", 100.0, 121.0, 99.0, 119.0))
    assert out["exit_reason"] == "target" and out["exit_price"] == 120.0
    update_record(rec, [_bar("2026-11-13", 100.0, 121.0, 99.0, 119.0)], "2026-11-13")
    assert rec["status"] == "exited" and rec["r_multiple"] == 2.0


def test_exit_stop_hit_records_negative_r():
    rec = _open_rec()
    out = evaluate_day(rec, _bar("2026-11-13", 100.0, 101.0, 89.0, 95.0))
    assert out["exit_reason"] == "stop" and out["exit_price"] == 90.0


def test_exit_gap_through_stop_fills_at_open():
    rec = _open_rec()
    out = evaluate_day(rec, _bar("2026-11-13", 85.0, 86.0, 84.0, 85.0))
    assert out["exit_reason"] == "stop" and out["exit_price"] == 85.0
    assert out["gap_through_stop"] is True


def test_exit_gap_through_target_fills_at_open():
    rec = _open_rec()
    out = evaluate_day(rec, _bar("2026-11-13", 125.0, 126.0, 124.0, 125.0))
    assert out["exit_reason"] == "target" and out["exit_price"] == 125.0
    assert out["gap_through_target"] is True


def test_exit_same_day_both_touched_is_ambiguous_and_ordered_by_open():
    rec = _open_rec()
    # Open above entry -> target checked first.
    out = evaluate_day(rec, _bar("2026-11-13", 110.0, 125.0, 85.0, 100.0))
    assert out["exit_reason"] == "target" and out["ambiguous_same_day_touch"] is True
    # Open below entry -> stop checked first.
    out = evaluate_day(rec, _bar("2026-11-13", 95.0, 125.0, 85.0, 100.0))
    assert out["exit_reason"] == "stop" and out["ambiguous_same_day_touch"] is True


def test_expiry_intraday_three_days_at_close():
    rec = _open_rec()
    bars = [_bar("2026-11-13", 100.0, 101.0, 99.0, 100.0),
            _bar("2026-11-14", 100.0, 101.0, 99.0, 100.0),
            _bar("2026-11-17", 100.0, 101.0, 99.0, 102.0)]
    update_record(rec, bars, "2026-11-17")
    assert rec["status"] == "exited" and rec["exit_reason"] == "expiry"
    assert rec["exit_date"] == "2026-11-17" and rec["exit_price"] == 102.0
    assert rec["hold_days"] == 3
    assert rec["r_multiple"] == 0.2  # (102-100)/10


def test_expiry_swing_twenty_days():
    rec = _open_rec(horizon="swing")
    assert MAX_HOLD_DAYS["swing"] == 20
    bars = [_bar(f"2026-11-{13 + i:02d}" if 13 + i <= 30 else f"2026-12-{13 + i - 30:02d}",
                 100.0, 101.0, 99.0, 100.0) for i in range(19)]
    update_record(rec, bars, "2026-12-10")
    assert rec["status"] == "open"  # 19 days < 20
    bars.append(_bar("2026-12-11", 100.0, 101.0, 99.0, 101.0))
    update_record(rec, bars, "2026-12-11")
    assert rec["status"] == "exited" and rec["exit_reason"] == "expiry"
    assert rec["hold_days"] == 20


def test_update_record_idempotent():
    rec = _rec()
    bars = [_bar("2026-11-13", 99.0, 101.0, 98.0, 100.0),
            _bar("2026-11-14", 100.0, 121.0, 99.0, 119.0)]
    update_record(rec, bars, "2026-11-14")
    first = dict(rec)
    update_record(rec, bars, "2026-11-14")
    assert rec == first
    # And a third run with an extra earlier bar changes nothing either.
    update_record(rec, [_bar("2026-11-12", 98.0, 99.0, 97.0, 98.0)] + bars, "2026-11-14")
    assert rec == first


def test_skipped_record_stays_skipped():
    rec = _rec()
    bars = [_bar("2026-11-13", 105.0, 106.0, 104.0, 105.0)]
    update_record(rec, bars, "2026-11-13")
    assert rec["status"] == "skipped" and rec["skip_reason"] == "open-missed"
    update_record(rec, bars + [_bar("2026-11-14", 99.0, 100.0, 98.0, 99.0)], "2026-11-14")
    assert rec["status"] == "skipped"  # never re-enters on a later day


# ---------------------------------------------------------------------------
# Summary math
# ---------------------------------------------------------------------------

def test_win_prob_bucket():
    assert win_prob_bucket(49.9) == "<50"
    assert win_prob_bucket(50.0) == "50-60"
    assert win_prob_bucket(65.0) == "60-70"
    assert win_prob_bucket(70.0) == ">70"
    assert win_prob_bucket(None) == "unknown"


def _settled(win_prob, r, horizon="intraday", month="2026-11"):
    rec = _rec()
    rec["candidate_date"] = f"{month}-12"
    rec["win_prob"] = win_prob
    rec["horizon"] = horizon
    rec["status"] = "exited"
    rec["r_multiple"] = r
    return rec


def test_compute_summary_math():
    recs = [
        _settled(65.0, 2.0),
        _settled(65.0, -1.0),
        _settled(75.0, 1.0),
        _settled(45.0, -1.0),
    ]
    s = compute_summary(recs, as_of_date="2026-11-14")
    t = s["totals"]
    assert t["settled"] == 4 and t["wins"] == 2 and t["win_rate"] == 0.5
    assert t["expectancy_r"] == 0.25  # (2-1+1-1)/4
    assert t["profit_factor"] == 1.5  # 3.0 / 2.0
    # Brier: (0.65-1)^2 + (0.65-0)^2 + (0.75-1)^2 + (0.45-0)^2, /4
    assert abs(t["brier"] - (0.1225 + 0.4225 + 0.0625 + 0.2025) / 4) < 1e-9
    assert s["buckets"]["60-70"]["n"] == 2
    assert s["buckets"]["60-70"]["win_rate"] == 0.5
    assert s["buckets"][">70"]["n"] == 1 and s["buckets"][">70"]["win_rate"] == 1.0
    assert s["buckets"]["<50"]["n"] == 1 and s["buckets"]["<50"]["win_rate"] == 0.0
    assert s["by_horizon"]["intraday"]["n"] == 4
    assert s["by_month"]["2026-11"]["n"] == 4
    assert len(s["settled_trades"]) == 4


def test_compute_summary_empty():
    s = compute_summary([], as_of_date="2026-11-14")
    assert s["totals"]["settled"] == 0
    assert s["totals"]["expectancy_r"] is None
    assert s["totals"]["brier"] is None
