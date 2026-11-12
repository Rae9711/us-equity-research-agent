"""Tests for the judgment-quality panel (src/paper/judgment_quality.py).

All fixtures are synthetic -- no live quotes, no state files.
"""

from __future__ import annotations

from src.paper.judgment_quality import (
    compute_panel,
    consistency,
    directional_scores,
    levels_race,
    realism,
)


def _bar(date, o, h, l, c):
    return {"date": date, "open": o, "high": h, "low": l, "close": c}


def _rec(cdate, sym, direction, entry, stop, target, win_prob=70.0,
         horizon="intraday", origin="morning", zone=None):
    return {
        "id": f"{cdate}|{origin}|{sym}|{direction}",
        "candidate_date": cdate, "origin": origin, "symbol": sym,
        "direction": direction, "horizon": horizon, "win_prob": win_prob,
        "entry": entry, "stop": stop, "target": target,
        "entry_zone": zone, "status": "awaiting-entry",
    }


def _bars_up():
    # closes: 99, 100, 101, 102, 103, 104 (rising); first bar is the day
    # before the 10-20 candidate, so it serves as the reference close.
    return [_bar(f"2026-10-{d:02d}", 99 + i, 100 + i, 98 + i, 99 + i)
            for i, d in enumerate([19, 20, 21, 22, 23, 24])]


def _bars_down():
    return [_bar(f"2026-10-{d:02d}", 101 - i, 102 - i, 100 - i, 101 - i)
            for i, d in enumerate([19, 20, 21, 22, 23, 24])]


def test_directional_long_rising_is_correct():
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104)
    out = directional_scores([rec], {"AAA": _bars_up()})
    assert out["1d"]["n"] == 1
    assert out["1d"]["accuracy"] == 1.0
    assert out["3d"]["n"] == 1 and out["3d"]["accuracy"] == 1.0
    # 5d needs 5 bars after 10-20; only 4 exist
    assert out["5d"]["n"] == 0


def test_directional_short_falling_is_correct_and_long_falling_is_wrong():
    short = _rec("2026-10-20", "AAA", "SHORT", 100, 102, 96)
    long_ = _rec("2026-10-20", "BBB", "LONG", 100, 98, 104)
    out = directional_scores([short, long_], {"AAA": _bars_down(), "BBB": _bars_down()})
    assert out["1d"]["n"] == 2
    assert out["1d"]["accuracy"] == 0.5


def test_directional_uses_close_before_candidate_date_as_reference():
    # candidate on 10-21; ref close is 10-20's close (100), fwd is 10-22 (102)
    rec = _rec("2026-10-21", "AAA", "LONG", 100, 98, 104)
    out = directional_scores([rec], {"AAA": _bars_up()})
    assert out["1d"]["accuracy"] == 1.0


def test_directional_buckets():
    hi = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104, win_prob=80.0)
    lo = _rec("2026-10-20", "BBB", "LONG", 100, 98, 104, win_prob=40.0)
    out = directional_scores([hi, lo], {"AAA": _bars_up(), "BBB": _bars_down()})
    assert out["1d"]["by_bucket"][">70"]["accuracy"] == 1.0
    assert out["1d"]["by_bucket"]["<50"]["accuracy"] == 0.0


def test_levels_race_target_first():
    # entry 100 touched on 10-21 (low 99), target 104 hit on 10-22 (high 102? no)
    bars = [
        _bar("2026-10-20", 100, 101, 99, 100),
        _bar("2026-10-21", 100, 101, 99, 100),   # touches entry 100
        _bar("2026-10-22", 103, 105, 102, 104),  # high 105 >= target 104
    ]
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104)
    out = levels_race([rec], {"AAA": bars}, "2026-10-22")
    assert out["counts"] == {"target_first": 1}
    assert out["target_first_rate"] == 1.0


def test_levels_race_stop_first_and_never_entered():
    bars = [
        _bar("2026-10-20", 100, 101, 99, 100),
        _bar("2026-10-21", 100, 101, 99, 100),
        _bar("2026-10-22", 99, 100, 97, 98),  # touches entry 100, low 97 <= stop 98
        _bar("2026-10-23", 99, 100, 98, 99),  # third day of the window
    ]
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104)
    out = levels_race([rec], {"AAA": bars}, "2026-10-23")
    assert out["counts"] == {"stop_first": 1}

    far = _rec("2026-10-20", "AAA", "LONG", 200, 198, 204)
    out2 = levels_race([far], {"AAA": bars}, "2026-10-23")
    assert out2["counts"] == {"never-entered": 1}


def test_levels_race_censored_when_bars_run_out():
    bars = [_bar("2026-10-20", 100, 101, 99, 100),
            _bar("2026-10-21", 100, 101, 99, 100)]
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104, horizon="swing")
    out = levels_race([rec], {"AAA": bars}, "2026-10-21")
    # entry touched on 10-21 but the 20-day window extends past available bars
    assert out["counts"] == {"censored": 1}


def test_consistency_counts_flips():
    recs = [
        _rec("2026-10-20", "AAA", "LONG", 100, 98, 104),
        _rec("2026-10-21", "AAA", "SHORT", 100, 102, 96),
        _rec("2026-10-22", "AAA", "SHORT", 100, 102, 96),
        _rec("2026-10-23", "AAA", "LONG", 100, 98, 104),
    ]
    out = consistency(recs)
    assert out["flips"] == 2
    assert out["flips_by_symbol"] == {"AAA": 2}
    assert out["flip_days"] == {"2026-10-21": 1, "2026-10-23": 1}


def test_realism_ratios():
    # 10 bars with daily range 2 on close 100 -> avg range 0.02
    bars = [_bar(f"2026-10-{d:02d}", 100, 101, 99, 100)
            for d in range(10, 20)]
    # stop 1 away (1%), target 2 away (2%) -> 0.5 and 1.0 days of range
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 99, 102)
    out = realism([rec], {"AAA": bars})
    assert out["stop_days_of_range"]["median"] == 0.5
    assert out["target_days_of_range"]["median"] == 1.0
    assert out["stop_under_half_day_pct"] == 0.0


def test_compute_panel_assembles_all_sections():
    rec = _rec("2026-10-20", "AAA", "LONG", 100, 98, 104)
    out = compute_panel([rec], {"AAA": _bars_up()}, "2026-10-24")
    assert out["candidates"] == 1
    assert set(out) == {"as_of_date", "candidates", "directional",
                        "levels_race", "consistency", "realism"}
    assert out["directional"]["1d"]["n"] == 1
