"""Tests for per-tick fallback across enterable candidates (2026-11-17).

Before this, one tick tried only the primary per book: when it was
calibration-filtered, MISSED/INVALIDATED, in stop-cooldown, or held by
the allocation floor, the tick tried nothing else. Now the tick walks
the pick's ``enter_candidates`` (calibrated-expectancy order, max 3
attempts, at most one new leg per book per tick).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json

import pytest

from src.paper.account import default_account, get_position
from src.paper.execution_agent import decide_and_act
from src.paper.signals import (
    _enter_candidates,
    pick_intraday_signal,
    pick_swing_signal,
)

# Same recalibration map as tests/test_calibration_filter.py (34 settled).
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


def _write_morning(root, trading_date, primary, swing=None, top_trades=None):
    day = root / "reports" / trading_date
    day.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_trades": {"primary": primary},
        "best_opportunity": primary,
        "advisory": True,
    }
    if top_trades is not None:
        payload["best_trades"]["top_trades"] = top_trades
    elif primary:
        payload["best_trades"]["top_trades"] = [primary]
    if swing:
        payload["swing_trade"] = swing
        payload["best_trades"]["swing"] = swing
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")


def _slot(symbol, *, entry=100.0, stop=95.0, target=110.0, win_prob=75,
          expected_return_pct=3.0, horizon="Intraday", rank=1, direction="LONG"):
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry,
        "entry_zone": {"low": entry - 1, "mid": entry, "high": entry + 1},
        "stop_price": stop,
        "target_price": target,
        "win_prob": win_prob,
        "expected_return_pct": expected_return_pct,
        "horizon": horizon,
        "rank": rank,
        "trade_action": "Small",
    }


def test_primary_calibration_filtered_then_second_enters(data_root):
    # #1 AAPL wins the edge-score primary (wp 43 x ER 10) but its
    # calibrated expectancy is negative (intraday <50: 0.45 x R:R 1.0 -
    # 0.55 = -0.10R) so the picker excludes it; #2 MSFT enters.
    _write_summary(data_root)
    primary = _slot("AAPL", stop=96.0, target=104.0, win_prob=43,
                    expected_return_pct=10.0, rank=1)
    second = _slot("MSFT", stop=95.0, target=110.0, win_prob=75,
                   expected_return_pct=2.0, rank=2)
    _write_morning(data_root, "2026-07-08", primary, top_trades=[primary, second])
    picked = pick_intraday_signal(
        "2026-07-08", session_phase="open", current_price_by_symbol={"AAPL": 100.0, "MSFT": 100.0}
    )
    assert picked["signal"]["symbol"] == "AAPL"  # edge-score primary
    cands = picked["enter_candidates"]
    assert [c["signal"]["symbol"] for c in cands] == ["MSFT"]  # #1 filtered out
    acct = default_account()
    decision = decide_and_act(acct, "2026-07-08", session_phase="open", force_price=100.0)
    assert decision["action"] == "ENTRY"
    assert get_position(acct, "intraday")["symbol"] == "MSFT"


def test_loop_after_allocation_floor_skip(data_root):
    # #1 AAPL passes calibration (0.45 x 3.0 - 0.55 = +0.80R, best) but
    # the allocation floor holds it (win_prob 40 < 48). The tick must
    # then try #2 MSFT (+0.21R) and enter it -- not stop at #1's SKIP.
    _write_summary(data_root)
    first = _slot("AAPL", stop=97.0, target=109.0, win_prob=40,
                  expected_return_pct=10.0, rank=1)
    second = _slot("MSFT", stop=95.0, target=106.0, win_prob=75,
                   expected_return_pct=3.0, rank=2)
    _write_morning(data_root, "2026-07-08", first, top_trades=[first, second])
    picked = pick_intraday_signal(
        "2026-07-08", session_phase="open", current_price_by_symbol={"AAPL": 100.0, "MSFT": 100.0}
    )
    assert [c["signal"]["symbol"] for c in picked["enter_candidates"]] == ["AAPL", "MSFT"]
    acct = default_account()
    decision = decide_and_act(acct, "2026-07-08", session_phase="open", force_price=100.0)
    assert decision["action"] == "ENTRY"
    assert get_position(acct, "intraday")["symbol"] == "MSFT"
    intra = [b for b in decision["books"] if b.get("book") == "intraday"][0]
    assert intra["action"] == "ENTRY"
    assert intra.get("fallback_used") is True
    assert len(intra["attempts"]) == 2
    assert intra["attempts"][0]["symbol"] == "AAPL"
    assert "胜率 40%" in (intra["attempts"][0]["reason"] or "")


def test_all_three_untradeable_still_skip_max_3_attempts(data_root):
    # Four candidates, all calibration-passing but all below the 48%
    # allocation floor. The tick tries at most 3 and ends SKIP with no leg.
    _write_summary(data_root)
    rows = [
        _slot(sym, stop=97.0, target=109.0, win_prob=wp,
              expected_return_pct=er, rank=i + 1)
        for i, (sym, wp, er) in enumerate(
            [("AAPL", 40, 10.0), ("MSFT", 42, 9.0), ("NVDA", 45, 8.0), ("AMZN", 47, 7.0)]
        )
    ]
    _write_morning(data_root, "2026-07-08", rows[0], top_trades=rows)
    acct = default_account()
    decision = decide_and_act(acct, "2026-07-08", session_phase="open", force_price=100.0)
    assert get_position(acct, "intraday") is None
    assert decision["action"] in ("SKIP", "HOLD", "WAIT")
    intra = [b for b in decision["books"] if b.get("book") == "intraday"][0]
    assert intra["action"] == "SKIP"
    assert len(intra["attempts"]) == 3
    assert [a["symbol"] for a in intra["attempts"]] == ["AAPL", "MSFT", "NVDA"]


def test_swing_fallback_from_top_trades(data_root):
    # Swing slot NVDA LONG is MISSED at the forced price (110, above its
    # 99-101 zone) while a top_trades Swing row (TQQQ SHORT, zone
    # 109-111) is READY: the swing book must enter TQQQ, not stop at the
    # slot. Intraday primary ARM LONG is MISSED at the same price.
    primary = _slot("ARM", win_prob=68, expected_return_pct=3.8, rank=1)
    swing = _slot("NVDA", win_prob=70, expected_return_pct=12.0, horizon="Swing")
    tqqq = _slot("TQQQ", entry=110.0, stop=115.0, target=100.0, win_prob=60,
                 expected_return_pct=9.0, horizon="Swing", rank=1, direction="SHORT")
    _write_morning(data_root, "2026-07-08", primary, swing=swing, top_trades=[tqqq])
    picked = pick_swing_signal(
        "2026-07-08", session_phase="open",
        current_price_by_symbol={"NVDA": 110.0, "TQQQ": 110.0},
    )
    assert picked["action"] == "skip"  # the slot itself is MISSED
    assert [c["signal"]["symbol"] for c in picked["enter_candidates"]] == ["TQQQ"]
    acct = default_account()
    decision = decide_and_act(acct, "2026-07-08", session_phase="open", force_price=110.0)
    pos = get_position(acct, "swing")
    assert pos is not None
    assert pos["symbol"] == "TQQQ"
    assert pos["direction"] == "SHORT"


def test_enter_candidates_ordering_by_calibrated_expectancy(data_root):
    # Unit: order is calibrated expectancy desc, calibration-skipped rows
    # excluded, tiebreak by edge score.
    _write_summary(data_root)

    def _row(sym, win_prob, entry, stop, target, er):
        return {
            "key": f"top_trade_{sym}",
            "label": sym,
            "signal": {
                "symbol": sym, "direction": "LONG", "entry_price": entry,
                "stop_price": stop, "target_price": target,
                "win_prob": win_prob, "expected_return_pct": er,
                "entry_zone": {"low": entry - 1, "mid": entry, "high": entry + 1},
            },
            "action": "enter",
            "entry_status": {"status": "READY"},
            "quote": entry,
            "quote_source": "test",
        }

    rows = [
        _row("MSFT", 75, 100.0, 95.0, 110.0, 2.0),   # +0.65R
        _row("AAPL", 43, 100.0, 96.0, 104.0, 10.0),  # -0.10R -> excluded
        _row("NVDA", 40, 100.0, 97.0, 109.0, 10.0),  # +0.80R
    ]
    out = _enter_candidates(rows, horizon="intraday")
    assert [c["signal"]["symbol"] for c in out] == ["NVDA", "MSFT"]
    assert out[0]["calibrated_expectancy_r"] == pytest.approx(0.80, abs=0.01)
    assert out[1]["calibrated_expectancy_r"] == pytest.approx(0.65, abs=0.01)
