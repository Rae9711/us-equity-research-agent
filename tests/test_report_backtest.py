"""Tests for the report-driven signal provider + report-mode backtest.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json

import pytest

from src.paper.backtest import Bar, resolve_fill_price, run_paper_backtest
from src.paper.report_signals import (
    available_event_ls_report_dates,
    available_report_dates,
    event_ls_signals_for_date,
    report_signals_for_date,
    report_symbols,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _write_morning(root, trading_date, primary, swing=None, top_trades=None):
    day = root / "reports" / trading_date
    day.mkdir(parents=True, exist_ok=True)
    payload = {"best_trades": {"primary": primary}, "best_opportunity": primary, "advisory": True}
    if top_trades is not None:
        payload["best_trades"]["top_trades"] = top_trades
    elif primary:
        payload["best_trades"]["top_trades"] = [primary]
    if swing:
        payload["swing_trade"] = swing
        payload["best_trades"]["swing"] = swing
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fill realism
# --------------------------------------------------------------------------- #
def test_resolve_fill_price_limit_touch():
    bar = Bar(date="2026-03-02", open=100.5, high=101.0, low=99.0, close=100.8)
    # LONG limit at 100 fills at min(open, entry)
    assert resolve_fill_price({"direction": "LONG", "entry_price": 100.0}, bar) == 100.0
    # LONG limit below the day's low never fills
    assert resolve_fill_price({"direction": "LONG", "entry_price": 98.0}, bar) is None
    # SHORT limit at 100 fills at max(open, entry)
    assert resolve_fill_price({"direction": "SHORT", "entry_price": 100.0}, bar) == 100.5
    # SHORT limit above the day's high never fills
    assert resolve_fill_price({"direction": "SHORT", "entry_price": 102.0}, bar) is None
    # No entry price → market on open
    assert resolve_fill_price({"direction": "LONG"}, bar) == 100.5


# --------------------------------------------------------------------------- #
# Report provider parsing
# --------------------------------------------------------------------------- #
def test_report_provider_parses_and_filters(data_root):
    primary = {
        "symbol": "NVDA", "direction": "LONG",
        "entry_price": 100.0, "stop_price": 95.0, "target_price": 110.0,
        "win_prob": 66, "expected_return_pct": 8.0, "risk_reward": 2.0,
        "trade_action": "BUY", "horizon": "Intraday",
        "targets": [{"price": 110.0, "label": "T1"}, {"price": 120.0, "label": "T2"}],
    }
    passer = {
        "symbol": "AMD", "direction": "LONG", "rank": 2,
        "entry_price": 50.0, "stop_price": 48.0, "target_price": 51.0,
        "trade_action": "Pass", "horizon": "Intraday",
    }
    bad_geometry = {
        "symbol": "MU", "direction": "LONG", "rank": 3,
        "entry_price": 50.0, "stop_price": 52.0, "target_price": 60.0,  # stop > entry
        "trade_action": "BUY", "horizon": "Intraday",
    }
    swing = {
        "symbol": "META", "direction": "LONG",
        "entry_price": 600.0, "stop_price": 560.0, "target_price": 700.0,
        "win_prob": 70, "expected_return_pct": 12.0, "horizon": "Swing",
    }
    _write_morning(data_root, "2026-03-02", primary, swing=swing,
                   top_trades=[primary, passer, bad_geometry])

    sigs = report_signals_for_date("2026-03-02")
    syms = {s["symbol"] for s in sigs}
    assert "NVDA" in syms and "META" in syms
    assert "AMD" not in syms      # Pass filtered
    assert "MU" not in syms       # bad geometry filtered

    nvda = next(s for s in sigs if s["symbol"] == "NVDA")
    assert nvda["target_price"] == 110.0
    assert nvda["target2"] == 120.0
    assert nvda["_edge"] > 0
    meta = next(s for s in sigs if s["symbol"] == "META")
    assert "swing" in meta["horizon"].lower()

    assert available_report_dates() == ["2026-03-02"]
    assert set(report_symbols(["2026-03-02"])) == {"NVDA", "META"}


def test_event_ls_report_preserves_multi_leg_targets(data_root):
    day = data_root / "reports" / "2026-03-03"
    day.mkdir(parents=True)
    payload = {
        "event_ls_portfolio": {
            "deploy": True,
            "as_of": "2026-03-03T09:25:00-05:00",
            "hold_days": [2, 10],
            "costs_gate": {"pass": True},
            "legs": [
                {
                    "symbol": "NVDA",
                    "direction": "LONG",
                    "weight_pct": 5.0,
                    "industry": "Technology",
                },
                {
                    "symbol": "AMD",
                    "direction": "SHORT",
                    "weight_pct": 5.0,
                    "industry": "Technology",
                },
            ],
            "hedge": {
                "symbol": "SPY",
                "direction": "SHORT",
                "weight": 0.02,
            },
        }
    }
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = event_ls_signals_for_date("2026-03-03")
    assert {row["symbol"] for row in rows} == {"NVDA", "AMD", "SPY"}
    assert next(row for row in rows if row["symbol"] == "NVDA")["target_weight"] == 0.05
    assert next(row for row in rows if row["symbol"] == "AMD")["target_weight"] == -0.05
    assert next(row for row in rows if row["symbol"] == "SPY")["is_hedge"] is True
    assert available_event_ls_report_dates() == ["2026-03-03"]

    payload["event_ls_portfolio"]["costs_gate"]["pass"] = False
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")
    assert event_ls_signals_for_date("2026-03-03") == []


# --------------------------------------------------------------------------- #
# End-to-end report-driven backtest
# --------------------------------------------------------------------------- #
def _bars_for_dates(dates, base, slope):
    bars = []
    prev = base
    for i, d in enumerate(dates):
        open_ = prev + 0.2
        close = base + slope * (i + 1)
        bars.append(Bar(date=d, open=round(open_, 2),
                        high=round(max(open_, close) + 0.8, 2),
                        low=round(min(open_, close) - 0.8, 2),
                        close=round(close, 2), volume=1e6))
        prev = close
    return bars


def test_report_driven_backtest_runs(data_root):
    dates = [f"2026-03-{i:02d}" for i in range(2, 28)]  # 26 pseudo-sessions
    nvda_bars = _bars_for_dates(dates, base=100.0, slope=1.0)
    qqq_bars = _bars_for_dates(dates, base=400.0, slope=0.6)

    # Write a daily LONG NVDA plan whose entry sits inside that day's range.
    for i, d in enumerate(dates):
        entry = nvda_bars[i].open
        _write_morning(data_root, d, {
            "symbol": "NVDA", "direction": "LONG",
            "entry_price": round(entry, 2),
            "stop_price": round(entry - 4, 2),
            "target_price": round(entry + 8, 2),  # 2R so EV/R:R gates clear
            "win_prob": 62, "expected_return_pct": 4.0, "risk_reward": 2.0,
            "trade_action": "BUY", "horizon": "Intraday",
            "targets": [{"price": round(entry + 8, 2), "label": "T1"},
                        {"price": round(entry + 12, 2), "label": "T2"}],
        })

    bars = {"NVDA": nvda_bars, "QQQ": qqq_bars}
    result = run_paper_backtest(
        symbols=["NVDA"], bars=bars, benchmark_symbol="QQQ",
        day_signal_provider=report_signals_for_date, walk_forward_folds=2,
    )
    assert result.get("error") is None
    assert result["signal_mode"] == "report"
    assert result["n_entries"] >= 1
    assert result["n_trades"] >= 1
    assert "excess_return_pct" in result
