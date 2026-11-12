"""Tests for the 60-day backtest replay (src/jobs/backtest_replay.py).

D4: the yfinance fallback used to set data_available/qqq_return but never
filled features.qqq_chg / features.vix, so should_trade (which needs both
non-None) could never fire on fallback days -- the July 2026 run showed
0 agent trades for that reason.

No network in these tests: _replay_day takes pre-fetched dicts, and the
yfinance download is stubbed with a fake module returning a small pandas
DataFrame. Raw-data absence is simulated by stubbing build_features to
return an empty FeaturesModel.
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from src.jobs.backtest_replay import (
    _compute_virtual_pnl,
    _fetch_historical_closes,
    _fetch_historical_prices,
    _replay_day,
)
from src.schemas.market_case import FeaturesModel


def _no_raw_data(monkeypatch):
    """Simulate a day with no raw data file: build_features yields nothing."""
    monkeypatch.setattr(
        "src.features.build.build_features", lambda d: FeaturesModel()
    )


def test_fallback_day_produces_should_trade_true(monkeypatch):
    _no_raw_data(monkeypatch)
    gaps: list[str] = []
    result = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": 0.5}},
        gaps,
        vix_closes={"2026-06-30": 18.5},
    )
    assert result["data_available"] is True
    assert result["qqq_return"] == 0.5
    # vix 18.5 < 22 and qqq_chg 0.5 > 0.3 -> trade signal fires
    assert result["should_trade"] is True


def test_fallback_day_produces_should_trade_false_when_vix_high(monkeypatch):
    _no_raw_data(monkeypatch)
    result = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": 0.5}},
        [],
        vix_closes={"2026-06-30": 25.0},
    )
    # vix 25.0 >= 22 -> no trade, but the decision is made (not None)
    assert result["should_trade"] is False


def test_fallback_day_produces_should_trade_false_when_qqq_negative(monkeypatch):
    _no_raw_data(monkeypatch)
    result = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": -0.4}},
        [],
        vix_closes={"2026-06-30": 18.5},
    )
    assert result["should_trade"] is False


def test_vix_uses_prior_day_close_not_same_day(monkeypatch):
    _no_raw_data(monkeypatch)
    # Same-day ^VIX close (30.0) must NOT be used: it is only known after
    # the day ends. The prior-day close (18.5) is the signal input.
    result = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": 0.5}},
        [],
        vix_closes={"2026-06-30": 18.5, "2026-07-01": 30.0},
    )
    assert result["should_trade"] is True


def test_vix_prior_close_bridges_weekend(monkeypatch):
    _no_raw_data(monkeypatch)
    # Mon 2026-06-29 has no prior trading-day row for the weekend; the
    # latest close strictly before it is Fri 2026-06-26 (19.0), not the
    # same-day close (40.0).
    result = _replay_day(
        date(2026, 6, 29),
        {"2026-06-29": {"QQQ": 0.5}},
        [],
        vix_closes={"2026-06-26": 19.0, "2026-06-29": 40.0},
    )
    assert result["should_trade"] is True


def test_no_fallback_data_still_yields_no_signal(monkeypatch):
    _no_raw_data(monkeypatch)
    result = _replay_day(date(2026, 7, 1), {}, [])
    assert result["should_trade"] is None
    assert result["qqq_return"] is None


def test_raw_features_are_not_overridden_by_fallback(monkeypatch):
    monkeypatch.setattr(
        "src.features.build.build_features",
        lambda d: FeaturesModel(qqq_chg=1.0, vix=20.0),
    )
    result = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": 0.5}},
        [],
        vix_closes={"2026-06-30": 30.0},
    )
    # Raw values win: qqq_chg 1.0 > 0.3 and vix 20.0 < 22 -> True.
    # (Without the guard the fallback vix 30.0 would flip this to False.)
    assert result["should_trade"] is True


def test_virtual_pnl_counts_fallback_trades(monkeypatch):
    _no_raw_data(monkeypatch)
    day_trade = _replay_day(
        date(2026, 7, 1),
        {"2026-07-01": {"QQQ": 0.5}},
        [],
        vix_closes={"2026-06-30": 18.5},
    )
    day_no_trade = _replay_day(
        date(2026, 7, 2),
        {"2026-07-02": {"QQQ": 0.8}},
        [],
        vix_closes={"2026-07-01": 25.0},
    )
    assert day_trade["should_trade"] is True
    assert day_no_trade["should_trade"] is False
    pnl = _compute_virtual_pnl([day_trade, day_no_trade])
    assert pnl["n_trading_days"] == 2
    assert pnl["n_agent_trades"] == 1


class _FakeYFinance:
    """Stand-in for the yfinance module: download returns fixed closes."""

    def __init__(self, closes: dict[str, float]):
        dates = sorted(closes)
        self._frame = pd.DataFrame(
            {"Close": [closes[d] for d in dates]},
            index=pd.to_datetime(dates),
        )

    def download(self, *args, **kwargs):
        return self._frame


def test_fetch_historical_closes_uses_mocked_yfinance(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        _FakeYFinance({"2026-06-30": 18.5, "2026-07-01": 19.25}),
    )
    closes = _fetch_historical_closes(["^VIX"], date(2026, 7, 1), date(2026, 7, 1))
    assert closes == {
        "2026-06-30": {"^VIX": 18.5},
        "2026-07-01": {"^VIX": 19.25},
    }


def test_fetch_historical_prices_returns_unchanged_by_refactor(monkeypatch):
    # Same stub: closes 100.0 -> 101.0 must still yield a +1.0% return.
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        _FakeYFinance({"2026-06-30": 100.0, "2026-07-01": 101.0}),
    )
    prices = _fetch_historical_prices(["QQQ"], date(2026, 7, 1), date(2026, 7, 1))
    assert prices == {"2026-07-01": {"QQQ": 1.0}}
