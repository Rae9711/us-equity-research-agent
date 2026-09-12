from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest

from src.paper.market_neutral_backtest import (
    BacktestConfig,
    DailyBar,
    MarketNeutralBacktester,
)
from src.strategies.market_neutral_ls import (
    Event,
    MarketNeutralLongShortStrategy,
    StrategyConfig,
)


START = date(2025, 1, 1)
SECTORS = {"AAA": "Technology", "BBB": "Technology"}


def _strategy():
    config = StrategyConfig(
        momentum_20_weight=0.0,
        momentum_60_weight=0.0,
        earnings_surprise_weight=0.0,
        earnings_revision_weight=0.0,
        pead_weight=0.0,
        news_weight=1.0,
        min_price=1.0,
        min_average_dollar_volume=1.0,
        names_per_sector_side=1,
        min_names_per_sector=2,
        gross_target=1.0,
        max_abs_net=0.1,
        max_name_weight=0.5,
        max_sector_gross=1.0,
        predicted_alpha_bps_per_score=1_000.0,
        cost_gate_multiple=0.0,
        default_spread_bps=0.0,
        default_impact_bps=0.0,
        default_borrow_bps=0.0,
    )
    return MarketNeutralLongShortStrategy(config, SECTORS)


def _bars(days=75, *, volume=1_000_000.0, pit=False):
    rows = []
    for offset in range(days):
        day = START + timedelta(days=offset)
        for symbol in ("AAA", "BBB", "SPY"):
            rows.append(DailyBar(
                symbol=symbol,
                date=day,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=volume,
                sector=SECTORS.get(symbol),
                beta=1.0,
                known_at=datetime.combine(day, time(16)) if pit else None,
                member_from=START if pit else None,
                member_to=START + timedelta(days=days + 1) if pit else None,
            ))
    return rows


def _events(flip=False):
    signal_day = START + timedelta(days=60)
    rows = [
        Event("AAA", datetime.combine(signal_day, time(16)), "news", sentiment=1.0),
        Event("BBB", datetime.combine(signal_day, time(16)), "news", sentiment=-1.0),
    ]
    if flip:
        flip_day = START + timedelta(days=61)
        rows.extend([
            Event("AAA", datetime.combine(flip_day, time(16)), "news", sentiment=-3.0),
            Event("BBB", datetime.combine(flip_day, time(16)), "news", sentiment=3.0),
        ])
    return rows


def _run(engine, bars, events, *, pit_borrow=False, borrow_rate=0.0):
    availability = {"AAA": True, "BBB": True, "SPY": True}
    rates = {"AAA": borrow_rate, "BBB": borrow_rate, "SPY": borrow_rate}
    ssr = {"AAA": False, "BBB": False, "SPY": False}
    forced = {"AAA": False, "BBB": False, "SPY": False}
    if pit_borrow:
        days = sorted({row.date for row in bars})
        availability = {day.isoformat(): dict(availability) for day in days}
        rates = {day.isoformat(): dict(rates) for day in days}
        ssr = {day.isoformat(): dict(ssr) for day in days}
        forced = {day.isoformat(): dict(forced) for day in days}
    return engine.run(
        bars,
        events,
        borrow_available=availability,
        borrow_rates=rates,
        ssr_restricted=ssr,
        forced_cover=forced,
    )


def _config(**changes):
    base = BacktestConfig(
        default_spread_bps=0.0,
        slippage_bps=0.0,
        default_impact_bps=0.0,
        impact_curve_bps=0.0,
        annual_borrow_rate=0.0,
        max_sector_gross=1.0,
        max_abs_beta=0.05,
    )
    return replace(base, **changes)


def _snapshot(result, offset):
    day = START + timedelta(days=offset)
    return next(row for row in result.snapshots if row.date == day)


def test_open_decision_uses_strictly_prior_day_features_and_trades_both_sides():
    bars = _bars()
    engine = MarketNeutralBacktester(_strategy(), _config())
    baseline = _run(engine, bars, _events())

    changed = list(bars)
    index = next(
        i for i, row in enumerate(changed)
        if row.date == START + timedelta(days=61) and row.symbol == "AAA"
    )
    changed[index] = replace(changed[index], close=140.0, high=141.0)
    future_news = Event(
        "BBB",
        datetime.combine(START + timedelta(days=61), time(10)),
        "news",
        sentiment=100.0,
    )
    contaminated = _run(engine, changed, _events() + [future_news])

    decision = _snapshot(baseline, 61)
    contaminated_decision = _snapshot(contaminated, 61)
    assert decision.target_weights == contaminated_decision.target_weights
    assert decision.feature_cutoff == START + timedelta(days=60)
    assert decision.positions["AAA"] > 0
    assert decision.positions["BBB"] < 0


def test_utc_event_timestamps_are_compared_to_new_york_open():
    day = START + timedelta(days=61)
    before = Event(
        "AAA", datetime.combine(day, time(14, 29), timezone.utc), "news"
    )
    after = Event(
        "BBB", datetime.combine(day, time(14, 31), timezone.utc), "news"
    )
    visible = MarketNeutralBacktester._visible_events([before, after], day)
    assert visible == [before]


def test_spread_slippage_impact_and_short_borrow_reduce_equity():
    frictionless = _run(
        MarketNeutralBacktester(_strategy(), _config()), _bars(), _events()
    )
    costly = _run(
        MarketNeutralBacktester(
            _strategy(),
            _config(
                default_spread_bps=10.0,
                slippage_bps=8.0,
                default_impact_bps=7.0,
                impact_curve_bps=25.0,
                annual_borrow_rate=0.252,
            ),
        ),
        _bars(),
        _events(),
        borrow_rate=0.252,
    )
    assert costly.metrics["ending_equity"] < frictionless.metrics["ending_equity"]
    assert costly.metrics["transaction_cost"] > 0
    assert costly.metrics["borrow_cost"] > 0


def test_missing_borrow_fee_blocks_short_and_fails_data_quality():
    bars = _bars()
    result = MarketNeutralBacktester(_strategy(), _config()).run(
        bars,
        _events(),
        borrow_available={"AAA": True, "BBB": True},
        borrow_rates={"AAA": 0.01},
    )
    decision = _snapshot(result, 61)
    assert "BBB" not in decision.positions
    assert result.metrics["borrow_data_complete"] is False
    assert result.metrics["pit_quality_passed"] is False


def test_ssr_blocks_new_short_and_forced_cover_exits_existing_short():
    bars = _bars(days=65)
    days = sorted({row.date for row in bars})
    availability = {
        day.isoformat(): {"AAA": True, "BBB": True, "SPY": True}
        for day in days
    }
    rates = {
        day.isoformat(): {"AAA": 0.0, "BBB": 0.0, "SPY": 0.0}
        for day in days
    }
    ssr = {
        day.isoformat(): {"AAA": False, "BBB": False, "SPY": False}
        for day in days
    }
    forced = {
        day.isoformat(): {"AAA": False, "BBB": False, "SPY": False}
        for day in days
    }
    entry_day = START + timedelta(days=61)
    ssr[entry_day.isoformat()]["BBB"] = True
    ssr_result = MarketNeutralBacktester(_strategy(), _config()).run(
        bars,
        _events(),
        borrow_available=availability,
        borrow_rates=rates,
        ssr_restricted=ssr,
        forced_cover=forced,
    )
    assert "BBB" not in _snapshot(ssr_result, 61).positions
    ssr[entry_day.isoformat()]["BBB"] = False

    forced_day = START + timedelta(days=62)
    forced[forced_day.isoformat()]["BBB"] = True
    result = MarketNeutralBacktester(
        _strategy(), _config(min_hold_days=10)
    ).run(
        bars,
        _events(),
        borrow_available=availability,
        borrow_rates=rates,
        ssr_restricted=ssr,
        forced_cover=forced,
    )
    assert _snapshot(result, 61).positions["BBB"] < 0
    assert "BBB" not in _snapshot(result, 62).positions
    assert any(
        fill.symbol == "BBB" and fill.reason == "forced_exit"
        for fill in _snapshot(result, 62).fills
    )


def test_minimum_hold_blocks_early_flip_and_maximum_hold_forces_exit():
    flip_result = _run(
        MarketNeutralBacktester(
            _strategy(), _config(min_hold_days=2, max_hold_days=10)
        ),
        _bars(),
        _events(flip=True),
    )
    first = _snapshot(flip_result, 61)
    too_early = _snapshot(flip_result, 62)
    allowed = _snapshot(flip_result, 63)
    assert first.positions["AAA"] > 0
    assert too_early.positions["AAA"] == pytest.approx(first.positions["AAA"])
    assert not any(fill.symbol == "AAA" for fill in too_early.fills)
    assert allowed.positions["AAA"] < 0

    forced = _run(
        MarketNeutralBacktester(
            _strategy(), _config(min_hold_days=0, max_hold_days=3)
        ),
        _bars(days=66),
        _events(),
    )
    assert any(
        fill.reason == "forced_exit"
        for row in forced.snapshots
        for fill in row.fills
    )


def test_drawdown_throttle_scales_daily_target():
    bars = _bars(days=64)
    shock_day = START + timedelta(days=62)
    shocked = [
        replace(row, open=90.0, high=100.0, low=89.0, close=90.0)
        if row.date == shock_day and row.symbol == "AAA" else row
        for row in bars
    ]
    result = _run(
        MarketNeutralBacktester(_strategy(), _config(min_hold_days=0)),
        shocked,
        _events(),
    )
    shock = _snapshot(result, 62)
    assert 4.0 <= shock.drawdown_pct < 6.0
    assert shock.throttle == 0.75
    assert sum(abs(value) for value in shock.target_weights.values()) <= 0.75 + 1e-12


def test_one_percent_daily_loss_blocks_new_risk_on_next_session():
    bars = _bars(days=65)
    shock_day = START + timedelta(days=62)
    shocked = [
        replace(row, close=96.0, low=95.0)
        if row.date == shock_day and row.symbol == "AAA" else row
        for row in bars
    ]
    result = _run(
        MarketNeutralBacktester(_strategy(), _config(min_hold_days=0)),
        shocked,
        _events(),
    )
    loss_day = _snapshot(result, 62)
    following = _snapshot(result, 63)
    assert loss_day.daily_return <= -0.01
    assert following.new_entries_halted is True
    prior_shares = loss_day.positions
    assert all(
        fill.symbol in prior_shares
        and fill.filled_shares * prior_shares[fill.symbol] <= 0
        for fill in following.fills
    )


def test_ten_percent_drawdown_liquidates_and_permanently_halts():
    bars = _bars(days=65)
    shock_day = START + timedelta(days=62)
    shocked = [
        replace(row, open=75.0, high=100.0, low=74.0, close=75.0)
        if row.date == shock_day and row.symbol == "AAA" else row
        for row in bars
    ]
    result = _run(
        MarketNeutralBacktester(_strategy(), _config(min_hold_days=10)),
        shocked,
        _events(),
    )
    shock = _snapshot(result, 62)
    following = _snapshot(result, 63)
    assert shock.halted
    assert shock.throttle == 0.0
    assert shock.positions == {}
    assert following.halted
    assert following.target_weights == {}


def test_thin_sample_acceptance_stays_failed():
    result = _run(
        MarketNeutralBacktester(_strategy(), _config()),
        _bars(days=65),
        _events(),
    )
    assert result.acceptance["passed"] is False
    assert result.metrics["oos_closed_trades"] == 0
    assert "oos_closed_trades" in result.acceptance["failures"]
    assert result.metrics["pit_quality_passed"] is False
    assert "pit_quality_passed" in result.acceptance["failures"]


def test_complete_bar_metadata_passes_pit_quality_gate():
    result = _run(
        MarketNeutralBacktester(_strategy(), _config()),
        _bars(days=65, pit=True),
        _events(),
        pit_borrow=True,
    )
    assert result.metrics["pit_quality_passed"] is True
    assert "pit_quality_passed" not in result.acceptance["failures"]
