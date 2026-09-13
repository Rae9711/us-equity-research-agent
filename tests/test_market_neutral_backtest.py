from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest

from src.paper.market_neutral_backtest import (
    BacktestConfig,
    DailyBar,
    MarketNeutralBacktester,
)
from src.jobs.market_neutral_backtest import main as backtest_main
from src.strategies.market_neutral_ls import (
    Bar as StrategyBar,
    Event,
    MarketNeutralLongShortStrategy,
    StrategyConfig,
)


START = date(2025, 1, 1)
SECTORS = {"AAA": "Technology", "BBB": "Technology"}
PIT_AUDIT = {
    "ok": True,
    "unavailable_partitions": [],
    "audit": {"ok": True},
    "data_gaps": {
        "ok": True,
        "required_categories": [
            "daily", "intraday", "news", "earnings",
            "borrow", "universe", "macro",
        ],
        "counts": {
            "daily": 1,
            "intraday": 1,
            "news": 1,
            "earnings": 1,
            "borrow": 1,
            "universe": 1,
            "macro": 1,
        },
        "missing_categories": [],
    },
}


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
                open_volume=volume,
                sector=SECTORS.get(symbol),
                beta=1.0,
                known_at=datetime.combine(day, time(16)) if pit else None,
                open_known_at=datetime.combine(day, time(9, 30)) if pit else None,
                member_from=START if pit else None,
                member_to=START + timedelta(days=days + 1) if pit else None,
            ))
    return rows


def _events(flip=False):
    signal_day = START + timedelta(days=60)
    rows = [
        Event(
            "AAA", datetime.combine(signal_day, time(16)), "news",
            sentiment=1.0, available_at=datetime.combine(signal_day, time(16)),
        ),
        Event(
            "BBB", datetime.combine(signal_day, time(16)), "news",
            sentiment=-1.0, available_at=datetime.combine(signal_day, time(16)),
        ),
    ]
    if flip:
        flip_day = START + timedelta(days=61)
        rows.extend([
            Event(
                "AAA", datetime.combine(flip_day, time(16)), "news",
                sentiment=-3.0, available_at=datetime.combine(flip_day, time(16)),
            ),
            Event(
                "BBB", datetime.combine(flip_day, time(16)), "news",
                sentiment=3.0, available_at=datetime.combine(flip_day, time(16)),
            ),
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
        pit_audit_report=PIT_AUDIT if pit_borrow else None,
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


def test_point_in_time_market_regime_classification_covers_required_states():
    engine = MarketNeutralBacktester(_strategy(), _config())

    def history(returns):
        price = 100.0
        rows = [StrategyBar("SPY", datetime(2025, 1, 1, 16), price, 1_000)]
        for index, value in enumerate(returns, start=1):
            price *= 1.0 + value
            rows.append(
                StrategyBar(
                    "SPY",
                    datetime(2025, 1, 1, 16) + timedelta(days=index),
                    price,
                    1_000,
                )
            )
        return rows

    assert engine._market_regime(history([0.002] * 60)) == "bull"
    assert engine._market_regime(history([-0.002] * 60)) == "bear"
    assert engine._market_regime(
        history([0.03 if index % 2 else -0.03 for index in range(60)])
    ) == "high_volatility"
    assert engine._market_regime(history([0.0] * 60)) == "sideways"


def test_future_sector_beta_and_completed_daily_volume_cannot_change_open_fill():
    bars = _bars()
    engine = MarketNeutralBacktester(_strategy(), _config())
    baseline = _run(engine, bars, _events())
    changed = [
        replace(
            row,
            sector="FutureSector",
            beta=9.0,
            volume=row.volume * 100,
        )
        if row.date == START + timedelta(days=61) and row.symbol == "AAA"
        else row
        for row in bars
    ]
    contaminated = _run(engine, changed, _events())
    expected = _snapshot(baseline, 61)
    actual = _snapshot(contaminated, 61)
    assert actual.target_weights == expected.target_weights
    assert [(f.symbol, f.filled_shares) for f in actual.fills] == [
        (f.symbol, f.filled_shares) for f in expected.fills
    ]


def test_atomic_partial_rebalance_keeps_all_legs_at_common_fill_ratio():
    bars = [
        replace(
            row,
            open_volume=1_000.0 if row.symbol == "AAA" else row.open_volume,
        )
        for row in _bars()
    ]
    result = _run(MarketNeutralBacktester(_strategy(), _config()), bars, _events())
    fills = _snapshot(result, 61).fills
    ratios = {
        round(abs(fill.filled_shares / fill.requested_shares), 8)
        for fill in fills
    }
    assert len(fills) >= 2
    assert len(ratios) == 1
    assert next(iter(ratios)) < 1.0
    assert result.metrics["portfolio_controls_passed"] is True


def test_utc_event_timestamps_are_compared_to_new_york_open():
    day = START + timedelta(days=61)
    before = Event(
        "AAA", datetime.combine(day, time(14, 24), timezone.utc), "news",
        available_at=datetime.combine(day, time(14, 24), timezone.utc),
    )
    after = Event(
        "BBB", datetime.combine(day, time(14, 26), timezone.utc), "news",
        available_at=datetime.combine(day, time(14, 26), timezone.utc),
    )
    visible = MarketNeutralBacktester._visible_events([before, after], day)
    assert visible == [before]


def test_aware_predecision_events_generate_signals_end_to_end():
    day = START + timedelta(days=61)
    events = [
        Event(
            "AAA", datetime.combine(day, time(14, 24), timezone.utc),
            "news", sentiment=1.0,
            available_at=datetime.combine(day, time(14, 24), timezone.utc),
        ),
        Event(
            "BBB", datetime.combine(day, time(14, 24), timezone.utc),
            "news", sentiment=-1.0,
            available_at=datetime.combine(day, time(14, 24), timezone.utc),
        ),
    ]
    result = _run(
        MarketNeutralBacktester(_strategy(), _config()), _bars(), events
    )
    assert _snapshot(result, 61).positions["AAA"] > 0
    assert _snapshot(result, 61).positions["BBB"] < 0


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
    assert sum(trade.pnl for trade in costly.trades) < sum(
        trade.pnl for trade in frictionless.trades
    )


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


def test_malformed_or_negative_borrow_inputs_fail_closed():
    with pytest.raises(ValueError, match="annual_borrow_rate cannot be negative"):
        BacktestConfig(annual_borrow_rate=-0.01)
    bars = _bars()
    days = sorted({row.date for row in bars})
    availability = {
        day.isoformat(): {"AAA": True, "BBB": "false", "SPY": True}
        for day in days
    }
    controls = {
        day.isoformat(): {"AAA": False, "BBB": None, "SPY": False}
        for day in days
    }
    rates = {
        day.isoformat(): {"AAA": 0.0, "BBB": -1.0, "SPY": 0.0}
        for day in days
    }
    result = MarketNeutralBacktester(_strategy(), _config()).run(
        bars,
        _events(),
        borrow_available=availability,
        borrow_rates=rates,
        ssr_restricted=controls,
        forced_cover=controls,
        pit_audit_report=PIT_AUDIT,
    )
    assert "BBB" not in _snapshot(result, 61).positions
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
        pit_audit_report=PIT_AUDIT,
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


def test_partial_exit_counts_one_round_trip_only_after_fully_closed():
    bars = [
        replace(row, open_volume=20_000.0)
        if row.date >= START + timedelta(days=64)
        else row
        for row in _bars(days=75)
    ]
    result = _run(
        MarketNeutralBacktester(
            _strategy(),
            _config(min_hold_days=0, max_hold_days=3, volume_participation=0.05),
        ),
        bars,
        _events(),
    )
    aaa_trades = [trade for trade in result.trades if trade.symbol == "AAA"]
    assert len(aaa_trades) == 1
    assert aaa_trades[0].fully_closed is True


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
    following = _snapshot(result, 63)
    assert 4.0 <= shock.drawdown_pct < 6.0
    assert shock.throttle == 1.0
    assert following.throttle == 0.75
    assert sum(abs(value) for value in following.target_weights.values()) <= 0.75 + 1e-12


def test_drawdown_reduction_overrides_minimum_holding_period():
    bars = _bars(days=64)
    shock_day = START + timedelta(days=62)
    shocked = [
        replace(row, open=90.0, high=100.0, low=89.0, close=90.0)
        if row.date == shock_day and row.symbol == "AAA" else row
        for row in bars
    ]
    result = _run(
        MarketNeutralBacktester(_strategy(), _config(min_hold_days=10)),
        shocked,
        _events(),
    )
    before = _snapshot(result, 61)
    reduced = _snapshot(result, 63)
    assert reduced.throttle == 0.75
    assert abs(reduced.positions["AAA"]) < abs(before.positions["AAA"])


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
    assert shock.halted is False
    assert following.halted
    assert following.throttle == 0.0
    assert following.target_weights == {}
    assert following.positions == {}


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


def test_walk_forward_folds_reset_state_and_freeze_parameters():
    bars = _bars(days=1_100, pit=True)
    events = []
    for offset in range(60, 1_090, 15):
        event_day = START + timedelta(days=offset)
        events.extend(
            [
                Event(
                    "AAA", datetime.combine(event_day, time(16)),
                    "news", sentiment=1.0,
                    available_at=datetime.combine(event_day, time(16)),
                ),
                Event(
                    "BBB", datetime.combine(event_day, time(16)),
                    "news", sentiment=-1.0,
                    available_at=datetime.combine(event_day, time(16)),
                ),
            ]
        )
    result = _run(
        MarketNeutralBacktester(_strategy(), _config()),
        bars,
        events,
        pit_borrow=True,
    )
    assert result.walk_forward
    assert all(fold["state_isolated"] for fold in result.walk_forward)
    assert all(fold["parameters_frozen"] for fold in result.walk_forward)
    assert all(fold["ending_positions_flat"] for fold in result.walk_forward)
    assert result.metrics["oos_liquidation_complete"] is True
    assert result.metrics["oos_closed_trades"] > 0
    report = result.markdown_report()
    assert "| Calmar |" in report
    assert "Out-of-sample regime contribution" in report


def test_cli_rejects_unknown_control_keys(tmp_path, capsys):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "bars": [
                    {
                        "symbol": "AAA",
                        "date": "2025-01-01",
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 1_000,
                    }
                ],
                "backtest_config": {"max_gros": 1.0},
            }
        ),
        encoding="utf-8",
    )
    assert backtest_main([str(dataset)]) == 2
    assert "unknown BacktestConfig keys: max_gros" in capsys.readouterr().err
