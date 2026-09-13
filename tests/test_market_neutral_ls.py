from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from src.strategies.market_neutral_ls import (
    Bar,
    Event,
    MarketNeutralLongShortStrategy,
    StrategyConfig,
    load_sector_map,
)


AS_OF = datetime(2026, 9, 12, 12, 0)
SYMBOLS = ("TA", "TB", "FA", "FB")
SECTORS = {"TA": "Tech", "TB": "Tech", "FA": "Finance", "FB": "Finance"}


def config(**changes):
    base = StrategyConfig(
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
        max_name_weight=0.30,
        max_sector_gross=0.60,
        default_spread_bps=1.0,
        default_impact_bps=1.0,
        default_borrow_bps=1.0,
        predicted_alpha_bps_per_score=100.0,
    )
    return replace(base, **changes)


def bars(beta_by_symbol=None):
    beta_by_symbol = beta_by_symbol or {symbol: 1.0 for symbol in SYMBOLS}
    rows = []
    prices = {symbol: 100.0 for symbol in SYMBOLS}
    spy = 100.0
    start = AS_OF - timedelta(days=75)
    for day in range(70):
        timestamp = (start + timedelta(days=day)).replace(hour=16)
        market_return = 0.002 if day % 2 == 0 else -0.001
        spy *= 1.0 + market_return
        rows.append(Bar("SPY", timestamp, spy, 10_000_000))
        for symbol in SYMBOLS:
            prices[symbol] *= 1.0 + beta_by_symbol[symbol] * market_return
            rows.append(Bar(symbol, timestamp, prices[symbol], 1_000_000))
    return rows


def events():
    timestamp = AS_OF - timedelta(days=1)
    return [
        Event("TA", timestamp, "news", sentiment=1.0, available_at=timestamp),
        Event("TB", timestamp, "news", sentiment=-1.0, available_at=timestamp),
        Event("FA", timestamp, "news", sentiment=0.8, available_at=timestamp),
        Event("FB", timestamp, "news", sentiment=-0.8, available_at=timestamp),
    ]


def strategy(**changes):
    return MarketNeutralLongShortStrategy(config(**changes), SECTORS)


def all_borrowable():
    return {symbol: True for symbol in SYMBOLS}


def test_yaml_configs_load():
    loaded = StrategyConfig.from_yaml()
    sectors = load_sector_map()
    assert loaded.min_holding_days == 2
    assert loaded.max_holding_days == 10
    assert sectors["AAPL"] == "Technology"


def test_no_lookahead_from_future_bar_or_news():
    engine = strategy()
    baseline = engine.generate(bars(), events(), AS_OF, all_borrowable())
    contaminated = engine.generate(
        bars()
        + [
            Bar(
                "TB",
                AS_OF + timedelta(minutes=1),
                10_000.0,
                1_000_000,
            )
        ],
        events()
        + [
            Event(
                "TB",
                AS_OF - timedelta(hours=1),
                "news",
                sentiment=100.0,
                available_at=AS_OF + timedelta(minutes=1),
            )
        ],
        AS_OF,
        all_borrowable(),
    )
    assert contaminated.weights == pytest.approx(baseline.weights)
    assert [signal.symbol for signal in contaminated.signals] == [
        signal.symbol for signal in baseline.signals
    ]


def test_stale_news_expires_and_future_reaction_is_ignored():
    engine = strategy(news_freshness_days=3)
    stale = [
        replace(
            event,
            timestamp=AS_OF - timedelta(days=4),
            available_at=AS_OF - timedelta(days=4),
        )
        for event in events()
    ]
    assert engine.generate(
        bars(), stale, AS_OF, all_borrowable()
    ).signals == []

    baseline = engine.generate(bars(), events(), AS_OF, all_borrowable())
    contaminated = engine.generate(
        bars(),
        [
            replace(
                event,
                reaction=-100.0,
                reaction_known_at=AS_OF + timedelta(minutes=1),
            )
            for event in events()
        ],
        AS_OF,
        all_borrowable(),
    )
    assert contaminated.weights == pytest.approx(baseline.weights)


def test_within_sector_selection_is_sector_neutral():
    portfolio = strategy().generate(bars(), events(), AS_OF, all_borrowable())
    for sector in ("Tech", "Finance"):
        sector_weight = sum(
            weight
            for symbol, weight in portfolio.weights.items()
            if SECTORS.get(symbol) == sector
        )
        assert sector_weight == pytest.approx(0.0, abs=1e-12)
    assert portfolio.gross_exposure == pytest.approx(1.0)
    assert abs(portfolio.net_exposure) <= portfolio.metadata["beta_tolerance"] + 0.10
    assert all(signal.hold_days == (2, 10) for signal in portfolio.signals)


def test_unborrowable_short_is_removed_with_unpaired_long():
    borrow = all_borrowable()
    borrow["TB"] = False
    portfolio = strategy().generate(bars(), events(), AS_OF, borrow)
    assert "TB" not in portfolio.weights
    assert "TA" not in portfolio.weights
    assert {signal.symbol for signal in portfolio.signals} == {"FA", "FB"}


def test_cost_gate_requires_alpha_greater_than_twice_total_cost():
    portfolio = strategy(predicted_alpha_bps_per_score=1.0).generate(
        bars(),
        events(),
        AS_OF,
        all_borrowable(),
        {symbol: 10.0 for symbol in SYMBOLS},
    )
    assert portfolio.signals == []
    assert portfolio.weights == {}


def test_rolling_beta_is_hedged_with_spy_to_near_zero():
    beta = {"TA": 1.1, "TB": 0.9, "FA": 1.1, "FB": 0.9}
    portfolio = strategy().generate(
        bars(beta),
        events(),
        AS_OF,
        all_borrowable(),
    )
    assert "SPY" in portfolio.weights
    assert portfolio.hedge_weight < 0
    assert abs(portfolio.estimated_beta) < 1e-10
    assert portfolio.gross_exposure == pytest.approx(1.0)
    assert abs(portfolio.net_exposure) <= 0.10
