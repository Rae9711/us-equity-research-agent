"""Pure point-in-time event-enhanced market-neutral long/short strategy.

The module intentionally performs no data fetching and has no model/LLM
dependency.  Callers must provide timestamped observations; every calculation
first applies the decision-time boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    close: float
    volume: float
    spread_bps: float = 0.0
    impact_bps: float = 0.0


@dataclass(frozen=True)
class Event:
    symbol: str
    timestamp: datetime
    event_type: str
    value: float = 0.0
    expected: Optional[float] = None
    prior: Optional[float] = None
    sentiment: float = 0.0
    available_at: Optional[datetime] = None
    relevance: float = 1.0
    reaction: Optional[float] = None


@dataclass(frozen=True)
class StrategyConfig:
    momentum_20_weight: float = 0.25
    momentum_60_weight: float = 0.15
    earnings_surprise_weight: float = 0.25
    earnings_revision_weight: float = 0.15
    pead_weight: float = 0.10
    news_weight: float = 0.10
    min_price: float = 5.0
    min_average_dollar_volume: float = 10_000_000.0
    liquidity_lookback: int = 20
    volatility_lookback: int = 20
    beta_lookback: int = 60
    max_same_day_move: float = 0.12
    names_per_sector_side: int = 1
    min_names_per_sector: int = 2
    gross_target: float = 1.0
    max_abs_net: float = 0.10
    max_name_weight: float = 0.10
    max_sector_gross: float = 0.30
    predicted_alpha_bps_per_score: float = 100.0
    cost_gate_multiple: float = 2.0
    default_spread_bps: float = 5.0
    default_impact_bps: float = 5.0
    default_borrow_bps: float = 3.0
    min_holding_days: int = 2
    max_holding_days: int = 10
    spy_symbol: str = "SPY"
    beta_tolerance: float = 0.03

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "StrategyConfig":
        source = path or (
            Path(__file__).resolve().parents[2] / "config" / "strategies" / "event_ls.yaml"
        )
        raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
        flat: Dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        flat.update({key: value for key, value in raw.items() if not isinstance(value, dict)})
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in flat.items() if key in valid})


@dataclass
class Signal:
    symbol: str
    sector: str
    side: int
    raw_score: float
    sector_score: float
    predicted_alpha_bps: float
    estimated_cost_bps: float
    weight: float = 0.0
    hold_days: Tuple[int, int] = (2, 10)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    as_of: datetime
    signals: List[Signal]
    weights: Dict[str, float]
    gross_exposure: float
    net_exposure: float
    estimated_beta: float
    hedge_weight: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_sector_map(path: Optional[Path] = None) -> Dict[str, str]:
    source = path or (Path(__file__).resolve().parents[2] / "config" / "sector_map.yaml")
    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    symbols = raw.get("symbols", raw)
    return {str(symbol).upper(): str(sector) for symbol, sector in symbols.items()}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _returns(bars: Sequence[Bar]) -> List[float]:
    result: List[float] = []
    for previous, current in zip(bars, bars[1:]):
        if previous.close > 0:
            result.append(current.close / previous.close - 1.0)
    return result


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class MarketNeutralLongShortStrategy:
    """Generate a constrained event-enhanced long/short portfolio."""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        sector_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.config = config or StrategyConfig.from_yaml()
        self.sector_map = {
            key.upper(): value for key, value in (sector_map or load_sector_map()).items()
        }

    @staticmethod
    def _before_decision_day(timestamp: datetime, as_of: datetime) -> bool:
        return timestamp < as_of and timestamp.date() < as_of.date()

    def _split_bars(
        self, bars: Iterable[Bar], as_of: datetime
    ) -> Tuple[Dict[str, List[Bar]], Dict[str, Bar]]:
        history: Dict[str, List[Bar]] = {}
        intraday: Dict[str, Bar] = {}
        for bar in bars:
            if bar.timestamp > as_of:
                continue
            symbol = bar.symbol.upper()
            if self._before_decision_day(bar.timestamp, as_of):
                history.setdefault(symbol, []).append(bar)
            elif bar.timestamp.date() == as_of.date():
                previous = intraday.get(symbol)
                if previous is None or previous.timestamp < bar.timestamp:
                    intraday[symbol] = bar
        for rows in history.values():
            rows.sort(key=lambda item: item.timestamp)
        return history, intraday

    def _momentum(self, rows: Sequence[Bar], lookback: int) -> float:
        # The interval contains exactly ``lookback`` prior-close changes.
        if len(rows) <= lookback or rows[-lookback - 1].close <= 0:
            return 0.0
        return rows[-1].close / rows[-lookback - 1].close - 1.0

    def _event_features(
        self,
        symbol: str,
        events: Iterable[Event],
        rows: Sequence[Bar],
        as_of: datetime,
    ) -> Dict[str, float]:
        visible = [
            event
            for event in events
            if event.symbol.upper() == symbol
            and event.timestamp <= as_of
            and (event.available_at or event.timestamp) <= as_of
        ]
        visible.sort(key=lambda item: (item.available_at or item.timestamp, item.timestamp))
        surprise = revision = news = pead = 0.0
        last_earnings: Optional[Event] = None
        for event in visible:
            kind = event.event_type.lower()
            if kind in ("earnings", "earnings_surprise"):
                denominator = abs(event.expected or 0.0)
                surprise = (
                    (event.value - float(event.expected)) / denominator
                    if event.expected is not None and denominator > 0
                    else event.value
                )
                last_earnings = event
            elif kind in ("revision", "earnings_revision", "estimate_revision"):
                denominator = abs(event.prior or 0.0)
                revision = (
                    (event.value - float(event.prior)) / denominator
                    if event.prior is not None and denominator > 0
                    else event.value
                )
            elif kind in ("news", "headline"):
                # Sentiment/value is known only when the event itself is available.
                direction = event.sentiment if event.sentiment != 0 else event.value
                reaction = 1.0 if event.reaction is None else event.reaction
                news += direction * reaction * _clip(event.relevance, 0.0, 1.0)
        if last_earnings is not None:
            before = [bar for bar in rows if bar.timestamp < last_earnings.timestamp]
            after = [bar for bar in rows if bar.timestamp >= last_earnings.timestamp]
            if before and after and before[-1].close > 0:
                pead = after[-1].close / before[-1].close - 1.0
        return {
            "surprise": _clip(surprise, -2.0, 2.0),
            "revision": _clip(revision, -1.0, 1.0),
            "pead": _clip(pead, -0.25, 0.25),
            "news": _clip(news, -3.0, 3.0),
        }

    def _raw_candidates(
        self,
        bars: Iterable[Bar],
        events: Iterable[Event],
        as_of: datetime,
        borrow_available: Mapping[str, bool],
        borrow_bps: Mapping[str, float],
    ) -> List[Signal]:
        cfg = self.config
        history, intraday = self._split_bars(bars, as_of)
        candidates: List[Signal] = []
        for symbol, rows in history.items():
            sector = self.sector_map.get(symbol)
            if sector is None or not rows:
                continue
            last = rows[-1]
            liquidity_rows = rows[-cfg.liquidity_lookback :]
            average_dollar_volume = _mean(
                [bar.close * bar.volume for bar in liquidity_rows]
            )
            if last.close < cfg.min_price or average_dollar_volume < cfg.min_average_dollar_volume:
                continue
            live = intraday.get(symbol)
            same_day_move = (
                live.close / last.close - 1.0 if live is not None and last.close > 0 else 0.0
            )
            if abs(same_day_move) > cfg.max_same_day_move:
                continue
            event_features = self._event_features(symbol, events, rows, as_of)
            momentum_20 = self._momentum(rows, 20)
            momentum_60 = self._momentum(rows, 60)
            raw_score = (
                cfg.momentum_20_weight * momentum_20
                + cfg.momentum_60_weight * momentum_60
                + cfg.earnings_surprise_weight * event_features["surprise"]
                + cfg.earnings_revision_weight * event_features["revision"]
                + cfg.pead_weight * event_features["pead"]
                + cfg.news_weight * event_features["news"]
            )
            spread = live.spread_bps if live is not None else last.spread_bps
            impact = live.impact_bps if live is not None else last.impact_bps
            spread = spread if spread > 0 else cfg.default_spread_bps
            impact = impact if impact > 0 else cfg.default_impact_bps
            cost = spread + impact + float(borrow_bps.get(symbol, cfg.default_borrow_bps))
            candidates.append(
                Signal(
                    symbol=symbol,
                    sector=sector,
                    side=0,
                    raw_score=raw_score,
                    sector_score=0.0,
                    predicted_alpha_bps=0.0,
                    estimated_cost_bps=cost,
                    hold_days=(cfg.min_holding_days, cfg.max_holding_days),
                    metadata={
                        "momentum_20": momentum_20,
                        "momentum_60": momentum_60,
                        "same_day_move": same_day_move,
                        "average_dollar_volume": average_dollar_volume,
                        **event_features,
                    },
                )
            )
        return self._rank_and_gate(candidates, borrow_available)

    def _rank_and_gate(
        self, candidates: Sequence[Signal], borrow_available: Mapping[str, bool]
    ) -> List[Signal]:
        cfg = self.config
        selected: List[Signal] = []
        by_sector: Dict[str, List[Signal]] = {}
        for signal in candidates:
            by_sector.setdefault(signal.sector, []).append(signal)
        for sector, rows in by_sector.items():
            if len(rows) < cfg.min_names_per_sector:
                continue
            values = [row.raw_score for row in rows]
            center, scale = _mean(values), _std(values)
            for row in rows:
                row.sector_score = (row.raw_score - center) / scale if scale > 0 else 0.0
            ordered = sorted(rows, key=lambda item: (item.sector_score, item.symbol))
            count = min(cfg.names_per_sector_side, len(ordered) // 2)
            shorts = ordered[:count]
            longs = list(reversed(ordered[-count:]))
            for side, sleeve in ((-1, shorts), (1, longs)):
                for row in sleeve:
                    if side < 0 and not borrow_available.get(row.symbol, False):
                        continue
                    row.side = side
                    row.predicted_alpha_bps = (
                        abs(row.sector_score) * cfg.predicted_alpha_bps_per_score
                    )
                    if row.predicted_alpha_bps <= (
                        cfg.cost_gate_multiple * row.estimated_cost_bps
                    ):
                        continue
                    selected.append(row)
        # Keep sectors paired after borrow and cost gates, ensuring sector neutrality.
        paired: List[Signal] = []
        for sector in sorted({row.sector for row in selected}):
            longs = [row for row in selected if row.sector == sector and row.side > 0]
            shorts = [row for row in selected if row.sector == sector and row.side < 0]
            count = min(len(longs), len(shorts))
            paired.extend(longs[:count])
            paired.extend(shorts[:count])
        return paired

    def _volatility(self, rows: Sequence[Bar]) -> float:
        returns = _returns(rows[-self.config.volatility_lookback - 1 :])
        return max(_std(returns), 0.005)

    def _initial_weights(
        self, signals: Sequence[Signal], history: Mapping[str, Sequence[Bar]]
    ) -> Dict[str, float]:
        cfg = self.config
        weights: Dict[str, float] = {}
        sectors = sorted({signal.sector for signal in signals})
        if not sectors:
            return weights
        sector_gross = min(cfg.max_sector_gross, cfg.gross_target / len(sectors))
        total_sector_gross = sector_gross * len(sectors)
        target_gross = min(cfg.gross_target, total_sector_gross)
        for sector in sectors:
            sector_rows = [signal for signal in signals if signal.sector == sector]
            for side in (-1, 1):
                sleeve = [signal for signal in sector_rows if signal.side == side]
                inverse_risk = {
                    signal.symbol: 1.0 / self._volatility(history[signal.symbol])
                    for signal in sleeve
                }
                allocation = target_gross / (2.0 * len(sectors))
                remaining = allocation
                active = dict(inverse_risk)
                allocated: Dict[str, float] = {}
                while active and remaining > 1e-12:
                    denominator = sum(active.values())
                    capped = [
                        symbol
                        for symbol, risk_weight in active.items()
                        if remaining * risk_weight / denominator >= cfg.max_name_weight
                    ]
                    if not capped:
                        for symbol, risk_weight in active.items():
                            allocated[symbol] = remaining * risk_weight / denominator
                        break
                    for symbol in capped:
                        allocated[symbol] = cfg.max_name_weight
                        remaining -= cfg.max_name_weight
                        del active[symbol]
                for symbol, value in allocated.items():
                    weights[symbol] = side * value
        return weights

    def _rolling_betas(
        self, history: Mapping[str, Sequence[Bar]]
    ) -> Dict[str, float]:
        spy = history.get(self.config.spy_symbol, ())
        spy_returns_by_time = {
            bar.timestamp: value
            for bar, value in zip(spy[1:], _returns(spy))
        }
        result: Dict[str, float] = {self.config.spy_symbol: 1.0}
        for symbol, rows in history.items():
            if symbol == self.config.spy_symbol:
                continue
            pairs: List[Tuple[float, float]] = []
            for previous, current in zip(rows, rows[1:]):
                market_return = spy_returns_by_time.get(current.timestamp)
                if market_return is not None and previous.close > 0:
                    pairs.append((current.close / previous.close - 1.0, market_return))
            pairs = pairs[-self.config.beta_lookback :]
            if len(pairs) < 2:
                result[symbol] = 1.0
                continue
            stock = [pair[0] for pair in pairs]
            market = [pair[1] for pair in pairs]
            market_mean = _mean(market)
            variance = sum((value - market_mean) ** 2 for value in market)
            covariance = sum(
                (left - _mean(stock)) * (right - market_mean)
                for left, right in pairs
            )
            result[symbol] = covariance / variance if variance > 0 else 1.0
        return result

    def generate(
        self,
        bars: Iterable[Bar],
        events: Iterable[Event],
        as_of: datetime,
        borrow_available: Optional[Mapping[str, bool]] = None,
        borrow_bps: Optional[Mapping[str, float]] = None,
    ) -> Portfolio:
        """Return the decision-time portfolio using observations available by ``as_of``."""
        bar_rows = list(bars)
        event_rows = list(events)
        borrow = {key.upper(): value for key, value in (borrow_available or {}).items()}
        borrow_cost = {key.upper(): value for key, value in (borrow_bps or {}).items()}
        signals = self._raw_candidates(bar_rows, event_rows, as_of, borrow, borrow_cost)
        history, _ = self._split_bars(bar_rows, as_of)
        weights = self._initial_weights(signals, history)
        betas = self._rolling_betas(history)
        stock_beta = sum(weights.get(symbol, 0.0) * betas.get(symbol, 1.0) for symbol in weights)
        hedge = _clip(-stock_beta, -self.config.max_abs_net, self.config.max_abs_net)
        if abs(hedge) > 1e-12:
            weights[self.config.spy_symbol] = weights.get(self.config.spy_symbol, 0.0) + hedge
        gross = sum(abs(value) for value in weights.values())
        if gross > self.config.gross_target:
            scale = self.config.gross_target / gross
            weights = {symbol: value * scale for symbol, value in weights.items()}
        for signal in signals:
            signal.weight = weights.get(signal.symbol, 0.0)
        gross = sum(abs(value) for value in weights.values())
        net = sum(weights.values())
        portfolio_beta = sum(
            value * (1.0 if symbol == self.config.spy_symbol else betas.get(symbol, 1.0))
            for symbol, value in weights.items()
        )
        return Portfolio(
            as_of=as_of,
            signals=signals,
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            estimated_beta=portfolio_beta,
            hedge_weight=weights.get(self.config.spy_symbol, 0.0),
            metadata={
                "point_in_time": True,
                "hold_days": (
                    self.config.min_holding_days,
                    self.config.max_holding_days,
                ),
                "beta_tolerance": self.config.beta_tolerance,
            },
        )

