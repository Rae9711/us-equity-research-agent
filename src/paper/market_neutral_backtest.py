"""Point-in-time daily portfolio backtester for the market-neutral strategy.

The engine is deliberately data-source agnostic: callers supply daily OHLCV
bars and timestamped strategy events.  Decisions at an opening auction only
receive bars and events from earlier calendar days.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.paper.acceptance import acceptance_markdown_report, evaluate_backtest_acceptance
from src.paper.portfolio_controls import drawdown_throttle, exposure_snapshot
from src.paper.walk_forward import WalkForwardConfig, splits_from_config
from src.strategies.market_neutral_ls import (
    Bar,
    Event,
    MarketNeutralLongShortStrategy,
)

_ET = ZoneInfo("America/New_York")
_DECISION_TIME = time(9, 25)
_REQUIRED_PIT_CATEGORIES = {
    "daily", "intraday", "news", "earnings", "borrow", "universe", "macro"
}


def _date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time())
    else:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    # The engine's opening decision is 09:30 New York time. Normalize every
    # aware vendor timestamp to that wall clock before dropping the timezone;
    # naive inputs are documented as New York local time.
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return parsed.astimezone(_ET).replace(tzinfo=None)
    return parsed


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _dated_values(
    values: Optional[Mapping[str, Any]],
    day: date,
) -> Tuple[Dict[str, Any], bool]:
    """Resolve a dated symbol map and report whether it is point-in-time keyed."""
    if not values:
        return {}, False
    dated = values.get(day.isoformat())
    if dated is None:
        dated = values.get(day)  # type: ignore[arg-type]
    if isinstance(dated, Mapping):
        return {str(key).upper(): value for key, value in dated.items()}, True
    if any(isinstance(value, Mapping) for value in values.values()):
        return {}, False
    return {str(key).upper(): value for key, value in values.items()}, False


@dataclass(frozen=True)
class DailyBar:
    """Generic daily market input; no vendor-specific fields are required."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_volume: Optional[float] = None
    open_spread_bps: Optional[float] = None
    open_impact_bps: Optional[float] = None
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    halted: bool = False
    sector: Optional[str] = None
    beta: Optional[float] = None
    known_at: Optional[datetime] = None
    open_known_at: Optional[datetime] = None
    member_from: Optional[date] = None
    member_to: Optional[date] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "date", _date(self.date))
        for name in ("open", "high", "low", "close", "volume"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be at least open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be at most open, close, and high")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        if self.open_volume is not None:
            object.__setattr__(
                self, "open_volume", _finite(self.open_volume, "open_volume")
            )
            if self.open_volume < 0:
                raise ValueError("open_volume cannot be negative")
        for name in ("open_spread_bps", "open_impact_bps"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
                if value < 0:
                    raise ValueError("{} cannot be negative".format(name))
        if self.known_at is not None:
            object.__setattr__(self, "known_at", _datetime(self.known_at))
        if self.open_known_at is not None:
            object.__setattr__(
                self, "open_known_at", _datetime(self.open_known_at)
            )
        if self.member_from is not None:
            object.__setattr__(self, "member_from", _date(self.member_from))
        if self.member_to is not None:
            object.__setattr__(self, "member_to", _date(self.member_to))
        if (
            self.member_from is not None
            and self.member_to is not None
            and self.member_from >= self.member_to
        ):
            raise ValueError("member_from must precede member_to")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "DailyBar":
        timestamp = row.get("date", row.get("timestamp"))
        if timestamp is None:
            raise ValueError("bar requires date or timestamp")
        return cls(
            symbol=str(row["symbol"]),
            date=_date(timestamp),
            open=float(row["open"]),
            high=float(row.get("high", max(float(row["open"]), float(row["close"])))),
            low=float(row.get("low", min(float(row["open"]), float(row["close"])))),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0) or 0.0),
            open_volume=(
                float(row["open_volume"])
                if row.get("open_volume") is not None else None
            ),
            open_spread_bps=(
                float(row["open_spread_bps"])
                if row.get("open_spread_bps") is not None else None
            ),
            open_impact_bps=(
                float(row["open_impact_bps"])
                if row.get("open_impact_bps") is not None else None
            ),
            spread_bps=float(row.get("spread_bps", 0.0) or 0.0),
            impact_bps=float(row.get("impact_bps", 0.0) or 0.0),
            halted=bool(row.get("halted", False)),
            sector=row.get("sector"),
            beta=float(row["beta"]) if row.get("beta") is not None else None,
            known_at=_datetime(row["known_at"]) if row.get("known_at") else None,
            open_known_at=(
                _datetime(row["open_known_at"])
                if row.get("open_known_at") else None
            ),
            member_from=_date(row["member_from"]) if row.get("member_from") else None,
            member_to=_date(row["member_to"]) if row.get("member_to") else None,
        )


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    volume_participation: float = 0.05
    default_spread_bps: float = 5.0
    slippage_bps: float = 2.0
    default_impact_bps: float = 3.0
    impact_curve_bps: float = 20.0
    annual_borrow_rate: float = 0.03
    trading_days_per_year: int = 252
    min_hold_days: int = 2
    max_hold_days: int = 10
    max_gross: float = 1.0
    max_abs_net: float = 0.10
    max_sector_gross: float = 0.15
    max_abs_beta: float = 0.05
    max_daily_loss_pct: float = 1.0
    purge_days: int = 5
    embargo_days: int = 5
    walk_forward_train_months: int = 24
    walk_forward_validation_months: int = 3
    walk_forward_test_months: int = 3
    high_volatility_annualized: float = 0.25
    bull_bear_lookback_return: float = 0.05

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < self.volume_participation <= 1:
            raise ValueError("volume_participation must be in (0, 1]")
        if self.min_hold_days < 0 or self.max_hold_days < self.min_hold_days:
            raise ValueError("invalid holding period")
        if min(self.max_gross, self.max_sector_gross) <= 0:
            raise ValueError("exposure limits must be positive")
        if self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be positive")
        for name in (
            "default_spread_bps",
            "slippage_bps",
            "default_impact_bps",
            "impact_curve_bps",
            "annual_borrow_rate",
            "high_volatility_annualized",
            "bull_bear_lookback_return",
        ):
            value = _finite(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value < 0:
                raise ValueError("{} cannot be negative".format(name))
        if self.high_volatility_annualized <= 0:
            raise ValueError("high_volatility_annualized must be positive")
        if self.bull_bear_lookback_return <= 0:
            raise ValueError("bull_bear_lookback_return must be positive")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")


@dataclass
class Position:
    symbol: str
    shares: float
    average_price: float
    opened_on: date
    opened_index: int
    sector: str
    beta: float
    realized_pnl: float = 0.0
    borrow_cost: float = 0.0
    closed_shares: float = 0.0

    def market_value(self, price: float) -> float:
        return self.shares * price


@dataclass(frozen=True)
class Fill:
    date: date
    symbol: str
    requested_shares: float
    filled_shares: float
    price: float
    reference_price: float
    cost: float
    reason: str
    partial: bool

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["date"] = self.date.isoformat()
        return row


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    side: str
    opened_on: date
    closed_on: date
    shares: float
    pnl: float
    holding_days: int
    reason: str
    sector: str = "UNKNOWN"
    fully_closed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["opened_on"] = self.opened_on.isoformat()
        row["closed_on"] = self.closed_on.isoformat()
        return row


@dataclass
class DailySnapshot:
    date: date
    equity: float
    cash: float
    daily_return: float
    drawdown_pct: float
    throttle: float
    gross_exposure: float
    net_exposure: float
    beta_exposure: float
    sector_exposure: Dict[str, float]
    turnover: float
    transaction_cost: float
    borrow_cost: float
    long_pnl: float
    short_pnl: float
    capacity_usd: float = 0.0
    target_weights: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, float] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    blocked_orders: Dict[str, str] = field(default_factory=dict)
    feature_cutoff: Optional[date] = None
    market_regime: str = "insufficient_history"
    halted: bool = False
    new_entries_halted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["date"] = self.date.isoformat()
        row["feature_cutoff"] = (
            self.feature_cutoff.isoformat() if self.feature_cutoff is not None else None
        )
        row["fills"] = [fill.to_dict() for fill in self.fills]
        return row


@dataclass
class BacktestResult:
    config: BacktestConfig
    snapshots: List[DailySnapshot]
    trades: List[ClosedTrade]
    metrics: Dict[str, Any]
    walk_forward: List[Dict[str, Any]]
    acceptance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "snapshots": [row.to_dict() for row in self.snapshots],
            "trades": [row.to_dict() for row in self.trades],
            "metrics": self.metrics,
            "walk_forward": self.walk_forward,
            "acceptance": self.acceptance,
        }

    def markdown_report(self) -> str:
        metrics = self.metrics
        lines = [
            "# Market-Neutral Backtest",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Ending equity | ${:,.2f} |".format(metrics["ending_equity"]),
            "| Total return | {:.2f}% |".format(metrics["total_return_pct"]),
            "| Sharpe | {:.3f} |".format(metrics["sharpe"]),
            "| Maximum drawdown | {:.2f}% |".format(metrics["max_drawdown_pct"]),
            "| Calmar | {:.3f} |".format(metrics["calmar"]),
            "| Profit factor | {:.3f} |".format(metrics["profit_factor"]),
            "| Closed trades | {} |".format(metrics["closed_trades"]),
            "| Annualized turnover | {:.2f}x |".format(metrics["annualized_turnover"]),
            "| Capacity estimate | ${:,.0f} |".format(metrics["capacity_estimate_usd"]),
            "| Long contribution | ${:,.2f} |".format(metrics["long_contribution"]),
            "| Short contribution | ${:,.2f} |".format(metrics["short_contribution"]),
            "",
            "## Out-of-sample summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Trades | {} |".format(metrics.get("oos_closed_trades", 0)),
            "| Maximum drawdown | {:.2f}% |".format(
                metrics.get("oos_max_drawdown_pct", 0.0)
            ),
            "| Sharpe | {:.3f} |".format(metrics.get("oos_sharpe", 0.0)),
            "| Calmar | {:.3f} |".format(metrics.get("oos_calmar", 0.0)),
            "| Profit factor | {:.3f} |".format(
                metrics.get("oos_profit_factor", 0.0)
            ),
            "| Annualized turnover | {:.2f}x |".format(
                metrics.get("oos_annualized_turnover", 0.0)
            ),
            "| Capacity estimate | ${:,.0f} |".format(
                metrics.get("oos_capacity_estimate_usd", 0.0)
            ),
            "| Long contribution | ${:,.2f} |".format(
                metrics.get("oos_long_contribution", 0.0)
            ),
            "| Short contribution | ${:,.2f} |".format(
                metrics.get("oos_short_contribution", 0.0)
            ),
            "",
            "## Out-of-sample monthly returns",
            "",
            "| Month | Return |",
            "|---|---:|",
        ]
        for row in metrics.get("oos_monthly_returns", []):
            lines.append("| {} | {:.2f}% |".format(row["month"], row["return_pct"]))
        lines.extend([
            "",
            "## Out-of-sample regime contribution",
            "",
            "| Regime | Trading days | Contribution |",
            "|---|---:|---:|",
        ])
        regime_days = metrics.get("oos_regime_days") or {}
        regime_contribution = metrics.get("oos_regime_contribution") or {}
        for regime in (
            "bull", "bear", "high_volatility", "sideways",
            "insufficient_history",
        ):
            lines.append("| {} | {} | ${:,.2f} |".format(
                regime,
                regime_days.get(regime, 0),
                regime_contribution.get(regime, 0.0),
            ))
        lines.extend(["", acceptance_markdown_report(
            evaluate_backtest_acceptance(metrics)
        ).rstrip()])
        return "\n".join(lines) + "\n"


class MarketNeutralBacktester:
    """Simulate daily opening rebalances with signed shares and explicit cash."""

    def __init__(
        self,
        strategy: MarketNeutralLongShortStrategy,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def _strategy_bars(
        self, rows: Sequence[DailyBar], decision_day: date
    ) -> List[Bar]:
        decision_at = datetime.combine(decision_day, _DECISION_TIME)
        return [
            Bar(
                symbol=row.symbol,
                timestamp=datetime.combine(row.date, time(16)),
                close=row.close,
                volume=row.volume,
                spread_bps=row.spread_bps,
                impact_bps=row.impact_bps,
            )
            for row in rows
            if row.date < decision_day
            and (row.known_at is None or row.known_at <= decision_at)
            and (
                row.member_from is None
                or row.member_to is None
                or row.member_from <= decision_day < row.member_to
            )
        ]

    @staticmethod
    def _visible_events(events: Sequence[Event], decision_day: date) -> List[Event]:
        decision_at = datetime.combine(decision_day, _DECISION_TIME)
        visible = []
        for event in events:
            if event.available_at is None:
                continue
            happened = _datetime(event.timestamp)
            available = _datetime(event.available_at)
            if happened <= decision_at and available <= decision_at:
                visible.append(event)
        return visible

    def _betas(
        self,
        history: Sequence[Bar],
        symbols: Iterable[str],
        strategy: Optional[MarketNeutralLongShortStrategy] = None,
    ) -> Dict[str, float]:
        by_symbol: Dict[str, List[Bar]] = {}
        for row in history:
            by_symbol.setdefault(row.symbol, []).append(row)
        calculated = (strategy or self.strategy)._rolling_betas(by_symbol)
        return {symbol: calculated.get(symbol, 1.0) for symbol in symbols}

    def _market_regime(self, history: Sequence[Bar]) -> str:
        spy = sorted(
            (row for row in history if row.symbol == "SPY"),
            key=lambda row: row.timestamp,
        )
        if len(spy) < 61:
            return "insufficient_history"
        recent_returns = [
            current.close / previous.close - 1.0
            for previous, current in zip(spy[-21:-1], spy[-20:])
            if previous.close > 0
        ]
        volatility = (
            statistics.stdev(recent_returns)
            * math.sqrt(self.config.trading_days_per_year)
            if len(recent_returns) > 1 else 0.0
        )
        if volatility >= self.config.high_volatility_annualized:
            return "high_volatility"
        lookback_return = spy[-1].close / spy[-61].close - 1.0
        if lookback_return >= self.config.bull_bear_lookback_return:
            return "bull"
        if lookback_return <= -self.config.bull_bear_lookback_return:
            return "bear"
        return "sideways"

    def _constrain_weights(
        self,
        raw: Mapping[str, float],
        sectors: Mapping[str, str],
        betas: Mapping[str, float],
        throttle: float,
    ) -> Dict[str, float]:
        cfg = self.config
        weights = {symbol: float(value) for symbol, value in raw.items() if abs(value) > 1e-12}
        # First cap sector gross. SPY is the beta hedge, not an industry sleeve.
        for sector in sorted(set(sectors.values())):
            names = [s for s in weights if s != "SPY" and sectors.get(s, "UNKNOWN") == sector]
            sector_gross = sum(abs(weights[s]) for s in names)
            cap = cfg.max_sector_gross * throttle
            if sector_gross > cap and sector_gross > 0:
                scale = cap / sector_gross
                for symbol in names:
                    weights[symbol] *= scale
        gross_cap = cfg.max_gross * throttle
        gross = sum(abs(value) for value in weights.values())
        if gross > gross_cap and gross > 0:
            weights = {symbol: value * gross_cap / gross for symbol, value in weights.items()}

        # Recompute the hedge after scaling the stock book.
        stock_beta = sum(
            value * betas.get(symbol, 1.0)
            for symbol, value in weights.items()
            if symbol != "SPY"
        )
        hedge = max(-cfg.max_abs_net, min(cfg.max_abs_net, -stock_beta))
        if abs(stock_beta + hedge) > cfg.max_abs_beta:
            # A hedge constrained by net exposure cannot fix this book; scale
            # stock risk to the maximum beta that the allowed hedge can offset.
            permitted = cfg.max_abs_net + cfg.max_abs_beta
            scale = min(1.0, permitted / abs(stock_beta)) if stock_beta else 1.0
            for symbol in list(weights):
                if symbol != "SPY":
                    weights[symbol] *= scale
            stock_beta *= scale
            hedge = max(-cfg.max_abs_net, min(cfg.max_abs_net, -stock_beta))
        if abs(hedge) > 1e-12:
            weights["SPY"] = hedge
        else:
            weights.pop("SPY", None)

        net = sum(weights.values())
        if abs(net) > cfg.max_abs_net and "SPY" in weights:
            weights["SPY"] -= math.copysign(abs(net) - cfg.max_abs_net, net)
        gross = sum(abs(value) for value in weights.values())
        if gross > gross_cap and gross > 0:
            weights = {symbol: value * gross_cap / gross for symbol, value in weights.items()}
        return weights

    def run(
        self,
        bars: Iterable[DailyBar],
        events: Iterable[Event] = (),
        *,
        borrow_available: Optional[Mapping[str, Any]] = None,
        borrow_rates: Optional[Mapping[str, Any]] = None,
        ssr_restricted: Optional[Mapping[str, Any]] = None,
        forced_cover: Optional[Mapping[str, Any]] = None,
        pit_audit_report: Optional[Mapping[str, Any]] = None,
        _evaluation_start: Optional[date] = None,
        _evaluation_end: Optional[date] = None,
        _compute_walk_forward: bool = True,
    ) -> BacktestResult:
        cfg = self.config
        rows = sorted(list(bars), key=lambda row: (row.date, row.symbol))
        event_rows = list(events)
        if not rows:
            raise ValueError("at least one bar is required")
        duplicates = [(row.date, row.symbol) for row in rows]
        if len(duplicates) != len(set(duplicates)):
            raise ValueError("only one daily bar per date and symbol is allowed")

        by_day: Dict[date, Dict[str, DailyBar]] = {}
        for row in rows:
            by_day.setdefault(row.date, {})[row.symbol] = row
        days = [
            day
            for day in sorted(by_day)
            if _evaluation_end is None or day <= _evaluation_end
        ]
        cash = cfg.initial_cash
        positions: Dict[str, Position] = {}
        snapshots: List[DailySnapshot] = []
        trades: List[ClosedTrade] = []
        peak = cfg.initial_cash
        previous_equity = cfg.initial_cash
        previous_closes: Dict[str, float] = {}
        permanently_halted = False
        capacity_samples: List[float] = []
        dated_borrow_complete = bool(
            borrow_available and borrow_rates and ssr_restricted and forced_cover
        )
        missing_position_days: Dict[str, int] = {}
        stale_mark_detected = False
        execution_strategy_config = replace(
            self.strategy.config,
            default_spread_bps=cfg.default_spread_bps,
            default_impact_bps=cfg.default_impact_bps,
            default_slippage_bps=cfg.slippage_bps,
        )

        for day_index, day in enumerate(days):
            market = by_day[day]
            for symbol in positions:
                if symbol in market:
                    missing_position_days[symbol] = 0
                else:
                    missing_position_days[symbol] = (
                        missing_position_days.get(symbol, 0) + 1
                    )
                    if missing_position_days[symbol] > 1:
                        stale_mark_detected = True
            decision_at = datetime.combine(day, _DECISION_TIME)
            known_history = [
                row
                for row in rows
                if row.date < day
                and (row.known_at is None or row.known_at <= decision_at)
            ]
            sectors: Dict[str, str] = {}
            explicit_betas: Dict[str, float] = {}
            for row in known_history:
                if row.sector:
                    sectors[row.symbol] = row.sector
                if row.beta is not None:
                    explicit_betas[row.symbol] = row.beta
            day_strategy = MarketNeutralLongShortStrategy(
                execution_strategy_config, sectors
            )
            availability_values, availability_is_dated = _dated_values(
                borrow_available, day
            )
            rate_values, rates_are_dated = _dated_values(borrow_rates, day)
            ssr_values, ssr_is_dated = _dated_values(ssr_restricted, day)
            forced_cover_values, forced_cover_is_dated = _dated_values(
                forced_cover, day
            )
            dated_borrow_complete = (
                dated_borrow_complete
                and availability_is_dated
                and rates_are_dated
                and ssr_is_dated
                and forced_cover_is_dated
                and set(market).issubset(availability_values)
                and set(market).issubset(rate_values)
                and set(market).issubset(ssr_values)
                and set(market).issubset(forced_cover_values)
            )
            valid_rates = {}
            for symbol, value in rate_values.items():
                try:
                    rate = _finite(value, "borrow_rate")
                except (TypeError, ValueError):
                    continue
                if rate >= 0:
                    valid_rates[symbol] = rate
            rates = valid_rates
            # A locate without its contemporaneous fee is incomplete and cannot
            # support a short return. Missing data always means unavailable.
            control_values_valid = (
                all(type(value) is bool for value in availability_values.values())
                and all(type(value) is bool for value in ssr_values.values())
                and all(type(value) is bool for value in forced_cover_values.values())
                and set(rate_values) == set(rates)
            )
            dated_borrow_complete = dated_borrow_complete and control_values_valid
            borrowable = {
                symbol: value is True and symbol in rates
                for symbol, value in availability_values.items()
            }
            short_entry_allowed = {
                symbol: allowed
                and ssr_values.get(symbol) is False
                and forced_cover_values.get(symbol) is False
                for symbol, allowed in borrowable.items()
            }
            decision_equity = cash + sum(
                position.shares * (
                    previous_closes[symbol]
                    if symbol in previous_closes else position.average_price
                )
                for symbol, position in positions.items()
            )
            peak = max(peak, decision_equity)
            decision_drawdown = 100.0 * max(
                0.0, 1.0 - decision_equity / peak
            )
            throttle = drawdown_throttle(decision_drawdown)
            if throttle == 0.0:
                permanently_halted = True
            new_entries_halted = bool(
                snapshots
                and snapshots[-1].daily_return
                <= -(cfg.max_daily_loss_pct / 100.0)
            )

            history = self._strategy_bars(rows, day)
            market_regime = self._market_regime(history)
            visible_events = self._visible_events(event_rows, day)
            target_weights: Dict[str, float] = {}
            betas = self._betas(
                history, set(market) | set(positions), day_strategy
            )
            betas.update(explicit_betas)
            betas["SPY"] = 1.0
            for symbol, position in positions.items():
                position.beta = betas.get(symbol, position.beta)
                position.sector = sectors.get(symbol, position.sector)
            evaluation_active = (
                (_evaluation_start is None or day >= _evaluation_start)
                and (_evaluation_end is None or day <= _evaluation_end)
            )
            final_evaluation_day = (
                _evaluation_end is not None and day == _evaluation_end
            )
            if (
                not permanently_halted
                and evaluation_active
                and not final_evaluation_day
            ):
                portfolio = day_strategy.generate(
                    history,
                    visible_events,
                    datetime.combine(day, _DECISION_TIME),
                    short_entry_allowed,
                    {symbol: (
                        rates.get(symbol, cfg.annual_borrow_rate)
                        * 10_000.0 * cfg.max_hold_days
                        / cfg.trading_days_per_year
                    )
                     for symbol in borrowable},
                )
                target_weights = self._constrain_weights(
                    portfolio.weights, sectors, betas, throttle
                )

            # Convert desired notionals into opening shares, then enforce hold
            # rules against the actual position (not the unfilled target).
            desired: Dict[str, float] = {}
            all_symbols = set(positions) | set(target_weights)
            forced_symbols = set()
            for symbol in all_symbols:
                bar = market.get(symbol)
                if bar is None:
                    continue
                target = (
                    target_weights.get(symbol, 0.0)
                    * decision_equity
                    / bar.open
                )
                current = positions.get(symbol)
                if current is not None:
                    held = day_index - current.opened_index
                    reducing_or_reversing = (
                        abs(target) < abs(current.shares)
                        or target * current.shares <= 0
                    )
                    if permanently_halted:
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif final_evaluation_day:
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif current.shares < 0 and not borrowable.get(symbol, False):
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif current.shares < 0 and (
                        forced_cover_values.get(symbol) is True
                    ):
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif held >= cfg.max_hold_days:
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif held < cfg.min_hold_days and reducing_or_reversing:
                        # Risk throttles may always reduce an existing sleeve;
                        # the minimum hold only blocks discretionary exits or
                        # reversals.
                        if not (
                            throttle < 1.0
                            and abs(target) < abs(current.shares)
                        ):
                            target = current.shares
                        elif target * current.shares <= 0:
                            target = 0.0
                    elif new_entries_halted:
                        if target * current.shares <= 0:
                            target = 0.0
                        elif abs(target) > abs(current.shares):
                            target = current.shares
                elif new_entries_halted:
                    target = 0.0
                desired[symbol] = target

            fills: List[Fill] = []
            blocked: Dict[str, str] = {}
            day_cost = 0.0
            day_turnover = 0.0
            day_long_pnl = 0.0
            day_short_pnl = 0.0
            day_capacity_samples: List[float] = []
            opening_shares = {
                symbol: position.shares for symbol, position in positions.items()
            }
            # Non-emergency rebalances are filled at one common participation
            # ratio. This keeps long/short sleeves and the hedge on the line
            # between the already-held book and the constrained target.
            rebalance_scale = 1.0
            regular_orders = []
            for symbol in sorted(all_symbols - forced_symbols):
                current_shares = positions[symbol].shares if symbol in positions else 0.0
                request = desired.get(symbol, 0.0) - current_shares
                if abs(request) <= 1e-10:
                    continue
                regular_orders.append(symbol)
                bar = market.get(symbol)
                if (
                    bar is None
                    or bar.halted
                    or bar.open_volume is None
                    or bar.open_volume <= 0
                ):
                    rebalance_scale = 0.0
                    break
                rebalance_scale = min(
                    rebalance_scale,
                    bar.open_volume * cfg.volume_participation / abs(request),
                )
            for symbol in sorted(all_symbols):
                current = positions.get(symbol)
                current_shares = current.shares if current else 0.0
                requested = desired.get(symbol, 0.0) - current_shares
                if abs(requested) <= 1e-10:
                    continue
                if symbol in regular_orders and rebalance_scale <= 0:
                    blocked[symbol] = "atomic_rebalance_blocked"
                    continue
                bar = market.get(symbol)
                if bar is None:
                    blocked[symbol] = "missing_bar"
                    continue
                if bar.halted:
                    blocked[symbol] = "halted"
                    continue
                if bar.open_volume is None or bar.open_volume <= 0:
                    blocked[symbol] = "missing_open_volume"
                    continue
                if requested < 0 and current_shares + requested < 0 and not borrowable.get(symbol, False):
                    blocked[symbol] = "borrow_unavailable"
                    continue
                if (
                    requested < 0
                    and current_shares + requested < min(current_shares, 0.0)
                    and ssr_values.get(symbol) is True
                ):
                    blocked[symbol] = "ssr_restricted"
                    continue
                max_fill = bar.open_volume * cfg.volume_participation
                fill_shares = (
                    requested * min(1.0, rebalance_scale)
                    if symbol in regular_orders
                    else math.copysign(min(abs(requested), max_fill), requested)
                )
                participation = abs(fill_shares) / bar.open_volume
                spread = (
                    bar.open_spread_bps
                    if bar.open_spread_bps is not None
                    else cfg.default_spread_bps
                )
                impact = (
                    (
                        bar.open_impact_bps
                        if bar.open_impact_bps is not None
                        else cfg.default_impact_bps
                    )
                    + cfg.impact_curve_bps * participation * participation
                )
                adverse_bps = spread / 2.0 + cfg.slippage_bps + impact
                execution_price = bar.open * (
                    1.0 + math.copysign(adverse_bps / 10_000.0, fill_shares)
                )
                cost = abs(fill_shares) * bar.open * adverse_bps / 10_000.0
                reason = "forced_exit" if symbol in forced_symbols else "rebalance"
                cash -= fill_shares * execution_price
                day_cost += cost
                day_turnover += abs(fill_shares) * bar.open

                old_shares = current_shares
                if current is not None and old_shares * fill_shares < 0:
                    closing = min(abs(fill_shares), abs(old_shares))
                    side_sign = 1.0 if old_shares > 0 else -1.0
                    pnl = closing * side_sign * (execution_price - current.average_price)
                    current.realized_pnl += pnl
                    current.closed_shares += closing
                new_shares = old_shares + fill_shares
                fully_closed = (
                    current is not None
                    and old_shares * fill_shares < 0
                    and (abs(new_shares) <= 1e-10 or old_shares * new_shares < 0)
                )
                if fully_closed and current is not None:
                    trades.append(ClosedTrade(
                        symbol=symbol,
                        side="LONG" if old_shares > 0 else "SHORT",
                        opened_on=current.opened_on,
                        closed_on=day,
                        shares=current.closed_shares,
                        pnl=current.realized_pnl - current.borrow_cost,
                        holding_days=day_index - current.opened_index,
                        reason=reason,
                        sector=current.sector,
                    ))
                if abs(new_shares) <= 1e-10:
                    positions.pop(symbol, None)
                elif current is None or old_shares * new_shares <= 0:
                    positions[symbol] = Position(
                        symbol, new_shares, execution_price, day, day_index,
                        sectors.get(symbol, "INDEX" if symbol == "SPY" else "UNKNOWN"),
                        betas.get(symbol, 1.0),
                    )
                elif abs(new_shares) > abs(old_shares):
                    added = abs(fill_shares)
                    current.average_price = (
                        abs(old_shares) * current.average_price + added * execution_price
                    ) / abs(new_shares)
                    current.shares = new_shares
                else:
                    current.shares = new_shares
                fills.append(Fill(
                    day, symbol, requested, fill_shares, execution_price, bar.open,
                    cost, reason, abs(fill_shares) + 1e-10 < abs(requested),
                ))
                capacity = bar.open * bar.open_volume * cfg.volume_participation
                capacity_samples.append(capacity)
                day_capacity_samples.append(capacity)

            # Existing shares earn the overnight move; post-rebalance shares
            # earn only the opening-to-close move.
            for symbol in set(opening_shares) | set(positions):
                bar = market.get(symbol)
                if bar is None:
                    continue
                before = opening_shares.get(symbol, 0.0)
                after = positions[symbol].shares if symbol in positions else 0.0
                overnight = before * (bar.open - previous_closes.get(symbol, bar.open))
                intraday = after * (bar.close - bar.open)
                if before > 0:
                    day_long_pnl += overnight
                elif before < 0:
                    day_short_pnl += overnight
                if after > 0:
                    day_long_pnl += intraday
                elif after < 0:
                    day_short_pnl += intraday
            for fill in fills:
                before = opening_shares.get(fill.symbol, 0.0)
                after = positions[fill.symbol].shares if fill.symbol in positions else 0.0
                if after > 0 or (after == 0 and before > 0):
                    day_long_pnl -= fill.cost
                else:
                    day_short_pnl -= fill.cost

            borrow_cost = 0.0
            for symbol, position in positions.items():
                if position.shares >= 0:
                    continue
                bar = market.get(symbol)
                mark = bar.close if bar is not None else previous_closes.get(symbol)
                if mark is None:
                    continue
                annual_rate = rates.get(symbol, cfg.annual_borrow_rate)
                charge = abs(position.shares) * mark * annual_rate / cfg.trading_days_per_year
                cash -= charge
                position.borrow_cost += charge
                borrow_cost += charge
                day_short_pnl -= charge

            marks = {
                symbol: market[symbol].close
                if symbol in market else previous_closes[symbol]
                for symbol in positions
                if symbol in market or symbol in previous_closes
            }
            equity = cash + sum(position.shares * marks[symbol]
                                for symbol, position in positions.items()
                                if symbol in marks)
            peak = max(peak, equity)
            drawdown = 100.0 * max(0.0, 1.0 - equity / peak)
            daily_return = equity / previous_equity - 1.0 if previous_equity else 0.0
            exposure_rows = [
                {
                    "shares": position.shares,
                    "price": marks[symbol],
                    "sector": position.sector,
                    "beta": position.beta,
                }
                for symbol, position in positions.items() if symbol in marks
            ]
            if exposure_rows and equity > 0:
                exposure = exposure_snapshot(exposure_rows, equity=equity)
                gross = exposure.gross_pct / 100.0
                net = exposure.net_pct / 100.0
                beta_exposure = exposure.beta
                sector_exposure = {
                    key: value / 100.0 for key, value in exposure.sector_pct.items()
                }
            else:
                gross = net = beta_exposure = 0.0
                sector_exposure = {}
            snapshots.append(DailySnapshot(
                day, equity, cash, daily_return, drawdown, throttle,
                gross, net, beta_exposure, sector_exposure,
                day_turnover / decision_equity if decision_equity > 0 else 0.0,
                day_cost, borrow_cost, day_long_pnl, day_short_pnl,
                min(day_capacity_samples) if day_capacity_samples else 0.0,
                target_weights=dict(sorted(target_weights.items())),
                positions={symbol: position.shares for symbol, position in sorted(positions.items())},
                fills=fills, blocked_orders=blocked,
                feature_cutoff=max((row.date for row in rows if row.date < day), default=None),
                market_regime=market_regime,
                halted=permanently_halted,
                new_entries_halted=new_entries_halted,
            ))
            previous_equity = equity
            previous_closes.update({symbol: bar.close for symbol, bar in market.items()})

        metrics = _performance_metrics(
            snapshots, trades, cfg.initial_cash, capacity_samples,
            cfg.trading_days_per_year,
        )
        metrics["borrow_data_complete"] = dated_borrow_complete
        bar_quality_passed = all(
            row.known_at is not None
            and row.open_known_at is not None
            and row.member_from is not None
            and row.member_to is not None
            and row.date <= row.known_at.date() <= row.date + timedelta(days=3)
            and row.open_known_at <= datetime.combine(row.date, time(9, 30))
            and row.member_from <= row.date < row.member_to
            for row in rows
        )
        audit_section = (pit_audit_report or {}).get("audit") or {}
        gap_section = (pit_audit_report or {}).get("data_gaps") or {}
        audit_categories = set(gap_section.get("required_categories") or ())
        audit_counts = gap_section.get("counts") or {}
        audit_quality_passed = bool(
            pit_audit_report
            and pit_audit_report.get("ok") is True
            and audit_section.get("ok") is True
            and gap_section.get("ok") is True
            and not gap_section.get("missing_categories")
            and not pit_audit_report.get("unavailable_partitions")
            and audit_categories == _REQUIRED_PIT_CATEGORIES
            and isinstance(audit_counts, Mapping)
            and all(int(audit_counts.get(category, 0)) > 0
                    for category in _REQUIRED_PIT_CATEGORIES)
        )
        metrics["pit_audit_passed"] = audit_quality_passed
        event_quality_passed = all(
            event.available_at is not None
            and _datetime(event.available_at) >= _datetime(event.timestamp)
            and (
                event.reaction is None
                or (
                    event.reaction_known_at is not None
                    and _datetime(event.reaction_known_at)
                    >= _datetime(event.timestamp)
                )
            )
            for event in event_rows
        )
        metrics["event_pit_quality_passed"] = event_quality_passed
        metrics["stale_mark_detected"] = stale_mark_detected
        metrics["pit_quality_passed"] = (
            dated_borrow_complete
            and bar_quality_passed
            and audit_quality_passed
            and event_quality_passed
            and not stale_mark_detected
        )
        tolerance = 1e-6
        metrics["portfolio_controls_passed"] = not stale_mark_detected and all(
            row.gross_exposure <= cfg.max_gross + tolerance
            and abs(row.net_exposure) <= cfg.max_abs_net + tolerance
            and abs(row.beta_exposure) <= cfg.max_abs_beta + tolerance
            and all(
                exposure <= cfg.max_sector_gross + tolerance
                for exposure in row.sector_exposure.values()
            )
            for row in snapshots
        )
        metrics["drawdown_throttle_passed"] = all(
            sum(abs(value) for value in row.target_weights.values())
            <= cfg.max_gross * row.throttle + tolerance
            and row.gross_exposure
            <= cfg.max_gross * row.throttle + tolerance
            and (row.throttle > 0 or not row.positions)
            for row in snapshots
        )
        walk_forward: List[Dict[str, Any]] = []
        oos_rows: List[DailySnapshot] = []
        oos_trades: List[ClosedTrade] = []
        oos_liquidation_complete = True
        oos_portfolio_controls_passed = True
        oos_drawdown_throttle_passed = True
        if _compute_walk_forward:
            wf_config = WalkForwardConfig(
                train_months=cfg.walk_forward_train_months,
                validation_months=cfg.walk_forward_validation_months,
                test_months=cfg.walk_forward_test_months,
                purge_days=cfg.purge_days,
                embargo_days=cfg.embargo_days,
            )
            for fold in splits_from_config(days, wf_config):
                test_start = min(fold.test_dates)
                test_end = max(fold.test_dates)
                isolated = MarketNeutralBacktester(self.strategy, cfg).run(
                    rows,
                    event_rows,
                    borrow_available=borrow_available,
                    borrow_rates=borrow_rates,
                    ssr_restricted=ssr_restricted,
                    forced_cover=forced_cover,
                    pit_audit_report=pit_audit_report,
                    _evaluation_start=test_start,
                    _evaluation_end=test_end,
                    _compute_walk_forward=False,
                )
                test_rows = [
                    row
                    for row in isolated.snapshots
                    if test_start <= row.date <= test_end
                ]
                test_trades = [
                    trade
                    for trade in isolated.trades
                    if trade.opened_on >= test_start and trade.closed_on <= test_end
                ]
                start_equity = (
                    test_rows[0].equity / (1.0 + test_rows[0].daily_return)
                    if test_rows else cfg.initial_cash
                )
                fold_row = fold.to_dict()
                fold_row["state_isolated"] = True
                fold_row["parameters_frozen"] = True
                fold_row["ending_positions_flat"] = not bool(
                    test_rows[-1].positions if test_rows else {}
                )
                oos_liquidation_complete = (
                    oos_liquidation_complete
                    and fold_row["ending_positions_flat"]
                )
                oos_portfolio_controls_passed = (
                    oos_portfolio_controls_passed
                    and isolated.metrics["portfolio_controls_passed"]
                )
                oos_drawdown_throttle_passed = (
                    oos_drawdown_throttle_passed
                    and isolated.metrics["drawdown_throttle_passed"]
                )
                fold_row["test_metrics"] = _performance_metrics(
                    test_rows, test_trades, start_equity, (),
                    cfg.trading_days_per_year,
                )
                walk_forward.append(fold_row)
                oos_rows.extend(test_rows)
                oos_trades.extend(test_trades)
        if oos_rows:
            oos_metrics = _performance_metrics(
                oos_rows, oos_trades,
                oos_rows[0].equity / (1 + oos_rows[0].daily_return),
                (), cfg.trading_days_per_year,
            )
            metrics.update({
                "oos_closed_trades": oos_metrics["closed_trades"],
                "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
                "oos_profit_factor": oos_metrics["profit_factor"],
                "oos_sharpe": oos_metrics["sharpe"],
                "oos_calmar": oos_metrics["calmar"],
                "oos_worst_month_pct": oos_metrics["worst_month_pct"],
                "oos_concentration_pct": oos_metrics["concentration_pct"],
                "oos_monthly_returns": oos_metrics["monthly_returns"],
                "oos_avg_monthly_return_pct": oos_metrics["avg_monthly_return_pct"],
                "oos_months_hit_10pct": oos_metrics["months_hit_10pct"],
                "oos_months_total": oos_metrics["months_total"],
                "oos_liquidation_complete": oos_liquidation_complete,
                "oos_portfolio_controls_passed": oos_portfolio_controls_passed,
                "oos_drawdown_throttle_passed": oos_drawdown_throttle_passed,
                "oos_turnover": oos_metrics["turnover"],
                "oos_annualized_turnover": oos_metrics["annualized_turnover"],
                "oos_capacity_estimate_usd": oos_metrics["capacity_estimate_usd"],
                "oos_long_contribution": oos_metrics["long_contribution"],
                "oos_short_contribution": oos_metrics["short_contribution"],
                "oos_regime_days": oos_metrics["regime_days"],
                "oos_regime_contribution": oos_metrics["regime_contribution"],
                "oos_regime_coverage_passed": oos_metrics["regime_coverage_passed"],
            })
        else:
            metrics["oos_closed_trades"] = 0
            metrics["oos_liquidation_complete"] = False
            metrics["oos_portfolio_controls_passed"] = False
            metrics["oos_drawdown_throttle_passed"] = False
            metrics["oos_regime_coverage_passed"] = False
        acceptance = evaluate_backtest_acceptance(metrics).to_dict()
        return BacktestResult(cfg, snapshots, trades, metrics, walk_forward, acceptance)


def _monthly_returns(snapshots: Sequence[DailySnapshot]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for row in snapshots:
        grouped.setdefault(row.date.strftime("%Y-%m"), []).append(row.daily_return)
    result = []
    for month, returns in sorted(grouped.items()):
        compounded = math.prod(1.0 + value for value in returns) - 1.0
        result.append({"month": month, "return_pct": round(compounded * 100.0, 6)})
    return result


def _performance_metrics(
    snapshots: Sequence[DailySnapshot],
    trades: Sequence[ClosedTrade],
    initial_equity: float,
    capacity_samples: Sequence[float],
    annualization: int,
) -> Dict[str, Any]:
    returns = [row.daily_return for row in snapshots]
    mean = statistics.mean(returns) if returns else 0.0
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = mean / deviation * math.sqrt(annualization) if deviation > 0 else 0.0
    compounded_return = math.prod(1.0 + value for value in returns) - 1.0
    annualized_return = (
        (1.0 + compounded_return) ** (annualization / len(returns)) - 1.0
        if returns and compounded_return > -1.0 else -1.0
    )
    wins = sum(max(trade.pnl, 0.0) for trade in trades)
    losses = abs(sum(min(trade.pnl, 0.0) for trade in trades))
    if losses > 0:
        profit_factor = wins / losses
    elif wins > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    monthly = _monthly_returns(snapshots)
    month_values = [row["return_pct"] for row in monthly]
    profitable_months = [max(value, 0.0) for value in month_values]
    month_profit = sum(profitable_months)
    period_concentration = (
        max(profitable_months) / month_profit * 100.0
        if month_profit > 0 and profitable_months
        else 100.0
    )
    by_sector: Dict[str, float] = {}
    for trade in trades:
        by_sector[trade.sector] = by_sector.get(trade.sector, 0.0) + max(trade.pnl, 0.0)
    sector_profit = sum(by_sector.values())
    sector_concentration = (
        max(by_sector.values()) / sector_profit * 100.0
        if sector_profit > 0 and by_sector
        else 100.0
    )
    ending = snapshots[-1].equity if snapshots else initial_equity
    curve = 1.0
    curve_peak = 1.0
    local_max_drawdown = 0.0
    for daily_return in returns:
        curve *= 1.0 + daily_return
        curve_peak = max(curve_peak, curve)
        if curve_peak > 0:
            local_max_drawdown = max(
                local_max_drawdown, 100.0 * (1.0 - curve / curve_peak)
            )
    max_exposure_concentration = max(
        (max(row.sector_exposure.values(), default=0.0) for row in snapshots),
        default=0.0,
    ) * 100.0
    total_turnover = sum(row.turnover for row in snapshots)
    snapshot_capacity = [
        row.capacity_usd for row in snapshots if row.capacity_usd > 0
    ]
    effective_capacity = snapshot_capacity or list(capacity_samples)
    calmar = (
        annualized_return / (local_max_drawdown / 100.0)
        if local_max_drawdown > 0 else (999.0 if annualized_return > 0 else 0.0)
    )
    regime_days: Dict[str, int] = {}
    regime_profit: Dict[str, float] = {}
    for row in snapshots:
        regime_days[row.market_regime] = regime_days.get(row.market_regime, 0) + 1
        prior_equity = (
            row.equity / (1.0 + row.daily_return)
            if row.daily_return > -1.0 else 0.0
        )
        regime_profit[row.market_regime] = (
            regime_profit.get(row.market_regime, 0.0)
            + row.daily_return * prior_equity
        )
    required_regimes = {"bull", "bear", "high_volatility", "sideways"}
    regime_coverage_passed = all(regime_days.get(name, 0) > 0 for name in required_regimes)
    return {
        "starting_equity": round(initial_equity, 6),
        "ending_equity": round(ending, 6),
        "total_return_pct": round((ending / initial_equity - 1.0) * 100.0, 6)
        if initial_equity else 0.0,
        "sharpe": round(sharpe, 6),
        "annualized_return_pct": round(annualized_return * 100.0, 6),
        "calmar": round(calmar, 6),
        "max_drawdown_pct": round(local_max_drawdown, 6),
        "profit_factor": round(profit_factor, 6),
        "closed_trades": len(trades),
        "monthly_returns": monthly,
        "worst_month_pct": min((row["return_pct"] for row in monthly), default=0.0),
        "long_contribution": round(sum(row.long_pnl for row in snapshots), 6),
        "short_contribution": round(sum(row.short_pnl for row in snapshots), 6),
        "transaction_cost": round(sum(row.transaction_cost for row in snapshots), 6),
        "borrow_cost": round(sum(row.borrow_cost for row in snapshots), 6),
        "average_gross_exposure": round(
            statistics.mean([row.gross_exposure for row in snapshots]), 6
        ) if snapshots else 0.0,
        "average_abs_net_exposure": round(
            statistics.mean([abs(row.net_exposure) for row in snapshots]), 6
        ) if snapshots else 0.0,
        "sector_profit_concentration_pct": round(sector_concentration, 6),
        "period_profit_concentration_pct": round(period_concentration, 6),
        "max_sector_exposure_pct": round(max_exposure_concentration, 6),
        "concentration_pct": round(
            max(sector_concentration, period_concentration), 6
        ),
        "turnover": round(total_turnover, 6),
        "annualized_turnover": round(
            total_turnover * annualization / len(snapshots), 6
        ) if snapshots else 0.0,
        "capacity_estimate_usd": round(min(effective_capacity), 2)
        if effective_capacity else 0.0,
        "avg_monthly_return_pct": round(
            statistics.mean(month_values) if month_values else 0.0, 6
        ),
        "months_hit_10pct": sum(value >= 10.0 for value in month_values),
        "months_total": len(month_values),
        "target_10pct_monthly_guaranteed": False,
        "regime_days": dict(sorted(regime_days.items())),
        "regime_contribution": {
            key: round(value, 6)
            for key, value in sorted(regime_profit.items())
        },
        "regime_coverage_passed": regime_coverage_passed,
    }


def run_market_neutral_backtest(
    bars: Iterable[DailyBar],
    events: Iterable[Event] = (),
    *,
    strategy: Optional[MarketNeutralLongShortStrategy] = None,
    config: Optional[BacktestConfig] = None,
    borrow_available: Optional[Mapping[str, Any]] = None,
    borrow_rates: Optional[Mapping[str, Any]] = None,
    ssr_restricted: Optional[Mapping[str, Any]] = None,
    forced_cover: Optional[Mapping[str, Any]] = None,
    pit_audit_report: Optional[Mapping[str, Any]] = None,
) -> BacktestResult:
    """Convenience API for callers that do not need to retain engine state."""

    return MarketNeutralBacktester(
        strategy or MarketNeutralLongShortStrategy(), config
    ).run(
        bars,
        events,
        borrow_available=borrow_available,
        borrow_rates=borrow_rates,
        ssr_restricted=ssr_restricted,
        forced_cover=forced_cover,
        pit_audit_report=pit_audit_report,
    )
