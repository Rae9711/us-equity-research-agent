"""Point-in-time daily portfolio backtester for the market-neutral strategy.

The engine is deliberately data-source agnostic: callers supply daily OHLCV
bars and timestamped strategy events.  Decisions at an opening auction only
receive bars and events from earlier calendar days.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.paper.acceptance import acceptance_markdown_report, evaluate_backtest_acceptance
from src.paper.portfolio_controls import drawdown_throttle, exposure_snapshot
from src.paper.walk_forward import WalkForwardConfig, splits_from_config
from src.strategies.market_neutral_ls import (
    Bar,
    Event,
    MarketNeutralLongShortStrategy,
)


def _date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    # Strategy comparisons must not mix aware and naive timestamps.  Wall-clock
    # dates are sufficient for daily bars, so normalize to naive UTC-like time.
    return parsed.replace(tzinfo=None)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


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
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    halted: bool = False
    sector: Optional[str] = None
    beta: Optional[float] = None
    known_at: Optional[datetime] = None
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
        if self.known_at is not None:
            object.__setattr__(self, "known_at", _datetime(self.known_at))
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
            spread_bps=float(row.get("spread_bps", 0.0) or 0.0),
            impact_bps=float(row.get("impact_bps", 0.0) or 0.0),
            halted=bool(row.get("halted", False)),
            sector=row.get("sector"),
            beta=float(row["beta"]) if row.get("beta") is not None else None,
            known_at=_datetime(row["known_at"]) if row.get("known_at") else None,
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
    target_weights: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, float] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    blocked_orders: Dict[str, str] = field(default_factory=dict)
    feature_cutoff: Optional[date] = None
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
            "| Profit factor | {:.3f} |".format(metrics["profit_factor"]),
            "| Closed trades | {} |".format(metrics["closed_trades"]),
            "| Annualized turnover | {:.2f}x |".format(metrics["annualized_turnover"]),
            "| Capacity estimate | ${:,.0f} |".format(metrics["capacity_estimate_usd"]),
            "",
            "## Monthly returns",
            "",
            "| Month | Return |",
            "|---|---:|",
        ]
        for row in metrics["monthly_returns"]:
            lines.append("| {} | {:.2f}% |".format(row["month"], row["return_pct"]))
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
        decision_at = datetime.combine(decision_day, time(9, 30))
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
        decision_at = datetime.combine(decision_day, time(9, 30))
        visible = []
        for event in events:
            happened = _datetime(event.timestamp)
            available = _datetime(event.available_at or event.timestamp)
            if happened <= decision_at and available <= decision_at:
                visible.append(event)
        return visible

    def _betas(
        self, history: Sequence[Bar], symbols: Iterable[str]
    ) -> Dict[str, float]:
        by_symbol: Dict[str, List[Bar]] = {}
        for row in history:
            by_symbol.setdefault(row.symbol, []).append(row)
        calculated = self.strategy._rolling_betas(by_symbol)
        return {symbol: calculated.get(symbol, 1.0) for symbol in symbols}

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
        borrow_available: Optional[Mapping[str, bool]] = None,
        borrow_rates: Optional[Mapping[str, float]] = None,
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
        sectors = dict(self.strategy.sector_map)
        explicit_betas: Dict[str, float] = {}
        for row in rows:
            by_day.setdefault(row.date, {})[row.symbol] = row
            if row.sector:
                sectors[row.symbol] = row.sector
            if row.beta is not None:
                explicit_betas[row.symbol] = row.beta
        days = sorted(by_day)
        borrowable = {
            key.upper(): bool(value) for key, value in (borrow_available or {}).items()
        }
        for symbol in {row.symbol for row in rows}:
            borrowable.setdefault(symbol, True)
        rates = {key.upper(): float(value) for key, value in (borrow_rates or {}).items()}

        cash = cfg.initial_cash
        positions: Dict[str, Position] = {}
        snapshots: List[DailySnapshot] = []
        trades: List[ClosedTrade] = []
        peak = cfg.initial_cash
        previous_equity = cfg.initial_cash
        previous_closes: Dict[str, float] = {}
        permanently_halted = False
        capacity_samples: List[float] = []

        for day_index, day in enumerate(days):
            market = by_day[day]
            open_equity = cash + sum(
                position.shares * (
                    market[symbol].open
                    if symbol in market else previous_closes[symbol]
                )
                for symbol, position in positions.items()
                if symbol in market or symbol in previous_closes
            )
            peak = max(peak, open_equity)
            open_drawdown = 100.0 * max(0.0, 1.0 - open_equity / peak)
            throttle = drawdown_throttle(open_drawdown)
            if throttle == 0.0:
                permanently_halted = True
            new_entries_halted = bool(
                snapshots
                and snapshots[-1].daily_return
                <= -(cfg.max_daily_loss_pct / 100.0)
            )

            history = self._strategy_bars(rows, day)
            visible_events = self._visible_events(event_rows, day)
            target_weights: Dict[str, float] = {}
            betas = self._betas(history, set(market) | set(positions))
            betas.update(explicit_betas)
            betas["SPY"] = 1.0
            if not permanently_halted:
                portfolio = self.strategy.generate(
                    history,
                    visible_events,
                    datetime.combine(day, time(9, 30)),
                    borrowable,
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
                target = target_weights.get(symbol, 0.0) * open_equity / bar.open
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
                    elif held >= cfg.max_hold_days:
                        target = 0.0
                        forced_symbols.add(symbol)
                    elif held < cfg.min_hold_days and reducing_or_reversing:
                        target = current.shares
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
            opening_shares = {
                symbol: position.shares for symbol, position in positions.items()
            }
            for symbol in sorted(all_symbols):
                current = positions.get(symbol)
                current_shares = current.shares if current else 0.0
                requested = desired.get(symbol, 0.0) - current_shares
                if abs(requested) <= 1e-10:
                    continue
                bar = market.get(symbol)
                if bar is None:
                    blocked[symbol] = "missing_bar"
                    continue
                if bar.halted:
                    blocked[symbol] = "halted"
                    continue
                if bar.volume <= 0:
                    blocked[symbol] = "missing_volume"
                    continue
                if requested < 0 and current_shares + requested < 0 and not borrowable.get(symbol, False):
                    blocked[symbol] = "borrow_unavailable"
                    continue
                max_fill = bar.volume * cfg.volume_participation
                fill_shares = math.copysign(min(abs(requested), max_fill), requested)
                participation = abs(fill_shares) / bar.volume
                spread = bar.spread_bps if bar.spread_bps > 0 else cfg.default_spread_bps
                impact = (
                    (bar.impact_bps if bar.impact_bps > 0 else cfg.default_impact_bps)
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
                    trades.append(ClosedTrade(
                        symbol=symbol,
                        side="LONG" if old_shares > 0 else "SHORT",
                        opened_on=current.opened_on,
                        closed_on=day,
                        shares=closing,
                        pnl=pnl,
                        holding_days=day_index - current.opened_index,
                        reason=reason,
                        sector=current.sector,
                    ))
                new_shares = old_shares + fill_shares
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
                capacity_samples.append(bar.close * bar.volume * cfg.volume_participation)

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
                day_turnover / open_equity if open_equity > 0 else 0.0,
                day_cost, borrow_cost, day_long_pnl, day_short_pnl,
                target_weights=dict(sorted(target_weights.items())),
                positions={symbol: position.shares for symbol, position in sorted(positions.items())},
                fills=fills, blocked_orders=blocked,
                feature_cutoff=max((row.date for row in rows if row.date < day), default=None),
                halted=permanently_halted,
                new_entries_halted=new_entries_halted,
            ))
            previous_equity = equity
            previous_closes.update({symbol: bar.close for symbol, bar in market.items()})

        metrics = _performance_metrics(
            snapshots, trades, cfg.initial_cash, capacity_samples,
            cfg.trading_days_per_year,
        )
        metrics["pit_quality_passed"] = all(
            row.known_at is not None
            and row.member_from is not None
            and row.member_to is not None
            and row.date <= row.known_at.date() <= row.date + timedelta(days=3)
            and row.member_from <= row.date < row.member_to
            for row in rows
        )
        walk_forward = _walk_forward_metrics(snapshots, trades, cfg)
        oos_indices = sorted({
            index
            for fold in walk_forward
            for index in fold.pop("_test_indices", [])
        })
        if oos_indices:
            oos_rows = [snapshots[index] for index in oos_indices]
            oos_days = {row.date for row in oos_rows}
            oos_trades = [trade for trade in trades if trade.closed_on in oos_days]
            oos_metrics = _performance_metrics(
                oos_rows, oos_trades,
                oos_rows[0].equity / (1 + oos_rows[0].daily_return),
                capacity_samples, cfg.trading_days_per_year,
            )
            metrics.update({
                "oos_closed_trades": oos_metrics["closed_trades"],
                "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
                "oos_profit_factor": oos_metrics["profit_factor"],
                "oos_sharpe": oos_metrics["sharpe"],
                "oos_worst_month_pct": oos_metrics["worst_month_pct"],
                "oos_concentration_pct": oos_metrics["concentration_pct"],
                "oos_monthly_returns": oos_metrics["monthly_returns"],
                "oos_avg_monthly_return_pct": oos_metrics["avg_monthly_return_pct"],
                "oos_months_hit_10pct": oos_metrics["months_hit_10pct"],
                "oos_months_total": oos_metrics["months_total"],
            })
        else:
            metrics["oos_closed_trades"] = 0
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
    return {
        "starting_equity": round(initial_equity, 6),
        "ending_equity": round(ending, 6),
        "total_return_pct": round((ending / initial_equity - 1.0) * 100.0, 6)
        if initial_equity else 0.0,
        "sharpe": round(sharpe, 6),
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
        "capacity_estimate_usd": round(min(capacity_samples), 2)
        if capacity_samples else 0.0,
        "avg_monthly_return_pct": round(
            statistics.mean(month_values) if month_values else 0.0, 6
        ),
        "months_hit_10pct": sum(value >= 10.0 for value in month_values),
        "months_total": len(month_values),
        "target_10pct_monthly_guaranteed": False,
    }


def _walk_forward_metrics(
    snapshots: Sequence[DailySnapshot],
    trades: Sequence[ClosedTrade],
    config: BacktestConfig,
) -> List[Dict[str, Any]]:
    wf_config = WalkForwardConfig(
        train_months=config.walk_forward_train_months,
        validation_months=config.walk_forward_validation_months,
        test_months=config.walk_forward_test_months,
        purge_days=config.purge_days,
        embargo_days=config.embargo_days,
    )
    folds = splits_from_config([row.date for row in snapshots], wf_config)
    output = []
    for fold in folds:
        test_rows = [snapshots[index] for index in fold.test]
        test_days = {item.date for item in test_rows}
        test_trades = [trade for trade in trades if trade.closed_on in test_days]
        start = test_rows[0].equity / (1.0 + test_rows[0].daily_return)
        row = fold.to_dict()
        row["test_metrics"] = _performance_metrics(
            test_rows, test_trades, start, (), config.trading_days_per_year
        )
        row["_test_indices"] = list(fold.test)
        output.append(row)
    return output


def run_market_neutral_backtest(
    bars: Iterable[DailyBar],
    events: Iterable[Event] = (),
    *,
    strategy: Optional[MarketNeutralLongShortStrategy] = None,
    config: Optional[BacktestConfig] = None,
    borrow_available: Optional[Mapping[str, bool]] = None,
    borrow_rates: Optional[Mapping[str, float]] = None,
) -> BacktestResult:
    """Convenience API for callers that do not need to retain engine state."""

    return MarketNeutralBacktester(
        strategy or MarketNeutralLongShortStrategy(), config
    ).run(
        bars,
        events,
        borrow_available=borrow_available,
        borrow_rates=borrow_rates,
    )
