"""Portfolio-level exposure and drawdown controls for paper research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional


def _get(position: Any, *names: str, default: Any = None) -> Any:
    if isinstance(position, Mapping):
        for name in names:
            if name in position and position[name] is not None:
                return position[name]
    else:
        for name in names:
            value = getattr(position, name, None)
            if value is not None:
                return value
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signed_market_value(position: Any) -> float:
    explicit = _get(position, "market_value", "notional", "value")
    if explicit is None:
        quantity = _number(_get(position, "quantity", "qty", "shares"))
        price = _number(_get(position, "price", "last_price", "mark", "avg_entry"))
        value = quantity * price
    else:
        value = _number(explicit)

    # A negative value or quantity already carries its sign. Otherwise infer it
    # from common direction/side labels.
    if value < 0:
        return value
    direction = str(_get(position, "direction", "side", default="LONG")).upper()
    return -value if direction in {"SHORT", "SELL", "-1"} else value


@dataclass(frozen=True)
class ExposureSnapshot:
    """Exposure values expressed as percentages of portfolio equity."""

    equity: float
    gross_pct: float
    net_pct: float
    sector_pct: dict[str, float]
    beta: float

    @property
    def beta_exposure(self) -> float:
        return self.beta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureLimits:
    max_gross_pct: float = 100.0
    max_abs_net_pct: float = 100.0
    max_sector_pct: float = 40.0
    max_abs_beta: float = 1.0


@dataclass(frozen=True)
class ExposureCheck:
    accepted: bool
    snapshot: ExposureSnapshot
    breaches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "passed": self.passed,
            "snapshot": self.snapshot.to_dict(),
            "breaches": list(self.breaches),
        }


def exposure_snapshot(
    positions: Iterable[Any], equity: Optional[float] = None
) -> ExposureSnapshot:
    """Calculate gross, net, sector and beta exposure from generic positions.

    Positions may be mappings or objects. Supported fields include
    ``market_value``/``notional`` or ``quantity``/``shares`` plus a price,
    ``direction``/``side``, ``sector`` and ``beta``.
    """

    rows = list(positions)
    if equity is None:
        equity = sum(abs(_signed_market_value(row)) for row in rows)
    equity_value = _number(equity)
    if equity_value <= 0:
        raise ValueError("equity must be positive")

    gross = 0.0
    net = 0.0
    beta_notional = 0.0
    sectors: dict[str, float] = {}
    for row in rows:
        value = _signed_market_value(row)
        gross += abs(value)
        net += value
        beta_notional += value * _number(_get(row, "beta"), default=0.0)
        sector = str(_get(row, "sector", default="UNKNOWN") or "UNKNOWN")
        sectors[sector] = sectors.get(sector, 0.0) + abs(value)

    scale = 100.0 / equity_value
    return ExposureSnapshot(
        equity=round(equity_value, 10),
        gross_pct=round(gross * scale, 10),
        net_pct=round(net * scale, 10),
        sector_pct={key: round(value * scale, 10) for key, value in sorted(sectors.items())},
        beta=round(beta_notional / equity_value, 10),
    )


def check_proposed_exposure(
    positions: Iterable[Any],
    proposed_position: Any,
    *,
    equity: Optional[float] = None,
    limits: Optional[ExposureLimits] = None,
) -> ExposureCheck:
    """Check the post-trade portfolio, including the proposed position."""

    limits = limits or ExposureLimits()
    snapshot = exposure_snapshot([*positions, proposed_position], equity=equity)
    breaches = []
    if snapshot.gross_pct > limits.max_gross_pct:
        breaches.append("gross_exposure")
    if abs(snapshot.net_pct) > limits.max_abs_net_pct:
        breaches.append("net_exposure")
    if any(value > limits.max_sector_pct for value in snapshot.sector_pct.values()):
        breaches.append("sector_concentration")
    if abs(snapshot.beta) > limits.max_abs_beta:
        breaches.append("beta_exposure")
    return ExposureCheck(not breaches, snapshot, tuple(breaches))


def proposed_exposure_check(
    positions: Iterable[Any],
    proposed_position: Any,
    *,
    equity: Optional[float] = None,
    limits: Optional[ExposureLimits] = None,
) -> ExposureCheck:
    """Singular-name compatibility wrapper for ``check_proposed_exposure``."""

    return check_proposed_exposure(
        positions, proposed_position, equity=equity, limits=limits
    )


def drawdown_throttle(drawdown_pct: float) -> float:
    """Return the new-risk multiplier for a positive drawdown percentage."""

    drawdown = abs(_number(drawdown_pct))
    if drawdown >= 10.0:
        return 0.0
    if drawdown >= 8.0:
        return 0.25
    if drawdown >= 6.0:
        return 0.5
    if drawdown >= 4.0:
        return 0.75
    return 1.0


compute_exposure_snapshot = exposure_snapshot
drawdown_throttle_multiplier = drawdown_throttle
