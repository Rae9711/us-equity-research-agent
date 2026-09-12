"""Objective acceptance gates for research and paper qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


BACKTEST_THRESHOLDS = {
    "oos_closed_trades": 300,
    "max_drawdown_pct": 10.0,
    "profit_factor": 1.3,
    "sharpe": 1.8,
    "worst_month_pct": -5.0,
    "concentration_pct": 40.0,
}

PAPER_THRESHOLDS = {
    "trading_days": 63,
    "closed_trades": 100,
}


def _first(metrics: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _monthly_values(metrics: Mapping[str, Any]) -> list[float]:
    rows = metrics.get("oos_monthly_returns") or metrics.get("monthly_returns") or []
    values = []
    for row in rows:
        value = row.get("return_pct") if isinstance(row, Mapping) else row
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _concentration(metrics: Mapping[str, Any]) -> Optional[float]:
    direct = _first(
        metrics,
        (
            "oos_concentration_pct",
            "concentration_pct",
            "max_concentration_pct",
            "largest_position_pct",
        ),
    )
    if direct is not None:
        return _float(direct)
    sectors = metrics.get("sector_pct") or metrics.get("sector_exposure_pct") or {}
    if isinstance(sectors, Mapping) and sectors:
        return max(_float(value) for value in sectors.values())
    return None


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    criteria: dict[str, bool]
    observed: dict[str, Any]
    failures: tuple[str, ...]
    disclosures: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "criteria": dict(self.criteria),
            "observed": dict(self.observed),
            "failures": list(self.failures),
            "disclosures": dict(self.disclosures),
        }


def evaluate_backtest_acceptance(metrics: Mapping[str, Any]) -> AcceptanceResult:
    """Evaluate the accepted out-of-sample backtest criteria."""

    monthly = _monthly_values(metrics)
    trades = _int(
        _first(metrics, ("oos_closed_trades", "oos_n_trades", "n_oos_trades", "n_trades"))
    )
    drawdown_raw = _first(
        metrics, ("oos_max_drawdown_pct", "max_drawdown_pct", "agent_max_drawdown_pct")
    )
    profit_factor_raw = _first(metrics, ("oos_profit_factor", "profit_factor"))
    sharpe_raw = _first(metrics, ("oos_sharpe", "sharpe", "agent_sharpe"))
    worst_month_raw = _first(
        metrics,
        ("oos_worst_month_pct", "worst_month_pct"),
        min(monthly) if monthly else None,
    )
    drawdown = abs(_float(drawdown_raw)) if drawdown_raw is not None else None
    profit_factor = _float(profit_factor_raw) if profit_factor_raw is not None else None
    sharpe = _float(sharpe_raw) if sharpe_raw is not None else None
    worst_month = _float(worst_month_raw) if worst_month_raw is not None else None
    concentration = _concentration(metrics)

    observed = {
        "oos_closed_trades": trades,
        "max_drawdown_pct": drawdown,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "worst_month_pct": worst_month,
        "concentration_pct": concentration,
    }
    criteria = {
        "oos_closed_trades": trades >= 300,
        "max_drawdown_pct": drawdown is not None and drawdown <= 10.0,
        "profit_factor": profit_factor is not None and profit_factor >= 1.3,
        "sharpe": sharpe is not None and sharpe >= 1.8,
        "worst_month_pct": worst_month is not None and worst_month >= -5.0,
        "concentration_pct": concentration is not None and concentration <= 40.0,
        "pit_quality_passed": metrics.get("pit_quality_passed") is True,
    }
    observed["pit_quality_passed"] = metrics.get("pit_quality_passed") is True

    average_monthly = _first(metrics, ("oos_avg_monthly_return_pct", "avg_monthly_return_pct"))
    if average_monthly is None:
        average_monthly = sum(monthly) / len(monthly) if monthly else 0.0
    hit_count = _int(_first(metrics, ("oos_months_hit_10pct", "months_hit_10pct")))
    month_count = _int(_first(metrics, ("oos_months_total", "months_total"), len(monthly)))
    if (
        monthly
        and "oos_months_hit_10pct" not in metrics
        and "months_hit_10pct" not in metrics
    ):
        hit_count = sum(value >= 10.0 for value in monthly)
    disclosures = {
        "avg_monthly_return_pct": round(_float(average_monthly), 4),
        "months_hit_10_count": hit_count,
        "months_total": month_count,
        "hit_10_ratio": round(hit_count / month_count, 4) if month_count else 0.0,
    }
    failures = tuple(name for name, passed in criteria.items() if not passed)
    return AcceptanceResult(not failures, criteria, observed, failures, disclosures)


def evaluate_paper_qualification(
    metrics: Mapping[str, Any],
    *,
    backtest_pass: Optional[bool] = None,
) -> AcceptanceResult:
    """Evaluate whether a paper run is qualified for the next review stage."""

    days = _int(_first(metrics, ("trading_days", "n_trading_days", "n_days")))
    trades = _int(_first(metrics, ("closed_trades", "n_closed_trades", "n_trades")))
    explicit_slippage = _first(metrics, ("slippage_within_assumption",))
    actual_slippage = _first(
        metrics, ("actual_slippage_bps", "realized_slippage_bps", "slippage_bps")
    )
    assumed_slippage = _first(
        metrics, ("assumed_slippage_bps", "slippage_assumption_bps")
    )
    if explicit_slippage is not None:
        slippage_ok = bool(explicit_slippage)
    elif actual_slippage is not None and assumed_slippage is not None:
        slippage_ok = _float(actual_slippage) <= _float(assumed_slippage)
    else:
        slippage_ok = False

    if backtest_pass is None:
        backtest_pass = bool(
            _first(metrics, ("backtest_pass", "backtest_passed"), False)
        )
    observed = {
        "trading_days": days,
        "closed_trades": trades,
        "actual_slippage_bps": actual_slippage,
        "assumed_slippage_bps": assumed_slippage,
        "slippage_within_assumption": slippage_ok,
        "backtest_pass": bool(backtest_pass),
    }
    criteria = {
        "trading_days": days >= 63,
        "closed_trades": trades >= 100,
        "slippage_within_assumption": slippage_ok,
        "backtest_pass": bool(backtest_pass),
    }
    failures = tuple(name for name, passed in criteria.items() if not passed)
    return AcceptanceResult(not failures, criteria, observed, failures, {})


def acceptance_markdown_report(
    backtest: AcceptanceResult,
    paper: Optional[AcceptanceResult] = None,
) -> str:
    """Render acceptance decisions and required disclosures as Markdown."""

    def section(title: str, result: AcceptanceResult) -> list[str]:
        rows = [
            "## {}".format(title),
            "",
            "**Result:** {}".format("PASS" if result.passed else "FAIL"),
            "",
            "| Criterion | Observed | Status |",
            "|---|---:|:---:|",
        ]
        for name, passed in result.criteria.items():
            rows.append(
                "| {} | {} | {} |".format(
                    name, result.observed.get(name, "—"), "PASS" if passed else "FAIL"
                )
            )
        return rows

    lines = ["# Research Acceptance Report", ""]
    lines.extend(section("Out-of-sample backtest", backtest))
    if backtest.disclosures:
        disclosures = backtest.disclosures
        lines.extend(
            [
                "",
                "### Disclosures",
                "",
                "- Average monthly return: {:.2f}%".format(
                    _float(disclosures.get("avg_monthly_return_pct"))
                ),
                "- Months at or above 10%: {} / {} ({:.1%})".format(
                    disclosures.get("months_hit_10_count", 0),
                    disclosures.get("months_total", 0),
                    _float(disclosures.get("hit_10_ratio")),
                ),
            ]
        )
    if paper is not None:
        lines.extend(["", *section("Paper qualification", paper)])
    return "\n".join(lines) + "\n"


def paper_metrics_from_account(
    account: Mapping[str, Any],
    *,
    strategy: str = "event_ls",
) -> dict[str, Any]:
    """Build qualification metrics from persisted fills, without inventing data."""
    trades = [
        row
        for row in (account.get("trades") or [])
        if not row.get("voided")
        and (
            row.get("strategy") == strategy
            or row.get("book") == strategy
        )
    ]
    closed = [
        row
        for row in trades
        if row.get("action") in ("EXIT", "SCALE_OUT")
        and row.get("pnl") is not None
    ]
    observed_dates = {
        str(row.get("trading_date"))[:10]
        for row in trades
        if row.get("trading_date")
    }
    slippages = [
        abs(_float(row.get("slippage_bps")))
        for row in trades
        if row.get("slippage_bps") is not None
    ]
    assumed = _float(
        (account.get("params") or {}).get("slippage_bps"),
        default=0.0,
    )
    return {
        "trading_days": len(observed_dates),
        "closed_trades": len(closed),
        "actual_slippage_bps": (
            sum(slippages) / len(slippages) if slippages else None
        ),
        "assumed_slippage_bps": assumed,
    }


def promotion_record(
    account: Mapping[str, Any],
    backtest_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an auditable promotion record; never enables trading by itself."""
    backtest = evaluate_backtest_acceptance(backtest_metrics)
    paper_metrics = paper_metrics_from_account(account)
    paper = evaluate_paper_qualification(
        paper_metrics,
        backtest_pass=backtest.passed,
    )
    if paper.passed:
        status = "PAPER_QUALIFIED"
        reason = "Eligible for explicit small-capital review; not auto-enabled"
    elif backtest.passed:
        status = "BACKTEST_ONLY"
        reason = "OOS passed; requires 63 paper days and 100 closed trades"
    else:
        status = "NOT_READY"
        reason = "Out-of-sample acceptance criteria not met"
    return {
        "status": status,
        "backtest_passed": backtest.passed,
        "paper_passed": paper.passed,
        "enabled": False,
        "reason": reason,
        "backtest": backtest.to_dict(),
        "paper": paper.to_dict(),
        "target_10pct_monthly_guaranteed": False,
    }


def refresh_event_ls_qualification(account: dict[str, Any]) -> dict[str, Any]:
    """Recompute and persist the research-only promotion state on an account."""
    qualifications = account.setdefault("strategy_qualification", {})
    previous = qualifications.get("event_ls") or {}
    metrics = previous.get("backtest_metrics") or {}
    record = promotion_record(account, metrics)
    if metrics:
        record["backtest_metrics"] = metrics
    qualifications["event_ls"] = record
    # Promotion never silently turns on an execution path.
    account.setdefault("params", {})["event_ls_enabled"] = False
    return record


def write_acceptance_report(
    path: Path,
    backtest: AcceptanceResult,
    paper: Optional[AcceptanceResult] = None,
) -> Path:
    """Persist a human-readable gate report for review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        acceptance_markdown_report(backtest, paper),
        encoding="utf-8",
    )
    return path


evaluate_acceptance = evaluate_backtest_acceptance
evaluate_backtest = evaluate_backtest_acceptance
evaluate_paper = evaluate_paper_qualification
markdown_report = acceptance_markdown_report
render_acceptance_report = acceptance_markdown_report
