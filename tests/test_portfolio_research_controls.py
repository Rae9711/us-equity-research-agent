"""Deterministic tests for portfolio and research acceptance controls."""

from datetime import date

import pytest

from src.paper.acceptance import (
    acceptance_markdown_report,
    evaluate_backtest_acceptance,
    evaluate_paper_qualification,
    paper_metrics_from_account,
    promotion_record,
)
from src.paper.portfolio_controls import (
    ExposureLimits,
    check_proposed_exposure,
    drawdown_throttle,
    exposure_snapshot,
)
from src.paper.walk_forward import purged_walk_forward_splits


def test_exposure_snapshot_from_generic_positions():
    positions = [
        {
            "symbol": "AAA",
            "shares": 300,
            "last_price": 100,
            "direction": "LONG",
            "sector": "Technology",
            "beta": 1.2,
        },
        {
            "symbol": "BBB",
            "market_value": 10_000,
            "side": "SHORT",
            "sector": "Energy",
            "beta": 0.8,
        },
    ]
    snapshot = exposure_snapshot(positions, equity=100_000)
    assert snapshot.gross_pct == pytest.approx(40)
    assert snapshot.net_pct == pytest.approx(20)
    assert snapshot.sector_pct == {"Energy": 10.0, "Technology": 30.0}
    assert snapshot.beta == pytest.approx(0.28)


def test_proposal_is_included_in_exposure_checks():
    positions = [
        {"market_value": 30_000, "sector": "Technology", "beta": 1.0}
    ]
    proposal = {
        "quantity": 150,
        "price": 100,
        "sector": "Technology",
        "beta": 1.0,
    }
    result = check_proposed_exposure(
        positions,
        proposal,
        equity=100_000,
        limits=ExposureLimits(
            max_gross_pct=60,
            max_abs_net_pct=60,
            max_sector_pct=40,
            max_abs_beta=0.75,
        ),
    )
    assert not result.accepted
    assert result.snapshot.gross_pct == pytest.approx(45)
    assert result.breaches == ("sector_concentration",)


@pytest.mark.parametrize(
    ("drawdown", "expected"),
    [
        (0, 1.0),
        (3.999, 1.0),
        (4, 0.75),
        (5.999, 0.75),
        (6, 0.5),
        (8, 0.25),
        (10, 0.0),
        (25, 0.0),
    ],
)
def test_drawdown_throttle_boundaries(drawdown, expected):
    assert drawdown_throttle(drawdown) == expected


def _monthly_dates(start_year, start_month, count):
    values = []
    year, month = start_year, start_month
    for _ in range(count):
        values.append(date(year, month, 1))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def test_purged_walk_forward_is_chronological_and_disjoint():
    dates = _monthly_dates(2020, 1, 42)
    folds = purged_walk_forward_splits(
        dates,
        train_months=24,
        validation_months=3,
        test_months=3,
        purge_days=5,
        embargo_days=7,
    )
    assert len(folds) >= 4
    for fold in folds:
        assert fold.train_dates
        assert fold.validation_dates
        assert fold.test_dates
        assert max(fold.train_dates) < min(fold.validation_dates)
        assert max(fold.validation_dates) < min(fold.test_dates)
        assert set(fold.train).isdisjoint(fold.validation)
        assert set(fold.train).isdisjoint(fold.test)
        assert set(fold.validation).isdisjoint(fold.test)


def test_backtest_acceptance_boundary_values_and_disclosures():
    result = evaluate_backtest_acceptance(
        {
            "oos_closed_trades": 300,
            "max_drawdown_pct": 10,
            "profit_factor": 1.3,
            "sharpe": 1.8,
            "worst_month_pct": -5,
            "concentration_pct": 40,
            "monthly_returns": [12, 8, 10],
            "pit_quality_passed": True,
        }
    )
    assert result.passed
    assert result.disclosures["avg_monthly_return_pct"] == pytest.approx(10)
    assert result.disclosures["hit_10_ratio"] == pytest.approx(2 / 3, abs=0.0001)
    assert all("guarantee" not in key for key in result.to_dict())


def test_backtest_acceptance_requires_explicit_pit_quality_pass():
    metrics = {
        "oos_closed_trades": 300,
        "max_drawdown_pct": 9,
        "profit_factor": 1.4,
        "sharpe": 2,
        "worst_month_pct": -4,
        "concentration_pct": 35,
    }
    missing = evaluate_backtest_acceptance(metrics)
    failed = evaluate_backtest_acceptance({**metrics, "pit_quality_passed": False})
    assert not missing.passed
    assert not failed.passed
    assert missing.failures == ("pit_quality_passed",)
    assert failed.failures == ("pit_quality_passed",)


def test_paper_qualification_requires_all_four_gates():
    qualified = evaluate_paper_qualification(
        {
            "trading_days": 63,
            "closed_trades": 100,
            "actual_slippage_bps": 4.0,
            "assumed_slippage_bps": 5.0,
        },
        backtest_pass=True,
    )
    assert qualified.passed

    failed = evaluate_paper_qualification(
        {
            "trading_days": 62,
            "closed_trades": 99,
            "actual_slippage_bps": 6.0,
            "assumed_slippage_bps": 5.0,
        },
        backtest_pass=False,
    )
    assert not failed.passed
    assert set(failed.failures) == {
        "trading_days",
        "closed_trades",
        "slippage_within_assumption",
        "backtest_pass",
    }


def test_markdown_report_contains_decisions_and_required_disclosures():
    backtest = evaluate_backtest_acceptance(
        {
            "oos_closed_trades": 300,
            "max_drawdown_pct": 9,
            "profit_factor": 1.4,
            "sharpe": 2,
            "worst_month_pct": -4,
            "concentration_pct": 35,
            "avg_monthly_return_pct": 3.25,
            "months_hit_10pct": 1,
            "months_total": 4,
            "pit_quality_passed": True,
        }
    )
    report = acceptance_markdown_report(backtest)
    assert "# Research Acceptance Report" in report
    assert "**Result:** PASS" in report
    assert "Average monthly return: 3.25%" in report
    assert "Months at or above 10%: 1 / 4 (25.0%)" in report


def test_promotion_record_uses_only_event_ls_paper_evidence_and_never_enables():
    account = {
        "params": {"slippage_bps": 5.0},
        "trades": [
            {
                "strategy": "event_ls",
                "action": "EXIT",
                "pnl": 10,
                "trading_date": f"2025-01-{day:02d}",
                "slippage_bps": 4.0,
            }
            for day in range(1, 32)
        ]
        + [
            {
                "strategy": "legacy",
                "action": "EXIT",
                "pnl": 10,
                "trading_date": "2025-02-01",
                "slippage_bps": 1.0,
            }
        ],
    }
    paper = paper_metrics_from_account(account)
    assert paper["trading_days"] == 31
    assert paper["closed_trades"] == 31
    assert paper["actual_slippage_bps"] == pytest.approx(4.0)

    record = promotion_record(
        account,
        {
            "oos_closed_trades": 300,
            "oos_max_drawdown_pct": 9,
            "oos_profit_factor": 1.4,
            "oos_sharpe": 2,
            "oos_worst_month_pct": -4,
            "oos_concentration_pct": 35,
            "pit_quality_passed": True,
        },
    )
    assert record["status"] == "BACKTEST_ONLY"
    assert record["backtest_passed"] is True
    assert record["paper_passed"] is False
    assert record["enabled"] is False
    assert record["target_10pct_monthly_guaranteed"] is False
