"""Tests for the verification-epoch restart (Rae's decision 2026-11-17).

Old epoch 2026-11-12 broke the zero-fault gate twice (11-13 day-breaker
trip; 11-16 Polygon 429 blocking all intraday entries). New epoch day 1 =
2026-11-17.

Counting convention (pinned here, matches the old epoch's recorded dates):
"第 60 个交易日" = the 60th NYSE session AFTER the start date, start day
excluded. Proof the old epoch used this convention: the recorded old
60th day 2027-02-10 satisfies count_trading_days(2026-11-13, 2027-02-10)
== 60, while inclusive counting would give 2027-02-09. Under the same
convention the new 60th day is 2027-02-16 (2027-02-15 is Presidents Day,
market closed). The 60-calendar-day checkpoint is likewise start + 60
days = 2027-01-16.
"""

from __future__ import annotations

from datetime import date

from src.paper.account import VERIFICATION_EPOCH_START, verification_epoch_summary
from src.utils.trading_calendar import count_trading_days, is_trading_day

NEW_START = date(2026, 11, 17)
DAY_60 = date(2027, 2, 16)
CHECKPOINT_60D = date(2027, 1, 16)


def test_epoch_start_constant_is_2026_11_17() -> None:
    assert VERIFICATION_EPOCH_START == "2026-11-17"


def test_old_epoch_convention_anchor() -> None:
    # Old epoch: start 2026-11-12, recorded 60th trading day 2027-02-10.
    # 60 sessions after the start day, start excluded.
    assert count_trading_days(date(2026, 11, 13), date(2027, 2, 10)) == 60
    # Inclusive counting would have given 2027-02-09 instead -- not the
    # recorded date, so the project convention is start-excluded.
    assert count_trading_days(date(2026, 11, 12), date(2027, 2, 9)) == 60


def test_new_epoch_day_60_is_2027_02_16() -> None:
    assert count_trading_days(date(2026, 11, 18), DAY_60) == 60
    assert is_trading_day(DAY_60)


def test_2027_02_15_is_presidents_day_holiday() -> None:
    assert not is_trading_day(date(2027, 2, 15))
    # The session before the holiday is the 59th after start; the holiday
    # itself must not be counted as a trading day.
    assert count_trading_days(date(2026, 11, 18), date(2027, 2, 12)) == 59


def test_60_calendar_day_checkpoint_is_2027_01_16() -> None:
    assert (CHECKPOINT_60D - NEW_START).days == 60
    assert CHECKPOINT_60D.weekday() == 5  # Saturday
    # ~41 trading days by the checkpoint (not the 60-trading-day gate).
    assert count_trading_days(NEW_START, CHECKPOINT_60D) == 41


def test_epoch_summary_filters_legs_by_started_on() -> None:
    account = {
        "verification_epoch": {"started_on": "2026-11-17"},
        "trades": [
            {
                "action": "EXIT",
                "symbol": "AMD",
                "trading_date": "2026-11-16",
                "pnl": -100.0,
                "r_multiple": -1.0,
            },
            {
                "action": "EXIT",
                "symbol": "AMD",
                "trading_date": "2026-11-17",
                "pnl": -128.23,
                "r_multiple": -1.0,
            },
            {
                "action": "EXIT",
                "symbol": "NVDA",
                "trading_date": "2026-11-18",
                "pnl": 50.0,
                "r_multiple": 1.5,
            },
        ],
    }
    summary = verification_epoch_summary(account)
    assert summary["started_on"] == "2026-11-17"
    # The 2026-11-16 leg (old epoch) is excluded; legs on/after the start
    # date are counted (boundary is inclusive: >= started_on).
    assert summary["closed_legs"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["realized_pnl"] == -78.23


def test_epoch_summary_without_started_on_counts_all_legs() -> None:
    account = {
        "trades": [
            {"action": "EXIT", "symbol": "AMD", "trading_date": "2026-11-12", "pnl": 10.0},
        ],
    }
    assert verification_epoch_summary(account)["closed_legs"] == 1
