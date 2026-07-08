"""Tests for point-in-time snapshot storage and quote resolution."""

from __future__ import annotations

from datetime import date, time
from unittest.mock import patch

import pytest

from src.utils import pit_snapshots as pit
from src.utils.quote_resolve import session_observation


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return tmp_path


def test_snapshot_path_hhmm(snap_dir):
    path = pit.snapshot_path(date(2026, 7, 7), time(8, 0))
    assert "2026-07-07/0800.json" in path


def test_save_snapshot_immutable(snap_dir):
    d = date(2026, 7, 7)
    raw_v1 = {"trading_date": "2026-07-07", "version": 1}
    raw_v2 = {"trading_date": "2026-07-07", "version": 2}

    pit.save_snapshot(d, "step0_0745", raw_v1)
    result = pit.save_snapshot(d, "step0_0745", raw_v2)
    assert result["data"]["version"] == 1

    pit.save_snapshot(d, "step0_0745", raw_v2, force=True)
    loaded = pit.load_snapshot(d, "step0_0745")
    assert loaded["data"]["version"] == 2


def test_load_snapshot_raw_step1_wraps_raw(snap_dir):
    d = date(2026, 7, 7)
    inner = {"trading_date": "2026-07-07", "market": {"quotes": {}}}
    pit.save_snapshot(d, "step1_0800", {"raw": inner, "morning": {}})
    assert pit.load_snapshot_raw(d, "step1_0800") == inner


def test_parse_as_of_defaults_to_step_schedule():
    assert pit.parse_as_of(None, step_num=1) == time(8, 0)
    assert pit.parse_as_of("now", step_num=1) == "now"
    assert pit.parse_as_of("09:30", step_num=1) == time(9, 30)


def test_session_observation_pit_uses_raw_not_live(snap_dir):
    """PIT mode (as_of=08:00) must not call live intraday when raw has quotes."""
    trading_day = date(2026, 7, 7)
    raw = {
        "trading_date": "2026-07-07",
        "collected_at": "2026-07-07T07:45:00-04:00",
        "market": {
            "quotes": {
                "TSLA": {
                    "ticker": "TSLA",
                    "close": 399.0,
                    "prev_close": 390.0,
                    "open": 395.0,
                    "date": "2026-07-07",
                }
            }
        },
    }
    prior_raw = {
        "market": {
            "quotes": {
                "TSLA": {"close": 390.0},
            }
        }
    }

    with patch("src.utils.quote_resolve.intraday_session_quote") as mock_intraday:
        mock_intraday.return_value = {
            "ticker": "TSLA",
            "last": 419.0,
            "change_pct": 7.4,
            "source": "intraday",
        }
        obs = session_observation(
            "TSLA",
            raw,
            prior_raw,
            trading_day,
            section="market",
            as_of_et=time(8, 0),
        )
        mock_intraday.assert_not_called()

    assert obs["source"] == "raw"
    assert obs["last"] == 399.0
    assert obs["change_pct"] == pytest.approx(2.31, abs=0.1)


def test_format_as_of_display_morning():
    label = pit.format_as_of_display(time(8, 0), step_num=1)
    assert "08:00 ET" in label
    assert "早盘决策" in label
