"""Unit tests for trading-day raw data validation."""

from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch

from src.utils.data_freshness import (
    macro_series_tier,
    validate_raw_for_trading_date,
)
from src.utils.trading_calendar import prior_trading_day


def _quote(
    *,
    session: date,
    prior: date,
    close: float = 100.0,
    prior_close: float = 99.0,
    session_type: str = "premarket",
) -> dict:
    change = round((close - prior_close) / prior_close * 100, 2)
    out = {
        "ticker": "TEST",
        "close": close,
        "prior_close": prior_close,
        "prior_close_date": prior.isoformat(),
        "change_pct": change,
        "quote_session_date": session.isoformat(),
        "date": session.isoformat(),
        "session_type": session_type,
        "data_as_of": session.isoformat() if session_type == "regular" else f"{session.isoformat()}T07:45:00-04:00",
    }
    if session == prior:
        out["prior_session"] = True
    return out


def _minimal_raw(trading_date: date, *, quotes: dict | None = None) -> dict:
    prior = prior_trading_day(trading_date)
    q = quotes or {"SPY": _quote(session=prior, prior=prior)}
    return {
        "collected_at": datetime(
            trading_date.year, trading_date.month, trading_date.day, 7, 45, tzinfo=None
        ).isoformat() + "-04:00",
        "trading_date": trading_date.isoformat(),
        "prior_trading_day": prior.isoformat(),
        "market": {"quotes": q},
        "sector": {"quotes": {}},
        "stocks": {"quotes": {}},
        "news": {
            "published_gte": "2026-07-02T20:00:00Z",
            "polygon": [],
        },
        "macro": {"series": {}},
    }


class MondayAfterHolidayTests(unittest.TestCase):
    """2026-07-06 Monday: prior session is Thursday 2026-07-02 (July 3–4 holiday)."""

    def test_prior_trading_day_is_thursday(self) -> None:
        monday = date(2026, 7, 6)
        prior = prior_trading_day(monday)
        self.assertEqual(prior, date(2026, 7, 2))

    def test_premarket_quotes_on_prior_session_pass(self) -> None:
        monday = date(2026, 7, 6)
        prior = prior_trading_day(monday)
        raw = _minimal_raw(monday)
        raw["collected_at"] = "2026-07-06T07:45:00-04:00"
        raw["market"]["quotes"] = {
            "SPY": _quote(session=prior, prior=prior, close=550.0, prior_close=545.0),
            "QQQ": _quote(session=prior, prior=prior, close=480.0, prior_close=475.0),
        }
        raw["news"]["published_gte"] = "2026-07-02T20:00:00Z"

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-02T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, monday)

        self.assertTrue(result.ok, result.reasons)
        self.assertEqual(
            raw["market"]["quotes"]["SPY"]["prior_close_date"],
            prior.isoformat(),
        )


class StaleQuoteTests(unittest.TestCase):
    def test_stale_quote_date_fails_validation(self) -> None:
        trading_date = date(2026, 7, 6)
        prior = prior_trading_day(trading_date)
        stale_day = date(2026, 6, 26)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-06T07:45:00-04:00"
        raw["market"]["quotes"]["SPY"] = {
            "ticker": "SPY",
            "close": 550.0,
            "prior_close": 545.0,
            "prior_close_date": prior.isoformat(),
            "change_pct": 0.92,
            "quote_session_date": stale_day.isoformat(),
            "date": stale_day.isoformat(),
            "session_type": "regular",
            "data_as_of": "2026-07-06T07:45:00-04:00",
        }

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-02T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertFalse(result.ok)
        self.assertTrue(any("早于上一交易日" in r or "不属于" in r for r in result.reasons))

    def test_stale_change_pct_fails_validation(self) -> None:
        trading_date = date(2026, 7, 6)
        prior = prior_trading_day(trading_date)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-06T07:45:00-04:00"
        q = _quote(session=prior, prior=prior, close=550.0, prior_close=500.0)
        q["change_pct"] = 1.0
        raw["market"]["quotes"]["SPY"] = q

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-02T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertFalse(result.ok)
        self.assertTrue(any("change_pct" in r for r in result.reasons))


class MacroFreshnessTierTests(unittest.TestCase):
    def test_monthly_macro_is_reference_not_attention(self) -> None:
        trading_date = date(2026, 7, 6)
        prior = prior_trading_day(trading_date)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-06T07:45:00-04:00"
        raw["macro"]["series"] = {
            "CPIAUCSL": {"series_id": "CPIAUCSL", "date": "2026-05-01", "value": "320.1"},
            "UNRATE": {"series_id": "UNRATE", "date": "2026-05-01", "value": "4.1"},
        }

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-02T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.macro_reference), 2)
        self.assertFalse(any("早于上一交易日" in a for a in result.attention))
        self.assertEqual(macro_series_tier("CPIAUCSL"), "monthly")

    def test_daily_dgs10_weekend_lag_is_attention_not_blocking(self) -> None:
        trading_date = date(2026, 7, 6)
        prior = prior_trading_day(trading_date)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-06T07:45:00-04:00"
        raw["macro"]["series"] = {
            "DGS10": {"series_id": "DGS10", "date": "2026-07-01", "value": "4.25"},
        }
        raw["market"]["treasury_10y_fred"] = {"date": "2026-07-01", "value": "4.25"}

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-02T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.attention), 2)
        self.assertTrue(any("FRED 最新" in a for a in result.attention))
        self.assertFalse(any("早于上一交易日" in a for a in result.attention))

    def test_icsa_stale_only_after_fourteen_days(self) -> None:
        # Wed Jul 15: prior Saturday week-ending Jul 4 is 11 days old — still normal
        # before Thursday's next release (week ending Jul 11).
        trading_date = date(2026, 7, 15)
        prior = prior_trading_day(trading_date)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-15T07:45:00-04:00"
        raw["news"]["published_gte"] = "2026-07-14T20:00:00Z"
        raw["macro"]["series"] = {
            "ICSA": {"series_id": "ICSA", "date": "2026-07-04", "value": "215000"},
        }

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-14T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.macro_reference), 1)
        self.assertEqual(result.attention, [])
        self.assertEqual(prior, date(2026, 7, 14))

        # 15 calendar days exceeds the 14-day weekly allowance.
        raw["macro"]["series"]["ICSA"]["date"] = "2026-06-27"
        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-14T20:00:00Z"):
            stale = validate_raw_for_trading_date(raw, trading_date)
        self.assertTrue(any("已超过" in a for a in stale.attention))


class BondProxyTests(unittest.TestCase):
    """^TNX is a bond yield proxy — stale bars must not block equity validation."""

    def test_stale_tnx_error_does_not_block(self) -> None:
        trading_date = date(2026, 7, 7)
        prior = prior_trading_day(trading_date)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-07T07:45:00-04:00"
        raw["trading_date"] = trading_date.isoformat()
        raw["prior_trading_day"] = prior.isoformat()
        raw["market"]["quotes"] = {
            "SPY": _quote(session=prior, prior=prior, close=550.0, prior_close=545.0),
            "^TNX": {
                "ticker": "^TNX",
                "error": f"stale bar date 2026-07-02 (expected {trading_date} or {prior})",
                "data_as_of": "2026-07-07T07:45:00-04:00",
            },
        }
        raw["news"]["published_gte"] = "2026-07-06T20:00:00Z"

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-06T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertTrue(result.ok, result.reasons)
        self.assertTrue(any("^TNX" in a and "债券数据正常滞后" in a for a in result.attention))
        self.assertFalse(any("^TNX" in r for r in result.reasons))

    def test_stale_tnx_session_date_is_attention_only(self) -> None:
        trading_date = date(2026, 7, 7)
        prior = prior_trading_day(trading_date)
        stale = date(2026, 7, 2)
        raw = _minimal_raw(trading_date)
        raw["collected_at"] = "2026-07-07T07:45:00-04:00"
        raw["market"]["quotes"] = {
            "SPY": _quote(session=prior, prior=prior),
            "^TNX": {
                "ticker": "^TNX",
                "close": 4.25,
                "quote_session_date": stale.isoformat(),
                "date": stale.isoformat(),
                "session_type": "fallback",
                "data_as_of": "2026-07-07T07:45:00-04:00",
            },
        }
        raw["news"]["published_gte"] = "2026-07-06T20:00:00Z"

        with patch("src.utils.data_freshness.prior_close_utc_iso", return_value="2026-07-06T20:00:00Z"):
            result = validate_raw_for_trading_date(raw, trading_date)

        self.assertTrue(result.ok, result.reasons)
        self.assertTrue(any("proxy 最新" in a for a in result.attention))


if __name__ == "__main__":
    unittest.main()
