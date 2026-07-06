"""Tests for morning rule engine — driver taxonomy and options logic."""

from __future__ import annotations

from src.research.rules import (
    DRIVER_TYPES,
    _build_p9_options,
    _daily_driver_type_and_driver,
)
from src.utils.driver_match import driver_match_level


def test_no_catalyst_day_not_macro():
    dtype, driver = _daily_driver_type_and_driver(
        catalysts_today=[],
        chip_selloff=False,
        smh_pct=0.8,
        strongest_sector="SMH",
        qqq_pct=0.2,
    )
    assert dtype == "Momentum"
    assert driver == "AI Momentum"
    assert dtype != "Macro"
    assert dtype in DRIVER_TYPES


def test_nfp_day_is_macro():
    dtype, driver = _daily_driver_type_and_driver(
        catalysts_today=[{"name": "NFP"}],
        chip_selloff=False,
        smh_pct=-1.0,
        strongest_sector="XLF",
        qqq_pct=0.1,
    )
    assert dtype == "Macro"
    assert driver == "NFP"


def test_p9_buy_no_implies_zero_dte_no():
    p9 = _build_p9_options(
        catalyst_today=True,
        pre_holiday=False,
        iv_level="Low",
        total=2,
        options={},
    )
    assert p9["buy_options"] == "No"
    assert p9["zero_dte"] == "No"


def test_vague_macro_driver_mismatch():
    assert driver_match_level("Macro", "AI Momentum", morning_driver_type="Macro") == "错"
