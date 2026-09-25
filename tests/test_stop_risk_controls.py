"""Regression tests for stop-loss execution and position risk controls.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Covers the 2026-08-24 ARM pathology (stop 239.49, late-tick fill 243.393 →
-3.67R) and the guards that prevent its recurrence:
  - stop-market fills use first-touch + bounded slippage, not the stale quote
  - true gaps through the stop still fill honestly at the open (flagged)
  - no same-symbol re-entry after a stop-out for the rest of the day
  - entries with absurdly tight / mis-sided stops are rejected
  - hard per-trade loss breaker when no valid stop exists
  - profitable stop exits are labeled distinctly from losing stops
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.paper.account import default_account, get_position
from src.paper.broker_sim import can_afford, execute_entry, size_shares
from src.paper.execution_agent import (
    _r_multiple,
    _stop_fill_price,
    _try_entry,
    _validate_stop_geometry,
    decide_and_act,
)
from src.utils.paths import raw_data_path
from src.utils.trading_calendar import prior_trading_day


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _raw_bar(symbol, trading_date, *, open, high, low, close):
    return {
        "stocks": {
            "quotes": {
                symbol: {
                    "open": open,
                    "high": high,
                    "low": low,
                    "close": close,
                    "date": trading_date,
                }
            }
        }
    }


def _write_prior_close(data_root, trading_date, symbol, close):
    prior_day = prior_trading_day(date.fromisoformat(trading_date)).isoformat()
    path = raw_data_path(prior_day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"stocks": {"quotes": {symbol: {"close": close, "date": prior_day}}}}
        ),
        encoding="utf-8",
    )
    return prior_day


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_stop_fill_price_caps_late_tick():
    # SHORT, stop touched during the bar, tick arrives late at 243.393
    fill, capped, gap = _stop_fill_price(
        "SHORT",
        239.49,
        243.393,
        {"open": 238.2, "high": 243.5, "low": 237.8},
        0.5,
    )
    assert fill == pytest.approx(239.49 * 1.005, abs=1e-6)
    assert capped is True
    assert gap is False


def test_stop_fill_price_true_gap_uses_open():
    # Gapped up through the stop overnight: fill at the open, honestly.
    fill, capped, gap = _stop_fill_price(
        "SHORT",
        239.49,
        243.6,
        {"open": 243.5, "high": 244.0, "low": 243.2, "prev_close": 238.0},
        0.5,
    )
    assert fill == pytest.approx(243.5)
    assert gap is True
    assert capped is False


def test_stop_fill_price_near_stop_uses_quote():
    fill, capped, gap = _stop_fill_price(
        "LONG",
        335.0,
        334.5,
        {"open": 339.0, "high": 340.0, "low": 334.0},
        0.5,
    )
    assert fill == pytest.approx(334.5)
    assert capped is False
    assert gap is False


def test_validate_stop_geometry():
    assert _validate_stop_geometry("LONG", 341.0, 335.0, 0.8) is None
    assert _validate_stop_geometry("SHORT", 238.03, 240.50, 0.8) is None  # 1.04% ok
    # The real 2026-08-24 ARM geometry: 0.61% stop — correctly rejected.
    r = _validate_stop_geometry("SHORT", 238.03, 239.49, 0.8)
    assert r is not None and "止损距离" in r
    # Too tight: 0.29% < 0.8%
    r = _validate_stop_geometry("LONG", 341.0, 340.0, 0.8)
    assert r is not None and "止损距离" in r
    # Mis-sided
    r = _validate_stop_geometry("LONG", 341.0, 342.0, 0.8)
    assert r is not None and "多头" in r
    r = _validate_stop_geometry("SHORT", 238.0, 237.0, 0.8)
    assert r is not None and "空头" in r
    # Missing levels: nothing to validate (breaker covers the tail)
    assert _validate_stop_geometry("LONG", 341.0, None, 0.8) is None


def test_r_multiple():
    assert _r_multiple("LONG", 341.0, 335.0, 334.5) == pytest.approx(-1.0833, abs=1e-3)
    assert _r_multiple("SHORT", 238.0286, 239.49, 240.687) == pytest.approx(
        -1.819, abs=0.01
    )
    assert _r_multiple("LONG", 258.4175, 269.8369, 269.80) > 0


# ---------------------------------------------------------------------------
# Integration: the 2026-08-24 ARM case
# ---------------------------------------------------------------------------


def test_arm_late_tick_stop_capped(data_root):
    """Regression: stop 239.49, late quote 243.393 must not print -3.67R."""
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="SHORT",
        price=238.0286,
        shares=19,
        stop=239.49,
        target=230.0,
        reason="test",
        trading_date="2026-08-24",
        book="intraday",
    )
    raw = _raw_bar(
        "ARM", "2026-08-24", open=238.2, high=243.5, low=237.8, close=243.393
    )
    decision = decide_and_act(acct, "2026-08-24", session_phase="open", raw=raw)

    assert decision["action"] == "EXIT"
    trade = decision["trade"]
    # First-touch model: stop + 0.5% slippage cap, not the 243.393 quote.
    assert trade["price"] == pytest.approx(239.49 * 1.005, abs=0.02)
    assert trade["slippage_capped"] is True
    assert trade["gap_through_stop"] is False
    assert trade["stop_trigger"] == pytest.approx(239.49)
    r = trade["r_multiple"]
    assert r is not None and r > -2.0  # was -3.67R before the fix
    assert "止损触发" in (decision["reason"] or "")
    assert get_position(acct, "intraday") is None
    # Cooldown recorded: same symbol must not re-enter today.
    cd = (acct.get("risk_state") or {}).get("stop_cooldown") or {}
    assert cd.get("2026-08-24:intraday:ARM") is True


def test_true_gap_still_fills_honestly(data_root):
    """A real overnight gap through the stop is a real loss — not hidden."""
    _write_prior_close(data_root, "2026-08-24", "ARM", 238.0)
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="SHORT",
        price=238.0286,
        shares=19,
        stop=239.49,
        target=230.0,
        reason="test",
        trading_date="2026-08-24",
        book="intraday",
    )
    raw = _raw_bar(
        "ARM", "2026-08-24", open=243.5, high=244.0, low=243.2, close=243.6
    )
    decision = decide_and_act(acct, "2026-08-24", session_phase="open", raw=raw)

    assert decision["action"] == "EXIT"
    trade = decision["trade"]
    assert trade["price"] == pytest.approx(243.5)
    assert trade["gap_through_stop"] is True
    assert trade["slippage_capped"] is False
    assert trade["r_multiple"] is not None and trade["r_multiple"] < -2.0
    assert "跳空" in (decision["reason"] or "")


def test_stop_cooldown_blocks_reentry(data_root):
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="SHORT",
        price=238.0286,
        shares=19,
        stop=239.49,
        target=230.0,
        reason="test",
        trading_date="2026-08-24",
        book="intraday",
    )
    raw = _raw_bar(
        "ARM", "2026-08-24", open=238.2, high=243.5, low=237.8, close=243.393
    )
    d1 = decide_and_act(acct, "2026-08-24", session_phase="open", raw=raw)
    assert d1["action"] == "EXIT"
    n_trades = len(acct["trades"])

    picked = {
        "action": "enter",
        "signal": {
            "symbol": "ARM",
            "direction": "SHORT",
            "entry_price": 238.03,
            "entry_zone": {"low": 237.5, "mid": 238.03, "high": 238.6},
            "stop_price": 239.49,
            "target_price": 230.0,
            "horizon": "Intraday",
            "source": "session_primary",
        },
        "quote": 238.5,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test re-entry",
    }
    out = _try_entry(acct, "intraday", picked, "2026-08-24")
    assert out["action"] == "SKIP"
    assert "冷却" in (out["reason"] or "")
    assert get_position(acct, "intraday") is None
    assert len(acct["trades"]) == n_trades


def test_entry_rejects_tight_stop(data_root):
    acct = default_account()
    picked = {
        "action": "enter",
        "signal": {
            "symbol": "ARM",
            "direction": "LONG",
            "entry_price": 341.0,
            "entry_zone": {"low": 340.0, "mid": 341.0, "high": 342.0},
            "stop_price": 340.0,  # 0.29% away — absurdly tight
            "target_price": 354.0,
            "horizon": "Intraday",
            "source": "morning_primary",
        },
        "quote": 341.0,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }
    out = _try_entry(acct, "intraday", picked, "2026-08-24")
    assert out["action"] == "SKIP"
    assert "止损距离" in (out["reason"] or "")
    assert get_position(acct, "intraday") is None


def test_entry_rejects_missided_stop(data_root):
    acct = default_account()
    picked = {
        "action": "enter",
        "signal": {
            "symbol": "ARM",
            "direction": "LONG",
            "entry_price": 341.0,
            "entry_zone": {"low": 340.0, "mid": 341.0, "high": 342.0},
            "stop_price": 342.0,  # above entry for a LONG — invalid
            "target_price": 354.0,
            "horizon": "Intraday",
            "source": "morning_primary",
        },
        "quote": 341.0,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }
    out = _try_entry(acct, "intraday", picked, "2026-08-24")
    assert out["action"] == "SKIP"
    assert "多头" in (out["reason"] or "")


def test_breaker_exits_without_valid_stop(data_root):
    """Hard backstop: no stop set, -5% equity unrealized → market exit."""
    acct = default_account()
    acct["positions"]["intraday"] = {
        "symbol": "ARM",
        "direction": "LONG",
        "shares": 50,
        "avg_entry": 200.0,
        "stop": None,
        "target": 250.0,
        "last_price": 200.0,
        "horizon": "Intraday",
        "book": "intraday",
        "signal_source": "test",
        "opened_date": "2026-08-24",
    }
    raw = _raw_bar(
        "ARM", "2026-08-24", open=192.0, high=195.0, low=189.0, close=190.0
    )
    decision = decide_and_act(acct, "2026-08-24", session_phase="open", raw=raw)

    assert decision["action"] == "EXIT"
    assert "熔断" in (decision["reason"] or "")
    assert decision["trade"]["pnl"] == pytest.approx(-500.0)
    assert get_position(acct, "intraday") is None


def test_profitable_stop_labeled_distinctly(data_root):
    """A stop that exits in profit (e.g. trailed above entry) is not a loss."""
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="LONG",
        price=258.4175,
        shares=6,
        stop=269.8369,  # above entry — trailed stop
        target=280.0,
        reason="test",
        trading_date="2026-09-09",
        book="intraday",
    )
    raw = _raw_bar(
        "ARM", "2026-09-09", open=270.0, high=271.0, low=269.5, close=269.80
    )
    decision = decide_and_act(acct, "2026-09-09", session_phase="open", raw=raw)

    assert decision["action"] == "EXIT"
    trade = decision["trade"]
    assert trade["pnl"] > 0
    assert trade["r_multiple"] is not None and trade["r_multiple"] >= 0
    assert "盈利/保本" in (decision["reason"] or "")


def test_slippage_reserve_shrinks_tight_stop_size():
    kw = dict(
        cash=100000.0,
        equity=10000.0,
        entry_price=100.0,
        stop_price=101.0,
        direction="SHORT",
        risk_pct=1.0,
        max_position_pct=100.0,
    )
    s0 = size_shares(**kw, slippage_reserve_pct=0.0)
    s1 = size_shares(**kw, slippage_reserve_pct=0.002)
    assert s1 < s0
    assert s0 == 100  # 100 // 1.0
    assert s1 == 83  # 100 // 1.2


def test_can_afford_uses_reserve_from_params(data_root):
    acct = default_account()  # params slippage_reserve_pct=0.2
    shares, err = can_afford(
        acct,
        price=100.0,
        stop=101.0,
        direction="SHORT",
        risk_pct=1.0,
        max_position_pct=100.0,
    )
    expect = size_shares(
        cash=10000.0,
        equity=10000.0,
        entry_price=100.0,
        stop_price=101.0,
        direction="SHORT",
        risk_pct=1.0,
        max_position_pct=100.0,
        slippage_reserve_pct=0.002,
    )
    assert err is None
    assert shares == expect


def test_nan_quote_never_poisons_equity(data_root):
    """Regression: 2026-09-23 20:14 ET tick wrote NaN equity from a NaN quote.

    mark_to_market must skip non-finite quotes, keep the last good mark,
    and never persist a NaN equity value.
    """
    import math

    from src.paper.account import mark_to_market

    acct = default_account()
    execute_entry(
        acct,
        symbol="MSFT",
        direction="SHORT",
        price=493.6319,
        shares=2,
        stop=509.95,
        target=472.16,
        reason="test",
        trading_date="2026-09-23",
        book="swing",
    )
    mark_to_market(acct, price_by_symbol={"MSFT": 500.59})
    good_equity = acct["equity"]
    good_unrealized = acct["unrealized_pnl"]
    assert math.isfinite(good_equity)
    # Short 2 @ 493.6319 marked at 500.59 -> unrealized ≈ -13.92
    assert good_unrealized == pytest.approx(-13.92, abs=0.05)

    # After-hours tick with a NaN quote must not poison anything.
    mark_to_market(acct, price_by_symbol={"MSFT": float("nan")})
    assert math.isfinite(acct["equity"])
    assert acct["equity"] == pytest.approx(good_equity, abs=0.01)
    assert math.isfinite(acct["unrealized_pnl"])

    # Even a persisted NaN mark must be quarantined, not propagated.
    pos = acct["positions"]["swing"]
    pos["last_price"] = float("nan")
    pos["unrealized_pnl"] = float("nan")
    acct["equity"] = float("nan")
    mark_to_market(acct, price_by_symbol={"MSFT": float("nan")})
    assert math.isfinite(acct["equity"]), "equity must fall back, never stay NaN"


def test_short_equity_does_not_double_count_sale_proceeds(data_root):
    """Regression (found 2026-09-25 by paper-intraday-watch).

    execute_entry credits short-sale proceeds to cash, so mark_to_market must
    value a short at its *negative* market value. The old formula added only
    the unrealized PnL (``cash + short unrealized``), double-counting the
    proceeds: while the 2026-09-21 MSFT short (2 @ 493.6319) was open,
    displayed equity was overstated by exactly 2*493.6319 ≈ $987.26, masking
    the true drawdown and inflating risk sizing.
    """
    from src.paper.account import mark_to_market

    acct = default_account()
    execute_entry(
        acct,
        symbol="MSFT",
        direction="SHORT",
        price=493.6319,
        shares=2,
        stop=509.95,
        target=472.16,
        reason="test",
        trading_date="2026-09-21",
        book="swing",
    )
    # Cash holds the $987.26 sale proceeds.
    assert acct["cash"] == pytest.approx(10000.0 + 2 * 493.6319, abs=0.01)

    mark_to_market(acct, price_by_symbol={"MSFT": 500.59})
    # Correct: 10987.26 - 2*500.59 = 9986.08 (unrealized -13.92).
    # Buggy:   10987.26 + (-13.92) = 10973.34 (proceeds double-counted).
    assert acct["equity"] == pytest.approx(10987.26 - 2 * 500.59, abs=0.05)
    assert acct["unrealized_pnl"] == pytest.approx(-13.92, abs=0.05)


def test_long_equity_is_cash_plus_market_value(data_root):
    """Long side of the same identity: equity = cash + shares * mark."""
    from src.paper.account import mark_to_market

    acct = default_account()
    execute_entry(
        acct,
        symbol="AMD",
        direction="LONG",
        price=633.35,
        shares=1,
        stop=599.52,
        target=693.56,
        reason="test",
        trading_date="2026-09-25",
        book="swing",
    )
    mark_to_market(acct, price_by_symbol={"AMD": 632.425})
    assert acct["cash"] == pytest.approx(10000.0 - 633.35, abs=0.01)
    assert acct["equity"] == pytest.approx(10000.0 - 633.35 + 632.425, abs=0.05)
    assert acct["unrealized_pnl"] == pytest.approx(-0.925, abs=0.05)
