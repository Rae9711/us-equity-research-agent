"""Tests for the six execution/record fixes (2026-11-12) + revised #2/#7.

Fix 1: tick scans 1-minute bars in order; first stop/target touch wins.
Fix 2 (revised): intraday legs hold overnight, max 3 trading days, then
  deterministically flatten at the close window (or first later tick).
Fix 3: every exit records r_multiple (stop, target, EOD flatten, breaker).
Fix 4: -$589.73 booked as an explicit history adjustment + epoch start.
Fix 5: portfolio-level planned-risk cap (existing + new <= 3% equity).
Fix 6: no new entries when Polygon intraday news data is missing/429.
Fix 7 (revised): intraday <= 3 legs, swing <= 2 legs, 5 total; the
  per-book notional cap is shared by the book's legs.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json

import pytest

from src.paper.account import (
    HISTORY_ADJUSTMENT_AMOUNT,
    default_account,
    ensure_history_adjustment,
    get_position,
    get_positions,
    load_account,
    record_history_adjustment,
    save_account,
    verification_epoch_summary,
)
from src.paper.allocation import apply_portfolio_risk_cap, portfolio_planned_risk
from src.paper.broker_sim import execute_entry, execute_exit
from src.paper.execution_agent import (
    _try_entry,
    decide_and_act,
    flatten_intraday_at_close,
)
from src.paper.news_gate import (
    news_block_reason,
    news_gate_for_date,
    parse_news_error_entries,
)
from src.utils.trading_calendar import holding_trading_days


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _bar(ts, open_, high, low, close):
    return {"ts": ts, "open": open_, "high": high, "low": low, "close": close}


def _open_leg(acct, symbol, *, book, price, shares, stop, target,
              trading_date, direction="LONG"):
    return execute_entry(
        acct,
        symbol=symbol,
        direction=direction,
        price=price,
        shares=shares,
        stop=stop,
        target=target,
        reason="test",
        trading_date=trading_date,
        signal={
            "source": "test",
            "horizon": "Swing" if book == "swing" else "Intraday",
            # Production _try_entry passes the full picked slot; the
            # broker's sane check anchors on the planned entry, so the
            # test signal must carry it too (a bare signal leaves only
            # stop/target anchors, which sit 10%+ away on wide-stop plans
            # like the live AMD leg 557.495/493.4/643.27).
            "entry_price": price,
            "stop_price": stop,
            "target_price": target,
            "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
        },
        book=book,
    )


def _picked(symbol, *, price=341.0, stop=335.0, target=354.0,
            direction="LONG", book="intraday", win_prob=60):
    # NOTE: win_prob is a PERCENT in this codebase (e.g. 60 = 60%).
    return {
        "action": "enter",
        "signal": {
            "symbol": symbol,
            "direction": direction,
            "entry_price": price,
            "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
            "stop_price": stop,
            "target_price": target,
            "horizon": "Swing" if book == "swing" else "Intraday",
            "source": "morning_primary",
            "win_prob": win_prob,
        },
        "quote": price,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }


def _fake_quote(price):
    return lambda *a, **k: (price, "test")


def _exit_row(decision, book):
    rows = [
        b for b in (decision.get("books") or [])
        if b.get("book") == book and b.get("action") == "EXIT"
    ]
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Fix 1: sequential 1-minute bar scan
# ---------------------------------------------------------------------------


def test_bar_scan_stop_first_touch_fills_at_trigger(data_root, monkeypatch):
    """A stop touched mid-interval exits at the bar's trigger, not the late
    tick quote — the 2026-08-24 ARM pathology (-3.67R on a stale quote)."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(338.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 340.0, 340.0, 334.0, 336.0),  # low 334 < stop
        _bar("2026-11-09 09:33", 336.0, 337.0, 335.0, 336.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    trade = row["trade"]
    # Trigger price 334 is inside the 0.5% slippage band (worst 333.325),
    # so the fill is the first-touch trigger, not the late quote.
    assert trade["price"] == pytest.approx(334.0)
    assert trade["bar_ts"] == "2026-11-09 09:32"
    assert trade["gap_through_stop"] is False
    assert trade["slippage_capped"] is False
    assert get_position(acct, "intraday") is None


def test_bar_scan_target_fills_at_target(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 341.0, 342.0, 340.0, 341.0),
        _bar("2026-11-09 09:32", 341.0, 354.5, 341.0, 353.0),  # high > target
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    trade = row["trade"]
    assert trade["price"] == pytest.approx(354.0)  # first-touch price
    assert trade["gap_through_target"] is False
    assert trade["bar_ts"] == "2026-11-09 09:32"


def test_bar_scan_gap_through_target_fills_at_open(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 349.0, 350.0, 349.0, 350.0),
        _bar("2026-11-09 09:32", 356.0, 358.0, 356.0, 357.0),  # gapped open
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    trade = row["trade"]
    assert trade["price"] == pytest.approx(356.0)  # the gap open, not 354
    assert trade["gap_through_target"] is True


def test_bar_scan_gap_through_stop_fills_at_open(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 340.0, 340.0),
        _bar("2026-11-09 09:32", 330.0, 330.0, 328.0, 329.0),  # gapped down
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    trade = row["trade"]
    assert trade["price"] == pytest.approx(330.0)  # honest gap loss at open
    assert trade["gap_through_stop"] is True
    assert trade["r_multiple"] == pytest.approx((330.0 - 341.0) / 6.0, abs=0.01)


def test_bar_scan_stop_before_target_in_same_bar(data_root, monkeypatch):
    """Within one minute the order is unknowable — the stop wins."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 341.0, 355.0, 334.0, 345.0),  # both touched
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert "止损" in (row["reason"] or "")
    assert row["trade"]["price"] == pytest.approx(334.0)


def test_bar_scan_no_touch_holds_and_fallback_single_price(
    data_root, monkeypatch
):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 341.0, 343.0, 339.0, 342.0),
        _bar("2026-11-09 09:32", 342.0, 344.0, 340.0, 343.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None

    # No bars at all (empty map) falls back to the single-price path: a
    # quote below the stop still exits, at the capped first-touch price.
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(334.0)
    )
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open", bars_by_symbol={},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert row["trade"]["price"] == pytest.approx(334.0)
    assert "bar_ts" not in (row["trade"] or {})


# ---------------------------------------------------------------------------
# Fix 3: r_multiple on every exit
# ---------------------------------------------------------------------------


def test_target_exit_records_r_multiple(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [_bar("2026-11-09 09:31", 341.0, 355.0, 341.0, 354.0)]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    # R = (354 - 341) / (341 - 335) = 13/6
    assert row["trade"]["r_multiple"] == pytest.approx(13 / 6, abs=0.01)
    assert "R=2.17" in (row["reason"] or "")


def test_short_target_exit_records_r_multiple(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=347.0, target=328.0, trading_date="2026-11-09",
              direction="SHORT")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [_bar("2026-11-09 09:31", 341.0, 341.0, 327.0, 329.0)]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    # R = (341 - 328) / (347 - 341) = 13/6
    assert row["trade"]["r_multiple"] == pytest.approx(13 / 6, abs=0.01)


def test_execute_exit_records_r_centrally(data_root):
    """Any exit routed through execute_exit carries R — the breaker path
    included — because R is set centrally, not per exit branch."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    trade = execute_exit(
        acct, price=345.0, reason="manual", trading_date="2026-11-09",
        book="intraday",
    )
    assert trade["r_multiple"] == pytest.approx((345.0 - 341.0) / 6.0, abs=0.01)


# ---------------------------------------------------------------------------
# Fix 4: history adjustment + verification epoch
# ---------------------------------------------------------------------------


def test_history_adjustment_recorded_and_idempotent(data_root):
    acct = default_account()
    assert record_history_adjustment(acct) is True
    adj = acct["history_adjustment"]
    assert adj["amount"] == HISTORY_ADJUSTMENT_AMOUNT == -589.73
    assert "2026-09-21" in adj["reason"]
    assert acct["verification_epoch"]["started_on"] == "2026-11-12"
    assert acct["equity"] == pytest.approx(10000.0)
    assert acct["realized_pnl"] == 0.0
    # Second call is a no-op: the record is never duplicated or rewritten.
    assert record_history_adjustment(acct) is False
    assert acct["history_adjustment"]["amount"] == -589.73
    assert len(acct["journal"]) >= 1


def test_ensure_history_adjustment_only_with_trades(data_root):
    fresh = default_account()
    assert ensure_history_adjustment(fresh) is False
    assert fresh.get("history_adjustment") is None

    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    assert ensure_history_adjustment(acct) is True
    assert acct["history_adjustment"]["amount"] == -589.73


def test_verification_epoch_summary_date_filter(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    # Pre-epoch leg (before 2026-11-12) must not count toward the epoch.
    execute_exit(acct, price=330.0, reason="pre-epoch",
                 trading_date="2026-11-09", book="intraday")
    _open_leg(acct, "NVDA", book="swing", price=170.0, shares=2,
              stop=160.0, target=190.0, trading_date="2026-11-12")
    execute_exit(acct, price=180.0, reason="epoch",
                 trading_date="2026-11-12", book="swing")
    record_history_adjustment(acct)
    summary = verification_epoch_summary(acct)
    assert summary["started_on"] == "2026-11-12"
    assert summary["closed_legs"] == 1
    assert summary["wins"] == 1
    assert summary["realized_pnl"] == pytest.approx(20.0)
    assert summary["r_multiples"] == [pytest.approx(1.0)]


def test_load_account_backfills_adjustment(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    assert acct.get("history_adjustment") is None
    save_account(acct)
    loaded = load_account()
    assert loaded["history_adjustment"]["amount"] == -589.73
    assert loaded["verification_epoch"]["started_on"] == "2026-11-12"
    # The open leg survives the list-schema migration intact.
    assert get_position(loaded, "intraday")["symbol"] == "ARM"


# ---------------------------------------------------------------------------
# Fix 5: portfolio-level planned-risk cap (3% of equity, all books)
# ---------------------------------------------------------------------------


def test_portfolio_cap_reduces_shares(data_root):
    acct = default_account()
    _open_leg(acct, "AMD", book="swing", price=557.495, shares=2,
              stop=493.4, target=643.27, trading_date="2026-11-09")
    summary = portfolio_planned_risk(acct)
    # 2 x |557.495 - 493.4| = $128.19 of the $300 cap (3% of $10,000)
    assert summary["total_planned_risk"] == pytest.approx(128.19, abs=0.01)
    assert summary["cap_dollars"] == pytest.approx(300.0)
    assert summary["headroom"] == pytest.approx(171.81, abs=0.01)

    shares, detail = apply_portfolio_risk_cap(
        acct, direction="LONG", entry_price=500.0, stop_price=400.0, shares=3,
    )
    # 3 x $100 = $300 new + $128.19 existing > $300 cap -> 1 share fits
    assert detail["capped"] is True
    assert detail["allowed_shares"] == 1
    assert shares == 1


def test_portfolio_cap_skips_when_one_share_breaches(data_root):
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=500.0, shares=2,
              stop=355.0, target=600.0, trading_date="2026-11-09")
    # Existing risk 2 x $145 = $290; headroom is only $10.
    shares, detail = apply_portfolio_risk_cap(
        acct, direction="LONG", entry_price=341.0, stop_price=300.0, shares=2,
    )
    assert shares == 0
    assert detail["capped"] is True
    assert "无法新开仓" in detail["reason_zh"]

    # The same breach via _try_entry is a SKIP, not a partial entry.
    # (stop 320 -> $21/share risk > $10 headroom; R:R 0.90 still deployable,
    # so the skip is the portfolio cap's doing, not a cash hold.)
    out = _try_entry(
        acct, "intraday", _picked("ARM", stop=320.0, target=360.0),
        "2026-11-09",
    )
    assert out["action"] == "SKIP"
    assert "组合计划风险" in (out["reason"] or "")
    assert get_position(acct, "intraday") is None


def test_portfolio_cap_allows_within_cap(data_root):
    acct = default_account()
    shares, detail = apply_portfolio_risk_cap(
        acct, direction="LONG", entry_price=341.0, stop_price=335.0, shares=3,
    )
    assert detail["capped"] is False
    assert shares == 3


def test_portfolio_cap_counts_both_books(data_root):
    acct = default_account()
    _open_leg(acct, "AMD", book="swing", price=557.495, shares=2,
              stop=493.4, target=643.27, trading_date="2026-11-09")
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    # $128.19 swing + $30 intraday = $158.19
    summary = portfolio_planned_risk(acct)
    assert summary["total_planned_risk"] == pytest.approx(158.19, abs=0.01)
    assert len(summary["legs"]) == 2
    shares, detail = apply_portfolio_risk_cap(
        acct, direction="LONG", entry_price=500.0, stop_price=400.0, shares=2,
    )
    # 2 x $100 = $200 new + $158.19 > $300 -> 1 share fits
    assert shares == 1
    assert detail["capped"] is True


# ---------------------------------------------------------------------------
# Fix 6: news-availability gate
# ---------------------------------------------------------------------------


def test_news_gate_blocks_symbol():
    parsed = parse_news_error_entries(
        ["AAPL:Client error '429 Too Many Requests' for url ..."]
    )
    assert parsed["blocked_symbols"] == {"AAPL"}
    assert parsed["global_block"] is False
    assert parsed["broad_failed"] is False


def test_news_universe_covers_traded_names():
    # 2026-11-12: AMD/MU/AVGO/ARM are traded names (shadow ledger) whose
    # per-ticker news was never collected, so the gate could never block
    # them. The universe now includes them (config + fallback in sync).
    from src.paper.news_gate import FALLBACK_NEWS_TICKERS, news_universe

    universe = news_universe()
    for sym in ("AMD", "MU", "AVGO", "ARM"):
        assert sym in universe
        assert sym in FALLBACK_NEWS_TICKERS
    # And a 429 on one of them now blocks that symbol.
    parsed = parse_news_error_entries(
        ["AMD:Client error '429 Too Many Requests' for url ..."]
    )
    assert parsed["blocked_symbols"] == {"AMD"}
    assert parsed["global_block"] is False


def test_news_gate_broad_only_not_global():
    parsed = parse_news_error_entries(
        ["broad:Client error '429 Too Many Requests' for url ..."]
    )
    assert parsed["broad_failed"] is True
    assert parsed["blocked_symbols"] == set()
    assert parsed["global_block"] is False


def test_news_gate_broad_plus_all_tickers_global():
    universe = frozenset(
        ["SPY", "QQQ", "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    )
    errors = ["broad:Client error '429'"] + [
        f"{t}:Client error '429'" for t in sorted(universe)
    ]
    parsed = parse_news_error_entries(errors, universe=universe)
    assert parsed["global_block"] is True


def test_news_gate_whole_collection_failure_global():
    parsed = parse_news_error_entries(
        ["collect_intraday_news raised RuntimeError('boom')"]
    )
    assert parsed["whole_collection_failed"] is True
    assert parsed["global_block"] is True


def test_try_entry_skips_blocked_symbol(data_root):
    gate = news_gate_for_date(
        "2026-11-12",
        step3={"intraday_news_errors": [
            "AAPL:Client error '429 Too Many Requests' for url ..."
        ]},
    )
    assert gate["source"] == "step3"
    assert gate["blocked_symbols"] == ["AAPL"]
    assert news_block_reason(gate, "AAPL") is not None

    out = _try_entry(
        default_account(), "intraday",
        _picked("AAPL", price=200.0, stop=195.0, target=210.0),
        "2026-11-12", news_gate=gate,
    )
    assert out["action"] == "SKIP"
    assert out.get("news_blocked") is True
    assert "新闻数据缺失" in (out["reason"] or "")

    # An unblocked symbol still enters.
    out = _try_entry(
        default_account(), "intraday",
        _picked("MSFT", price=500.0, stop=490.0, target=520.0),
        "2026-11-12", news_gate=gate,
    )
    assert out["action"] == "ENTRY"


def test_try_entry_global_block_skips_all(data_root):
    gate = news_gate_for_date(
        "2026-11-12",
        step3={"intraday_news_errors": [
            "collect_intraday_news raised RuntimeError('boom')"
        ]},
    )
    assert gate["global_block"] is True
    out = _try_entry(
        default_account(), "intraday",
        _picked("MSFT", price=500.0, stop=490.0, target=520.0),
        "2026-11-12", news_gate=gate,
    )
    assert out["action"] == "SKIP"
    assert "全局新闻数据缺失" in (out["reason"] or "")


def test_news_gate_exits_unaffected(data_root, monkeypatch):
    """A blocked symbol's open leg still exits on its stop — the gate only
    blocks *new* entries."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-12")
    # D1: pin the entry time before the 09:31 bar — execute_entry stamps
    # opened_at with the wall-clock run time (mid-session today), which
    # would make the 09:31 touch pre-entry and correctly filtered out.
    get_position(acct, "intraday")["opened_at"] = "2026-11-12T09:30:00-05:00"
    gate = {
        "blocked_symbols": ["ARM"],
        "global_block": False,
        "broad_failed": False,
        "whole_collection_failed": False,
        "source": "step3",
        "reasons": ["ARM 新闻数据缺失（Polygon 429/失败）"],
    }
    monkeypatch.setattr(
        "src.paper.execution_agent.news_gate_for_date",
        lambda *a, **k: gate,
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(341.0)
    )
    bars = [_bar("2026-11-12 09:31", 341.0, 341.0, 334.0, 336.0)]
    decision = decide_and_act(
        acct, "2026-11-12", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    assert _exit_row(decision, "intraday") is not None
    assert get_position(acct, "intraday") is None


# ---------------------------------------------------------------------------
# Fix 2 (revised 2026-11-12): intraday legs hold overnight, max 3 trading
# days, then flatten at the close window (or first later tick if overdue).
# ---------------------------------------------------------------------------


def _tick_hold(acct, trading_date, *, close_window, quote=341.0,
               monkeypatch):
    """Run one tick with a neutral quote and a pinned close-window flag."""
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(quote)
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.session_in_close_window",
        lambda d: close_window,
    )
    return decide_and_act(
        acct, trading_date, session_phase="open", bars_by_symbol={},
    )


def test_intraday_holds_day1_and_day2(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    # Day 1 (Mon 11-09): close window reached, but only 1 day held.
    decision = _tick_hold(acct, "2026-11-09", close_window=True,
                          monkeypatch=monkeypatch)
    assert get_position(acct, "intraday") is not None
    assert _exit_row(decision, "intraday") is None
    # Day 2 (Tue 11-10): still held.
    decision = _tick_hold(acct, "2026-11-10", close_window=True,
                          monkeypatch=monkeypatch)
    assert get_position(acct, "intraday") is not None
    assert _exit_row(decision, "intraday") is None


def test_intraday_flattens_on_day3_close(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-11-09")
    # Day 3 (Wed 11-11): close window -> deterministic flatten, swing untouched.
    decision = _tick_hold(acct, "2026-11-11", close_window=True, quote=345.0,
                          monkeypatch=monkeypatch)
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert "3 个交易日" in (row["reason"] or "")
    assert row["trade"]["price"] == pytest.approx(345.0)
    # R = (345 - 341) / (341 - 335)
    assert row["trade"]["r_multiple"] == pytest.approx(4 / 6, abs=0.01)
    assert get_position(acct, "intraday") is None
    assert get_position(acct, "swing")["symbol"] == "NVDA"


def test_intraday_overdue_flattens_next_morning(data_root, monkeypatch):
    """If the day-3 close window was missed (no tick ran), the first later
    tick flattens the leg instead of letting it run to a 4th full day."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    # Thu 11-12, morning tick, close window NOT reached, 4 trading days held.
    decision = _tick_hold(acct, "2026-11-12", close_window=False,
                          monkeypatch=monkeypatch)
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert "满 4 个交易日收盘平仓" in (row["reason"] or "")
    assert get_position(acct, "intraday") is None


def test_flatten_direct_skips_when_not_due(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    out = flatten_intraday_at_close(
        acct, "2026-11-09", price=345.0, symbol="ARM",
    )
    assert out is not None
    assert out["action"] == "SKIP"
    assert "未到 3 个交易日" in (out["reason"] or "")
    assert get_position(acct, "intraday") is not None

    # The guard is explicit: min_holding_days=None flattens regardless.
    out = flatten_intraday_at_close(
        acct, "2026-11-09", price=345.0, symbol="ARM", min_holding_days=None,
    )
    assert out["action"] == "EXIT"
    assert get_position(acct, "intraday") is None


def test_swing_never_flattens_on_day3(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-11-09")
    decision = _tick_hold(acct, "2026-11-11", close_window=True, quote=345.0,
                          monkeypatch=monkeypatch)
    assert _exit_row(decision, "swing") is None
    assert get_position(acct, "swing") is not None


# ---------------------------------------------------------------------------
# Fix 2 (extended 2026-11-12): swing legs also have a max hold — 20 trading
# days, matching the shadow ledger. Flatten in the close window on day 20
# (or first later tick if overdue). A time exit sets no stop cooldown.
# ---------------------------------------------------------------------------


def test_swing_holds_through_day19(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-10-20")
    # Day 19 (Fri 11-13): close window reached, but only 19 days held.
    decision = _tick_hold(acct, "2026-11-13", close_window=True, quote=345.0,
                          monkeypatch=monkeypatch)
    assert _exit_row(decision, "swing") is None
    assert get_position(acct, "swing") is not None


def test_swing_flattens_on_day20_close(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-10-20")
    # Day 20 (Mon 11-16): close window -> deterministic flatten.
    decision = _tick_hold(acct, "2026-11-16", close_window=True, quote=345.0,
                          monkeypatch=monkeypatch)
    row = _exit_row(decision, "swing")
    assert row is not None
    assert "20 个交易日" in (row["reason"] or "")
    assert row["trade"]["price"] == pytest.approx(345.0)
    # R = (345 - 340) / (340 - 330)
    assert row["trade"]["r_multiple"] == pytest.approx(0.5, abs=0.01)
    assert get_position(acct, "swing") is None


def test_swing_overdue_flattens_next_morning(data_root, monkeypatch):
    """If the day-20 close window was missed (no tick ran), the first later
    tick flattens the leg instead of letting it run to a 21st full day."""
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-10-20")
    # Tue 11-17, morning tick, close window NOT reached, 21 trading days held.
    decision = _tick_hold(acct, "2026-11-17", close_window=False, quote=345.0,
                          monkeypatch=monkeypatch)
    row = _exit_row(decision, "swing")
    assert row is not None
    assert "满 21 个交易日收盘平仓" in (row["reason"] or "")
    assert get_position(acct, "swing") is None


def test_swing_time_exit_sets_no_cooldown(data_root, monkeypatch):
    """A max-hold flatten is neutral like a target exit: the same symbol
    may re-enter the same book the same day (no stop cooldown)."""
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=340.0, shares=2,
              stop=330.0, target=360.0, trading_date="2026-10-20")
    decision = _tick_hold(acct, "2026-11-16", close_window=True, quote=345.0,
                          monkeypatch=monkeypatch)
    assert _exit_row(decision, "swing") is not None
    assert get_position(acct, "swing") is None
    cooldown = (acct.get("risk_state") or {}).get("stop_cooldown") or {}
    assert "2026-11-16:swing:NVDA" not in cooldown
    reentry = _try_entry(
        acct,
        "swing",
        _picked("NVDA", price=345.0, stop=335.0, target=360.0,
                direction="LONG", book="swing"),
        "2026-11-16",
    )
    assert reentry["action"] == "ENTRY"
    assert "冷却" not in (reentry["reason"] or "")


# ---------------------------------------------------------------------------
# Quote outage: 3 consecutive ticks with no usable quote force an exit at
# the last good mark (2026-11-12) — a leg that cannot be priced cannot be
# managed, and its stop may already have been hit.
# ---------------------------------------------------------------------------


def test_quote_outage_forces_exit_after_three_ticks(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    for _ in range(2):
        decision = _tick_hold(acct, "2026-11-09", close_window=False,
                              quote=None, monkeypatch=monkeypatch)
        assert _exit_row(decision, "intraday") is None
        assert get_position(acct, "intraday") is not None
    decision = _tick_hold(acct, "2026-11-09", close_window=False,
                          quote=None, monkeypatch=monkeypatch)
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert "数据中断" in (row["reason"] or "")
    assert "最后有效价" in (row["reason"] or "")
    assert row["trade"]["price"] == pytest.approx(341.0)
    assert row["trade"].get("quote_outage_exit") is True
    assert get_position(acct, "intraday") is None


def test_quote_outage_resets_on_good_quote(data_root, monkeypatch):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    for _ in range(2):
        _tick_hold(acct, "2026-11-09", close_window=False,
                   quote=None, monkeypatch=monkeypatch)
    # One good tick (340, inside the plan) resets the streak.
    decision = _tick_hold(acct, "2026-11-09", close_window=False,
                          quote=340.0, monkeypatch=monkeypatch)
    assert _exit_row(decision, "intraday") is None
    # Two more bad ticks: streak is 2, still below the 3-tick threshold.
    for _ in range(2):
        decision = _tick_hold(acct, "2026-11-09", close_window=False,
                              quote=None, monkeypatch=monkeypatch)
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None


def test_quote_outage_insane_quote_counts_too(data_root, monkeypatch):
    # A rejected quote with no usable bar fallback counts toward the
    # streak the same way a missing quote does.
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    for _ in range(2):
        decision = _tick_hold(acct, "2026-11-09", close_window=False,
                              quote=100.0, monkeypatch=monkeypatch)
        assert _exit_row(decision, "intraday") is None
    # The rejected quote must never become the position's mark: last_price
    # stays at the entry (341), so the garbage quote is still rejected on
    # the next tick instead of validating itself as an anchor.
    assert get_position(acct, "intraday")["last_price"] == pytest.approx(341.0)
    decision = _tick_hold(acct, "2026-11-09", close_window=False,
                          quote=100.0, monkeypatch=monkeypatch)
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert "数据中断" in (row["reason"] or "")
    assert get_position(acct, "intraday") is None


def test_quote_outage_without_last_price_keeps_holding(data_root, monkeypatch):
    # No last good mark: there is no price to exit at, so the leg keeps
    # holding (and the streak keeps counting) rather than inventing one.
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    get_position(acct, "intraday")["last_price"] = None
    for _ in range(4):
        decision = _tick_hold(acct, "2026-11-09", close_window=False,
                              quote=None, monkeypatch=monkeypatch)
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None


def test_holding_days_counts_trading_days_only():
    # Fri 11-06 -> Mon 11-09 is 2 trading days (weekend skipped).
    assert holding_trading_days("2026-11-06", "2026-11-09") == 2
    # Wed 11-25 -> Fri 11-27 is 2 (Thanksgiving 11-26 is a holiday).
    assert holding_trading_days("2026-11-25", "2026-11-27") == 2
    assert holding_trading_days("2026-11-09", "2026-11-09") == 1
    assert holding_trading_days(None, "2026-11-09") is None
    assert holding_trading_days("2026-11-09", "2026-11-06") == 0


# ---------------------------------------------------------------------------
# Fix 7 (revised 2026-11-12): intraday <= 3 legs, swing <= 2 legs, 5 total;
# the per-book notional cap is shared by the book's legs.
# ---------------------------------------------------------------------------


def test_second_leg_same_book_opens(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    out = _try_entry(
        acct, "intraday",
        _picked("NVDA", price=500.0, stop=490.0, target=520.0),
        "2026-11-09",
    )
    assert out["action"] == "ENTRY"
    syms = [p["symbol"] for p in get_positions(acct, "intraday")]
    assert syms == ["ARM", "NVDA"]


def test_third_intraday_leg_opens(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    _open_leg(acct, "NVDA", book="intraday", price=500.0, shares=2,
              stop=490.0, target=520.0, trading_date="2026-11-09")
    out = _try_entry(
        acct, "intraday",
        _picked("TSLA", price=300.0, stop=290.0, target=320.0),
        "2026-11-09",
    )
    assert out["action"] == "ENTRY"
    assert len(get_positions(acct, "intraday")) == 3


def test_fourth_intraday_leg_rejected(data_root):
    acct = default_account()
    for sym, price in [("ARM", 341.0), ("NVDA", 500.0), ("TSLA", 300.0)]:
        _open_leg(acct, sym, book="intraday", price=price, shares=2,
                  stop=price * 0.97, target=price * 1.05,
                  trading_date="2026-11-09")
    out = _try_entry(
        acct, "intraday",
        _picked("META", price=670.0, stop=650.0, target=700.0),
        "2026-11-09",
    )
    assert out["action"] == "SKIP"
    assert "账本已满" in (out["reason"] or "")
    assert len(get_positions(acct, "intraday")) == 3
    with pytest.raises(ValueError, match="Book intraday is full"):
        _open_leg(acct, "META", book="intraday", price=670.0, shares=1,
                  stop=650.0, target=700.0, trading_date="2026-11-09")


def test_swing_second_leg_opens_third_rejected(data_root):
    acct = default_account()
    _open_leg(acct, "NVDA", book="swing", price=500.0, shares=2,
              stop=490.0, target=520.0, trading_date="2026-11-09")
    out = _try_entry(
        acct, "swing",
        _picked("META", price=670.0, stop=650.0, target=700.0, book="swing"),
        "2026-11-09",
    )
    assert out["action"] == "ENTRY"
    assert len(get_positions(acct, "swing")) == 2
    out = _try_entry(
        acct, "swing",
        _picked("AMD", price=550.0, stop=530.0, target=590.0, book="swing"),
        "2026-11-09",
    )
    assert out["action"] == "SKIP"
    assert "账本已满" in (out["reason"] or "")
    with pytest.raises(ValueError, match="Book swing is full"):
        _open_leg(acct, "AMD", book="swing", price=550.0, shares=1,
                  stop=530.0, target=590.0, trading_date="2026-11-09")


def test_same_symbol_second_leg_skipped(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    out = _try_entry(acct, "intraday", _picked("ARM"), "2026-11-09")
    assert out["action"] == "SKIP"
    assert "已在短线持仓" in (out["reason"] or "")
    assert len(get_positions(acct, "intraday")) == 1


def test_shared_book_notional_cap_reduces_entry(data_root):
    picked = _picked("NVDA", price=500.0, stop=490.0, target=520.0)
    control = _try_entry(default_account(), "intraday", picked, "2026-11-09")
    assert control["action"] == "ENTRY"
    control_cap = control["allocation"]["max_position_pct"]

    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    out = _try_entry(acct, "intraday", picked, "2026-11-09")
    assert out["action"] == "ENTRY"
    # ARM notional $1,705 = 17.05% of equity. The intraday book cap is
    # 40% x confidence factor (~37.1%), so the new leg's headroom is
    # ~20.0% — tighter than its own per-entry cap (~23.2%).
    assert out["allocation"]["book_notional_used_pct"] == pytest.approx(
        17.05, abs=0.1
    )
    assert out["allocation"]["book_cap_pct"] == pytest.approx(37.1, abs=0.5)
    assert out["allocation"]["max_position_pct"] == pytest.approx(20.0, abs=0.5)
    assert out["allocation"]["max_position_pct"] < control_cap
    assert "账本名义已用" in (out["reason"] or "")


def test_shared_book_notional_cap_full_skips(data_root):
    acct = default_account()
    # One swing leg using 72% notional — more than any second leg's cap.
    _open_leg(acct, "NVDA", book="swing", price=120.0, shares=60,
              stop=110.0, target=140.0, trading_date="2026-11-09")
    out = _try_entry(
        acct, "swing",
        _picked("META", price=670.0, stop=650.0, target=700.0, book="swing"),
        "2026-11-09",
    )
    assert out["action"] == "SKIP"
    assert "账本名义上限已满" in (out["reason"] or "")
    assert len(get_positions(acct, "swing")) == 1


def test_portfolio_cap_with_five_legs(data_root):
    """The 3% portfolio cap still brakes entries when all 5 slots are full."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")   # $30
    _open_leg(acct, "NVDA", book="intraday", price=500.0, shares=2,
              stop=490.0, target=520.0, trading_date="2026-11-09")  # $20
    _open_leg(acct, "TSLA", book="intraday", price=300.0, shares=2,
              stop=290.0, target=320.0, trading_date="2026-11-09")  # $20
    _open_leg(acct, "AMD", book="swing", price=557.495, shares=2,
              stop=493.4, target=643.27, trading_date="2026-11-09")  # $128.19
    _open_leg(acct, "META", book="swing", price=670.0, shares=1,
              stop=640.0, target=700.0, trading_date="2026-11-09")   # $30
    assert len(get_positions(acct, "intraday")) == 3
    assert len(get_positions(acct, "swing")) == 2
    summary = portfolio_planned_risk(acct)
    assert len(summary["legs"]) == 5
    assert summary["total_planned_risk"] == pytest.approx(228.19, abs=0.01)
    # A 6th leg at $20/share risk: headroom $71.81 -> 3 shares fit, not 5.
    shares, detail = apply_portfolio_risk_cap(
        acct, direction="LONG", entry_price=400.0, stop_price=380.0, shares=5,
    )
    assert detail["capped"] is True
    assert detail["allowed_shares"] == 3
    assert shares == 3


def test_execute_exit_closes_correct_leg(data_root):
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    _open_leg(acct, "NVDA", book="intraday", price=500.0, shares=2,
              stop=490.0, target=520.0, trading_date="2026-11-09")
    trade = execute_exit(
        acct, price=505.0, reason="test", trading_date="2026-11-09",
        book="intraday", symbol="NVDA",
    )
    assert trade["symbol"] == "NVDA"
    assert [p["symbol"] for p in get_positions(acct, "intraday")] == ["ARM"]
    execute_exit(
        acct, price=345.0, reason="test", trading_date="2026-11-09",
        book="intraday", symbol="ARM",
    )
    assert get_positions(acct, "intraday") == []
