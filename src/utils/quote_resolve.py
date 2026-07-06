"""Resolve session quotes across holidays and stale Step 0 snapshots."""

from __future__ import annotations

from datetime import date
from typing import Any

import yfinance as yf

from src.utils.trading_calendar import prior_trading_day


def _safe_float(d: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    obj: Any = d
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _pct_chg(current: float | None, prev: float | None) -> float | None:
    if current is None or prev is None or prev == 0:
        return None
    return (current - prev) / abs(prev) * 100.0


def _quotes(section: dict[str, Any]) -> dict[str, Any]:
    return section.get("quotes") or section.get("prices") or {}


def _quote(section: dict[str, Any], ticker: str) -> dict[str, Any]:
    return _quotes(section).get(ticker) or {}


def quote_bar_date(q: dict[str, Any]) -> date | None:
    raw = q.get("date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def prior_close_from_raw(
    prior_raw: dict[str, Any],
    *,
    section: str,
    ticker: str,
    prior_section: str | None = None,
) -> float | None:
    if not prior_raw:
        return None
    prior_sec = prior_raw.get(prior_section or section, {})
    prior_q = _quote(prior_sec, ticker)
    return _safe_float(prior_q, "close") or _safe_float(prior_q, "current_price")


def prior_close_from_yfinance(ticker: str, prior_day: date) -> float | None:
    hist = yf.Ticker(ticker).history(period="10d")
    if hist.empty:
        return None
    for idx in reversed(hist.index):
        bar_day = idx.date()
        if bar_day <= prior_day:
            return float(hist.loc[idx, "Close"])
    return None


def _gap_label(gap_pct: float | None) -> str:
    if gap_pct is None:
        return "N/A"
    if gap_pct > 0.05:
        return "Up"
    if gap_pct < -0.05:
        return "Down"
    return "Flat"


def observation_from_quote(
    q: dict[str, Any],
    *,
    prev_close: float | None = None,
) -> dict[str, Any]:
    if "error" in q:
        return q
    prev = prev_close if prev_close is not None else _safe_float(q, "prev_close")
    open_px = _safe_float(q, "open")
    last_px = _safe_float(q, "close") or _safe_float(q, "current_price") or _safe_float(q, "last")
    gap_pct = _pct_chg(open_px, prev)
    change_pct = _pct_chg(last_px, prev)
    return {
        "ticker": q.get("ticker"),
        "prev_close": round(prev, 4) if prev is not None else None,
        "open": round(open_px, 4) if open_px is not None else None,
        "last": round(last_px, 4) if last_px is not None else None,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "gap": _gap_label(gap_pct),
    }


def intraday_session_quote(
    ticker: str,
    trading_date: date,
    prior_close: float | None = None,
) -> dict[str, Any]:
    """Today's open/last vs prior trading session close (yfinance 1m bars)."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1d", interval="1m", prepost=True)
    if hist.empty:
        return {"ticker": ticker, "error": "no intraday data"}

    prev = prior_close
    if prev is None:
        raw_prev = t.fast_info.get("previous_close") or t.fast_info.get(
            "regular_market_previous_close"
        )
        if raw_prev is not None:
            prev = float(raw_prev)
    if prev is None:
        prev = prior_close_from_yfinance(ticker, prior_trading_day(trading_date))
    if prev is None:
        return {"ticker": ticker, "error": "no prior close"}

    prev_f = float(prev)
    open_px = float(hist.iloc[0]["Open"])
    last_px = float(hist.iloc[-1]["Close"])
    gap_pct = round((open_px - prev_f) / prev_f * 100, 2)
    chg_pct = round((last_px - prev_f) / prev_f * 100, 2)
    return {
        "ticker": ticker,
        "prev_close": round(prev_f, 4),
        "open": round(open_px, 4),
        "last": round(last_px, 4),
        "gap_pct": gap_pct,
        "change_pct": chg_pct,
        "gap": _gap_label(gap_pct),
        "source": "intraday",
    }


def session_observation(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_date: date,
    *,
    section: str = "market",
    prior_section: str | None = None,
) -> dict[str, Any]:
    """Best-effort quote for trading_date: intraday first, then same-day raw."""
    prior_day = prior_trading_day(trading_date)
    prior_close = prior_close_from_raw(
        prior_raw,
        section=section,
        ticker=ticker,
        prior_section=prior_section,
    )
    if prior_close is None:
        prior_close = prior_close_from_yfinance(ticker, prior_day)

    intraday = intraday_session_quote(ticker, trading_date, prior_close)
    if "error" not in intraday:
        return intraday

    sec = raw.get(section, {})
    q = _quote(sec, ticker)
    if "error" in q or q.get("close") is None:
        return intraday

    bar_day = quote_bar_date(q)
    if bar_day is not None and bar_day < trading_date:
        return {
            "ticker": ticker,
            "error": "stale raw quote",
            "prev_close": round(prior_close, 4) if prior_close is not None else None,
            "quote_date": bar_day.isoformat(),
        }

    pc = prior_close if prior_close is not None else _safe_float(q, "prev_close")
    obs = observation_from_quote(q, prev_close=pc)
    obs["source"] = "raw"
    return obs


def session_change_pct(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_date: date,
    *,
    section: str = "market",
    prior_section: str | None = None,
) -> float | None:
    obs = session_observation(
        ticker,
        raw,
        prior_raw,
        trading_date,
        section=section,
        prior_section=prior_section,
    )
    if "error" in obs:
        return None
    return obs.get("change_pct")
