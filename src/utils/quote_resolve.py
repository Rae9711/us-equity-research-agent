"""Resolve session quotes across holidays and stale Step 0 snapshots.

Point-in-time (PIT): when *as_of_et* is a scheduled time (not ``now``), quotes
are resolved from frozen snapshots / raw only — never live intraday.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Any, Literal, Union

import yfinance as yf

from src.utils.trading_calendar import ET, prior_trading_day

logger = logging.getLogger(__name__)

AsOf = Union[time, Literal["now"], None]


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


def _as_of_cutoff(trading_date: date, as_of_et: time) -> datetime:
    return ET.localize(datetime.combine(trading_date, as_of_et))


def _filter_yfinance_bars(hist, trading_date: date, as_of_et: time):
    """Return hist rows up to as_of_et (inclusive)."""
    if hist.empty:
        return hist
    cutoff = _as_of_cutoff(trading_date, as_of_et)
    mask = []
    for idx in hist.index:
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        else:
            ts = ts.astimezone(ET)
        mask.append(ts <= cutoff)
    return hist[mask]


def intraday_session_quote(
    ticker: str,
    trading_date: date,
    prior_close: float | None = None,
    *,
    as_of_et: AsOf = None,
) -> dict[str, Any]:
    """Today's open/last vs prior close (yfinance 1m bars).

    When *as_of_et* is a time, only bars up to that moment are used.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="1d", interval="1m", prepost=True)
    if as_of_et is not None and as_of_et != "now":
        hist = _filter_yfinance_bars(hist, trading_date, as_of_et)
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
    obs: dict[str, Any] = {
        "ticker": ticker,
        "prev_close": round(prev_f, 4),
        "open": round(open_px, 4),
        "last": round(last_px, 4),
        "gap_pct": gap_pct,
        "change_pct": chg_pct,
        "gap": _gap_label(gap_pct),
        "source": "intraday",
    }
    if as_of_et is not None and as_of_et != "now":
        obs["price_as_of"] = _as_of_cutoff(trading_date, as_of_et).isoformat()
    return obs


def _observation_from_raw(
    ticker: str,
    raw: dict[str, Any],
    trading_date: date,
    *,
    section: str,
    prior_close: float | None,
) -> dict[str, Any] | None:
    """Same-day raw quote observation, or None if missing/stale."""
    sec = raw.get(section, {})
    q = _quote(sec, ticker)
    if "error" in q or q.get("close") is None:
        return None

    bar_day = quote_bar_date(q)
    if bar_day is not None and bar_day < trading_date:
        return None

    pc = prior_close if prior_close is not None else _safe_float(q, "prev_close")
    obs = observation_from_quote(q, prev_close=pc)
    obs["source"] = "raw"
    price_as_of = q.get("data_as_of") or raw.get("collected_at") or raw.get("data_as_of")
    if price_as_of:
        obs["price_as_of"] = price_as_of
    return obs


def _is_pit(as_of_et: AsOf, prefer_raw: bool) -> bool:
    if as_of_et == "now":
        return False
    if as_of_et is not None:
        return True
    return prefer_raw


def get_quote(
    symbol: str,
    trading_date: date,
    as_of_et: AsOf = None,
    *,
    raw: dict[str, Any] | None = None,
    prior_raw: dict[str, Any] | None = None,
    section: str = "market",
    prior_section: str | None = None,
) -> dict[str, Any]:
    """Resolve a quote as-of a decision boundary.

    Never uses live intraday unless *as_of_et* is ``now``.
    """
    if as_of_et == "now":
        logger.warning(
            "get_quote(%s, %s, as_of=now) — using LIVE intraday (debug only)",
            symbol,
            trading_date,
        )
    prefer_raw = _is_pit(as_of_et, prefer_raw=False)
    return session_observation(
        symbol,
        raw or {},
        prior_raw or {},
        trading_date,
        section=section,
        prior_section=prior_section,
        prefer_raw=prefer_raw,
        as_of_et=as_of_et,
    )


def session_observation(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_date: date,
    *,
    section: str = "market",
    prior_section: str | None = None,
    prefer_raw: bool = False,
    as_of_et: AsOf = None,
) -> dict[str, Any]:
    """Best-effort quote for trading_date at a decision boundary."""
    pit = _is_pit(as_of_et, prefer_raw)
    prior_day = prior_trading_day(trading_date)
    prior_close = prior_close_from_raw(
        prior_raw,
        section=section,
        ticker=ticker,
        prior_section=prior_section,
    )
    if prior_close is None and not pit:
        prior_close = prior_close_from_yfinance(ticker, prior_day)

    raw_obs = _observation_from_raw(
        ticker, raw, trading_date, section=section, prior_close=prior_close
    )

    if pit:
        if raw_obs is not None:
            if as_of_et is not None and as_of_et != "now":
                raw_obs["price_as_of"] = _as_of_cutoff(trading_date, as_of_et).isoformat()
            return raw_obs
        if as_of_et is not None and as_of_et != "now":
            intraday = intraday_session_quote(
                ticker, trading_date, prior_close, as_of_et=as_of_et
            )
            if "error" not in intraday:
                return intraday
        return {
            "ticker": ticker,
            "error": "no PIT quote available",
            "prev_close": round(prior_close, 4) if prior_close is not None else None,
        }

    intraday = intraday_session_quote(ticker, trading_date, prior_close)
    if "error" not in intraday:
        intraday["source"] = "intraday"
        return intraday

    if raw_obs is not None:
        return raw_obs

    bar_day = quote_bar_date(_quote(raw.get(section, {}), ticker))
    return {
        "ticker": ticker,
        "error": "stale raw quote",
        "prev_close": round(prior_close, 4) if prior_close is not None else None,
        "quote_date": bar_day.isoformat() if bar_day else None,
    }


def session_change_pct(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_date: date,
    *,
    section: str = "market",
    prior_section: str | None = None,
    as_of_et: AsOf = None,
) -> float | None:
    obs = session_observation(
        ticker,
        raw,
        prior_raw,
        trading_date,
        section=section,
        prior_section=prior_section,
        as_of_et=as_of_et,
    )
    if "error" in obs:
        return None
    return obs.get("change_pct")
