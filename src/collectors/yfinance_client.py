from __future__ import annotations

from datetime import date, datetime
from typing import Any

import yfinance as yf

from src.utils.trading_calendar import ET, market_open_et, prior_trading_day, today_et


def _bar_on_day(hist, day: date):
    for idx in reversed(hist.index):
        if idx.date() == day:
            return hist.loc[idx]
    return None


def _premarket_price(t: yf.Ticker, trading_date: date) -> float | None:
    try:
        fi = t.fast_info
        for key in ("pre_market_price", "last_price", "lastPrice", "regular_market_price"):
            val = fi.get(key)
            if val is not None:
                return float(val)
    except Exception:  # noqa: BLE001
        pass

    try:
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist.empty:
            return None
        for idx in reversed(hist.index):
            if idx.date() == trading_date:
                return float(hist.loc[idx, "Close"])
    except Exception:  # noqa: BLE001
        pass
    return None


def fetch_quote(ticker: str, *, trading_date: date | None = None) -> dict[str, Any]:
    """Daily / pre-market quote with explicit session dating."""
    trading_date = trading_date or today_et()
    prior_day = prior_trading_day(trading_date)
    data_as_of = datetime.now(ET).isoformat()
    now_et = datetime.now(ET)

    t = yf.Ticker(ticker)
    hist = t.history(period="10d", auto_adjust=True)
    if hist.empty:
        return {"ticker": ticker, "error": "no data", "data_as_of": data_as_of}

    prior_bar = _bar_on_day(hist, prior_day)
    today_bar = _bar_on_day(hist, trading_date)
    market_open = market_open_et(trading_date)
    is_premarket = now_et < market_open

    if prior_bar is None and today_bar is None:
        last_idx = hist.index[-1]
        fallback = hist.loc[last_idx]
        bar_date = last_idx.date()
        close = float(fallback["Close"])
        prev_bar = hist.iloc[-2] if len(hist) > 1 else None
        out: dict[str, Any] = {
            "ticker": ticker,
            "date": bar_date.isoformat(),
            "quote_session_date": bar_date.isoformat(),
            "session_type": "fallback",
            "open": round(float(fallback["Open"]), 4),
            "high": round(float(fallback["High"]), 4),
            "low": round(float(fallback["Low"]), 4),
            "close": round(close, 4),
            "volume": int(fallback["Volume"]),
            "data_as_of": data_as_of,
        }
        if prev_bar is not None:
            prev_close = float(prev_bar["Close"])
            out["prev_close"] = round(prev_close, 4)
            out["change_pct"] = round((close - prev_close) / prev_close * 100, 2)
        return out

    prior_close = float(prior_bar["Close"]) if prior_bar is not None else None

    if today_bar is not None and not is_premarket:
        last = today_bar
        quote_session_date = trading_date
        session_type = "regular"
        close = float(last["Close"])
        open_px = float(last["Open"])
        high = float(last["High"])
        low = float(last["Low"])
        volume = int(last["Volume"])
    elif is_premarket or today_bar is None:
        pre = _premarket_price(t, trading_date)
        if pre is not None and prior_close is not None:
            quote_session_date = prior_day
            session_type = "premarket"
            close = pre
            open_px = pre
            high = pre
            low = pre
            volume = 0
        elif prior_bar is not None:
            last = prior_bar
            quote_session_date = prior_day
            session_type = "prior_close"
            close = float(last["Close"])
            open_px = float(last["Open"])
            high = float(last["High"])
            low = float(last["Low"])
            volume = int(last["Volume"])
        elif today_bar is not None:
            last = today_bar
            quote_session_date = trading_date
            session_type = "regular"
            close = float(last["Close"])
            open_px = float(last["Open"])
            high = float(last["High"])
            low = float(last["Low"])
            volume = int(last["Volume"])
            prior_close = None
        else:
            return {"ticker": ticker, "error": "no session bar", "data_as_of": data_as_of}
    else:
        last = today_bar
        quote_session_date = trading_date
        session_type = "regular"
        close = float(last["Close"])
        open_px = float(last["Open"])
        high = float(last["High"])
        low = float(last["Low"])
        volume = int(last["Volume"])

    out = {
        "ticker": ticker,
        "date": quote_session_date.isoformat(),
        "quote_session_date": quote_session_date.isoformat(),
        "session_type": session_type,
        "open": round(open_px, 4),
        "high": round(high, 4),
        "low": round(low, 4),
        "close": round(close, 4),
        "volume": volume,
        "data_as_of": data_as_of,
    }
    if prior_close is not None:
        out["prior_close"] = round(prior_close, 4)
        out["change_pct"] = round((close - prior_close) / prior_close * 100, 2)
    return out


def fetch_many(tickers: list[str], *, trading_date: date | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[str] = []
    for ticker in tickers:
        try:
            results[ticker] = fetch_quote(ticker, trading_date=trading_date)
            if "error" in results[ticker]:
                errors.append(ticker)
        except Exception as exc:  # noqa: BLE001
            results[ticker] = {"ticker": ticker, "error": str(exc)}
            errors.append(ticker)
    return {"quotes": results, "errors": errors}
