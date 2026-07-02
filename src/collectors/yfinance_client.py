from __future__ import annotations

from typing import Any

import yfinance as yf


def fetch_quote(ticker: str) -> dict[str, Any]:
    """Daily OHLCV + change for a ticker via yfinance."""
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", auto_adjust=True)
    if hist.empty:
        return {"ticker": ticker, "error": "no data"}

    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else None
    close = float(last["Close"])
    out: dict[str, Any] = {
        "ticker": ticker,
        "date": str(hist.index[-1].date()),
        "open": round(float(last["Open"]), 4),
        "high": round(float(last["High"]), 4),
        "low": round(float(last["Low"]), 4),
        "close": round(close, 4),
        "volume": int(last["Volume"]),
    }
    if prev is not None:
        prev_close = float(prev["Close"])
        out["prev_close"] = round(prev_close, 4)
        out["change_pct"] = round((close - prev_close) / prev_close * 100, 2)
    return out


def fetch_many(tickers: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[str] = []
    for ticker in tickers:
        try:
            results[ticker] = fetch_quote(ticker)
            if "error" in results[ticker]:
                errors.append(ticker)
        except Exception as exc:  # noqa: BLE001
            results[ticker] = {"ticker": ticker, "error": str(exc)}
            errors.append(ticker)
    return {"quotes": results, "errors": errors}
