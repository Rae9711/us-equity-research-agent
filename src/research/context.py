from __future__ import annotations

from typing import Any


def _quote(raw: dict[str, Any], ticker: str) -> dict[str, Any]:
    return (raw.get("market") or {}).get("quotes", {}).get(ticker) or {}


def _sector_quotes(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return (raw.get("sector") or {}).get("quotes") or {}


def _stock_quotes(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return (raw.get("stocks") or {}).get("quotes") or {}


def build_research_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Compact context for the LLM — avoids sending the full raw JSON."""
    news_polygon = (raw.get("news") or {}).get("polygon") or []
    headlines = [
        {
            "publisher": n.get("publisher"),
            "title": n.get("title"),
            "ticker": n.get("ticker"),
            "published_utc": n.get("published_utc"),
            "sentiment": n.get("sentiment"),
        }
        for n in news_polygon[:20]
    ]

    macro_series = (raw.get("macro") or {}).get("series") or {}
    calendar = (raw.get("macro") or {}).get("economic_calendar") or []

    sectors = []
    for name, q in _sector_quotes(raw).items():
        if "error" not in q:
            sectors.append(
                {
                    "sector": name,
                    "ticker": q.get("ticker"),
                    "close": q.get("close"),
                    "change_pct": q.get("change_pct"),
                }
            )
    sectors.sort(key=lambda x: x.get("change_pct") or 0, reverse=True)

    mag7 = []
    for sym, q in _stock_quotes(raw).items():
        if "error" not in q:
            mag7.append(
                {
                    "symbol": sym,
                    "close": q.get("close"),
                    "change_pct": q.get("change_pct"),
                }
            )
    mag7.sort(key=lambda x: x.get("change_pct") or 0, reverse=True)

    options = raw.get("options") or {}

    return {
        "trading_date": raw.get("trading_date"),
        "prior_trading_day": raw.get("prior_trading_day"),
        "data_ready": raw.get("data_ready"),
        "market": {
            "SPY": _quote(raw, "SPY"),
            "QQQ": _quote(raw, "QQQ"),
            "TQQQ": _quote(raw, "TQQQ"),
            "VIX": _quote(raw, "^VIX"),
            "DXY": _quote(raw, "DX-Y.NYB"),
            "treasury_10y": (raw.get("market") or {}).get("treasury_10y_fred"),
        },
        "macro": {
            "series": macro_series,
            "calendar": calendar[:12],
            "nonfarm_days_until": (raw.get("macro") or {}).get("nonfarm_days_until"),
        },
        "sectors": sectors,
        "mag7": mag7,
        "headlines": headlines,
        "options": {
            "source": options.get("source"),
            "put_call_ratio": options.get("put_call_ratio"),
            "avg_iv": options.get("avg_implied_volatility"),
            "contracts_count": options.get("contracts_count"),
        },
    }
