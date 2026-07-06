from __future__ import annotations

from datetime import date
from typing import Any

from src.collectors.config import load_symbols
from src.collectors.quote_freshness import quote_fresh_for_checklist
from src.collectors.yfinance_client import fetch_many
from src.utils.trading_calendar import prior_trading_day, today_et


def _quote_ok(
    quotes: dict[str, Any],
    ticker: str,
    *,
    trading_date: date,
    prior_day: date,
) -> bool:
    q = quotes.get(ticker) or {}
    return quote_fresh_for_checklist(q, trading_date, prior_day)


def collect_sectors(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    prior_day = prior_trading_day(trading_date)
    cfg = load_symbols()
    tickers = list(cfg["sectors"].values())
    batch = fetch_many(tickers, trading_date=trading_date)
    quotes = batch["quotes"]
    checklist = {
        name: _quote_ok(quotes, sym, trading_date=trading_date, prior_day=prior_day)
        for name, sym in cfg["sectors"].items()
    }
    batch["checklist"] = checklist
    batch["ok"] = all(checklist.values()) and len(batch["errors"]) == 0
    return batch
