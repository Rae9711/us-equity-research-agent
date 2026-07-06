from __future__ import annotations

from datetime import date
from typing import Any

from src.collectors.config import load_symbols
from src.collectors.fred_client import FredClient
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


def collect_market(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    prior_day = prior_trading_day(trading_date)
    cfg = load_symbols()
    yf_tickers = [t for t in cfg["market"].values() if not str(t).startswith("DGS")]
    batch = fetch_many(yf_tickers, trading_date=trading_date)

    fred = FredClient()
    ten_y = fred.latest_observation(cfg["fred"]["DGS10"])
    batch["treasury_10y_fred"] = ten_y

    quotes = batch["quotes"]
    tnx_key = cfg["market"].get("TEN_Y", "^TNX")
    tnx_fresh = _quote_ok(quotes, tnx_key, trading_date=trading_date, prior_day=prior_day)
    ten_y_ok = ten_y is not None and ten_y.get("value") not in (None, ".")
    checklist = {
        "SPY": _quote_ok(quotes, cfg["market"]["SPY"], trading_date=trading_date, prior_day=prior_day),
        "QQQ": _quote_ok(quotes, cfg["market"]["QQQ"], trading_date=trading_date, prior_day=prior_day),
        "DIA": _quote_ok(quotes, cfg["market"]["DIA"], trading_date=trading_date, prior_day=prior_day),
        "ES_futures": _quote_ok(quotes, cfg["market"]["ES"], trading_date=trading_date, prior_day=prior_day),
        "TQQQ": _quote_ok(quotes, cfg["market"]["TQQQ"], trading_date=trading_date, prior_day=prior_day),
        "VIX": _quote_ok(quotes, cfg["market"]["VIX"], trading_date=trading_date, prior_day=prior_day),
        "DXY": _quote_ok(quotes, cfg["market"]["DXY"], trading_date=trading_date, prior_day=prior_day),
        "10Y": ten_y_ok or tnx_fresh,
    }
    batch["checklist"] = checklist
    batch["ok"] = all(checklist.values()) and len(batch["errors"]) == 0
    return batch
