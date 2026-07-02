from __future__ import annotations

from typing import Any

from src.collectors.config import load_symbols
from src.collectors.fred_client import FredClient
from src.collectors.yfinance_client import fetch_many


def _quote_ok(quotes: dict[str, Any], ticker: str) -> bool:
    q = quotes.get(ticker) or {}
    return "error" not in q and q.get("close") is not None


def collect_market() -> dict[str, Any]:
    cfg = load_symbols()
    yf_tickers = [t for t in cfg["market"].values() if not str(t).startswith("DGS")]
    batch = fetch_many(yf_tickers)

    fred = FredClient()
    ten_y = fred.latest_observation(cfg["fred"]["DGS10"])
    batch["treasury_10y_fred"] = ten_y

    quotes = batch["quotes"]
    checklist = {
        "SPY": _quote_ok(quotes, cfg["market"]["SPY"]),
        "QQQ": _quote_ok(quotes, cfg["market"]["QQQ"]),
        "TQQQ": _quote_ok(quotes, cfg["market"]["TQQQ"]),
        "VIX": _quote_ok(quotes, cfg["market"]["VIX"]),
        "DXY": _quote_ok(quotes, cfg["market"]["DXY"]),
        "10Y": ten_y is not None and ten_y.get("value") not in (None, "."),
    }
    batch["checklist"] = checklist
    batch["ok"] = all(checklist.values()) and len(batch["errors"]) == 0
    return batch
