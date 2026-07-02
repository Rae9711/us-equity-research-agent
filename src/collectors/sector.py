from __future__ import annotations

from typing import Any

from src.collectors.config import load_symbols
from src.collectors.yfinance_client import fetch_many


def _quote_ok(quotes: dict[str, Any], ticker: str) -> bool:
    q = quotes.get(ticker) or {}
    return "error" not in q and q.get("close") is not None


def collect_sectors() -> dict[str, Any]:
    cfg = load_symbols()
    tickers = list(cfg["sectors"].values())
    batch = fetch_many(tickers)
    quotes = batch["quotes"]
    checklist = {name: _quote_ok(quotes, sym) for name, sym in cfg["sectors"].items()}
    batch["checklist"] = checklist
    batch["ok"] = all(checklist.values()) and len(batch["errors"]) == 0
    return batch
