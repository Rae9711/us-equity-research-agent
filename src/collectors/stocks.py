from __future__ import annotations

from typing import Any

from src.collectors.config import load_symbols
from src.collectors.yfinance_client import fetch_many


def _quote_ok(quotes: dict[str, Any], ticker: str) -> bool:
    q = quotes.get(ticker) or {}
    return "error" not in q and q.get("close") is not None


def collect_stocks() -> dict[str, Any]:
    cfg = load_symbols()
    batch = fetch_many(cfg["stocks"])
    quotes = batch["quotes"]
    checklist = {sym: _quote_ok(quotes, sym) for sym in cfg["stocks"]}
    batch["checklist"] = checklist
    batch["ok"] = all(checklist.values()) and len(batch["errors"]) == 0
    return batch
