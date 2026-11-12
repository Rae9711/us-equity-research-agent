from __future__ import annotations

import os
from typing import Any

import httpx


class PolygonClient:
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {"status": "ERROR", "error": "POLYGON_API_KEY not set"}
        q = dict(params or {})
        q["apiKey"] = self.api_key
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{self.BASE}{path}", params=q)
            r.raise_for_status()
            return r.json()

    def prev_day_agg(self, ticker: str) -> dict[str, Any]:
        return self.get(f"/v2/aggs/ticker/{ticker}/prev")

    def last_trade(self, ticker: str) -> dict[str, Any]:
        """Last trade (``results.p``). ~15-min delayed on the free tier —
        a live-quote fallback when yfinance is down, fresher than the
        morning snapshot."""
        return self.get(f"/v2/last/trade/{ticker}")

    def minute_aggs(self, ticker: str, day: str) -> dict[str, Any]:
        """1-minute aggregates for one day (delayed on the free tier)."""
        return self.get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}",
            {"adjusted": "true", "sort": "asc", "limit": "5000"},
        )

    def news(
        self,
        ticker: str | None = None,
        limit: int = 5,
        *,
        published_gte: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "order": "desc"}
        if ticker:
            params["ticker"] = ticker
        if published_gte:
            params["published_utc.gte"] = published_gte
        return self.get("/v2/reference/news", params)

    def options_snapshot(self, underlying: str) -> dict[str, Any]:
        return self.get(f"/v3/snapshot/options/{underlying}")
