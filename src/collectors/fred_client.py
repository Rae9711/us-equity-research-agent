from __future__ import annotations

import os
from typing import Any

import httpx


class FredClient:
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {"error": "FRED_API_KEY not set"}
        q = dict(params or {})
        q["api_key"] = self.api_key
        q.setdefault("file_type", "json")
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{self.BASE}{path}", params=q)
            r.raise_for_status()
            return r.json()

    def latest_observation(self, series_id: str) -> dict[str, Any] | None:
        data = self.get(
            "/series/observations",
            {
                "series_id": series_id,
                "limit": 1,
                "sort_order": "desc",
            },
        )
        obs = data.get("observations") or []
        return obs[0] if obs else None

    def release_dates(self, limit: int = 15) -> dict[str, Any]:
        return self.get("/releases/dates", {"limit": limit, "sort_order": "desc"})
