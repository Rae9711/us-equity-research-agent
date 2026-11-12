"""Tests for the Polygon live-quote / minute-bar fallback (2026-11-12).

Live quotes and 1-minute bars used to depend on yfinance alone; when it
failed, the tick fell back to the (hours-old) morning snapshot or to no
bars at all. Polygon's last trade and minute aggregates now sit in the
chain as a second source, failing soft to the existing fallbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.paper.signals import resolve_minute_bars, resolve_quote


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _raise_fetch_quote(*a, **k):
    raise RuntimeError("yfinance down")


def _empty_fetch_quote(*a, **k):
    return {}


class _RaisingTicker:
    def __init__(self, *a, **k):
        pass

    def history(self, *a, **k):
        raise RuntimeError("yfinance down")


def _ms(dt_str: str) -> int:
    dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def test_resolve_quote_falls_back_to_polygon_last_trade(data_root, monkeypatch):
    monkeypatch.setattr(
        "src.collectors.yfinance_client.fetch_quote", _raise_fetch_quote
    )
    monkeypatch.setattr(
        "src.collectors.polygon_client.PolygonClient.last_trade",
        lambda self, ticker: {"status": "OK", "results": {"p": 345.0}},
    )
    px, source = resolve_quote("AMD", "2026-11-12", prefer_live=True)
    assert px == pytest.approx(345.0)
    assert source == "polygon_last_trade"


def test_resolve_quote_polygon_failure_falls_through(data_root, monkeypatch):
    # Polygon 429 + yfinance down + no snapshot -> (None, "none"), no crash.
    monkeypatch.setattr(
        "src.collectors.yfinance_client.fetch_quote", _raise_fetch_quote
    )
    monkeypatch.setattr(
        "src.collectors.polygon_client.PolygonClient.last_trade",
        lambda self, ticker: (_ for _ in ()).throw(RuntimeError("429")),
    )
    px, source = resolve_quote("AMD", "2026-11-12", prefer_live=True)
    assert px is None
    assert source == "none"


def test_resolve_quote_polygon_error_payload_falls_through(data_root, monkeypatch):
    # No API key -> PolygonClient.get returns an ERROR payload, not a raise.
    monkeypatch.setattr(
        "src.collectors.yfinance_client.fetch_quote", _empty_fetch_quote
    )
    monkeypatch.setattr(
        "src.collectors.polygon_client.PolygonClient.last_trade",
        lambda self, ticker: {"status": "ERROR", "error": "POLYGON_API_KEY not set"},
    )
    px, source = resolve_quote("AMD", "2026-11-12", prefer_live=True)
    assert px is None
    assert source == "none"


def test_resolve_minute_bars_falls_back_to_polygon(data_root, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _RaisingTicker)
    # 09:30 bar (in session) + 09:00 bar (premarket, must be filtered out).
    monkeypatch.setattr(
        "src.collectors.polygon_client.PolygonClient.minute_aggs",
        lambda self, ticker, day: {
            "status": "OK",
            "results": [
                {"t": _ms("2026-11-12T14:00:00"), "o": 340.0, "h": 341.0,
                 "l": 339.0, "c": 340.5},
                {"t": _ms("2026-11-12T14:30:00"), "o": 340.5, "h": 342.0,
                 "l": 340.0, "c": 341.0},
            ],
        },
    )
    bars = resolve_minute_bars("ARM", "2026-11-12")
    assert len(bars) == 1
    assert bars[0]["open"] == pytest.approx(340.5)
    assert bars[0]["close"] == pytest.approx(341.0)
    assert bars[0]["ts"].startswith("2026-11-12T09:30:00")


def test_resolve_minute_bars_empty_when_both_sources_fail(data_root, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _RaisingTicker)
    monkeypatch.setattr(
        "src.collectors.polygon_client.PolygonClient.minute_aggs",
        lambda self, ticker, day: (_ for _ in ()).throw(RuntimeError("429")),
    )
    assert resolve_minute_bars("ARM", "2026-11-12") == []
