"""Measured ETF reaction around macro release windows (QQQ / SMH)."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from pytz import timezone

from src.utils.paths import raw_data_path

logger = logging.getLogger(__name__)

ET = timezone("America/New_York")
REACTION_SYMBOLS = ("QQQ", "SMH")


def _parse_scheduled_et(scheduled_et: str, trading_date: date) -> datetime | None:
    try:
        hh, mm = scheduled_et.strip().split(":")
        return ET.localize(
            datetime(trading_date.year, trading_date.month, trading_date.day, int(hh), int(mm))
        )
    except (ValueError, AttributeError):
        return None


def _release_windows(release_dt: datetime) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Pre ~5 min before release; post ~5–10 min after (e.g. 8:25–8:29 vs 8:35–8:40)."""
    pre_start = release_dt - timedelta(minutes=5)
    pre_end = release_dt - timedelta(minutes=1)
    post_start = release_dt + timedelta(minutes=5)
    post_end = release_dt + timedelta(minutes=10)
    return (pre_start, pre_end), (post_start, post_end)


def _avg_close_in_window(
    bars: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> float | None:
    """Average close for Polygon-style bars with ms timestamp `t` and close `c`."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    prices: list[float] = []
    for bar in bars:
        ts = bar.get("t")
        close = bar.get("c")
        if ts is None or close is None:
            continue
        if start_ms <= int(ts) <= end_ms:
            try:
                prices.append(float(close))
            except (TypeError, ValueError):
                continue
    if not prices:
        return None
    return sum(prices) / len(prices)


def _pct_change(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None or pre == 0:
        return None
    return round((post - pre) / abs(pre) * 100, 2)


def _polygon_minute_bars(ticker: str, day_start: datetime, day_end: datetime) -> list[dict[str, Any]]:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        return []
    from_ms = int(day_start.timestamp() * 1000)
    to_ms = int(day_end.timestamp() * 1000)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{from_ms}/{to_ms}"
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params={"apiKey": api_key, "sort": "asc", "limit": 50000})
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Polygon minute %s failed: %s", ticker, exc)
        return []
    return list(data.get("results") or [])


def _yfinance_minute_bars(ticker: str, trading_date: date) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    start = ET.localize(datetime(trading_date.year, trading_date.month, trading_date.day, 4, 0))
    end = start + timedelta(days=1)
    try:
        df = yf.Ticker(ticker).history(
            start=start.astimezone(timezone("UTC")).replace(tzinfo=None),
            end=end.astimezone(timezone("UTC")).replace(tzinfo=None),
            interval="1m",
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance 1m %s failed: %s", ticker, exc)
        return []
    if df is None or df.empty:
        return []

    bars: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        else:
            ts = ts.astimezone(ET)
        bars.append({"t": int(ts.timestamp() * 1000), "c": float(row["Close"])})
    return bars


def _quote_change_pct(raw: dict[str, Any], symbol: str) -> float | None:
    for section_key in ("market", "sector", "stocks"):
        quotes = (raw.get(section_key) or {}).get("quotes") or {}
        q = quotes.get(symbol) or {}
        chg = q.get("change_pct")
        if chg is not None:
            try:
                return round(float(chg), 2)
            except (TypeError, ValueError):
                pass
    return None


def _reaction_from_minute_bars(
    bars: list[dict[str, Any]],
    pre_window: tuple[datetime, datetime],
    post_window: tuple[datetime, datetime],
) -> float | None:
    pre_avg = _avg_close_in_window(bars, *pre_window)
    post_avg = _avg_close_in_window(bars, *post_window)
    return _pct_change(pre_avg, post_avg)


def compute_measured_reaction(
    trading_date: date,
    scheduled_et: str,
    symbols: tuple[str, ...] = REACTION_SYMBOLS,
) -> dict[str, Any] | None:
    """QQQ/SMH % move pre- vs post-release window; degrades to daily Step 0 quotes."""
    release_dt = _parse_scheduled_et(scheduled_et, trading_date)
    if release_dt is None:
        return None

    pre_window, post_window = _release_windows(release_dt)
    window_label = f"{scheduled_et} release"
    day_lo = ET.localize(datetime(trading_date.year, trading_date.month, trading_date.day, 4, 0))
    day_hi = day_lo + timedelta(hours=20)

    result: dict[str, Any] = {"window": window_label}
    symbol_pcts: dict[str, float] = {}
    source = "polygon"

    for sym in symbols:
        bars = _polygon_minute_bars(sym, day_lo, day_hi)
        pct = _reaction_from_minute_bars(bars, pre_window, post_window) if bars else None
        if pct is None:
            bars = _yfinance_minute_bars(sym, trading_date)
            pct = _reaction_from_minute_bars(bars, pre_window, post_window) if bars else None
            if pct is not None:
                source = "yfinance"
        if pct is not None:
            symbol_pcts[sym] = pct
            result[sym] = f"{pct:+.1f}%"

    if symbol_pcts:
        result["source"] = source
        for sym, pct in symbol_pcts.items():
            result[f"{sym}_pct"] = pct
        return result

    # Last resort: Step 0 daily change (not intraday — label clearly)
    raw_path = raw_data_path(trading_date.isoformat())
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None

    daily_any = False
    for sym in symbols:
        chg = _quote_change_pct(raw, sym)
        if chg is not None:
            daily_any = True
            symbol_pcts[sym] = chg
            result[sym] = f"{chg:+.1f}%"
            result[f"{sym}_pct"] = chg

    if not daily_any:
        return None

    result["source"] = "daily"
    result["window"] = f"{window_label} (日涨跌 · Step 0)"
    return result
