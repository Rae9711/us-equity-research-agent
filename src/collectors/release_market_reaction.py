"""Measured ETF reaction around macro release windows (QQQ / SMH)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

from pytz import timezone

from src.utils.paths import raw_data_path

logger = logging.getLogger(__name__)

ET = timezone("America/New_York")
REACTION_SYMBOLS = ("QQQ", "SMH")
POST_RELEASE_MINUTES = 5


def _parse_scheduled_et(scheduled_et: str, trading_date: date) -> datetime | None:
    try:
        hh, mm = scheduled_et.strip().split(":")
        return ET.localize(
            datetime(trading_date.year, trading_date.month, trading_date.day, int(hh), int(mm))
        )
    except (ValueError, AttributeError):
        return None


def _bar_ts_et(ts_raw: Any, trading_date: date) -> datetime | None:
    if ts_raw is None:
        return None
    if isinstance(ts_raw, datetime):
        dt = ts_raw
    else:
        text = str(ts_raw).strip()
        if not text:
            return None
        if "T" in text or "+" in text or text.endswith("Z"):
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        elif ":" in text:
            try:
                hh, mm = text.split(":")[:2]
                dt = datetime(trading_date.year, trading_date.month, trading_date.day, int(hh), int(mm))
            except (TypeError, ValueError):
                return None
        else:
            return None
    if dt.tzinfo is None:
        return ET.localize(dt)
    return dt.astimezone(ET)


def _parse_minute_bars(bars: Any, trading_date: date) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    if not isinstance(bars, list):
        return out
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        px = bar.get("close") or bar.get("price") or bar.get("last") or bar.get("c")
        try:
            price = float(px)
        except (TypeError, ValueError):
            continue
        ts_raw = bar.get("ts") or bar.get("time") or bar.get("datetime") or bar.get("t")
        if isinstance(ts_raw, (int, float)) and ts_raw > 1_000_000_000_000:
            ts = datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=ET)
        else:
            ts = _bar_ts_et(ts_raw, trading_date)
        if ts is not None:
            out.append((ts, price))
    out.sort(key=lambda x: x[0])
    return out


def _minute_bars_from_raw(raw: dict[str, Any], symbol: str, trading_date: date) -> list[tuple[datetime, float]]:
    sym = symbol.upper()
    intraday = raw.get("intraday") or {}
    top_bars = (intraday.get("bars") or {}).get(sym)
    if top_bars:
        parsed = _parse_minute_bars(top_bars, trading_date)
        if parsed:
            return parsed

    for section_key in ("market", "sector", "stocks"):
        section = raw.get(section_key) or {}
        section_bars = (section.get("minute_bars") or {}).get(sym)
        if section_bars:
            parsed = _parse_minute_bars(section_bars, trading_date)
            if parsed:
                return parsed
        quotes = section.get("quotes") or {}
        q = quotes.get(sym) or {}
        if isinstance(q, dict) and q.get("minute_bars"):
            parsed = _parse_minute_bars(q["minute_bars"], trading_date)
            if parsed:
                return parsed
    return []


def _minute_bars_yfinance(ticker: str, trading_date: date) -> list[tuple[datetime, float]]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        df = yf.Ticker(ticker).history(
            start=trading_date.isoformat(),
            end=(trading_date + timedelta(days=1)).isoformat(),
            interval="1m",
            prepost=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance 1m %s failed: %s", ticker, exc)
        return []
    if df is None or df.empty:
        return []

    out: list[tuple[datetime, float]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        else:
            ts = ts.astimezone(ET)
        out.append((ts, float(row["Close"])))
    out.sort(key=lambda x: x[0])
    return out


def _price_at_or_before(bars: list[tuple[datetime, float]], target: datetime) -> float | None:
    best: float | None = None
    for ts, px in bars:
        if ts <= target:
            best = px
        else:
            break
    return best


def _price_at_or_after(bars: list[tuple[datetime, float]], target: datetime) -> float | None:
    for ts, px in bars:
        if ts >= target:
            return px
    return None


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
        close = q.get("close")
        prev = q.get("prev_close")
        if close is not None and prev not in (None, 0):
            try:
                return round((float(close) - float(prev)) / abs(float(prev)) * 100, 2)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    return None


def compute_measured_reaction(
    trading_date: date,
    scheduled_et: str,
    symbols: tuple[str, ...] = REACTION_SYMBOLS,
    *,
    window_minutes: int = POST_RELEASE_MINUTES,
) -> dict[str, Any] | None:
    """QQQ/SMH % move from scheduled_et to scheduled_et + window; raw 1m first, then yfinance."""
    release_dt = _parse_scheduled_et(scheduled_et, trading_date)
    if release_dt is None:
        return None

    after_dt = release_dt + timedelta(minutes=window_minutes)
    raw: dict[str, Any] = {}
    raw_path = raw_data_path(trading_date.isoformat())
    if raw_path.exists():
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}

    result: dict[str, Any] = {"window_minutes": window_minutes}
    symbol_pcts: dict[str, float] = {}
    source = "intraday"

    for sym in symbols:
        bars = _minute_bars_from_raw(raw, sym, trading_date) if raw else []
        if not bars:
            bars = _minute_bars_yfinance(sym, trading_date)
            if bars:
                source = "yfinance"
        if bars:
            px_before = _price_at_or_before(bars, release_dt)
            px_after = _price_at_or_after(bars, after_dt)
            if px_before and px_after and px_before > 0:
                pct = round((px_after - px_before) / px_before * 100, 2)
                symbol_pcts[sym] = pct
                result[sym] = f"{pct:+.1f}%"
                result[f"{sym}_pct"] = pct
                continue

    if symbol_pcts:
        result["source"] = source
        return result

    if not raw:
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
    return result


def format_measured_reaction(reaction: dict[str, Any] | None) -> str:
    if not reaction:
        return "—"
    parts: list[str] = []
    for sym in REACTION_SYMBOLS:
        val = reaction.get(sym)
        if val:
            parts.append(f"{sym} {val}" if not str(val).startswith(sym) else str(val))
    if not parts:
        return "—"
    text = " / ".join(parts)
    if reaction.get("source") == "daily":
        return f"{text} (全日)"
    window = reaction.get("window_minutes", POST_RELEASE_MINUTES)
    return f"{text} ({window}min post-release)"


def measure_post_release_move(
    trading_date: str,
    scheduled_et: str,
    symbols: list[str] | None = None,
    *,
    window_minutes: int = POST_RELEASE_MINUTES,
) -> dict[str, Any]:
    """Wrapper returning display string for UI callers."""
    d = date.fromisoformat(trading_date)
    sym_tuple = tuple(symbols) if symbols else REACTION_SYMBOLS
    reaction = compute_measured_reaction(
        d, scheduled_et, sym_tuple, window_minutes=window_minutes
    )
    return {
        "display": format_measured_reaction(reaction),
        "source": (reaction or {}).get("source"),
        "moves": {
            sym: reaction[f"{sym}_pct"]
            for sym in sym_tuple
            if reaction and f"{sym}_pct" in reaction
        },
        "reaction": reaction,
    }
