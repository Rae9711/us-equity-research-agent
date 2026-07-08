"""Auditable price levels with source tags (VWAP, ORB, prior day, expected close).

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

import yfinance as yf

from src.utils.trading_calendar import ET, market_open_et, prior_trading_day

ORB_MINUTES = 30
ENTRY_ZONE_TOLERANCE_PCT = 0.3

SOURCE_LABELS: dict[str, str] = {
    "vwap": "VWAP",
    "orb_low": "ORB低点",
    "orb_high": "ORB高点",
    "prev_low": "昨日低点",
    "prev_high": "昨日高点",
    "prior_close": "昨收",
    "expected_close": "预期收盘",
    "expected_high": "预期高点",
    "expected_low": "预期低点",
    "current": "现价",
}


def source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def format_tagged(price: float, source: str, *, prefix: str = "") -> str:
    """Human-readable level with source tag, e.g. '255.0 (VWAP)' or 'Above 711 (昨日低点)'."""
    label = source_label(source)
    px = f"{price:.1f}"
    if prefix:
        return f"{prefix} {px} ({label})"
    return f"{px} ({label})"


@dataclass
class LevelAnchors:
    vwap: float | None = None
    orb_high: float | None = None
    orb_low: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prior_close: float | None = None
    orb_from_minute: bool = False


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
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
                dt = datetime(
                    trading_date.year, trading_date.month, trading_date.day, int(hh), int(mm)
                )
            except (TypeError, ValueError):
                return None
        else:
            return None
    if dt.tzinfo is None:
        return ET.localize(dt)
    return dt.astimezone(ET)


def _ohlcv_from_raw_bars(bars: Any, trading_date: date) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(bars, list):
        return out
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        close = _safe_float(bar.get("close") or bar.get("price") or bar.get("c"))
        if close is None:
            continue
        high = _safe_float(bar.get("high") or bar.get("h")) or close
        low = _safe_float(bar.get("low") or bar.get("l")) or close
        vol = _safe_float(bar.get("volume") or bar.get("v")) or 0.0
        ts_raw = bar.get("ts") or bar.get("time") or bar.get("datetime") or bar.get("t")
        if isinstance(ts_raw, (int, float)) and ts_raw > 1_000_000_000_000:
            ts = datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=ET)
        else:
            ts = _bar_ts_et(ts_raw, trading_date)
        if ts is not None:
            out.append({"ts": ts, "high": high, "low": low, "close": close, "volume": vol})
    out.sort(key=lambda b: b["ts"])
    return out


def _ohlcv_bars_from_raw(raw: dict[str, Any], symbol: str, trading_date: date) -> list[dict[str, Any]]:
    sym = symbol.upper()
    intraday = raw.get("intraday") or {}
    top_bars = (intraday.get("bars") or {}).get(sym)
    if top_bars:
        parsed = _ohlcv_from_raw_bars(top_bars, trading_date)
        if parsed:
            return parsed

    for section_key in ("market", "sector", "stocks"):
        section = raw.get(section_key) or {}
        section_bars = (section.get("minute_bars") or {}).get(sym)
        if section_bars:
            parsed = _ohlcv_from_raw_bars(section_bars, trading_date)
            if parsed:
                return parsed
        quotes = section.get("quotes") or {}
        q = quotes.get(sym) or {}
        if isinstance(q, dict) and q.get("minute_bars"):
            parsed = _ohlcv_from_raw_bars(q["minute_bars"], trading_date)
            if parsed:
                return parsed
    return []


def _filter_bars_as_of(
    bars: list[dict[str, Any]],
    trading_date: date,
    as_of_et: time | Literal["now"] | None,
) -> list[dict[str, Any]]:
    if not bars or as_of_et is None or as_of_et == "now":
        return bars
    cutoff = ET.localize(datetime.combine(trading_date, as_of_et))
    return [b for b in bars if b["ts"] <= cutoff]


def _ohlcv_bars_yfinance(
    ticker: str,
    trading_date: date,
    *,
    as_of_et: time | Literal["now"] | None = None,
) -> list[dict[str, Any]]:
    try:
        df = yf.Ticker(ticker).history(
            start=trading_date.isoformat(),
            end=(trading_date + timedelta(days=1)).isoformat(),
            interval="1m",
            prepost=True,
        )
    except Exception:
        return []
    if df is None or df.empty:
        return []

    out: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        else:
            ts = ts.astimezone(ET)
        out.append(
            {
                "ts": ts,
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume") or 0),
            }
        )
    return _filter_bars_as_of(out, trading_date, as_of_et)


def _compute_vwap(bars: list[dict[str, Any]]) -> float | None:
    num = 0.0
    den = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = b.get("volume") or 0.0
        if vol <= 0:
            vol = 1.0
        num += tp * vol
        den += vol
    if den <= 0:
        return None
    return round(num / den, 2)


def _compute_orb(bars: list[dict[str, Any]], trading_date: date) -> tuple[float | None, float | None]:
    if not bars:
        return None, None
    open_et = market_open_et(trading_date)
    orb_end = open_et + timedelta(minutes=ORB_MINUTES)
    orb_bars = [b for b in bars if open_et <= b["ts"] <= orb_end]
    if not orb_bars:
        orb_bars = bars[: min(len(bars), ORB_MINUTES)]
    if not orb_bars:
        return None, None
    return (
        round(max(b["high"] for b in orb_bars), 2),
        round(min(b["low"] for b in orb_bars), 2),
    )


def _prior_quote(
    prior_raw: dict[str, Any],
    symbol: str,
    *,
    section: str,
) -> dict[str, Any]:
    sec = prior_raw.get(section) or {}
    quotes = sec.get("quotes") or {}
    sym = symbol.upper()
    if sym in quotes:
        return quotes[sym]
    for ticker, q in quotes.items():
        if str(ticker).upper() == sym:
            return q
    return {}


def compute_anchors(
    symbol: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date,
    *,
    section: str,
    q: dict[str, Any],
    obs: dict[str, Any],
    as_of_et: time | Literal["now"] | None = None,
) -> LevelAnchors:
    """VWAP / ORB from minute bars when available; else prior-day high/low fallback."""
    pit = as_of_et is not None and as_of_et != "now"
    prior_q = _prior_quote(prior_raw, symbol, section=section)
    prev_high = _safe_float(prior_q.get("high")) or _safe_float(q.get("high"))
    prev_low = _safe_float(prior_q.get("low")) or _safe_float(q.get("low"))
    prior_close = (
        _safe_float(obs.get("prev_close"))
        or _safe_float(q.get("prior_close"))
        or _safe_float(prior_q.get("close"))
    )

    bars = _ohlcv_bars_from_raw(raw, symbol, trading_day)
    bars = _filter_bars_as_of(bars, trading_day, as_of_et)
    if not bars and not pit:
        bars = _ohlcv_bars_yfinance(symbol, trading_day, as_of_et=as_of_et)
    elif not bars and pit and as_of_et is not None and as_of_et != "now":
        bars = _ohlcv_bars_yfinance(symbol, trading_day, as_of_et=as_of_et)

    vwap = _compute_vwap(bars) if bars else None
    orb_high, orb_low = _compute_orb(bars, trading_day) if bars else (None, None)
    orb_from_minute = orb_high is not None and orb_low is not None

    if not orb_from_minute:
        orb_high = prev_high
        orb_low = prev_low

    return LevelAnchors(
        vwap=vwap,
        orb_high=orb_high,
        orb_low=orb_low,
        prev_high=prev_high,
        prev_low=prev_low,
        prior_close=prior_close,
        orb_from_minute=orb_from_minute,
    )


def compute_entry_zone(
    direction: str,
    anchors: LevelAnchors,
    entry_px: float,
    entry_src: str,
    *,
    tolerance_pct: float = ENTRY_ZONE_TOLERANCE_PCT,
) -> dict[str, Any]:
    """Entry zone from ORB/VWAP confluence near chosen entry (±tolerance)."""
    anchor_map: list[tuple[str, float | None]] = [
        ("vwap", anchors.vwap),
        ("orb_high", anchors.orb_high),
        ("orb_low", anchors.orb_low),
        ("prev_high", anchors.prev_high),
        ("prev_low", anchors.prev_low),
        ("prior_close", anchors.prior_close),
    ]
    matched: list[dict[str, Any]] = []
    for src, px in anchor_map:
        if px is not None and _near(px, entry_px, tolerance_pct):
            matched.append(
                {
                    "source": src,
                    "label": source_label(src),
                    "price": round(px, 2),
                }
            )

    prices = [entry_px] + [a["price"] for a in matched]
    low = min(prices)
    high = max(prices)

    if low == high:
        band = entry_px * tolerance_pct / 100.0
        low = round(entry_px - band, 2)
        high = round(entry_px + band, 2)

    mid = round((low + high) / 2, 2)
    spread = high - low
    if spread >= 1.0:
        display = f"Entry Zone {low:.0f}–{high:.0f}"
    else:
        display = f"Entry Zone {low:.2f}–{high:.2f}"

    return {
        "low": round(low, 2),
        "high": round(high, 2),
        "mid": mid,
        "anchors": matched,
        "display": display,
        "entry_source": entry_src,
        "tolerance_pct": tolerance_pct,
    }


def derive_trade_levels(
    direction: str,
    anchors: LevelAnchors,
    *,
    current: float,
    expected_high: float,
    expected_low: float,
    expected_close: float,
) -> dict[str, Any]:
    """Entry / stop / target with level_source tags."""
    if direction == "LONG":
        if anchors.vwap is not None and current >= anchors.vwap * 0.998:
            entry_px = anchors.vwap
            entry_src = "vwap"
            entry_prefix = "Above"
        elif anchors.orb_high is not None:
            entry_px = anchors.orb_high
            entry_src = "orb_high" if anchors.orb_from_minute else "prev_high"
            entry_prefix = "Above"
        else:
            entry_px = current
            entry_src = "current"
            entry_prefix = "Above"

        if anchors.orb_low is not None and anchors.orb_from_minute:
            stop_px = anchors.orb_low
            stop_src = "orb_low"
        elif anchors.prev_low is not None:
            stop_px = anchors.prev_low
            stop_src = "prev_low"
        else:
            stop_px = expected_low
            stop_src = "expected_low"

        target_px = expected_close
        target_src = "expected_close"
        if expected_high > target_px * 1.002:
            target_px = expected_high
            target_src = "expected_high"

    elif direction == "SHORT":
        if anchors.vwap is not None and current <= anchors.vwap * 1.002:
            entry_px = anchors.vwap
            entry_src = "vwap"
            entry_prefix = "Below"
        elif anchors.orb_low is not None:
            entry_px = anchors.orb_low
            entry_src = "orb_low" if anchors.orb_from_minute else "prev_low"
            entry_prefix = "Below"
        else:
            entry_px = current
            entry_src = "current"
            entry_prefix = "Below"

        if anchors.orb_high is not None and anchors.orb_from_minute:
            stop_px = anchors.orb_high
            stop_src = "orb_high"
        elif anchors.prev_high is not None:
            stop_px = anchors.prev_high
            stop_src = "prev_high"
        else:
            stop_px = expected_high
            stop_src = "expected_high"

        target_px = expected_close
        target_src = "expected_close"
        if expected_low < target_px * 0.998:
            target_px = expected_low
            target_src = "expected_low"
    else:
        return {
            "entry": "—",
            "entry_source": None,
            "stop": "—",
            "stop_source": None,
            "target": "—",
            "target_source": None,
            "targets": [],
            "level_anchors": anchors,
        }

    entry = format_tagged(entry_px, entry_src, prefix=entry_prefix)
    stop = format_tagged(stop_px, stop_src)
    target = format_tagged(target_px, target_src)
    entry_zone = compute_entry_zone(direction, anchors, entry_px, entry_src)

    return {
        "entry": entry,
        "entry_source": entry_src,
        "entry_price": entry_zone["mid"],
        "entry_zone": entry_zone,
        "stop": stop,
        "stop_source": stop_src,
        "stop_price": round(stop_px, 2),
        "target": target,
        "target_source": target_src,
        "target_price": round(target_px, 2),
        "targets": [target],
        "level_anchors": {
            "vwap": anchors.vwap,
            "orb_high": anchors.orb_high,
            "orb_low": anchors.orb_low,
            "prev_high": anchors.prev_high,
            "prev_low": anchors.prev_low,
            "prior_close": anchors.prior_close,
            "orb_from_minute": anchors.orb_from_minute,
        },
    }


def format_if_level(price: float, source: str) -> str:
    """P16 IF-THEN level with source tag, e.g. '711 (昨日低点)'."""
    return f"{price:.1f} ({source_label(source)})"


def _near(px: float | None, target: float, tolerance_pct: float = 0.6) -> bool:
    if px is None or target <= 0:
        return False
    return abs(px - target) / target * 100.0 <= tolerance_pct


def _level_reason_row(
    source: str,
    label: str,
    price: float | None,
    *,
    chosen_src: str,
    level_px: float,
    primary_weight: float = 35.0,
    confluence_weight: float = 15.0,
) -> dict[str, Any] | None:
    if price is None:
        return None
    is_primary = source == chosen_src
    matched = is_primary or _near(price, level_px)
    if not matched:
        return None
    weight = primary_weight if is_primary else confluence_weight
    return {
        "source": source,
        "label": label,
        "price": round(price, 2),
        "weight_pct": weight,
        "matched": True,
        "primary": is_primary,
    }


def build_level_reasons(
    direction: str,
    anchors: LevelAnchors,
    *,
    entry_px: float,
    entry_src: str,
    stop_px: float,
    stop_src: str,
    target_px: float,
    target_src: str,
    current: float,
) -> dict[str, Any]:
    """Auditable entry/stop/target with confluence weights."""
    if direction == "LONG":
        entry_anchors = [
            ("vwap", "VWAP", anchors.vwap),
            ("orb_high", "ORB High", anchors.orb_high),
            ("prev_high", "Yesterday High", anchors.prev_high),
            ("prior_close", "Prior Close", anchors.prior_close),
        ]
        stop_anchors = [
            ("orb_low", "ORB Low", anchors.orb_low),
            ("prev_low", "Yesterday Low", anchors.prev_low),
            ("vwap", "VWAP", anchors.vwap),
        ]
        target_anchors = [
            ("expected_close", "Expected Close", target_px if target_src == "expected_close" else None),
            ("expected_high", "Expected High", target_px if target_src == "expected_high" else None),
            ("prev_high", "Yesterday High", anchors.prev_high),
        ]
        if current >= (anchors.vwap or 0):
            entry_anchors.append(("current", "Above Current", current))
    elif direction == "SHORT":
        entry_anchors = [
            ("orb_low", "ORB Low", anchors.orb_low),
            ("vwap", "VWAP", anchors.vwap),
            ("prev_low", "Yesterday Low", anchors.prev_low),
            ("prior_close", "Prior Close", anchors.prior_close),
        ]
        stop_anchors = [
            ("orb_high", "ORB High", anchors.orb_high),
            ("prev_high", "Yesterday High", anchors.prev_high),
            ("vwap", "VWAP", anchors.vwap),
        ]
        target_anchors = [
            ("expected_close", "Expected Close", target_px if target_src == "expected_close" else None),
            ("expected_low", "Expected Low", target_px if target_src == "expected_low" else None),
            ("prev_low", "Yesterday Low", anchors.prev_low),
        ]
        if current <= (anchors.vwap or float("inf")):
            entry_anchors.append(("current", "Below Current", current))
    else:
        return {}

    def _build_level(
        anchors_list: list[tuple[str, str, float | None]],
        chosen: str,
        level_px: float,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for src, label, px in anchors_list:
            row = _level_reason_row(src, label, px, chosen_src=chosen, level_px=level_px)
            if row:
                rows.append(row)
        total_w = sum(r["weight_pct"] for r in rows) or 1.0
        for r in rows:
            r["weight_pct"] = round(r["weight_pct"] / total_w * 100.0, 0)
        confidence = min(98, 55 + len(rows) * 12 + (10 if any(r["primary"] for r in rows) else 0))
        return {
            "price": round(level_px, 2),
            "source": chosen,
            "confidence": confidence,
            "reasons": rows,
        }

    return {
        "entry": _build_level(entry_anchors, entry_src, entry_px),
        "stop": _build_level(stop_anchors, stop_src, stop_px),
        "target": _build_level(target_anchors, target_src, target_px),
    }
