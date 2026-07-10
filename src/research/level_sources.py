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
    target_price: float | None = None,
) -> dict[str, Any]:
    """Entry zone from ORB/VWAP confluence near chosen entry (±tolerance).

    Anchors on the wrong side of target are excluded so the zone cannot
    cross the target, but the structural entry_px itself is never moved.
    """
    del direction  # reserved for callers / future direction-specific banding
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
        if px is None or not _near(px, entry_px, tolerance_pct):
            continue
        # Do not expand Ideal Entry past the target (would invent upside/downside).
        if target_price is not None and target_price > 0:
            if entry_px <= target_price and px > target_price:
                continue
            if entry_px >= target_price and px < target_price:
                continue
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
        # Keep single-point band from crossing target
        if target_price is not None and target_price > 0:
            if entry_px < target_price:
                high = min(high, round(target_price * 0.999, 2))
                low = min(low, high)
            elif entry_px > target_price:
                low = max(low, round(target_price * 1.001, 2))
                high = max(high, low)

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


def trade_levels_valid(
    direction: str,
    *,
    entry_price: float | None,
    stop_price: float | None,
    target_price: float | None,
    entry_zone: dict[str, Any] | None = None,
) -> bool:
    """Hard geometry invariants for directional trade levels.

    LONG:  stop < entry ≤ target (and entry_zone.high < target when zone present)
    SHORT: target ≤ entry < stop (and entry_zone.low > target when zone present)
    """
    if entry_price is None or stop_price is None or target_price is None:
        return False
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        return False

    if direction == "LONG":
        if not (stop_price < entry_price <= target_price):
            return False
        if entry_zone:
            zone_high = entry_zone.get("high")
            if zone_high is not None and float(zone_high) >= target_price:
                return False
        return True

    if direction == "SHORT":
        if not (target_price <= entry_price < stop_price):
            return False
        if entry_zone:
            zone_low = entry_zone.get("low")
            if zone_low is not None and float(zone_low) <= target_price:
                return False
        return True

    return False


def _anchor_near_market(px: float | None, current: float, *, max_dev_pct: float = 15.0) -> bool:
    """Ignore anchors that are wildly inconsistent with the session price."""
    if px is None or current <= 0:
        return False
    return abs(px - current) / current * 100.0 <= max_dev_pct


def _long_entry_candidates(
    anchors: LevelAnchors,
    current: float,
) -> list[tuple[float, str, str]]:
    """Preferred LONG entries (highest structural first), then current fallback."""
    out: list[tuple[float, str, str]] = []
    if (
        anchors.vwap is not None
        and _anchor_near_market(anchors.vwap, current)
        and current >= anchors.vwap * 0.998
    ):
        out.append((anchors.vwap, "vwap", "Above"))
    if anchors.orb_high is not None and _anchor_near_market(anchors.orb_high, current):
        src = "orb_high" if anchors.orb_from_minute else "prev_high"
        out.append((anchors.orb_high, src, "Above"))
    out.append((current, "current", "Above"))
    seen: set[float] = set()
    uniq: list[tuple[float, str, str]] = []
    for px, src, prefix in out:
        key = round(px, 4)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((px, src, prefix))
    return uniq


def _short_entry_candidates(
    anchors: LevelAnchors,
    current: float,
) -> list[tuple[float, str, str]]:
    out: list[tuple[float, str, str]] = []
    if (
        anchors.vwap is not None
        and _anchor_near_market(anchors.vwap, current)
        and current <= anchors.vwap * 1.002
    ):
        out.append((anchors.vwap, "vwap", "Below"))
    if anchors.orb_low is not None and _anchor_near_market(anchors.orb_low, current):
        src = "orb_low" if anchors.orb_from_minute else "prev_low"
        out.append((anchors.orb_low, src, "Below"))
    out.append((current, "current", "Below"))
    seen: set[float] = set()
    uniq: list[tuple[float, str, str]] = []
    for px, src, prefix in out:
        key = round(px, 4)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((px, src, prefix))
    return uniq


def _long_stop_candidates(
    anchors: LevelAnchors,
    current: float,
    expected_low: float,
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    if (
        anchors.orb_low is not None
        and anchors.orb_from_minute
        and _anchor_near_market(anchors.orb_low, current)
    ):
        out.append((anchors.orb_low, "orb_low"))
    if anchors.prev_low is not None and _anchor_near_market(anchors.prev_low, current):
        out.append((anchors.prev_low, "prev_low"))
    out.append((expected_low, "expected_low"))
    # Soft structural stop below current if expected_low is unusable
    out.append((current * 0.992, "current"))
    seen: set[float] = set()
    uniq: list[tuple[float, str]] = []
    for px, src in out:
        key = round(px, 4)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((px, src))
    return uniq


def _short_stop_candidates(
    anchors: LevelAnchors,
    current: float,
    expected_high: float,
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    if (
        anchors.orb_high is not None
        and anchors.orb_from_minute
        and _anchor_near_market(anchors.orb_high, current)
    ):
        out.append((anchors.orb_high, "orb_high"))
    if anchors.prev_high is not None and _anchor_near_market(anchors.prev_high, current):
        out.append((anchors.prev_high, "prev_high"))
    out.append((expected_high, "expected_high"))
    out.append((current * 1.008, "current"))
    seen: set[float] = set()
    uniq: list[tuple[float, str]] = []
    for px, src in out:
        key = round(px, 4)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((px, src))
    return uniq


def derive_trade_levels(
    direction: str,
    anchors: LevelAnchors,
    *,
    current: float,
    expected_high: float,
    expected_low: float,
    expected_close: float,
) -> dict[str, Any]:
    """Entry / stop / target with level_source tags.

    Chooses an entry/stop pair that respects target geometry when possible. If no
    stop < entry ≤ target (LONG) or target ≤ entry < stop (SHORT) setup
    exists, returns levels_valid=False so callers can Pass the trade.
    """
    empty = {
        "entry": "—",
        "entry_source": None,
        "entry_price": None,
        "entry_zone": None,
        "stop": "—",
        "stop_source": None,
        "stop_price": None,
        "target": "—",
        "target_source": None,
        "target_price": None,
        "targets": [],
        "levels_valid": False,
        "level_anchors": anchors,
    }

    if direction == "LONG":
        target_px = expected_close
        target_src = "expected_close"
        if expected_high > target_px * 1.002:
            target_px = expected_high
            target_src = "expected_high"

        entry_cands = _long_entry_candidates(anchors, current)
        stop_cands = _long_stop_candidates(anchors, current, expected_low)
        entry_px, entry_src, entry_prefix = entry_cands[0]
        stop_px, stop_src = stop_cands[0]
        chosen = False
        for cand_stop, cand_stop_src in stop_cands:
            for cand_px, cand_src, cand_prefix in entry_cands:
                if cand_stop < cand_px <= target_px:
                    entry_px, entry_src, entry_prefix = cand_px, cand_src, cand_prefix
                    stop_px, stop_src = cand_stop, cand_stop_src
                    chosen = True
                    break
            if chosen:
                break

    elif direction == "SHORT":
        target_px = expected_close
        target_src = "expected_close"
        if expected_low < target_px * 0.998:
            target_px = expected_low
            target_src = "expected_low"

        entry_cands = _short_entry_candidates(anchors, current)
        stop_cands = _short_stop_candidates(anchors, current, expected_high)
        entry_px, entry_src, entry_prefix = entry_cands[0]
        stop_px, stop_src = stop_cands[0]
        chosen = False
        for cand_stop, cand_stop_src in stop_cands:
            for cand_px, cand_src, cand_prefix in entry_cands:
                if target_px <= cand_px < cand_stop:
                    entry_px, entry_src, entry_prefix = cand_px, cand_src, cand_prefix
                    stop_px, stop_src = cand_stop, cand_stop_src
                    chosen = True
                    break
            if chosen:
                break
    else:
        return empty

    entry = format_tagged(entry_px, entry_src, prefix=entry_prefix)
    stop = format_tagged(stop_px, stop_src)
    target = format_tagged(target_px, target_src)
    entry_zone = compute_entry_zone(
        direction, anchors, entry_px, entry_src, target_price=target_px
    )
    # Prefer structural entry for ER math; zone mid only when it stays consistent.
    entry_mid = entry_zone["mid"]
    if abs(entry_mid - entry_px) / max(entry_px, 1e-9) > 0.01:
        entry_for_price = round(entry_px, 2)
    else:
        entry_for_price = entry_mid

    valid = trade_levels_valid(
        direction,
        entry_price=entry_for_price,
        stop_price=round(stop_px, 2),
        target_price=round(target_px, 2),
        entry_zone=entry_zone,
    )
    # If zone expansion broke validity but structural entry is fine, drop zone check
    if not valid and trade_levels_valid(
        direction,
        entry_price=round(entry_px, 2),
        stop_price=round(stop_px, 2),
        target_price=round(target_px, 2),
        entry_zone=None,
    ):
        entry_for_price = round(entry_px, 2)
        entry_zone = {
            "low": round(entry_px, 2),
            "high": round(entry_px, 2),
            "mid": round(entry_px, 2),
            "anchors": [],
            "display": f"Entry Zone {entry_px:.2f}–{entry_px:.2f}",
            "entry_source": entry_src,
            "tolerance_pct": ENTRY_ZONE_TOLERANCE_PCT,
        }
        valid = True

    return {
        "entry": entry,
        "entry_source": entry_src,
        "entry_price": entry_for_price,
        "entry_zone": entry_zone,
        "stop": stop,
        "stop_source": stop_src,
        "stop_price": round(stop_px, 2),
        "target": target,
        "target_source": target_src,
        "target_price": round(target_px, 2),
        "targets": [target],
        "levels_valid": valid,
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
