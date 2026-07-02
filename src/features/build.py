"""Feature engineering: build features dict from raw data + historical OHLCV.

Features are rule-based, no LLM involved.
All values use T-day or prior data only (no future leakage).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

from src.schemas.market_case import FeaturesModel
from src.utils.paths import raw_data_path
from src.utils.trading_calendar import prior_trading_day, today_et

logger = logging.getLogger(__name__)


def _safe_float(d: dict, *keys: str, default: float | None = None) -> float | None:
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _pct_chg(current: float | None, prev: float | None) -> float | None:
    if current is None or prev is None or prev == 0:
        return None
    return (current - prev) / abs(prev) * 100.0


def _quotes(section: dict[str, Any]) -> dict[str, Any]:
    return section.get("quotes") or section.get("prices") or {}


def _quote(section: dict[str, Any], ticker: str) -> dict[str, Any]:
    return _quotes(section).get(ticker) or {}


def _quote_chg(q: dict[str, Any], prior_q: dict[str, Any] | None = None) -> float | None:
    chg = _safe_float(q, "change_pct")
    if chg is not None:
        return chg
    cur = _safe_float(q, "close") or _safe_float(q, "current_price")
    prv = _safe_float(prior_q or {}, "close") or _safe_float(prior_q or {}, "current_price")
    if prv is None:
        prv = _safe_float(q, "prev_close")
    return _pct_chg(cur, prv)


def build_features(trading_date: date | None = None) -> FeaturesModel:
    """Build features from raw data files for given trading date."""
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()

    raw_path = raw_data_path(date_str)
    if not raw_path.exists():
        logger.warning("No raw data file for %s, returning empty features", date_str)
        return FeaturesModel()

    raw: dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))

    prior = prior_trading_day(trading_date)
    prior_str = prior.isoformat()
    prior_raw: dict[str, Any] = {}
    if raw_data_path(prior_str).exists():
        try:
            prior_raw = json.loads(raw_data_path(prior_str).read_text(encoding="utf-8"))
        except Exception:
            pass

    macro = raw.get("macro", {})
    market = raw.get("market", {})
    sector = raw.get("sector", {})
    stocks = raw.get("stocks", {})
    options = raw.get("options", {})

    prior_macro = prior_raw.get("macro", {})
    prior_market = prior_raw.get("market", {})
    prior_sector = prior_raw.get("sector", {})
    prior_stocks = prior_raw.get("stocks", {})

    # DGS10 — FRED 10Y treasury rate
    dgs10 = _safe_float(macro, "rates", "DGS10")
    if dgs10 is None:
        dgs10 = _safe_float(market.get("treasury_10y_fred") or {}, "value")
    if dgs10 is None:
        dgs10 = _safe_float((macro.get("series") or {}).get("DGS10") or {}, "value")

    # VIX
    vix_q = _quote(market, "^VIX")
    prior_vix_q = _quote(prior_market, "^VIX")
    vix = _safe_float(vix_q, "close") or _safe_float(vix_q, "current_price")
    vix_chg = _quote_chg(vix_q, prior_vix_q)

    # QQQ
    qqq_q = _quote(market, "QQQ")
    prior_qqq_q = _quote(prior_market, "QQQ")
    qqq_close = _safe_float(qqq_q, "close") or _safe_float(qqq_q, "current_price")
    qqq_open = _safe_float(qqq_q, "open")
    prior_qqq = _safe_float(prior_qqq_q, "close") or _safe_float(prior_qqq_q, "current_price")
    if prior_qqq is None:
        prior_qqq = _safe_float(qqq_q, "prev_close")
    qqq_chg = _quote_chg(qqq_q, prior_qqq_q)
    qqq_gap = _pct_chg(qqq_open, prior_qqq) if qqq_open and prior_qqq else None

    # SPY
    spy_q = _quote(market, "SPY")
    prior_spy_q = _quote(prior_market, "SPY")
    spy_chg = _quote_chg(spy_q, prior_spy_q)

    # DXY (dollar)
    dxy_q = _quote(market, "DX-Y.NYB")
    prior_dxy_q = _quote(prior_market, "DX-Y.NYB")
    dxy = _safe_float(dxy_q, "close") or _safe_float(dxy_q, "current_price")
    dxy_chg = _quote_chg(dxy_q, prior_dxy_q)

    # SMH (semiconductor ETF)
    smh_q = _quote(sector, "SMH")
    prior_smh_q = _quote(prior_sector, "SMH")
    smh_chg = _quote_chg(smh_q, prior_smh_q)

    # NVDA
    nvda_q = _quote(stocks, "NVDA")
    prior_nvda_q = _quote(prior_stocks, "NVDA")
    nvda_chg = _quote_chg(nvda_q, prior_nvda_q)

    # Oil (XLE as proxy)
    xle_q = _quote(sector, "XLE")
    prior_xle_q = _quote(prior_sector, "XLE")
    oil_chg = _quote_chg(xle_q, prior_xle_q)

    # Breadth proxy: ratio of up-moving Mag7 stocks
    mag7 = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    up_count = 0
    valid_count = 0
    for sym in mag7:
        cur_p = _quote(stocks, sym)
        prv_p = _quote(prior_stocks, sym)
        chg = _quote_chg(cur_p, prv_p)
        if chg is not None:
            valid_count += 1
            if chg > 0:
                up_count += 1
    breadth_proxy = (up_count / valid_count) if valid_count > 0 else None

    return FeaturesModel(
        dgs10=dgs10,
        vix=vix,
        vix_chg=vix_chg,
        qqq_chg=qqq_chg,
        smh_chg=smh_chg,
        nvda_chg=nvda_chg,
        spy_chg=spy_chg,
        oil_chg=oil_chg,
        dxy_chg=dxy_chg,
        breadth_proxy=breadth_proxy,
        qqq_gap=qqq_gap,
    )
