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
    prior_dgs10 = _safe_float(prior_macro, "rates", "DGS10")

    # VIX
    vix_prices = market.get("prices", {}).get("^VIX", {})
    vix = _safe_float(vix_prices, "close") or _safe_float(vix_prices, "current_price")
    prior_vix_prices = prior_market.get("prices", {}).get("^VIX", {})
    prior_vix = _safe_float(prior_vix_prices, "close") or _safe_float(prior_vix_prices, "current_price")
    vix_chg = _pct_chg(vix, prior_vix)

    # QQQ
    qqq_prices = market.get("prices", {}).get("QQQ", {})
    qqq_close = _safe_float(qqq_prices, "close") or _safe_float(qqq_prices, "current_price")
    qqq_open = _safe_float(qqq_prices, "open")
    prior_qqq_prices = prior_market.get("prices", {}).get("QQQ", {})
    prior_qqq = _safe_float(prior_qqq_prices, "close") or _safe_float(prior_qqq_prices, "current_price")
    qqq_chg = _pct_chg(qqq_close, prior_qqq)
    qqq_gap = _pct_chg(qqq_open, prior_qqq) if qqq_open and prior_qqq else None

    # SPY
    spy_prices = market.get("prices", {}).get("SPY", {})
    spy_close = _safe_float(spy_prices, "close") or _safe_float(spy_prices, "current_price")
    prior_spy_prices = prior_market.get("prices", {}).get("SPY", {})
    prior_spy = _safe_float(prior_spy_prices, "close") or _safe_float(prior_spy_prices, "current_price")
    spy_chg = _pct_chg(spy_close, prior_spy)

    # DXY (dollar)
    dxy_prices = market.get("prices", {}).get("DX-Y.NYB", {})
    dxy = _safe_float(dxy_prices, "close") or _safe_float(dxy_prices, "current_price")
    prior_dxy_prices = prior_market.get("prices", {}).get("DX-Y.NYB", {})
    prior_dxy = _safe_float(prior_dxy_prices, "close") or _safe_float(prior_dxy_prices, "current_price")
    dxy_chg = _pct_chg(dxy, prior_dxy)

    # SMH (semiconductor ETF)
    smh_prices = sector.get("prices", {}).get("SMH", {})
    smh_close = _safe_float(smh_prices, "close") or _safe_float(smh_prices, "current_price")
    prior_smh_prices = prior_sector.get("prices", {}).get("SMH", {})
    prior_smh = _safe_float(prior_smh_prices, "close") or _safe_float(prior_smh_prices, "current_price")
    smh_chg = _pct_chg(smh_close, prior_smh)

    # NVDA
    nvda_prices = stocks.get("prices", {}).get("NVDA", {})
    nvda_close = _safe_float(nvda_prices, "close") or _safe_float(nvda_prices, "current_price")
    prior_nvda_prices = prior_stocks.get("prices", {}).get("NVDA", {})
    prior_nvda = _safe_float(prior_nvda_prices, "close") or _safe_float(prior_nvda_prices, "current_price")
    nvda_chg = _pct_chg(nvda_close, prior_nvda)

    # Oil (XLE as proxy)
    xle_prices = sector.get("prices", {}).get("XLE", {})
    xle_close = _safe_float(xle_prices, "close") or _safe_float(xle_prices, "current_price")
    prior_xle_prices = prior_sector.get("prices", {}).get("XLE", {})
    prior_xle = _safe_float(prior_xle_prices, "close") or _safe_float(prior_xle_prices, "current_price")
    oil_chg = _pct_chg(xle_close, prior_xle)

    # Breadth proxy: ratio of up-moving Mag7 stocks
    mag7 = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    up_count = 0
    valid_count = 0
    for sym in mag7:
        cur_p = stocks.get("prices", {}).get(sym, {})
        prv_p = prior_stocks.get("prices", {}).get(sym, {})
        cur = _safe_float(cur_p, "close") or _safe_float(cur_p, "current_price")
        prv = _safe_float(prv_p, "close") or _safe_float(prv_p, "current_price")
        if cur is not None and prv is not None and prv != 0:
            valid_count += 1
            if cur > prv:
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
