from __future__ import annotations

from typing import Any

import yfinance as yf

from src.collectors.config import load_symbols
from src.collectors.polygon_client import PolygonClient


def _from_polygon(underlying: str) -> dict[str, Any] | None:
    polygon = PolygonClient()
    try:
        snap = polygon.options_snapshot(underlying)
    except Exception:
        return None

    if snap.get("status") == "ERROR" or snap.get("status_code") == 403:
        return None

    results = snap.get("results") or []
    if not results:
        return None

    call_oi = 0
    put_oi = 0
    ivs: list[float] = []

    for contract in results[:200]:
        details = contract.get("details") or {}
        oi = (contract.get("open_interest") or 0) or 0
        iv = contract.get("implied_volatility")
        if details.get("contract_type") == "call":
            call_oi += int(oi)
        elif details.get("contract_type") == "put":
            put_oi += int(oi)
        if iv is not None:
            try:
                ivs.append(float(iv))
            except (TypeError, ValueError):
                pass

    pc_ratio = round(put_oi / call_oi, 3) if call_oi else None
    avg_iv = round(sum(ivs) / len(ivs), 4) if ivs else None

    return {
        "source": "polygon",
        "underlying": underlying,
        "contracts_count": len(results),
        "put_call_ratio": pc_ratio,
        "avg_implied_volatility": avg_iv,
        "total_call_oi": int(call_oi),
        "total_put_oi": int(put_oi),
        "expiration": None,
    }


def _from_yfinance(underlying: str) -> dict[str, Any]:
    ticker = yf.Ticker(underlying)
    expirations = list(ticker.options or [])
    if not expirations:
        raise ValueError("no option expirations")

    expiration = expirations[0]
    chain = ticker.option_chain(expiration)
    calls = chain.calls
    puts = chain.puts

    call_oi = int(calls["openInterest"].fillna(0).sum())
    put_oi = int(puts["openInterest"].fillna(0).sum())
    iv_series = calls["impliedVolatility"].dropna().tolist() + puts["impliedVolatility"].dropna().tolist()
    ivs = [float(v) for v in iv_series if v == v]
    pc_ratio = round(put_oi / call_oi, 3) if call_oi else None
    avg_iv = round(sum(ivs) / len(ivs), 4) if ivs else None

    return {
        "source": "yfinance",
        "underlying": underlying,
        "contracts_count": len(calls) + len(puts),
        "put_call_ratio": pc_ratio,
        "avg_implied_volatility": avg_iv,
        "total_call_oi": call_oi,
        "total_put_oi": put_oi,
        "expiration": expiration,
    }


def collect_options() -> dict[str, Any]:
    cfg = load_symbols()
    underlying = cfg["options"]["underlying"]
    errors: list[str] = []

    data = _from_polygon(underlying)
    if data is None:
        try:
            data = _from_yfinance(underlying)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            return {
                "underlying": underlying,
                "error": "; ".join(errors),
                "checklist": {
                    "qqq_option_chain": False,
                    "iv": False,
                    "put_call_ratio": False,
                    "open_interest": False,
                },
                "ok": False,
            }

    checklist = {
        "qqq_option_chain": data["contracts_count"] > 0,
        "iv": data["avg_implied_volatility"] is not None,
        "put_call_ratio": data["put_call_ratio"] is not None,
        "open_interest": (data["total_call_oi"] + data["total_put_oi"]) > 0,
    }

    return {
        **data,
        "checklist": checklist,
        "errors": errors,
        "ok": checklist["qqq_option_chain"],
    }
