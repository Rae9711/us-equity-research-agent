"""P18 Trade Candidates + Best Opportunity — Decision Agent rules engine.

ADVISORY ONLY — 不构成投资建议. No auto-trading.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from src.utils.paths import data_root
from src.utils.quote_resolve import session_change_pct
from src.utils.trading_calendar import prior_trading_day

CANDIDATE_SYMBOLS = ["NVDA", "QQQ", "SMH", "TSLA", "SPY", "TQQQ"]
ADVISORY_TAG = "ADVISORY — 不构成投资建议"

_SYMBOL_SECTION: dict[str, str] = {
    "QQQ": "market",
    "SPY": "market",
    "TQQQ": "market",
    "SMH": "sector",
    "NVDA": "stocks",
    "TSLA": "stocks",
}


def _quote(raw: dict[str, Any], symbol: str) -> dict[str, Any]:
    sym = symbol.upper()
    section = _SYMBOL_SECTION.get(sym, "stocks")
    quotes = ((raw.get(section) or {}).get("quotes") or {})
    if sym in quotes:
        return quotes[sym]
    for ticker, q in quotes.items():
        if str(ticker).upper() == sym:
            return q
    return {}


def _pct(
    symbol: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date | None,
    q: dict[str, Any],
) -> float | None:
    section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
    if trading_day is not None:
        pct = session_change_pct(symbol, raw, prior_raw, trading_day, section=section)
        if pct is not None:
            return pct
    chg = q.get("change_pct")
    if chg is not None:
        try:
            return float(chg)
        except (TypeError, ValueError):
            pass
    return None


def _load_prior_raw(trading_day: date) -> dict[str, Any]:
    from src.utils.paths import data_root

    prior_path = data_root() / "raw" / f"{prior_trading_day(trading_day).isoformat()}.json"
    if not prior_path.exists():
        return {}
    try:
        return json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bias_stars(bias: str) -> str:
    b = (bias or "").lower()
    if "strong bear" in b:
        return "★☆☆☆☆"
    if "bear" in b:
        return "★★☆☆☆"
    if b == "bull":
        return "★★★★★"
    if "bullish" in b:
        return "★★★★☆"
    return "★★★☆☆"


def _p16_gate(parts: dict[str, Any], total: int) -> str:
    """Return Trade | Wait | No Trade from P16 text or score heuristics."""
    p16 = parts.get("P16") or {}
    text = " ".join(
        str(p16.get(k) or "") for k in ("judgment", "one_liner", "body_md")
    ).lower()
    if any(w in text for w in ("不交易", "放弃", "不追", "no trade", "hold off")):
        return "No Trade"
    if any(w in text for w in ("买", "call", "做多", "trade", "入场", "long")):
        return "Trade"
    edge = (parts.get("P13") or {}).get("judgment") or ""
    if total >= 2 and "Edge：YES" in edge:
        return "Trade"
    if total >= 0:
        return "Wait"
    return "No Trade"


def _instrument(
    symbol: str,
    direction: str,
    p9: dict[str, Any],
) -> str:
    if direction == "NO TRADE":
        return "—"
    buy_options = p9.get("buy_options") == "Yes"
    zero_dte = p9.get("zero_dte") == "Yes"
    buy_call = p9.get("buy_call") == "Yes"
    buy_put = p9.get("buy_put") == "Yes"

    if direction == "LONG":
        if zero_dte and buy_call:
            return f"{symbol} 0DTE Call"
        if buy_options and buy_call:
            return f"{symbol} Call"
        return "Stock" if symbol in ("NVDA", "TSLA") else "ETF"
    if direction == "SHORT":
        if zero_dte and buy_put:
            return f"{symbol} 0DTE Put"
        if buy_options and buy_put:
            return f"{symbol} Put"
        return "ETF"
    return "—"


def _price_levels(
    q: dict[str, Any],
    direction: str,
) -> tuple[str, str, list[str]]:
    """Entry/stop/target from prior bar levels or % from close (ADVISORY proxy)."""
    close = q.get("close")
    prev = q.get("prev_close") or q.get("previous_close")
    high = q.get("high") or q.get("day_high")
    low = q.get("low") or q.get("day_low")

    try:
        px = float(close) if close is not None else None
    except (TypeError, ValueError):
        px = None

    if px is None or px <= 0:
        return "—", "—", []

    try:
        prev_f = float(prev) if prev is not None else px * 0.995
    except (TypeError, ValueError):
        prev_f = px * 0.995

    atr_proxy = max(abs(px - prev_f), px * 0.004)
    try:
        hi = float(high) if high is not None else px + atr_proxy * 0.5
        lo = float(low) if low is not None else px - atr_proxy * 0.5
    except (TypeError, ValueError):
        hi, lo = px + atr_proxy * 0.5, px - atr_proxy * 0.5

    if direction == "LONG":
        entry_px = max(px, hi)
        stop_px = min(lo, entry_px - atr_proxy * 0.6)
        t1 = entry_px + atr_proxy * 1.2
        t2 = entry_px + atr_proxy * 2.0
        entry = f"Above {entry_px:.1f}"
        stop = f"{stop_px:.1f}"
        targets = [f"{t1:.1f}", f"{t2:.1f}"]
    elif direction == "SHORT":
        entry_px = min(px, lo)
        stop_px = max(hi, entry_px + atr_proxy * 0.6)
        t1 = entry_px - atr_proxy * 1.2
        t2 = entry_px - atr_proxy * 2.0
        entry = f"Below {entry_px:.1f}"
        stop = f"{stop_px:.1f}"
        targets = [f"{t1:.1f}", f"{t2:.1f}"]
    else:
        return "—", "—", []

    return entry, stop, targets


def _rr_ratio(entry: str, stop: str, targets: list[str]) -> str:
    try:
        if "Above" in entry:
            ep = float(entry.replace("Above", "").strip())
            sp = float(stop)
            tp = float(targets[0]) if targets else ep
            risk = ep - sp
            reward = tp - ep
        elif "Below" in entry:
            ep = float(entry.replace("Below", "").strip())
            sp = float(stop)
            tp = float(targets[0]) if targets else ep
            risk = sp - ep
            reward = ep - tp
        else:
            return "—"
        if risk <= 0:
            return "—"
        return f"1:{reward / risk:.1f}"
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"


def _score_symbol(
    symbol: str,
    *,
    driver_type: str,
    driver: str,
    smh_pct: float | None,
    qqq_pct: float | None,
    nvda_pct: float | None,
    mag7_leader: bool,
    vix_score: int,
    bond_score: int,
    regime_label: str,
    sym_pct: float | None,
) -> tuple[int, list[str]]:
    score = 50
    reasons: list[str] = []

    dt = (driver_type or "").lower()
    drv = (driver or "").lower()

    # Driver type alignment
    if dt in ("momentum", "ai") and symbol in ("QQQ", "SMH", "NVDA", "TQQQ"):
        score += 12
        reasons.append(f"{driver_type} driver favors {symbol}")
    elif dt == "macro" and symbol in ("SPY", "QQQ"):
        score += 6
        reasons.append("Macro day → index ETF")
    elif dt == "no catalyst" and symbol in ("QQQ", "SPY"):
        score += 4
        reasons.append("No catalyst → broad index")

    # Sector strength SMH vs QQQ
    if smh_pct is not None and qqq_pct is not None:
        smh_vs = smh_pct - qqq_pct
        if symbol == "SMH" and smh_vs > 0.1:
            score += 10
            reasons.append(f"SMH leads QQQ ({smh_vs:+.2f}%)")
        elif symbol == "QQQ" and smh_vs <= 0.1:
            score += 6
            reasons.append("QQQ breadth proxy")
        elif symbol == "NVDA" and smh_vs > 0:
            score += 8
            reasons.append("Semis strength → NVDA")

    # Mag7 leaders
    if mag7_leader and symbol in ("NVDA", "TSLA"):
        score += 8
        reasons.append("Mag7 leader momentum")

    # VIX
    if vix_score > 0:
        score += 5
        reasons.append("VIX down → risk-on")
    elif vix_score < 0:
        score -= 8
        reasons.append("VIX up → caution")

    # Bond headwind
    if bond_score < 0:
        score -= 5
        reasons.append("Bond headwind")
    elif bond_score > 0:
        score += 3
        reasons.append("Bond tailwind")

    # Regime
    if regime_label == "AI Expansion" and symbol in ("NVDA", "SMH", "QQQ", "TQQQ"):
        score += 8
        reasons.append("AI Expansion regime")
    elif regime_label == "Macro Fear" and symbol in ("SPY", "QQQ"):
        score -= 5
        reasons.append("Macro Fear regime")

    # Symbol session momentum
    if sym_pct is not None:
        if sym_pct > 0.3:
            score += 5
            reasons.append(f"{symbol} +{sym_pct:.2f}%")
        elif sym_pct < -0.3:
            score -= 5
            reasons.append(f"{symbol} {sym_pct:.2f}% weak")

    # Leveraged only when strong momentum + low VIX
    if symbol == "TQQQ":
        if vix_score > 0 and (sym_pct or 0) > 0.2 and dt in ("momentum", "ai"):
            score += 6
            reasons.append("Leveraged momentum day")
        else:
            score -= 10

    score = max(0, min(100, score))
    return score, reasons[:4]


def _no_trade_reasons(
    *,
    p16_gate: str,
    total: int,
    bias: str,
    iv_level: str,
    p9: dict[str, Any],
    regime_label: str,
) -> list[str]:
    reasons: list[str] = []
    if p16_gate == "No Trade":
        reasons.append("P16 交易计划：No Trade")
    if p16_gate == "Wait":
        reasons.append("P16 建议 Wait — 无明确 edge")
    if total < 2:
        reasons.append(f"Total score {total:+d} — edge 不足")
    if "bear" in (bias or "").lower():
        reasons.append(f"Bias {bias}")
    if iv_level == "High":
        reasons.append("IV 偏高")
    if p9.get("buy_options") == "No" and total < 3:
        reasons.append("期权环境不佳")
    if regime_label == "Macro Fear" and total < 3:
        reasons.append("Macro Fear regime")
    if not reasons:
        reasons.append("无足够风险回报 edge")
    return reasons


def compute_trade_decision(
    raw: dict[str, Any],
    *,
    rule_bundle: dict[str, Any],
    parts: dict[str, Any],
    regime_label: str = "Range",
    regime_confidence: float = 0.5,
) -> dict[str, Any]:
    """Rank trade candidates and pick best opportunity for morning.json."""
    trading_date = str(raw.get("trading_date") or "")
    trading_day: date | None = None
    if trading_date:
        try:
            trading_day = date.fromisoformat(trading_date)
        except ValueError:
            trading_day = None
    prior_raw = _load_prior_raw(trading_day) if trading_day else {}

    bias = rule_bundle.get("bias") or "Neutral"
    total = int(rule_bundle.get("total") or 0)
    driver_type = rule_bundle.get("driver_type") or ""
    driver = rule_bundle.get("daily_driver") or ""
    p9 = (parts.get("P9") or {})
    p11 = (parts.get("P11") or {})
    scores = p11.get("scores") or {}
    vix_score = int(scores.get("VIX") or 0)
    bond_score = int(scores.get("Bond") or 0)

    market_q = (raw.get("market") or {}).get("quotes") or {}
    sector_q = (raw.get("sector") or {}).get("quotes") or {}
    qqq_q = market_q.get("QQQ") or {}
    smh_q = sector_q.get("SMH") or {}
    nvda_q = ((raw.get("stocks") or {}).get("quotes") or {}).get("NVDA") or {}

    qqq_pct = _pct("QQQ", raw, prior_raw, trading_day, qqq_q)
    smh_pct = _pct("SMH", raw, prior_raw, trading_day, smh_q)
    nvda_pct = _pct("NVDA", raw, prior_raw, trading_day, nvda_q)

    mag7_syms = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    stock_q = (raw.get("stocks") or {}).get("quotes") or {}
    mag7_pcts = []
    for sym in mag7_syms:
        q = stock_q.get(sym) or {}
        p = _pct(sym, raw, prior_raw, trading_day, q)
        if p is not None:
            mag7_pcts.append((sym, p))
    mag7_pcts.sort(key=lambda x: x[1], reverse=True)
    mag7_leader_syms = {s for s, _ in mag7_pcts[:2]}

    options = raw.get("options") or {}
    avg_iv = options.get("avg_implied_volatility")
    iv_level = "Medium"
    if avg_iv is not None:
        if avg_iv < 0.25:
            iv_level = "Low"
        elif avg_iv > 0.45:
            iv_level = "High"

    p16_gate = _p16_gate(parts, total)

    # Direction from bias
    if "bear" in bias.lower():
        direction = "SHORT"
    elif "bull" in bias.lower():
        direction = "LONG"
    else:
        direction = "LONG" if total >= 0 else "SHORT"

    # Rank candidates
    ranked: list[dict[str, Any]] = []
    for sym in CANDIDATE_SYMBOLS:
        q = _quote(raw, sym)
        sym_pct = _pct(sym, raw, prior_raw, trading_day, q)
        score, why_parts = _score_symbol(
            sym,
            driver_type=driver_type,
            driver=driver,
            smh_pct=smh_pct,
            qqq_pct=qqq_pct,
            nvda_pct=nvda_pct,
            mag7_leader=sym in mag7_leader_syms,
            vix_score=vix_score,
            bond_score=bond_score,
            regime_label=regime_label,
            sym_pct=sym_pct,
        )
        sym_dir = direction if score >= 45 else "NO TRADE"
        entry, stop, targets = _price_levels(q, sym_dir if sym_dir != "NO TRADE" else "LONG")
        inst = _instrument(sym, sym_dir, p9)
        ranked.append(
            {
                "symbol": sym,
                "score": score,
                "why": " · ".join(why_parts) if why_parts else "—",
                "detail": {
                    "direction": sym_dir,
                    "instrument": inst,
                    "entry": entry,
                    "stop": stop,
                    "target": " / ".join(targets) if targets else "—",
                    "targets": targets,
                    "conviction": score,
                    "rr_ratio": _rr_ratio(entry, stop, targets),
                    "reasons": why_parts,
                    "avoid": _avoid_list(sym, p9, iv_level, driver_type),
                },
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    # Best opportunity gate
    no_edge = (
        p16_gate in ("No Trade", "Wait")
        or total < 2
        or iv_level == "High"
        or (regime_label == "Macro Fear" and total < 3)
    )
    if "strong bear" in bias.lower() or bias == "Bear":
        no_edge = no_edge or total < 0

    best_sym = ranked[0]["symbol"] if ranked else "QQQ"
    best_score = ranked[0]["score"] if ranked else 0

    if no_edge or best_score < 55:
        why = _no_trade_reasons(
            p16_gate=p16_gate,
            total=total,
            bias=bias,
            iv_level=iv_level,
            p9=p9,
            regime_label=regime_label,
        )
        best: dict[str, Any] = {
            "direction": "NO TRADE",
            "symbol": "—",
            "instrument": "—",
            "confidence": max(30, min(70, 50 + total * 3)),
            "entry": "—",
            "stop": "—",
            "target": "—",
            "targets": [],
            "duration": "—",
            "why": why,
            "why_chain": " · ".join(why),
            "avoid": ["All directional trades"],
            "one_liner": "今日无交易 — edge 不足或环境不利。",
            "p16_gate": p16_gate,
            "advisory": True,
        }
    else:
        pick = ranked[0]
        q = _quote(raw, pick["symbol"])
        entry, stop, targets = _price_levels(q, direction)
        inst = _instrument(pick["symbol"], direction, p9)
        duration = "Intraday" if p9.get("zero_dte") == "Yes" else (
            "Swing" if abs(total) >= 4 else "Intraday"
        )
        why_parts = pick["detail"]["reasons"] + [
            f"Driver {driver_type}",
            f"Regime {regime_label}",
        ]
        if vix_score > 0:
            why_parts.append("VIX down")
        if bond_score >= 0:
            why_parts.append("No Macro bond headwind" if bond_score == 0 else "Bond tailwind")

        conf = int(
            min(
                95,
                max(
                    55,
                    best_score * 0.6
                    + regime_confidence * 20
                    + max(0, total) * 3,
                ),
            )
        )
        avoid = _avoid_list(pick["symbol"], p9, iv_level, driver_type)
        dir_word = "buy" if direction == "LONG" else "sell"
        inst_word = inst.lower() if inst not in ("Stock", "ETF") else f"{pick['symbol']} {inst.lower()}"
        one_liner = f"Today {dir_word} {inst_word} {entry.lower()}."

        best = {
            "direction": direction,
            "symbol": pick["symbol"],
            "instrument": inst,
            "confidence": conf,
            "entry": entry,
            "stop": stop,
            "target": " / ".join(targets) if targets else "—",
            "targets": targets,
            "duration": duration,
            "why": why_parts[:5],
            "why_chain": " · ".join(why_parts[:5]),
            "avoid": avoid,
            "one_liner": one_liner,
            "p16_gate": p16_gate,
            "advisory": True,
        }

    return {
        "trade_candidates": ranked,
        "best_opportunity": best,
        "bias_stars": _bias_stars(bias),
    }


def _avoid_list(symbol: str, p9: dict[str, Any], iv_level: str, driver_type: str) -> list[str]:
    avoid: list[str] = []
    if p9.get("buy_put") == "No" and p9.get("buy_call") == "Yes":
        avoid.append("Long Put")
    if p9.get("buy_call") == "No":
        avoid.append("Long Call")
    if iv_level == "High":
        avoid.append("Options (IV high)")
    if driver_type in ("AI", "Momentum") and symbol not in ("XLE",):
        avoid.append("Energy")
    if p9.get("zero_dte") == "No":
        avoid.append("0DTE")
    return avoid[:4] or ["—"]


def build_p18_part(
    trade_candidates: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, Any]:
    """Format P18 part for morning parts dict."""
    lines = [
        "| Rank | Symbol | Score | Why |",
        "|------|--------|-------|-----|",
    ]
    for row in trade_candidates:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['score']} | {row['why']} |"
        )
    body = "\n".join(lines)
    judgment = (
        f"Best：{best.get('symbol', '—')} · {best.get('direction', '—')} · "
        f"conf {best.get('confidence', 0)}%"
    )
    return {
        "judgment": judgment,
        "confidence": (best.get("confidence") or 0) / 100.0,
        "one_liner": best.get("one_liner", "—"),
        "body_md": body,
        "trade_candidates": trade_candidates,
        "best_opportunity": best,
    }


def build_executive_summary(
    *,
    bias: str,
    bias_stars: str,
    driver_type: str,
    driver: str,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Eight-field executive summary for homepage decision card."""
    return {
        "bias": bias,
        "bias_stars": bias_stars,
        "driver_type": driver_type,
        "driver": driver,
        "driver_display": " — ".join(x for x in (driver_type, driver) if x) or "—",
        "best_trade": (
            f"{best.get('symbol', '—')} · {best.get('direction', '—')} · "
            f"{best.get('instrument', '—')}"
        ),
        "confidence": best.get("confidence"),
        "entry": best.get("entry", "—"),
        "stop": best.get("stop", "—"),
        "target": best.get("target", "—"),
        "why_chain": best.get("why_chain", "—"),
        "one_liner": best.get("one_liner", "—"),
        "advisory": ADVISORY_TAG,
    }
