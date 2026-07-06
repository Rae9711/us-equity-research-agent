"""P18 Trade Candidates + Best Opportunity — Decision Agent v2 rules engine.

Scoring uses today's tradeability (pre-market gap, RS, expected return) not yesterday strength.

ADVISORY ONLY — 不构成投资建议. No auto-trading.

final_score formula (documented):
    rr_weight = clamp(risk_reward / 2, 0.5, 1.5)
    final_score = win_prob * max(expected_return_pct, 0) * rr_weight / 100
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from src.research.edges import compute_edges
from src.utils.paths import data_root
from src.utils.quote_resolve import session_observation
from src.utils.trading_calendar import prior_trading_day

CANDIDATE_SYMBOLS = ["TSLA", "NVDA", "SMH", "QQQ", "SPY", "TQQQ"]
ADVISORY_TAG = "ADVISORY — 不构成投资建议"
FINAL_SCORE_THRESHOLD = 2.5
EXTENDED_GAP_PCT = 4.0
MIN_UPSIDE_PCT = 1.0

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


def _load_prior_raw(trading_day: date) -> dict[str, Any]:
    prior_path = data_root() / "raw" / f"{prior_trading_day(trading_day).isoformat()}.json"
    if not prior_path.exists():
        return {}
    try:
        return json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _prior_day_change(
    symbol: str,
    prior_raw: dict[str, Any],
    prior_day: date,
) -> float | None:
    section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
    q = _quote(prior_raw, symbol)
    if q.get("change_pct") is not None:
        try:
            return float(q["change_pct"])
        except (TypeError, ValueError):
            pass
    obs = session_observation(symbol, prior_raw, {}, prior_day, section=section)
    if "error" not in obs:
        return obs.get("change_pct")
    return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _news_hits(raw: dict[str, Any], symbol: str) -> int:
    sym = symbol.upper()
    count = 0
    news = raw.get("news") or {}
    for article in (news.get("polygon") or []) + (news.get("rss") or []):
        ticker = str(article.get("ticker") or "").upper()
        title = str(article.get("title") or "").upper()
        if ticker == sym or sym in title:
            count += 1
    return count


def _observation(
    symbol: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date,
) -> dict[str, Any]:
    section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
    return session_observation(symbol, raw, prior_raw, trading_day, section=section)


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
    edges = (parts.get("P13") or {}).get("edges") or {}
    any_edge = any(
        (edges.get(k) or {}).get("edge") == "YES"
        for k in ("macro_edge", "index_edge", "sector_edge", "stock_edge")
    )
    if not any_edge:
        edge_j = (parts.get("P13") or {}).get("judgment") or ""
        any_edge = "Edge：YES" in edge_j or (
            "/4 YES" in edge_j and edge_j != "Edge：0/4 YES"
        )
    if total >= 2 and any_edge:
        return "Trade"
    if total >= 0:
        return "Wait"
    return "No Trade"


def _instrument(symbol: str, direction: str, p9: dict[str, Any]) -> str:
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


def _price_levels_from_obs(
    obs: dict[str, Any],
    direction: str,
    expected_high: float,
    expected_low: float,
) -> tuple[str, str, list[str]]:
    current = _safe_float(obs.get("last")) or _safe_float(obs.get("close"))
    if current is None or current <= 0:
        return "—", "—", []

    if direction == "LONG":
        entry = f"Above {current:.1f}"
        stop = f"{expected_low:.1f}"
        targets = [f"{expected_high:.1f}"]
    elif direction == "SHORT":
        entry = f"Below {current:.1f}"
        stop = f"{expected_high:.1f}"
        targets = [f"{expected_low:.1f}"]
    else:
        return "—", "—", []

    return entry, stop, targets


def _rr_numeric(upside_pct: float, downside_pct: float) -> float:
    if downside_pct <= 0:
        return 0.0
    return round(upside_pct / downside_pct, 2)


def _rr_weight(rr: float) -> float:
    return max(0.5, min(1.5, rr / 2.0))


def _score_candidate_v2(
    symbol: str,
    *,
    obs: dict[str, Any],
    prior_day_chg: float | None,
    rs_vs_qqq: float | None,
    gap_pct: float | None,
    news_count: int,
    volume_ok: bool,
    vix_chg: float | None,
    driver_type: str,
    has_news_catalyst: bool,
    q: dict[str, Any],
) -> dict[str, Any]:
    """Score one symbol for today's tradeability at ~8:00 AM."""
    current = _safe_float(obs.get("last")) or _safe_float(obs.get("close"))
    prior_close = _safe_float(obs.get("prev_close")) or _safe_float(q.get("prior_close"))
    if current is None or current <= 0:
        current = _safe_float(q.get("close")) or 1.0
    if prior_close is None:
        prior_close = current

    if gap_pct is None and prior_close:
        gap_pct = (current - prior_close) / prior_close * 100.0

    prior_high = _safe_float(q.get("high")) or current * 1.01
    prior_low = _safe_float(q.get("low")) or current * 0.99

    why_factors: list[str] = []

    win_prob = 50.0
    if rs_vs_qqq is not None:
        if rs_vs_qqq > 0.5:
            win_prob += 12
            why_factors.append(f"RS vs QQQ {rs_vs_qqq:+.2f}%")
        elif rs_vs_qqq > 0.15:
            win_prob += 6
            why_factors.append(f"RS vs QQQ {rs_vs_qqq:+.2f}%")
        elif rs_vs_qqq < -0.5:
            win_prob -= 10
            why_factors.append(f"RS 弱于 QQQ {rs_vs_qqq:+.2f}%")

    if prior_day_chg is not None and prior_day_chg > 2.0:
        if gap_pct is not None and abs(gap_pct) < EXTENDED_GAP_PCT:
            win_prob += 10
            why_factors.append(f"昨日强势 {prior_day_chg:+.1f}% 今日 gap 可控")
        elif gap_pct is not None and gap_pct >= EXTENDED_GAP_PCT and not has_news_catalyst:
            win_prob -= 8
            why_factors.append(f"昨日涨后 gap 过大 {gap_pct:+.1f}%")

    if gap_pct is not None:
        if abs(gap_pct) < 1.5:
            win_prob += 5
            why_factors.append("Gap 未过度延伸")
        elif gap_pct >= EXTENDED_GAP_PCT and not has_news_catalyst:
            win_prob -= 12
            why_factors.append(f"Extended gap {gap_pct:+.1f}%")

    if news_count >= 1:
        win_prob += 5
        why_factors.append(f"News {news_count}")
    if volume_ok:
        win_prob += 4
        why_factors.append("Volume 信号")
    if vix_chg is not None and vix_chg < -3:
        win_prob += 4
        why_factors.append("VIX 回落")
    elif vix_chg is not None and vix_chg > 5:
        win_prob -= 6
        why_factors.append("VIX 走高")

    dt = (driver_type or "").lower()
    if dt in ("momentum", "ai") and symbol in ("NVDA", "SMH", "TSLA", "TQQQ"):
        win_prob += 5
        why_factors.append(f"{driver_type} driver")

    win_prob = max(15.0, min(92.0, win_prob))

    mom_adj = 0.0
    if rs_vs_qqq is not None:
        mom_adj += rs_vs_qqq * 0.20
    if prior_day_chg is not None:
        mom_adj += prior_day_chg * 0.06
    if gap_pct is not None:
        mom_adj += gap_pct * 0.08

    # Continuation: prior strength + controlled gap → today's ER not capped by yesterday alone
    if (
        prior_day_chg is not None
        and prior_day_chg > 3.0
        and gap_pct is not None
        and abs(gap_pct) < EXTENDED_GAP_PCT
    ):
        mom_adj += min(prior_day_chg * 0.35, 5.0)

    expected_close = current * (1 + mom_adj / 100.0)
    expected_high = max(expected_close, prior_high, current * (1 + max(mom_adj, 0.5) / 100.0))
    expected_low = min(expected_close, prior_low, current * (1 - max(abs(mom_adj), 0.8) / 100.0))

    expected_return_pct = (expected_close - current) / current * 100.0

    if gap_pct is not None and gap_pct > 3.0:
        if expected_return_pct < 0.5 and not has_news_catalyst:
            expected_return_pct *= 0.3
            why_factors.append("Gap>3% 且剩余空间小")
        elif gap_pct > EXTENDED_GAP_PCT and not has_news_catalyst:
            expected_return_pct *= 0.6
            why_factors.append("Extended gap 压缩 ER")

    upside_pct = (expected_high - current) / current * 100.0
    downside_risk_pct = (current - expected_low) / current * 100.0
    if downside_risk_pct < 0.1:
        downside_risk_pct = 0.8

    risk_reward = _rr_numeric(upside_pct, downside_risk_pct)
    rr_w = _rr_weight(risk_reward)
    final_score = round(win_prob * max(expected_return_pct, 0) * rr_w / 100.0, 2)

    if expected_return_pct < MIN_UPSIDE_PCT:
        trade_action = "Pass"
        why_factors.append(f"上行空间 <{MIN_UPSIDE_PCT}%")
    elif final_score >= 8.0 and expected_return_pct >= 1.5:
        trade_action = "BUY"
    elif final_score >= FINAL_SCORE_THRESHOLD or expected_return_pct >= 1.2:
        trade_action = "Small"
    else:
        trade_action = "Pass"

    return {
        "symbol": symbol,
        "win_prob": round(win_prob, 1),
        "expected_return_pct": round(expected_return_pct, 2),
        "expected_high": round(expected_high, 2),
        "expected_low": round(expected_low, 2),
        "expected_close": round(expected_close, 2),
        "current_price": round(current, 2),
        "upside_pct": round(upside_pct, 2),
        "downside_risk_pct": round(downside_risk_pct, 2),
        "risk_reward": risk_reward,
        "final_score": final_score,
        "trade_action": trade_action,
        "trade": trade_action,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "relative_strength": round(rs_vs_qqq, 2) if rs_vs_qqq is not None else None,
        "prior_day_change_pct": round(prior_day_chg, 2) if prior_day_chg is not None else None,
        "why_factors": why_factors[:6],
        "why": " · ".join(why_factors[:4]) if why_factors else "—",
        "score": round(final_score * 10),
        "news_count": news_count,
        "advisory": True,
    }


def _pick_direction(bias: str, total: int) -> str:
    if "bear" in (bias or "").lower():
        return "SHORT"
    if "bull" in (bias or "").lower():
        return "LONG"
    return "LONG" if total >= 0 else "SHORT"


def _rank_key(row: dict[str, Any]) -> float:
    return row.get("expected_return_pct", 0) * row.get("win_prob", 0)


def _build_trade_slot(
    row: dict[str, Any],
    *,
    rank: int,
    direction: str,
    p9: dict[str, Any],
    obs: dict[str, Any],
) -> dict[str, Any]:
    entry, stop, targets = _price_levels_from_obs(
        obs,
        direction,
        row["expected_high"],
        row["expected_low"],
    )
    inst = _instrument(row["symbol"], direction, p9)
    conf = int(min(95, max(40, row["win_prob"])))
    return {
        "rank": rank,
        "symbol": row["symbol"],
        "direction": direction,
        "instrument": inst,
        "confidence": conf,
        "expected_move": f"{row['expected_return_pct']:+.2f}%",
        "expected_return_pct": row["expected_return_pct"],
        "win_prob": row["win_prob"],
        "risk_reward": row["risk_reward"],
        "final_score": row["final_score"],
        "trade_action": row["trade_action"],
        "entry": entry,
        "stop": stop,
        "target": " / ".join(targets) if targets else "—",
        "targets": targets,
        "why": row["why_factors"],
        "why_chain": row["why"],
        "expected_high": row["expected_high"],
        "expected_low": row["expected_low"],
        "expected_close": row["expected_close"],
        "current_price": row["current_price"],
        "upside_pct": row["upside_pct"],
        "why_factors": row["why_factors"],
        "advisory": True,
    }


def _index_trade_label(
    ranked: list[dict[str, Any]],
    edges: dict[str, Any],
    direction: str,
) -> str:
    index_syms = {"QQQ", "SPY", "TQQQ"}
    index_rows = [r for r in ranked if r["symbol"] in index_syms and r["trade_action"] != "Pass"]
    if not index_rows:
        return "NO TRADE"
    top = index_rows[0]
    if top["trade_action"] == "Pass":
        return "NO TRADE"
    verb = "LONG" if direction == "LONG" else "SHORT"
    return f"{verb} {top['symbol']}"


def _select_stock_picks(
    ranked: list[dict[str, Any]],
    edges: dict[str, Any],
    tradeable: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick primary/secondary/watchlist stocks — independent of P16 index gate."""
    macro_no = edges.get("macro_edge", {}).get("edge") == "NO"
    index_no = edges.get("index_edge", {}).get("edge") == "NO"
    stock_yes = edges.get("stock_edge", {}).get("edge") == "YES"
    top = ranked[0] if ranked else None
    score_ok = bool(top and top["final_score"] > FINAL_SCORE_THRESHOLD)

    if not stock_yes and not score_ok:
        return []

    if tradeable:
        if macro_no and index_no:
            stock_only = [r for r in tradeable if r["symbol"] in ("NVDA", "TSLA")]
            return (stock_only or tradeable)[:3]
        return tradeable[:3]

    if stock_yes:
        stocks = [r for r in ranked if r["symbol"] in ("NVDA", "TSLA")]
        if stocks:
            return stocks[:3]
    if score_ok and top:
        return [top]
    return []


def _decision_tree(
    ranked: list[dict[str, Any]],
    edges: dict[str, Any],
    *,
    direction: str,
    p9: dict[str, Any],
    obs_by_sym: dict[str, dict[str, Any]],
    p16_gate: str,
) -> dict[str, Any]:
    tradeable = [r for r in ranked if r["trade_action"] in ("BUY", "Small")]
    tradeable.sort(key=lambda r: (r["final_score"], _rank_key(r)), reverse=True)

    picks = _select_stock_picks(ranked, edges, tradeable)
    threshold_msg: str | None = None
    if not picks:
        threshold_msg = "今日无任何标的达到交易阈值"

    # P16 No Trade / Wait gates index exposure only — stock picks stay independent.
    if p16_gate in ("No Trade", "Wait"):
        index_trade = "NO TRADE"
    else:
        index_trade = _index_trade_label(ranked, edges, direction)

    slots: dict[str, Any] = {
        "primary": None,
        "secondary": None,
        "watchlist": None,
        "index_trade": index_trade,
        "threshold_message": threshold_msg,
        "stock_trades": [],
        "advisory": True,
    }

    slot_names = ("primary", "secondary", "watchlist")
    for i, row in enumerate(picks):
        slot = _build_trade_slot(
            row,
            rank=i + 1,
            direction=direction,
            p9=p9,
            obs=obs_by_sym.get(row["symbol"], {}),
        )
        slots[slot_names[i]] = slot
        slots["stock_trades"].append({
            "rank": slot["rank"],
            "symbol": slot["symbol"],
            "direction": slot["direction"],
            "confidence": slot["confidence"],
            "expected_move": slot["expected_move"],
            "why": slot["why_chain"],
        })

    return slots


def _to_best_opportunity(
    primary: dict[str, Any] | None,
    *,
    threshold_message: str | None,
    p16_gate: str,
    duration: str = "Intraday",
) -> dict[str, Any]:
    if primary:
        dir_word = "buy" if primary["direction"] == "LONG" else "sell"
        one_liner = (
            f"Today {dir_word} {primary['symbol']} {primary['entry'].lower()}, "
            f"expected {primary['expected_move']}"
        )
        return {
            "direction": primary["direction"],
            "symbol": primary["symbol"],
            "instrument": primary["instrument"],
            "confidence": primary["confidence"],
            "entry": primary["entry"],
            "stop": primary["stop"],
            "target": primary["target"],
            "targets": primary.get("targets") or [],
            "duration": duration,
            "why": primary.get("why") or [],
            "why_chain": primary.get("why_chain", "—"),
            "avoid": [],
            "one_liner": one_liner,
            "p16_gate": p16_gate,
            "expected_return_pct": primary.get("expected_return_pct"),
            "win_prob": primary.get("win_prob"),
            "risk_reward": primary.get("risk_reward"),
            "final_score": primary.get("final_score"),
            "advisory": True,
        }

    why = [threshold_message] if threshold_message else ["今日无任何标的达到交易阈值"]
    return {
        "direction": "NO TRADE",
        "symbol": "—",
        "instrument": "—",
        "confidence": 45,
        "entry": "—",
        "stop": "—",
        "target": "—",
        "targets": [],
        "duration": "—",
        "why": why,
        "why_chain": why[0],
        "avoid": [],
        "one_liner": why[0],
        "p16_gate": p16_gate,
        "threshold_message": threshold_message,
        "advisory": True,
    }


def compute_trade_decision(
    raw: dict[str, Any],
    *,
    rule_bundle: dict[str, Any],
    parts: dict[str, Any],
    regime_label: str = "Range",
    regime_confidence: float = 0.5,
    edges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank trade candidates and pick best trades for morning.json (v2)."""
    del regime_label, regime_confidence  # kept for API compat

    trading_date = str(raw.get("trading_date") or "")
    trading_day: date | None = None
    if trading_date:
        try:
            trading_day = date.fromisoformat(trading_date)
        except ValueError:
            trading_day = None
    prior_raw = _load_prior_raw(trading_day) if trading_day else {}
    prior_day = prior_trading_day(trading_day) if trading_day else None

    bias = rule_bundle.get("bias") or "Neutral"
    total = int(rule_bundle.get("total") or 0)
    driver_type = rule_bundle.get("driver_type") or ""
    p9 = parts.get("P9") or {}
    catalysts = rule_bundle.get("catalysts_today") or []

    market_q = (raw.get("market") or {}).get("quotes") or {}
    vix_q = market_q.get("^VIX") or market_q.get("VIX") or {}
    vix_chg = _safe_float(vix_q.get("change_pct"))

    sym_pcts: dict[str, float | None] = {}
    obs_by_sym: dict[str, dict[str, Any]] = {}
    if trading_day:
        for sym in CANDIDATE_SYMBOLS:
            obs = _observation(sym, raw, prior_raw, trading_day)
            obs_by_sym[sym] = obs
            if "error" not in obs:
                sym_pcts[sym] = obs.get("change_pct")

    qqq_pct = sym_pcts.get("QQQ")
    smh_pct = sym_pcts.get("SMH")
    spy_pct = sym_pcts.get("SPY")

    if edges is None:
        edges = compute_edges(
            raw,
            catalysts_today=catalysts,
            qqq_pct=qqq_pct,
            smh_pct=smh_pct,
            spy_pct=spy_pct,
            sym_pcts=sym_pcts,
            driver_type=driver_type,
        )

    p16_gate = _p16_gate(parts, total)
    direction = _pick_direction(bias, total)
    duration = "Intraday" if p9.get("zero_dte") == "Yes" else (
        "Swing" if abs(total) >= 4 else "Intraday"
    )

    ranked: list[dict[str, Any]] = []
    for sym in CANDIDATE_SYMBOLS:
        q = _quote(raw, sym)
        obs = obs_by_sym.get(sym, {})
        sym_pct = sym_pcts.get(sym)
        prior_chg = _prior_day_change(sym, prior_raw, prior_day) if prior_day else None
        rs = (sym_pct - qqq_pct) if sym_pct is not None and qqq_pct is not None else None
        gap_pct = obs.get("gap_pct")
        prior_close = _safe_float(obs.get("prev_close")) or _safe_float(q.get("prior_close"))
        open_px = _safe_float(obs.get("open")) or _safe_float(q.get("open"))
        if open_px and prior_close:
            gap_from_open = (open_px - prior_close) / prior_close * 100.0
            if gap_pct is None or (
                sym_pct is not None and abs(float(gap_pct) - float(sym_pct)) < 0.15
            ):
                gap_pct = round(gap_from_open, 2)
        news_n = _news_hits(raw, sym)
        vol_ok = bool(q.get("volume")) or str(q.get("session_type")) == "premarket"
        has_catalyst = news_n >= 1 or bool(catalysts)

        row = _score_candidate_v2(
            sym,
            obs=obs,
            prior_day_chg=prior_chg,
            rs_vs_qqq=rs,
            gap_pct=gap_pct,
            news_count=news_n,
            volume_ok=vol_ok,
            vix_chg=vix_chg,
            driver_type=driver_type,
            has_news_catalyst=has_catalyst,
            q=q,
        )
        ranked.append(row)

    ranked.sort(key=lambda r: (r["final_score"], _rank_key(r)), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    best_trades = _decision_tree(
        ranked,
        edges,
        direction=direction,
        p9=p9,
        obs_by_sym=obs_by_sym,
        p16_gate=p16_gate,
    )
    best_opportunity = _to_best_opportunity(
        best_trades.get("primary"),
        threshold_message=best_trades.get("threshold_message"),
        p16_gate=p16_gate,
        duration=duration,
    )

    return {
        "edges": edges,
        "trade_candidates": ranked,
        "best_trades": best_trades,
        "best_opportunity": best_opportunity,
        "bias_stars": _bias_stars(bias),
        "index_trade": best_trades.get("index_trade"),
        "stock_trades": best_trades.get("stock_trades") or [],
    }


def build_p18_part(
    trade_candidates: list[dict[str, Any]],
    best: dict[str, Any],
    *,
    best_trades: dict[str, Any] | None = None,
    edges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lines = [
        "| Rank | Symbol | Win% | ER% | R:R | Score | Trade |",
        "|------|--------|------|-----|-----|-------|-------|",
    ]
    for row in trade_candidates:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['win_prob']} | "
            f"{row['expected_return_pct']} | {row['risk_reward']} | "
            f"{row['final_score']} | {row['trade_action']} |"
        )
    body = "\n".join(lines)
    primary = (best_trades or {}).get("primary")
    if primary:
        judgment = (
            f"Primary：{primary.get('symbol')} · {primary.get('direction')} · "
            f"ER {primary.get('expected_move')} · conf {primary.get('confidence')}%"
        )
    else:
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
        "best_trades": best_trades,
        "edges": edges,
    }


def build_executive_summary(
    *,
    bias: str,
    bias_stars: str,
    driver_type: str,
    driver: str,
    best: dict[str, Any],
    best_trades: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = (best_trades or {}).get("primary")
    threshold_msg = (best_trades or {}).get("threshold_message")

    if primary:
        best_trade = (
            f"{primary['symbol']} · {primary['direction']} · "
            f"{primary.get('instrument', '—')} · ER {primary.get('expected_move')}"
        )
        one_liner = best.get("one_liner") or (
            f"Today LONG {primary['symbol']} {primary.get('entry', '')}, "
            f"expected {primary.get('expected_move')}"
        )
        confidence = primary.get("confidence")
        entry = primary.get("entry", "—")
        stop = primary.get("stop", "—")
        target = primary.get("target", "—")
        why_chain = primary.get("why_chain", "—")
    elif threshold_msg:
        best_trade = threshold_msg
        one_liner = threshold_msg
        confidence = None
        entry = stop = target = "—"
        why_chain = threshold_msg
    else:
        best_trade = (
            f"{best.get('symbol', '—')} · {best.get('direction', '—')} · "
            f"{best.get('instrument', '—')}"
        )
        one_liner = best.get("one_liner", "—")
        confidence = best.get("confidence")
        entry = best.get("entry", "—")
        stop = best.get("stop", "—")
        target = best.get("target", "—")
        why_chain = best.get("why_chain", "—")

    return {
        "bias": bias,
        "bias_stars": bias_stars,
        "driver_type": driver_type,
        "driver": driver,
        "driver_display": " — ".join(x for x in (driver_type, driver) if x) or "—",
        "best_trade": best_trade,
        "primary_trade": primary,
        "confidence": confidence,
        "entry": entry,
        "stop": stop,
        "target": target,
        "why_chain": why_chain,
        "one_liner": one_liner,
        "threshold_message": threshold_msg,
        "index_trade": (best_trades or {}).get("index_trade"),
        "advisory": ADVISORY_TAG,
    }
