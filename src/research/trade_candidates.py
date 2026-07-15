"""P18 Trade Candidates + Best Opportunity — Decision Agent v2 rules engine.

Scoring uses today's tradeability (pre-market gap, RS, expected return) not yesterday strength.

ADVISORY ONLY — 不构成投资建议. No auto-trading.

final_score formula (documented):
    rr_weight = clamp(risk_reward / 2, 0.5, 1.5)
    final_score = win_prob * max(expected_return_pct, 0) * rr_weight / 100
"""

from __future__ import annotations

import json
from datetime import date, time
from typing import Any, Literal

from src.research.decision_transparency import (
    add_win_prob_delta,
    build_decision_transparency,
    build_rr_display,
    build_top5_board,
    enrich_trade_slot,
    finalize_win_prob_breakdown,
    init_win_prob_breakdown,
)
from src.research.edges import compute_edges
from src.research.level_sources import (
    build_level_reasons,
    compute_anchors,
    derive_trade_levels,
    trade_levels_valid,
)
from src.research.macro_calendar import has_geopolitical_risk
from src.utils.paths import data_root
from src.utils.quote_resolve import session_observation
from src.utils.trading_calendar import prior_trading_day

CANDIDATE_SYMBOLS = [
    "TSLA", "NVDA", "AMD", "MU", "AVGO", "META", "ARM",
    "SMH", "QQQ", "SPY", "TQQQ",
]
ADVISORY_TAG = "ADVISORY — 不构成投资建议"
FINAL_SCORE_THRESHOLD = 2.5
EXTENDED_GAP_PCT = 4.0
MIN_UPSIDE_PCT = 1.0

_STOCK_SYMBOLS = frozenset({"TSLA", "NVDA", "AMD", "MU", "AVGO", "META", "ARM"})
_SEMI_SYMBOLS = frozenset({"NVDA", "AMD", "MU", "AVGO", "ARM", "SMH"})
# Liquid names preferred for multi-day / swing positions (not 0DTE).
_SWING_PREFERRED = ("NVDA", "META", "QQQ", "AVGO", "AMD", "TSLA", "SMH", "SPY")
_SWING_MIN_QUALITY = 2.0  # independent of intraday BUY/Small threshold

_SYMBOL_SECTION: dict[str, str] = {
    "QQQ": "market",
    "SPY": "market",
    "TQQQ": "market",
    "SMH": "sector",
    "NVDA": "stocks",
    "TSLA": "stocks",
    "AMD": "stocks",
    "MU": "stocks",
    "AVGO": "stocks",
    "META": "stocks",
    "ARM": "stocks",
}


def _has_market_data(obs: dict[str, Any], q: dict[str, Any]) -> bool:
    if obs.get("error") and not q:
        return False
    current = (
        _safe_float(obs.get("last"))
        or _safe_float(obs.get("close"))
        or _safe_float(q.get("close"))
        or _safe_float(q.get("last"))
    )
    return current is not None and current > 0


def _pass_stub_row(symbol: str, reason: str = "No quote data") -> dict[str, Any]:
    """Minimal ranked row for symbols lacking usable market data.

    Must include every key `_build_trade_slot` / transparency expect so universe
    expansion (extra tickers without quotes) cannot KeyError on Pass stubs.
    """
    return {
        "symbol": symbol,
        "win_prob": 50.0,
        "win_prob_breakdown": None,
        "expected_return_pct": 0.0,
        "expected_high": 0.0,
        "expected_low": 0.0,
        "expected_close": 0.0,
        "current_price": None,
        "upside_pct": 0.0,
        "downside_risk_pct": 0.0,
        "risk_reward": 0.0,
        "rr_display": None,
        "final_score": 0.0,
        "trade_action": "Pass",
        "trade": "Pass",
        "gap_pct": None,
        "relative_strength": None,
        "relative_strength_vs_smh": None,
        "relative_weakness_score": 0,
        "prior_day_change_pct": None,
        "why_factors": [reason],
        "why": reason,
        "score": 0,
        "news_count": 0,
        "factor_breakdown": {},
        "edge_type": None,
        "score_formula_display": None,
        "rr_weight": None,
        "why_vs_runner_up": "—",
        "advisory": True,
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
    *,
    as_of_et: time | Literal["now"] | None = None,
) -> dict[str, Any]:
    section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
    return session_observation(
        symbol,
        raw,
        prior_raw,
        trading_day,
        section=section,
        prefer_raw=True,
        as_of_et=as_of_et,
    )


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
        return "Stock" if symbol in _STOCK_SYMBOLS else "ETF"
    if direction == "SHORT":
        if zero_dte and buy_put:
            return f"{symbol} 0DTE Put"
        if buy_options and buy_put:
            return f"{symbol} Put"
        # Stock shorts are not 0DTE options — label underlying correctly
        return "Stock" if symbol in _STOCK_SYMBOLS else "ETF"
    return "—"


def _trade_horizon(
    *,
    direction: str,
    p9: dict[str, Any],
    total: int,
    instrument: str | None = None,
) -> str:
    """Explicit holding horizon: 0DTE | Intraday | Swing.

    0DTE only when P9 actually authorizes zero_dte options for this direction.
    Never imply stock/ETF shorts are 0DTE Puts.
    """
    if direction == "NO TRADE":
        return "—"
    buy_options = p9.get("buy_options") == "Yes"
    zero_dte = p9.get("zero_dte") == "Yes"
    inst = instrument or ""
    if zero_dte and buy_options and "0DTE" in inst:
        return "0DTE"
    if (
        zero_dte
        and buy_options
        and (
            (direction == "LONG" and p9.get("buy_call") == "Yes")
            or (direction == "SHORT" and p9.get("buy_put") == "Yes")
        )
    ):
        return "0DTE"
    if abs(int(total or 0)) >= 4:
        return "Swing"
    return "Intraday"


def _rr_weight_display(rr: float) -> float:
    return round(_rr_weight(rr), 2)


def _score_formula_display(win_prob: float, expected_return_pct: float, rr: float) -> str:
    rr_w = _rr_weight_display(rr)
    return (
        f"final = win_prob({win_prob:.0f}%) × ER({expected_return_pct:.1f}%) "
        f"× rr_weight({rr_w}) / 100"
    )


def _beta_proxy(sym_pct: float | None, qqq_pct: float | None) -> float | None:
    if sym_pct is None or qqq_pct is None or abs(qqq_pct) < 0.05:
        return None
    return round(sym_pct / qqq_pct, 2)


def _geo_context_penalty(
    symbol: str,
    direction: str,
    *,
    macro_calendar: dict[str, Any] | None,
    driver_tree: dict[str, Any] | None,
) -> tuple[float, str | None]:
    """Penalize trades that fight geopolitical/macro context (e.g. blind NVDA short on Iran day)."""
    macro_calendar = macro_calendar or {}
    driver_tree = driver_tree or {}
    primary = driver_tree.get("primary") or {}
    primary_type = str(primary.get("type") or "")
    primary_label = str(primary.get("label") or "").lower()
    geo = has_geopolitical_risk(macro_calendar) or primary_type == "Political" or "geo" in primary_label

    if not geo:
        return 0.0, None

    sym = symbol.upper()
    if direction == "SHORT" and sym in ("NVDA", "SMH", "TQQQ"):
        return 18.0, f"Geo-risk day — avoid blind {sym} short vs escalation"
    if direction == "LONG" and sym in ("XLE", "USO"):
        return 0.0, None
    if direction == "LONG" and sym in ("NVDA", "SMH") and "oil" in primary_label:
        return 8.0, "Oil shock day — semis long needs confirmation"
    return 0.0, None


def _catalyst_for_symbol(symbol: str, macro_calendar: dict[str, Any] | None) -> str:
    macro_calendar = macro_calendar or {}
    cats = macro_calendar.get("catalysts") or []
    if not cats:
        return "—"
    names = [c.get("name") for c in cats[:2] if c.get("name")]
    sym = symbol.upper()
    if sym in _SEMI_SYMBOLS and any(
        any(k in n for k in ("Chip", "AI", "Semi", "HBM", "Memory"))
        for n in names
    ):
        return " / ".join(names)
    if sym == "META" and any("Social" in n or "Ad" in n for n in names):
        return " / ".join(names)
    return " · ".join(names) if names else "—"


def _invalidation_for_slot(slot: dict[str, Any], macro_calendar: dict[str, Any] | None) -> str:
    stop = slot.get("stop_price")
    sym = slot.get("symbol", "")
    direction = slot.get("direction", "")
    parts: list[str] = []
    if stop is not None:
        if direction == "LONG":
            parts.append(f"Close below stop ${stop}")
        elif direction == "SHORT":
            parts.append(f"Close above stop ${stop}")
    geo = has_geopolitical_risk(macro_calendar or {})
    if geo:
        parts.append("Geopolitical escalation invalidates mean-reversion")
    if not parts:
        return f"{sym} thesis breaks on stop breach"
    return " · ".join(parts)


def _extended_gap_penalty(gap_pct: float | None, has_news_catalyst: bool) -> float:
    if gap_pct is None or has_news_catalyst:
        return 0.0
    if gap_pct >= EXTENDED_GAP_PCT:
        return 12.0
    if gap_pct > 3.0:
        return 6.0
    return 0.0


def _infer_edge_type(
    symbol: str,
    *,
    rs_vs_qqq: float | None,
    driver_type: str,
    edges: dict[str, Any] | None,
) -> str:
    edges = edges or {}
    if symbol in _STOCK_SYMBOLS and edges.get("stock_edge", {}).get("edge") == "YES":
        return "Stock Edge"
    if edges.get("sector_edge", {}).get("edge") == "YES" and symbol in _SEMI_SYMBOLS:
        return "Sector Edge"
    if rs_vs_qqq is not None and rs_vs_qqq < -0.3:
        return "Relative Weakness"
    if rs_vs_qqq is not None and rs_vs_qqq > 0.5:
        return "Relative Strength"
    dt = (driver_type or "").lower()
    if dt in ("momentum", "ai"):
        return "Momentum"
    if symbol in ("QQQ", "SPY", "TQQQ"):
        return "Index Edge"
    return "Range"


def _why_vs_runner_up(top: dict[str, Any], runner_up: dict[str, Any] | None) -> str:
    if not runner_up:
        return "—"
    parts: list[str] = []
    top_rs = top.get("relative_strength")
    run_rs = runner_up.get("relative_strength")
    if top_rs is not None and run_rs is not None:
        parts.append(
            f"{top['symbol']} RS {top_rs:+.1f}% vs {runner_up['symbol']} {run_rs:+.1f}%"
        )
    top_er = top.get("expected_return_pct")
    run_er = runner_up.get("expected_return_pct")
    if top_er is not None and run_er is not None and top_er > run_er:
        parts.append(f"ER {top_er:.1f}% vs {run_er:.1f}%")
    top_wp = top.get("win_prob")
    run_wp = runner_up.get("win_prob")
    if top_wp is not None and run_wp is not None and top_wp > run_wp:
        parts.append(f"Win% {top_wp:.0f} vs {run_wp:.0f}")
    beta = top.get("factor_breakdown", {}).get("beta_proxy")
    run_beta = runner_up.get("factor_breakdown", {}).get("beta_proxy")
    if beta is not None and run_beta is not None and beta > run_beta:
        parts.append(f"higher beta ({beta:.1f} vs {run_beta:.1f})")
    if top_rs is not None and top_rs < -0.2:
        parts.append("relative weakness play")
    elif top_rs is not None and top_rs > 0.3:
        parts.append("relative strength")
    return "；".join(parts) if parts else f"Score {top.get('final_score')} vs {runner_up.get('final_score')}"


def _build_factor_breakdown(
    *,
    rs_vs_qqq: float | None,
    rs_vs_smh: float | None = None,
    gap_pct: float | None,
    beta: float | None,
    volume_ok: bool,
    news_count: int,
    extended_gap_penalty: float,
    vix_chg: float | None,
    prior_day_chg: float | None,
    relative_weakness_score: float | None = None,
) -> dict[str, Any]:
    return {
        "relative_strength_vs_qqq": round(rs_vs_qqq, 2) if rs_vs_qqq is not None else None,
        "relative_strength_vs_smh": round(rs_vs_smh, 2) if rs_vs_smh is not None else None,
        "relative_weakness_score": (
            round(relative_weakness_score, 2) if relative_weakness_score is not None else None
        ),
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "beta_proxy": beta,
        "volume_signal": volume_ok,
        "news_count": news_count,
        "extended_gap_penalty": round(extended_gap_penalty, 1),
        "vix_change_pct": round(vix_chg, 2) if vix_chg is not None else None,
        "prior_day_change_pct": round(prior_day_chg, 2) if prior_day_chg is not None else None,
    }


def _rr_numeric(upside_pct: float, downside_pct: float) -> float:
    if downside_pct <= 0:
        return 0.0
    return round(upside_pct / downside_pct, 2)


def _expected_return_from_prices(
    direction: str,
    entry_price: float | None,
    target_price: float | None,
) -> float | None:
    """Single source of truth: ER from entry/target prices."""
    if entry_price is None or target_price is None or entry_price <= 0:
        return None
    if direction == "LONG":
        return round((target_price - entry_price) / entry_price * 100.0, 2)
    if direction == "SHORT":
        return round((entry_price - target_price) / entry_price * 100.0, 2)
    return None


def _return_calculation_string(
    direction: str,
    entry_price: float,
    target_price: float,
    expected_return_pct: float,
) -> str:
    if direction == "LONG":
        return (
            f"({target_price}-{entry_price})/{entry_price}"
            f"={expected_return_pct:.2f}%"
        )
    if direction == "SHORT":
        return (
            f"({entry_price}-{target_price})/{entry_price}"
            f"={expected_return_pct:.2f}%"
        )
    return ""


def _risk_reward_from_levels(
    entry_price: float | None,
    stop_price: float | None,
    expected_return_pct: float,
) -> float | None:
    if entry_price is None or stop_price is None or entry_price <= 0:
        return None
    risk_pct = abs(entry_price - stop_price) / entry_price * 100.0
    if risk_pct <= 0:
        return None
    return round(expected_return_pct / risk_pct, 2)


def _apply_price_based_return(
    slot: dict[str, Any],
    *,
    direction: str,
    heuristic_er: float,
) -> dict[str, Any]:
    """Reconcile displayed ER / R:R with entry, target, and stop prices."""
    entry_px = _safe_float(slot.get("entry_price"))
    target_px = _safe_float(slot.get("target_price"))
    stop_px = _safe_float(slot.get("stop_price"))
    price_er = _expected_return_from_prices(direction, entry_px, target_px)
    if price_er is None:
        slot["expected_return_pct"] = heuristic_er
        slot["expected_move"] = f"{heuristic_er:+.2f}%"
        return slot

    slot["expected_return_pct"] = price_er
    slot["expected_move"] = f"{price_er:+.2f}%"
    slot["return_calculation"] = _return_calculation_string(
        direction, entry_px, target_px, price_er
    )
    slot["heuristic_expected_return_pct"] = heuristic_er

    rr_from_stop = _risk_reward_from_levels(entry_px, stop_px, price_er)
    if rr_from_stop is not None:
        slot["risk_reward"] = rr_from_stop
        risk_pct = abs(entry_px - stop_px) / entry_px * 100.0 if entry_px else None
        slot["rr_display"] = build_rr_display(
            reward_pct=price_er,
            risk_pct=risk_pct,
            reward_risk_ratio=rr_from_stop,
        )

    sym = slot.get("symbol", "")
    if direction == "LONG":
        slot["target_action"] = "卖出/获利"
        slot["trade_summary_cn"] = (
            f"做多 {sym}：在 {entry_px} 附近入场，目标 {target_px}，预期 {price_er:+.2f}%"
        )
    elif direction == "SHORT":
        slot["target_action"] = "买入/平空仓"
        slot["trade_summary_cn"] = (
            f"做空 {sym}：在 {entry_px} 附近入场，目标 {target_px}，预期 {price_er:+.2f}%"
        )
    return slot


def _enforce_level_invariants(slot: dict[str, Any]) -> dict[str, Any]:
    """Reject slots whose entry/stop/target geometry or price ER is inconsistent.

    LONG: stop < entry ≤ target; SHORT: target ≤ entry < stop.
    Actionable BUY/Small requires ER ≥ MIN_UPSIDE_PCT after price reconciliation.
    """
    direction = str(slot.get("direction") or "")
    entry_px = _safe_float(slot.get("entry_price"))
    stop_px = _safe_float(slot.get("stop_price"))
    target_px = _safe_float(slot.get("target_price"))
    entry_zone = slot.get("entry_zone")
    if not isinstance(entry_zone, dict):
        entry_zone = None

    levels_ok = slot.get("levels_valid")
    if levels_ok is None:
        levels_ok = trade_levels_valid(
            direction,
            entry_price=entry_px,
            stop_price=stop_px,
            target_price=target_px,
            entry_zone=entry_zone,
        )
    slot["levels_valid"] = bool(levels_ok)

    er = _safe_float(slot.get("expected_return_pct"))
    reasons: list[str] = []
    if not levels_ok:
        reasons.append("Levels invalid: entry/stop/target geometry")
    if er is not None and er < 0:
        reasons.append(f"Price ER {er:+.2f}% < 0")
    if er is not None and er < MIN_UPSIDE_PCT:
        reasons.append(f"Price ER {er:.2f}% < {MIN_UPSIDE_PCT}%")

    actionable = slot.get("trade_action") in ("BUY", "Small")
    if actionable and reasons:
        slot["trade_action"] = "Pass"
        slot["trade"] = "Pass"
        why = list(slot.get("why_factors") or [])
        for r in reasons:
            if r not in why:
                why.append(r)
        slot["why_factors"] = why[:8]
        slot["why"] = " · ".join(why[:4]) if why else slot.get("why", "—")
        slot["invalid_levels_reason"] = reasons[0]
    return slot


def _slot_is_actionable(slot: dict[str, Any]) -> bool:
    return (
        slot.get("trade_action") in ("BUY", "Small")
        and slot.get("levels_valid", True)
        and (_safe_float(slot.get("expected_return_pct")) or 0) >= MIN_UPSIDE_PCT
    )


def _rr_weight(rr: float) -> float:
    return max(0.5, min(1.5, rr / 2.0))


def _relative_weakness_score(
    rs_vs_qqq: float | None,
    rs_vs_smh: float | None,
) -> float | None:
    """Higher = weaker vs benchmarks (better short candidate)."""
    parts: list[float] = []
    if rs_vs_qqq is not None:
        parts.append(-rs_vs_qqq)
    if rs_vs_smh is not None:
        parts.append(-rs_vs_smh * 0.6)
    if not parts:
        return None
    return round(sum(parts), 2)


def _apply_rs_win_prob(
    wp: dict[str, float],
    why_factors: list[str],
    *,
    rs_vs_qqq: float | None,
    rs_vs_smh: float | None,
    symbol: str,
    is_short: bool,
) -> None:
    """Direction-aware RS adjustments — shorts reward weakness, longs reward strength."""
    if rs_vs_qqq is not None:
        if is_short:
            if rs_vs_qqq < -0.5:
                add_win_prob_delta(wp, "rs", 12)
                why_factors.append(f"RS弱于QQQ {rs_vs_qqq:+.2f}%")
            elif rs_vs_qqq < -0.15:
                add_win_prob_delta(wp, "rs", 6)
                why_factors.append(f"RS弱于QQQ {rs_vs_qqq:+.2f}%")
            elif rs_vs_qqq > 0.5:
                add_win_prob_delta(wp, "rs", -12)
                why_factors.append(f"板块内相对强势 {rs_vs_qqq:+.2f}%")
            elif rs_vs_qqq > 0.15:
                add_win_prob_delta(wp, "rs", -6)
        else:
            if rs_vs_qqq > 0.5:
                add_win_prob_delta(wp, "rs", 12)
                why_factors.append(f"RS vs QQQ {rs_vs_qqq:+.2f}%")
            elif rs_vs_qqq > 0.15:
                add_win_prob_delta(wp, "rs", 6)
                why_factors.append(f"RS vs QQQ {rs_vs_qqq:+.2f}%")
            elif rs_vs_qqq < -0.5:
                add_win_prob_delta(wp, "rs", -10)
                why_factors.append(f"RS 弱于 QQQ {rs_vs_qqq:+.2f}%")

    if rs_vs_smh is not None and symbol in _SEMI_SYMBOLS:
        if is_short:
            if rs_vs_smh < -0.5:
                add_win_prob_delta(wp, "rs", 8)
                why_factors.append(f"RS弱于SMH {rs_vs_smh:+.2f}%")
            elif rs_vs_smh < -0.15:
                add_win_prob_delta(wp, "rs", 4)
            elif rs_vs_smh > 0.5:
                add_win_prob_delta(wp, "rs", -10)
                why_factors.append(f"半导体内领涨 {rs_vs_smh:+.2f}%")
        else:
            if rs_vs_smh > 0.5:
                add_win_prob_delta(wp, "rs", 6)
                why_factors.append(f"RS vs SMH {rs_vs_smh:+.2f}%")
            elif rs_vs_smh < -0.5:
                add_win_prob_delta(wp, "rs", -6)


def _score_candidate_v2(
    symbol: str,
    *,
    obs: dict[str, Any],
    prior_day_chg: float | None,
    rs_vs_qqq: float | None,
    rs_vs_smh: float | None = None,
    gap_pct: float | None,
    news_count: int,
    volume_ok: bool,
    vix_chg: float | None,
    driver_type: str,
    has_news_catalyst: bool,
    q: dict[str, Any],
    qqq_pct: float | None = None,
    edges: dict[str, Any] | None = None,
    direction: str = "LONG",
    macro_calendar: dict[str, Any] | None = None,
    driver_tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one symbol for today's tradeability at ~8:00 AM."""
    is_short = direction == "SHORT"
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

    wp = init_win_prob_breakdown()
    _apply_rs_win_prob(
        wp,
        why_factors,
        rs_vs_qqq=rs_vs_qqq,
        rs_vs_smh=rs_vs_smh,
        symbol=symbol,
        is_short=is_short,
    )

    if prior_day_chg is not None and prior_day_chg > 2.0:
        if gap_pct is not None and abs(gap_pct) < EXTENDED_GAP_PCT:
            if not is_short:
                add_win_prob_delta(wp, "trend", 10)
                why_factors.append(f"昨日强势 {prior_day_chg:+.1f}% 今日 gap 可控")
            else:
                add_win_prob_delta(wp, "trend", -6)
                why_factors.append(f"昨日强势 {prior_day_chg:+.1f}% 做空逆风")
        elif gap_pct is not None and gap_pct >= EXTENDED_GAP_PCT and not has_news_catalyst:
            add_win_prob_delta(wp, "trend", -8 if not is_short else 4)
            why_factors.append(f"昨日涨后 gap 过大 {gap_pct:+.1f}%")

    if prior_day_chg is not None and prior_day_chg < -2.0 and is_short:
        if gap_pct is not None and abs(gap_pct) < EXTENDED_GAP_PCT:
            add_win_prob_delta(wp, "trend", 8)
            why_factors.append(f"昨日弱势 {prior_day_chg:+.1f}% 延续下行")

    if gap_pct is not None:
        if abs(gap_pct) < 1.5:
            add_win_prob_delta(wp, "gap", 5)
            why_factors.append("Gap 未过度延伸")
        elif gap_pct >= EXTENDED_GAP_PCT and not has_news_catalyst:
            if is_short and gap_pct > 0:
                add_win_prob_delta(wp, "gap", 6)
                why_factors.append(f"Extended gap {gap_pct:+.1f}% 回落空间")
            else:
                add_win_prob_delta(wp, "gap", -12)
                why_factors.append(f"Extended gap {gap_pct:+.1f}%")

    if news_count >= 1:
        if is_short:
            add_win_prob_delta(wp, "catalyst", -5)
            why_factors.append(f"News {news_count} (利空做空)")
        else:
            add_win_prob_delta(wp, "catalyst", 5)
            why_factors.append(f"News {news_count}")
    if volume_ok:
        add_win_prob_delta(wp, "volume", 4)
        why_factors.append("Volume 信号")
    if vix_chg is not None and vix_chg < -3:
        if not is_short:
            add_win_prob_delta(wp, "macro", 4)
            why_factors.append("VIX 回落")
        else:
            add_win_prob_delta(wp, "macro", -3)
            why_factors.append("VIX 回落 (做空逆风)")
    elif vix_chg is not None and vix_chg > 5:
        if is_short:
            add_win_prob_delta(wp, "macro", 6)
            why_factors.append("VIX 走高")
        else:
            add_win_prob_delta(wp, "macro", -6)
            why_factors.append("VIX 走高")

    dt = (driver_type or "").lower()
    if dt in ("momentum", "ai") and symbol in (*_SEMI_SYMBOLS, "TSLA", "TQQQ"):
        if is_short:
            add_win_prob_delta(wp, "catalyst", -4)
            why_factors.append(f"{driver_type} driver (做空逆风)")
        else:
            add_win_prob_delta(wp, "catalyst", 5)
            why_factors.append(f"{driver_type} driver")

    geo_penalty, geo_reason = _geo_context_penalty(
        symbol,
        direction,
        macro_calendar=macro_calendar,
        driver_tree=driver_tree,
    )
    if geo_penalty:
        add_win_prob_delta(wp, "macro", -geo_penalty)
        if geo_reason:
            why_factors.append(geo_reason)

    raw_win_prob = sum(wp.values())
    win_prob = max(15.0, min(92.0, raw_win_prob))
    win_prob_breakdown = finalize_win_prob_breakdown(wp, clamped=win_prob)

    mom_adj = 0.0
    if rs_vs_qqq is not None:
        mom_adj += (-rs_vs_qqq if is_short else rs_vs_qqq) * 0.20
    if rs_vs_smh is not None and symbol in _SEMI_SYMBOLS:
        mom_adj += (-rs_vs_smh if is_short else rs_vs_smh) * 0.12
    if prior_day_chg is not None:
        mom_adj += (-prior_day_chg if is_short else prior_day_chg) * 0.06
    if gap_pct is not None:
        mom_adj += (-gap_pct if is_short else gap_pct) * 0.08

    # Continuation: prior strength + controlled gap → today's ER not capped by yesterday alone
    if (
        not is_short
        and prior_day_chg is not None
        and prior_day_chg > 3.0
        and gap_pct is not None
        and abs(gap_pct) < EXTENDED_GAP_PCT
    ):
        mom_adj += min(prior_day_chg * 0.35, 5.0)
    if (
        is_short
        and prior_day_chg is not None
        and prior_day_chg < -3.0
        and gap_pct is not None
        and abs(gap_pct) < EXTENDED_GAP_PCT
    ):
        mom_adj += min(abs(prior_day_chg) * 0.35, 5.0)

    expected_close = current * (1 + (-mom_adj if is_short else mom_adj) / 100.0)
    if is_short:
        expected_high = max(current * (1 + max(abs(mom_adj), 0.5) / 100.0), prior_high, current)
        expected_low = min(expected_close, prior_low, current * (1 - max(mom_adj, 0.8) / 100.0))
        expected_return_pct = (current - expected_close) / current * 100.0
        upside_pct = (current - expected_low) / current * 100.0
        downside_risk_pct = (expected_high - current) / current * 100.0
    else:
        expected_high = max(expected_close, prior_high, current * (1 + max(mom_adj, 0.5) / 100.0))
        expected_low = min(expected_close, prior_low, current * (1 - max(abs(mom_adj), 0.8) / 100.0))
        expected_return_pct = (expected_close - current) / current * 100.0
        upside_pct = (expected_high - current) / current * 100.0
        downside_risk_pct = (current - expected_low) / current * 100.0

    if gap_pct is not None and gap_pct > 3.0:
        if expected_return_pct < 0.5 and not has_news_catalyst:
            expected_return_pct *= 0.3
            why_factors.append("Gap>3% 且剩余空间小")
        elif gap_pct > EXTENDED_GAP_PCT and not has_news_catalyst and not is_short:
            expected_return_pct *= 0.6
            why_factors.append("Extended gap 压缩 ER")

    if downside_risk_pct < 0.1:
        downside_risk_pct = 0.8

    risk_reward = _rr_numeric(upside_pct, downside_risk_pct)
    rr_w = _rr_weight(risk_reward)
    rr_display = build_rr_display(
        reward_pct=upside_pct,
        risk_pct=downside_risk_pct,
        reward_risk_ratio=risk_reward,
    )
    final_score = round(win_prob * max(expected_return_pct, 0) * rr_w / 100.0, 2)

    weakness = _relative_weakness_score(rs_vs_qqq, rs_vs_smh)
    sym_pct = obs.get("change_pct")
    beta = _beta_proxy(
        _safe_float(sym_pct) if sym_pct is not None else None,
        qqq_pct,
    )
    gap_penalty = _extended_gap_penalty(gap_pct, has_news_catalyst)
    factor_breakdown = _build_factor_breakdown(
        rs_vs_qqq=rs_vs_qqq,
        rs_vs_smh=rs_vs_smh,
        gap_pct=gap_pct,
        beta=beta,
        volume_ok=volume_ok,
        news_count=news_count,
        extended_gap_penalty=gap_penalty,
        vix_chg=vix_chg,
        prior_day_chg=prior_day_chg,
        relative_weakness_score=weakness,
    )
    edge_type = _infer_edge_type(
        symbol, rs_vs_qqq=rs_vs_qqq, driver_type=driver_type, edges=edges
    )
    score_formula = _score_formula_display(win_prob, expected_return_pct, risk_reward)

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
        "win_prob_breakdown": win_prob_breakdown,
        "expected_return_pct": round(expected_return_pct, 2),
        "expected_high": round(expected_high, 2),
        "expected_low": round(expected_low, 2),
        "expected_close": round(expected_close, 2),
        "current_price": round(current, 2),
        "upside_pct": round(upside_pct, 2),
        "downside_risk_pct": round(downside_risk_pct, 2),
        "risk_reward": risk_reward,
        "rr_display": rr_display,
        "final_score": final_score,
        "trade_action": trade_action,
        "trade": trade_action,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "relative_strength": round(rs_vs_qqq, 2) if rs_vs_qqq is not None else None,
        "relative_strength_vs_smh": round(rs_vs_smh, 2) if rs_vs_smh is not None else None,
        "relative_weakness_score": weakness,
        "prior_day_change_pct": round(prior_day_chg, 2) if prior_day_chg is not None else None,
        "why_factors": why_factors[:6],
        "why": " · ".join(why_factors[:4]) if why_factors else "—",
        "score": round(final_score * 10),
        "news_count": news_count,
        "factor_breakdown": factor_breakdown,
        "edge_type": edge_type,
        "score_formula_display": score_formula,
        "rr_weight": _rr_weight_display(risk_reward),
        "why_vs_runner_up": "—",
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
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date,
    section: str,
    q: dict[str, Any],
    as_of_et: time | Literal["now"] | None = None,
    macro_calendar: dict[str, Any] | None = None,
    total_score: int = 0,
) -> dict[str, Any]:
    sym = row.get("symbol") or ""
    current = (
        _safe_float(obs.get("last"))
        or _safe_float(obs.get("close"))
        or _safe_float(row.get("current_price"))
        or _safe_float(q.get("close"))
        or _safe_float(q.get("last"))
    )
    why_factors = list(row.get("why_factors") or [])
    why_chain = row.get("why") or (" · ".join(why_factors[:4]) if why_factors else "—")
    er = float(row.get("expected_return_pct") or 0.0)
    win_prob = float(row.get("win_prob") or 50.0)
    risk_reward = float(row.get("risk_reward") or 0.0)
    final_score = float(row.get("final_score") or 0.0)
    trade_action = row.get("trade_action") or "Pass"
    expected_high = _safe_float(row.get("expected_high"))
    expected_low = _safe_float(row.get("expected_low"))
    expected_close = _safe_float(row.get("expected_close"))
    upside_pct = float(row.get("upside_pct") or 0.0)

    # No usable price (Pass stub / missing quote after universe expansion):
    # emit a transparency Pass slot without calling level derivation.
    if current is None or current <= 0:
        if "No quote data" not in why_factors:
            why_factors = (why_factors + ["No quote data"])[:8]
            why_chain = " · ".join(why_factors[:4]) if why_factors else why_chain
        inst = _instrument(sym, direction, p9)
        score = int(row.get("_total_score") if row.get("_total_score") is not None else total_score)
        horizon = row.get("horizon") or _trade_horizon(
            direction=direction, p9=p9, total=score, instrument=inst
        )
        conf = int(min(95, max(40, win_prob)))
        conf_stars = "★" * min(5, max(1, conf // 20)) + "☆" * (5 - min(5, max(1, conf // 20)))
        return {
            "rank": rank,
            "symbol": sym,
            "direction": direction,
            "instrument": inst,
            "horizon": horizon,
            "confidence": conf,
            "confidence_stars": conf_stars,
            "expected_move": f"{er:+.2f}%",
            "expected_return_pct": er,
            "win_prob": win_prob,
            "win_prob_breakdown": row.get("win_prob_breakdown"),
            "risk_reward": risk_reward,
            "final_score": final_score,
            "trade_action": "Pass",
            "trade": "Pass",
            "relative_strength": row.get("relative_strength"),
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
            "level_anchors": None,
            "levels_valid": False,
            "why": why_factors,
            "why_chain": why_chain,
            "why_vs_runner_up": row.get("why_vs_runner_up", "—"),
            "factor_breakdown": row.get("factor_breakdown") or {},
            "edge_type": row.get("edge_type"),
            "score_formula_display": row.get("score_formula_display"),
            "expected_high": expected_high if expected_high is not None else 0.0,
            "expected_low": expected_low if expected_low is not None else 0.0,
            "expected_close": expected_close if expected_close is not None else 0.0,
            "current_price": None,
            "upside_pct": upside_pct,
            "downside_risk_pct": row.get("downside_risk_pct"),
            "why_factors": why_factors,
            "why_today": row.get("why_chain") or why_chain,
            "gap_pct": row.get("gap_pct"),
            "rr_display": row.get("rr_display"),
            "catalyst": _catalyst_for_symbol(sym, macro_calendar),
            "invalidation": "—",
            "invalid_levels_reason": "No quote data",
            "advisory": True,
        }

    anchors = compute_anchors(
        sym,
        raw,
        prior_raw,
        trading_day,
        section=section,
        q=q,
        obs=obs,
        as_of_et=as_of_et,
    )
    levels = derive_trade_levels(
        direction,
        anchors,
        current=current,
        expected_high=expected_high if expected_high is not None else current,
        expected_low=expected_low if expected_low is not None else current,
        expected_close=expected_close if expected_close is not None else current,
    )
    inst = _instrument(sym, direction, p9)
    conf = int(min(95, max(40, win_prob)))
    conf_stars = "★" * min(5, max(1, conf // 20)) + "☆" * (5 - min(5, max(1, conf // 20)))
    score = int(row.get("_total_score") if row.get("_total_score") is not None else total_score)
    horizon = row.get("horizon") or _trade_horizon(
        direction=direction, p9=p9, total=score, instrument=inst
    )
    slot = {
        "rank": rank,
        "symbol": sym,
        "direction": direction,
        "instrument": inst,
        "horizon": horizon,
        "confidence": conf,
        "confidence_stars": conf_stars,
        "expected_move": f"{er:+.2f}%",
        "expected_return_pct": er,
        "win_prob": win_prob,
        "win_prob_breakdown": row.get("win_prob_breakdown"),
        "risk_reward": risk_reward,
        "final_score": final_score,
        "trade_action": trade_action,
        "relative_strength": row.get("relative_strength"),
        "entry": levels["entry"],
        "entry_source": levels.get("entry_source"),
        "entry_price": levels.get("entry_price"),
        "entry_zone": levels.get("entry_zone"),
        "stop": levels["stop"],
        "stop_source": levels.get("stop_source"),
        "stop_price": levels.get("stop_price"),
        "target": levels["target"],
        "target_source": levels.get("target_source"),
        "target_price": levels.get("target_price"),
        "targets": levels.get("targets") or [],
        "level_anchors": levels.get("level_anchors"),
        "levels_valid": levels.get("levels_valid", True),
        "why": why_factors,
        "why_chain": why_chain,
        "why_vs_runner_up": row.get("why_vs_runner_up", "—"),
        "factor_breakdown": row.get("factor_breakdown") or {},
        "edge_type": row.get("edge_type"),
        "score_formula_display": row.get("score_formula_display"),
        "expected_high": expected_high if expected_high is not None else current,
        "expected_low": expected_low if expected_low is not None else current,
        "expected_close": expected_close if expected_close is not None else current,
        "current_price": round(current, 2),
        "upside_pct": upside_pct,
        "downside_risk_pct": row.get("downside_risk_pct"),
        "why_factors": why_factors,
        "why_today": row.get("why_chain") or why_chain,
        "gap_pct": row.get("gap_pct"),
        "rr_display": row.get("rr_display"),
        "catalyst": _catalyst_for_symbol(sym, macro_calendar),
        "invalidation": "—",
        "advisory": True,
    }
    if levels.get("entry_price") is not None:
        slot["level_reasons"] = build_level_reasons(
            direction,
            anchors,
            entry_px=levels["entry_price"],
            entry_src=levels.get("entry_source") or "",
            stop_px=levels.get("stop_price") or 0,
            stop_src=levels.get("stop_source") or "",
            target_px=levels.get("target_price") or 0,
            target_src=levels.get("target_source") or "",
            current=current,
        )
    slot = _apply_price_based_return(
        slot,
        direction=direction,
        heuristic_er=er,
    )
    slot["levels_valid"] = bool(levels.get("levels_valid", True))
    slot = _enforce_level_invariants(slot)
    slot["invalidation"] = _invalidation_for_slot(slot, macro_calendar)
    return slot


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
    """Pick primary/secondary/watchlist stocks from opportunity ranking (not theme leader)."""
    stock_yes = edges.get("stock_edge", {}).get("edge") == "YES"
    top = ranked[0] if ranked else None
    score_ok = bool(top and top["final_score"] > FINAL_SCORE_THRESHOLD)

    stock_tradeable = [r for r in tradeable if r["symbol"] in _STOCK_SYMBOLS]
    if stock_tradeable:
        return stock_tradeable[:3]

    if not stock_yes and not score_ok:
        return []

    stocks = [
        r for r in ranked
        if r["symbol"] in _STOCK_SYMBOLS and r["trade_action"] != "Pass"
    ]
    if stocks:
        return stocks[:3]
    if score_ok and top and top["symbol"] in _STOCK_SYMBOLS:
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
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date | None,
    quote_by_sym: dict[str, dict[str, Any]],
    as_of_et: time | Literal["now"] | None = None,
    macro_calendar: dict[str, Any] | None = None,
    vix_chg: float | None = None,
    exclude_date: str | None = None,
    total_score: int = 0,
) -> dict[str, Any]:
    tradeable = [r for r in ranked if r["trade_action"] in ("BUY", "Small")]
    tradeable.sort(key=lambda r: (r["final_score"], _rank_key(r)), reverse=True)

    picks = _select_stock_picks(ranked, edges, tradeable)
    top_trades: list[dict[str, Any]] = []
    seen_syms: set[str] = set()

    def _make_slot(row: dict[str, Any], rank: int) -> dict[str, Any]:
        sym = row["symbol"]
        section = _SYMBOL_SECTION.get(sym.upper(), "stocks")
        row = dict(row)
        row["_total_score"] = total_score
        row["horizon"] = _trade_horizon(
            direction=direction,
            p9=p9,
            total=total_score,
            instrument=_instrument(sym, direction, p9),
        )
        slot = _build_trade_slot(
            row,
            rank=rank,
            direction=direction,
            p9=p9,
            obs=obs_by_sym.get(sym, {}),
            raw=raw,
            prior_raw=prior_raw,
            trading_day=trading_day or date.today(),
            section=section,
            q=quote_by_sym.get(sym, {}),
            as_of_et=as_of_et,
            macro_calendar=macro_calendar,
            total_score=total_score,
        )
        return enrich_trade_slot(
            slot,
            ranked=ranked,
            trade_action=slot.get("trade_action") or row["trade_action"],
            exclude_date=exclude_date,
            vix_chg=vix_chg,
            macro_calendar=macro_calendar,
        )

    # Prefer tradeable rows; backfill from ranked if levels invalidate a slot.
    candidate_rows = list(tradeable) if tradeable else list(ranked)
    for row in candidate_rows:
        if len(top_trades) >= 5:
            break
        sym = row["symbol"]
        if sym in seen_syms:
            continue
        slot = _make_slot(row, rank=len(top_trades) + 1)
        seen_syms.add(sym)
        # Sync ranked row so Pass after level check is visible elsewhere
        if slot.get("trade_action") == "Pass" and row.get("trade_action") != "Pass":
            row["trade_action"] = "Pass"
            row["trade"] = "Pass"
            if slot.get("invalid_levels_reason"):
                why = list(row.get("why_factors") or [])
                reason = slot["invalid_levels_reason"]
                if reason not in why:
                    why.append(reason)
                row["why_factors"] = why[:8]
        if not _slot_is_actionable(slot):
            continue
        top_trades.append(slot)

    # If nothing actionable, still show top ranked slots (as Pass) for transparency
    if not top_trades:
        for i, row in enumerate((tradeable or ranked)[:5]):
            slot = _make_slot(row, rank=i + 1)
            top_trades.append(slot)

    watchlist: list[dict[str, Any]] = []
    top5_syms = {t["symbol"] for t in top_trades}
    for row in ranked:
        if row["symbol"] in top5_syms:
            continue
        if row.get("trade_action") == "Pass" or row.get("final_score", 0) < FINAL_SCORE_THRESHOLD * 0.6:
            watchlist.append(
                {
                    "symbol": row["symbol"],
                    "rank": row.get("rank"),
                    "win_prob": row.get("win_prob"),
                    "expected_return_pct": row.get("expected_return_pct"),
                    "trade_action": row.get("trade_action"),
                    "why": row.get("why") or "—",
                    "note": "Monitor — below trade threshold",
                }
            )
        if len(watchlist) >= 5:
            break

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
        "p16_gate": p16_gate,
        "threshold_message": threshold_msg,
        "stock_trades": [],
        "top_trades": top_trades,
        "watchlist_items": watchlist,
        "advisory": True,
    }

    slot_names = ("primary", "secondary", "watchlist")
    valid_picks: list[dict[str, Any]] = []
    for row in picks:
        sym = row["symbol"]
        section = _SYMBOL_SECTION.get(sym.upper(), "stocks")
        # Reuse top_trades slot when available (already invariant-checked)
        existing = next((t for t in top_trades if t.get("symbol") == sym), None)
        if existing is not None:
            slot = {**existing, "rank": len(valid_picks) + 1}
        else:
            slot = _make_slot(row, rank=len(valid_picks) + 1)
        if not _slot_is_actionable(slot):
            if slot.get("trade_action") == "Pass":
                row["trade_action"] = "Pass"
                row["trade"] = "Pass"
            continue
        valid_picks.append(slot)
        if len(valid_picks) >= 3:
            break

    # Backfill primary/secondary from actionable top_trades if picks were invalidated
    if len(valid_picks) < 3:
        for t in top_trades:
            if len(valid_picks) >= 3:
                break
            if any(p.get("symbol") == t.get("symbol") for p in valid_picks):
                continue
            if not _slot_is_actionable(t):
                continue
            if t.get("symbol") not in _STOCK_SYMBOLS:
                continue
            valid_picks.append({**t, "rank": len(valid_picks) + 1})

    if not valid_picks and not threshold_msg:
        threshold_msg = "今日无任何标的达到交易阈值"
        slots["threshold_message"] = threshold_msg

    for i, slot in enumerate(valid_picks[:3]):
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


def _fetch_swing_context(symbol: str, *, current: float) -> dict[str, Any]:
    """Daily ATR(14) + prior-week high/low for swing level sizing.

    Falls back to a ~1.5% price ATR proxy when Yahoo history is unavailable
    (tests / offline). Never raises — swing is advisory and optional.
    """
    atr_proxy = max(current * 0.015, 0.01)
    out: dict[str, Any] = {
        "atr": atr_proxy,
        "atr_source": "proxy_1.5pct",
        "week_high": None,
        "week_low": None,
        "prior_5d_chg_pct": None,
    }
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="30d", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 3:
            return out
        highs = hist["High"].astype(float)
        lows = hist["Low"].astype(float)
        closes = hist["Close"].astype(float)
        prev_close = closes.shift(1)
        tr = (highs - lows).to_frame("hl")
        tr["hc"] = (highs - prev_close).abs()
        tr["lc"] = (lows - prev_close).abs()
        true_range = tr.max(axis=1)
        atr = float(true_range.tail(14).mean())
        if atr > 0:
            out["atr"] = atr
            out["atr_source"] = "ATR14"
        week = hist.tail(5)
        out["week_high"] = round(float(week["High"].max()), 2)
        out["week_low"] = round(float(week["Low"].min()), 2)
        if len(closes) >= 6 and float(closes.iloc[-6]) > 0:
            out["prior_5d_chg_pct"] = round(
                (float(closes.iloc[-1]) / float(closes.iloc[-6]) - 1.0) * 100.0, 2
            )
    except Exception:  # noqa: BLE001
        pass
    return out


def _swing_direction(
    *,
    bias_direction: str,
    prior_day_chg: float | None,
    rs_vs_qqq: float | None,
    prior_5d_chg: float | None,
) -> str:
    """Prefer multi-day trend for position trades; fall back to morning bias.

    Does not force the opposite of a weak intraday bias when the stock's own
    multi-day momentum is clear (LONG on strength / SHORT on weakness).
    """
    mom = 0.0
    if prior_5d_chg is not None:
        mom += float(prior_5d_chg)
    if prior_day_chg is not None:
        mom += float(prior_day_chg) * 0.5
    if rs_vs_qqq is not None:
        mom += float(rs_vs_qqq)
    if mom >= 0.8:
        return "LONG"
    if mom <= -0.8:
        return "SHORT"
    if bias_direction in ("LONG", "SHORT"):
        return bias_direction
    return "LONG"


def derive_swing_levels(
    direction: str,
    *,
    current: float,
    atr: float,
    week_high: float | None = None,
    week_low: float | None = None,
) -> dict[str, Any]:
    """Wider multi-day entry / stop / targets (documented formula).

    Stop: 1.5–2× ATR from ideal entry, or prior-week low (LONG) / high (SHORT)
    when that level sits inside the ATR band.
    Targets: measured move toward 5–15% or 2–3× ATR (whichever is larger,
    capped at ~15%).
    """
    atr = max(float(atr or 0), current * 0.01, 0.01)
    if direction == "LONG":
        entry_mid = round(current * 0.997, 2)
        zone_low = round(entry_mid - 0.5 * atr, 2)
        zone_high = round(min(current, entry_mid + 0.35 * atr), 2)
        if zone_low >= zone_high:
            zone_low = round(entry_mid * 0.99, 2)
            zone_high = round(entry_mid * 1.005, 2)
        stop_atr = entry_mid - 1.75 * atr
        stop_candidates = [stop_atr, entry_mid - 1.5 * atr, entry_mid - 2.0 * atr]
        if week_low is not None and week_low < entry_mid:
            stop_candidates.append(float(week_low))
        # Prefer structural week low when within 1.5–2.5 ATR of entry
        stop_px = stop_atr
        for cand in sorted(stop_candidates, reverse=True):
            dist = entry_mid - cand
            if 1.2 * atr <= dist <= 2.6 * atr:
                stop_px = cand
                break
        else:
            stop_px = entry_mid - 1.75 * atr
        stop_src = "week_low" if week_low is not None and abs(stop_px - float(week_low)) < 0.02 else "1.75xATR"
        t1_pct = max(0.05 * entry_mid, 2.0 * atr)
        t2_pct = min(0.15 * entry_mid, max(0.10 * entry_mid, 3.0 * atr))
        target1 = round(entry_mid + t1_pct, 2)
        target2 = round(entry_mid + t2_pct, 2)
        if target2 <= target1:
            target2 = round(target1 * 1.04, 2)
        er = (target1 - entry_mid) / entry_mid * 100.0
        risk = (entry_mid - stop_px) / entry_mid * 100.0
        stop_label = f"{stop_px:.1f} ({stop_src})"
        entry_prefix = "Near"
    elif direction == "SHORT":
        entry_mid = round(current * 1.003, 2)
        zone_high = round(entry_mid + 0.5 * atr, 2)
        zone_low = round(max(current, entry_mid - 0.35 * atr), 2)
        if zone_low >= zone_high:
            zone_low = round(entry_mid * 0.995, 2)
            zone_high = round(entry_mid * 1.01, 2)
        stop_atr = entry_mid + 1.75 * atr
        stop_candidates = [stop_atr, entry_mid + 1.5 * atr, entry_mid + 2.0 * atr]
        if week_high is not None and week_high > entry_mid:
            stop_candidates.append(float(week_high))
        stop_px = stop_atr
        for cand in sorted(stop_candidates):
            dist = cand - entry_mid
            if 1.2 * atr <= dist <= 2.6 * atr:
                stop_px = cand
                break
        else:
            stop_px = entry_mid + 1.75 * atr
        stop_src = "week_high" if week_high is not None and abs(stop_px - float(week_high)) < 0.02 else "1.75xATR"
        t1_pct = max(0.05 * entry_mid, 2.0 * atr)
        t2_pct = min(0.15 * entry_mid, max(0.10 * entry_mid, 3.0 * atr))
        target1 = round(entry_mid - t1_pct, 2)
        target2 = round(entry_mid - t2_pct, 2)
        if target2 >= target1:
            target2 = round(target1 * 0.96, 2)
        er = (entry_mid - target1) / entry_mid * 100.0
        risk = (stop_px - entry_mid) / entry_mid * 100.0
        stop_label = f"{stop_px:.1f} ({stop_src})"
        entry_prefix = "Near"
    else:
        return {
            "levels_valid": False,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
        }

    rr = round(er / risk, 2) if risk > 0 else 0.0
    entry_zone = {
        "low": round(min(zone_low, zone_high), 2),
        "high": round(max(zone_low, zone_high), 2),
        "mid": entry_mid,
        "display": f"${min(zone_low, zone_high):.2f}–${max(zone_low, zone_high):.2f}",
        "source": "swing_pullback",
    }
    return {
        "levels_valid": True,
        "entry_price": entry_mid,
        "entry": f"{entry_prefix} {entry_mid:.1f} (swing zone)",
        "entry_source": "swing_zone",
        "entry_zone": entry_zone,
        "stop_price": round(stop_px, 2),
        "stop": stop_label,
        "stop_source": stop_src,
        "target_price": target1,
        "target": f"{target1:.1f} (T1 ~{er:.1f}%)",
        "target_source": "2xATR_or_5pct",
        "targets": [
            {"price": target1, "label": "T1", "pct": round(er, 2)},
            {
                "price": target2,
                "label": "T2",
                "pct": round(abs(target2 - entry_mid) / entry_mid * 100.0, 2),
            },
        ],
        "expected_return_pct": round(er, 2),
        "risk_reward": rr,
        "atr": round(atr, 4),
        "level_formula": (
            "stop=1.5–2×ATR or prior-week extreme; "
            "T1=max(5%, 2×ATR); T2=min(15%, max(10%, 3×ATR))"
        ),
    }


def _swing_quality_score(
    row: dict[str, Any],
    *,
    direction: str,
    prior_5d_chg: float | None,
    has_catalyst: bool,
) -> float:
    """Rank swing candidates: liquidity + multi-day momentum + catalyst + setup."""
    sym = str(row.get("symbol") or "")
    score = float(row.get("final_score") or 0.0) * 0.35
    if sym in _SWING_PREFERRED:
        score += 1.5 + (0.15 * (len(_SWING_PREFERRED) - _SWING_PREFERRED.index(sym)))
    prior = _safe_float(row.get("prior_day_change_pct"))
    rs = _safe_float(row.get("relative_strength"))
    if direction == "LONG":
        if prior is not None and prior > 0:
            score += min(2.0, prior * 0.35)
        if rs is not None and rs > 0:
            score += min(1.5, rs * 0.6)
        if prior_5d_chg is not None and prior_5d_chg > 0:
            score += min(2.5, prior_5d_chg * 0.25)
    else:
        if prior is not None and prior < 0:
            score += min(2.0, abs(prior) * 0.35)
        if rs is not None and rs < 0:
            score += min(1.5, abs(rs) * 0.6)
        if prior_5d_chg is not None and prior_5d_chg < 0:
            score += min(2.5, abs(prior_5d_chg) * 0.25)
    if has_catalyst:
        score += 0.8
    if row.get("trade_action") in ("BUY", "Small"):
        score += 0.4
    # Prefer stocks/ETFs over ultrashort levered for swing
    if sym == "TQQQ":
        score -= 1.0
    return round(score, 3)


def compute_swing_opportunity(
    ranked: list[dict[str, Any]],
    *,
    bias_direction: str,
    total_score: int,
    obs_by_sym: dict[str, dict[str, Any]] | None = None,
    quote_by_sym: dict[str, dict[str, Any]] | None = None,
    catalysts: list[Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select best Swing / Position candidate independent of Intraday Primary.

    Can return a setup even when intraday primary is NO TRADE, as long as
    swing quality clears ``_SWING_MIN_QUALITY``. Horizon is always ``Swing``.
    """
    del total_score  # reserved: |total|>=4 already maps slots to Swing via _trade_horizon
    obs_by_sym = obs_by_sym or {}
    quote_by_sym = quote_by_sym or {}
    catalysts = catalysts or []
    catalyst_text = " ".join(str(c) for c in catalysts).upper()

    # Cheap shortlist first (no Yahoo); ATR/week extremes only for top few.
    prelim: list[tuple[float, dict[str, Any], float, str]] = []
    for row in ranked:
        sym = str(row.get("symbol") or "")
        if not sym or sym == "TQQQ":
            continue
        obs = obs_by_sym.get(sym) or {}
        q = quote_by_sym.get(sym) or _quote(raw or {}, sym)
        current = (
            _safe_float(obs.get("last"))
            or _safe_float(obs.get("close"))
            or _safe_float(row.get("current_price"))
            or _safe_float(q.get("close"))
            or _safe_float(q.get("last"))
        )
        if current is None or current <= 0:
            continue
        direction = _swing_direction(
            bias_direction=bias_direction,
            prior_day_chg=_safe_float(row.get("prior_day_change_pct")),
            rs_vs_qqq=_safe_float(row.get("relative_strength")),
            prior_5d_chg=None,
        )
        has_cat = bool(row.get("news_count")) or sym in catalyst_text
        q_score = _swing_quality_score(
            row, direction=direction, prior_5d_chg=None, has_catalyst=has_cat
        )
        if q_score < _SWING_MIN_QUALITY and sym not in _SWING_PREFERRED:
            continue
        if q_score < _SWING_MIN_QUALITY * 0.75:
            continue
        prelim.append((q_score, row, current, direction))

    if not prelim:
        return None
    prelim.sort(key=lambda x: x[0], reverse=True)

    best: tuple[float, dict[str, Any], dict[str, Any], str, float] | None = None
    for _, row, current, direction in prelim[:5]:
        sym = str(row["symbol"])
        ctx = _fetch_swing_context(sym, current=current)
        direction = _swing_direction(
            bias_direction=bias_direction,
            prior_day_chg=_safe_float(row.get("prior_day_change_pct")),
            rs_vs_qqq=_safe_float(row.get("relative_strength")),
            prior_5d_chg=_safe_float(ctx.get("prior_5d_chg_pct")),
        )
        has_cat = bool(row.get("news_count")) or sym in catalyst_text
        refined = _swing_quality_score(
            row,
            direction=direction,
            prior_5d_chg=ctx.get("prior_5d_chg_pct"),
            has_catalyst=has_cat,
        )
        if refined < _SWING_MIN_QUALITY * 0.75:
            continue
        if best is None or refined > best[0]:
            best = (refined, row, ctx, direction, current)

    if best is None:
        return None
    q_score, row, ctx, direction, current = best
    sym = str(row["symbol"])
    levels = derive_swing_levels(
        direction,
        current=current,
        atr=float(ctx.get("atr") or current * 0.015),
        week_high=_safe_float(ctx.get("week_high")),
        week_low=_safe_float(ctx.get("week_low")),
    )
    if not levels.get("levels_valid"):
        return None

    instrument = "Stock" if sym in _STOCK_SYMBOLS else "ETF"
    why: list[str] = []
    if sym in _SWING_PREFERRED:
        why.append(f"高流动性波段标的 {sym}")
    prior = row.get("prior_day_change_pct")
    if prior is not None:
        why.append(f"Prior day {prior:+.1f}%")
    p5 = ctx.get("prior_5d_chg_pct")
    if p5 is not None:
        why.append(f"5D momentum {p5:+.1f}%")
    rs = row.get("relative_strength")
    if rs is not None:
        why.append(f"RS vs QQQ {rs:+.2f}%")
    if row.get("news_count"):
        why.append(f"News/{row['news_count']} catalyst hints")
    if not why:
        why.append("Multi-day setup quality vs peer universe")
    why.append(f"Swing score {q_score:.1f}")

    entry_px = levels["entry_price"]
    stop_px = levels["stop_price"]
    target_px = levels["target_price"]
    er = levels["expected_return_pct"]
    conf = int(min(90, max(45, float(row.get("win_prob") or 55) + 5)))
    invalidation = (
        f"Close beyond stop ${stop_px} or thesis broken (trend flip / catalyst fade)"
        if stop_px is not None
        else "Stop breach or thesis broken"
    )
    trade_summary_cn = (
        f"波段{'做多' if direction == 'LONG' else '做空'} {sym}："
        f"理想入场 {(levels.get('entry_zone') or {}).get('display', entry_px)}，"
        f"止损 ${stop_px}，目标 ${target_px}（约 {er:+.1f}%），持仓天数–数周"
    )
    return {
        "symbol": sym,
        "direction": direction,
        "instrument": instrument,
        "horizon": "Swing",
        "duration": "Swing (days–weeks)",
        "confidence": conf,
        "entry": levels["entry"],
        "entry_source": levels.get("entry_source"),
        "entry_price": entry_px,
        "entry_zone": levels.get("entry_zone"),
        "stop": levels["stop"],
        "stop_source": levels.get("stop_source"),
        "stop_price": stop_px,
        "target": levels["target"],
        "target_source": levels.get("target_source"),
        "target_price": target_px,
        "targets": levels.get("targets") or [],
        "expected_return_pct": er,
        "expected_move": f"{er:+.2f}%",
        "risk_reward": levels.get("risk_reward"),
        "win_prob": row.get("win_prob"),
        "final_score": row.get("final_score"),
        "swing_quality": q_score,
        "atr": levels.get("atr"),
        "atr_source": ctx.get("atr_source"),
        "level_formula": levels.get("level_formula"),
        "current_price": round(current, 2),
        "why": why[:6],
        "why_chain": " · ".join(why[:4]),
        "invalidation": invalidation,
        "trade_summary_cn": trade_summary_cn,
        "one_liner": trade_summary_cn,
        "edge_type": row.get("edge_type") or "Swing / Position",
        "advisory": True,
        "label": "长线 · Swing Trade",
        "separate_from_intraday": True,
    }


def _to_best_opportunity(
    primary: dict[str, Any] | None,
    *,
    threshold_message: str | None,
    p16_gate: str,
    duration: str = "Intraday",
    horizon: str | None = None,
) -> dict[str, Any]:
    if primary:
        entry_px = primary.get("entry_price")
        target_px = primary.get("target_price")
        er = primary.get("expected_return_pct")
        direction = primary["direction"]
        trade_summary_cn = primary.get("trade_summary_cn")
        if not trade_summary_cn and entry_px is not None and target_px is not None and er is not None:
            if direction == "SHORT":
                trade_summary_cn = (
                    f"做空 {primary['symbol']}：在 {entry_px} 附近入场，"
                    f"目标 {target_px}，预期 {er:+.2f}%"
                )
            else:
                trade_summary_cn = (
                    f"做多 {primary['symbol']}：在 {entry_px} 附近入场，"
                    f"目标 {target_px}，预期 {er:+.2f}%"
                )
        one_liner = trade_summary_cn or (
            f"Today {'buy' if direction == 'LONG' else 'sell short'} "
            f"{primary['symbol']} near ${entry_px}, target ${target_px}, "
            f"expected {primary['expected_move']}"
        )
        hz = horizon or primary.get("horizon") or duration
        return {
            "direction": direction,
            "symbol": primary["symbol"],
            "instrument": primary["instrument"],
            "horizon": hz,
            "confidence": primary["confidence"],
            "confidence_stars": primary.get("confidence_stars"),
            "entry": primary["entry"],
            "entry_source": primary.get("entry_source"),
            "entry_price": primary.get("entry_price"),
            "stop": primary["stop"],
            "stop_source": primary.get("stop_source"),
            "stop_price": primary.get("stop_price"),
            "target": primary["target"],
            "target_source": primary.get("target_source"),
            "target_price": primary.get("target_price"),
            "target_action": primary.get("target_action"),
            "targets": primary.get("targets") or [],
            "duration": hz,
            "why": primary.get("why") or [],
            "why_chain": primary.get("why_chain", "—"),
            "why_vs_runner_up": primary.get("why_vs_runner_up", "—"),
            "factor_breakdown": primary.get("factor_breakdown") or {},
            "edge_type": primary.get("edge_type"),
            "score_formula_display": primary.get("score_formula_display"),
            "win_prob": primary.get("win_prob"),
            "win_prob_breakdown": primary.get("win_prob_breakdown"),
            "expected_return_pct": primary.get("expected_return_pct"),
            "expected_move": primary.get("expected_move"),
            "return_calculation": primary.get("return_calculation"),
            "trade_summary_cn": trade_summary_cn,
            "risk_reward": primary.get("risk_reward"),
            "rr_display": primary.get("rr_display"),
            "entry_zone": primary.get("entry_zone"),
            "rank_summary": primary.get("rank_summary"),
            "win_prob_source": primary.get("win_prob_source"),
            "calibration": primary.get("calibration"),
            "similar_days": primary.get("similar_days"),
            "ev_distribution": primary.get("ev_distribution"),
            "final_score": primary.get("final_score"),
            "level_anchors": primary.get("level_anchors"),
            "level_reasons": primary.get("level_reasons"),
            "trade_economics": primary.get("trade_economics"),
            "position_sizing": primary.get("position_sizing"),
            "why_wins_today": primary.get("why_wins_today"),
            "why_not_alternatives": primary.get("why_not_alternatives"),
            "avoid": [],
            "one_liner": one_liner,
            "p16_gate": p16_gate,
            "advisory": True,
        }

    why = [threshold_message] if threshold_message else ["今日无任何标的达到交易阈值"]
    return {
        "direction": "NO TRADE",
        "symbol": "—",
        "instrument": "—",
        "horizon": "—",
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
    as_of_et: time | Literal["now"] | None = None,
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
    macro_calendar = rule_bundle.get("macro_calendar") or {}
    driver_tree = rule_bundle.get("driver_tree") or {}
    p9 = parts.get("P9") or {}
    catalysts = rule_bundle.get("catalysts_today") or []

    market_q = (raw.get("market") or {}).get("quotes") or {}
    vix_q = market_q.get("^VIX") or market_q.get("VIX") or {}
    vix_chg = _safe_float(vix_q.get("change_pct"))

    sym_pcts: dict[str, float | None] = {}
    obs_by_sym: dict[str, dict[str, Any]] = {}
    if trading_day:
        for sym in CANDIDATE_SYMBOLS:
            obs = _observation(sym, raw, prior_raw, trading_day, as_of_et=as_of_et)
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
            macro_calendar=macro_calendar,
            qqq_pct=qqq_pct,
            smh_pct=smh_pct,
            spy_pct=spy_pct,
            sym_pcts=sym_pcts,
            driver_type=driver_type,
        )

    p16_gate = _p16_gate(parts, total)
    direction = _pick_direction(bias, total)
    # Per-trade horizon is set on slots; best_opportunity uses primary's horizon
    default_horizon = _trade_horizon(direction=direction, p9=p9, total=total)

    ranked: list[dict[str, Any]] = []
    quote_by_sym: dict[str, dict[str, Any]] = {}
    for sym in CANDIDATE_SYMBOLS:
        q = _quote(raw, sym)
        quote_by_sym[sym] = q
        obs = obs_by_sym.get(sym, {})
        if not _has_market_data(obs, q):
            ranked.append(_pass_stub_row(sym, "No quote data"))
            continue
        sym_pct = sym_pcts.get(sym)
        prior_chg = _prior_day_change(sym, prior_raw, prior_day) if prior_day else None
        rs = (sym_pct - qqq_pct) if sym_pct is not None and qqq_pct is not None else None
        rs_smh = (sym_pct - smh_pct) if sym_pct is not None and smh_pct is not None else None
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
            rs_vs_smh=rs_smh,
            gap_pct=gap_pct,
            news_count=news_n,
            volume_ok=vol_ok,
            vix_chg=vix_chg,
            driver_type=driver_type,
            has_news_catalyst=has_catalyst,
            q=q,
            qqq_pct=qqq_pct,
            edges=edges,
            direction=direction,
            macro_calendar=macro_calendar,
            driver_tree=driver_tree,
        )
        ranked.append(row)

    ranked.sort(key=lambda r: (r["final_score"], _rank_key(r)), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
        row["direction"] = direction
    if len(ranked) >= 2:
        ranked[0]["why_vs_runner_up"] = _why_vs_runner_up(ranked[0], ranked[1])

    best_trades = _decision_tree(
        ranked,
        edges,
        direction=direction,
        p9=p9,
        obs_by_sym=obs_by_sym,
        p16_gate=p16_gate,
        raw=raw,
        prior_raw=prior_raw,
        trading_day=trading_day,
        quote_by_sym=quote_by_sym,
        as_of_et=as_of_et,
        macro_calendar=macro_calendar,
        vix_chg=vix_chg,
        exclude_date=trading_date or None,
        total_score=total,
    )
    best_opportunity = _to_best_opportunity(
        best_trades.get("primary"),
        threshold_message=best_trades.get("threshold_message"),
        p16_gate=p16_gate,
        duration=default_horizon,
        horizon=(best_trades.get("primary") or {}).get("horizon") or default_horizon,
    )

    swing_trade = compute_swing_opportunity(
        ranked,
        bias_direction=direction,
        total_score=total,
        obs_by_sym=obs_by_sym,
        quote_by_sym=quote_by_sym,
        catalysts=catalysts,
        raw=raw,
    )
    if swing_trade:
        best_trades["swing"] = swing_trade

    gap_pct_market = None
    qqq_obs = obs_by_sym.get("QQQ", {})
    if qqq_obs.get("gap_pct") is not None:
        gap_pct_market = _safe_float(qqq_obs.get("gap_pct"))

    transparency = build_decision_transparency(
        ranked=ranked,
        best_trades=best_trades,
        edges=edges,
        p16_gate=p16_gate,
        index_trade=best_trades.get("index_trade"),
        direction=direction,
        catalysts=catalysts,
        vix_chg=vix_chg,
        qqq_pct=qqq_pct,
        smh_pct=smh_pct,
        total_score=int(rule_bundle.get("total") or 0),
        gap_pct_market=gap_pct_market,
        macro_calendar=macro_calendar,
        trade_plan=(parts.get("P16") or {}).get("trade_plan"),
    )

    return {
        "edges": edges,
        "trade_candidates": ranked,
        "best_trades": best_trades,
        "best_opportunity": best_opportunity,
        "swing_trade": swing_trade,
        "top_trades": best_trades.get("top_trades") or [],
        "watchlist": best_trades.get("watchlist_items") or [],
        "bias_stars": _bias_stars(bias),
        "index_trade": best_trades.get("index_trade"),
        "stock_trades": best_trades.get("stock_trades") or [],
        "transparency": transparency,
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
    driver_tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = (best_trades or {}).get("primary")
    threshold_msg = (best_trades or {}).get("threshold_message")
    why_factors: list[str] = []
    entry_source = stop_source = target_source = edge_type = why_vs_runner_up = None

    if primary:
        best_trade = (
            f"{primary['symbol']} · {primary['direction']} · "
            f"{primary.get('instrument', '—')} · ER {primary.get('expected_move')}"
        )
        one_liner = best.get("one_liner") or primary.get("trade_summary_cn") or (
            f"Today {primary['direction']} {primary['symbol']} "
            f"entry ${primary.get('entry_price')} → target ${primary.get('target_price')}, "
            f"expected {primary.get('expected_move')}"
        )
        confidence = primary.get("confidence")
        entry = primary.get("entry", "—")
        stop = primary.get("stop", "—")
        target = primary.get("target", "—")
        why_chain = primary.get("why_chain", "—")
        why_factors = primary.get("why_factors") or primary.get("why") or []
        entry_source = primary.get("entry_source")
        stop_source = primary.get("stop_source")
        target_source = primary.get("target_source")
        edge_type = primary.get("edge_type")
        why_vs_runner_up = primary.get("why_vs_runner_up")
        entry_price = primary.get("entry_price")
        target_price = primary.get("target_price")
        stop_price = primary.get("stop_price")
        expected_return_pct = primary.get("expected_return_pct")
        return_calculation = primary.get("return_calculation")
        trade_summary_cn = primary.get("trade_summary_cn")
        target_action = primary.get("target_action")
    elif threshold_msg:
        best_trade = threshold_msg
        one_liner = threshold_msg
        confidence = None
        entry = stop = target = "—"
        why_chain = threshold_msg
        entry_price = target_price = stop_price = None
        expected_return_pct = return_calculation = trade_summary_cn = target_action = None
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
        entry_price = best.get("entry_price")
        target_price = best.get("target_price")
        stop_price = best.get("stop_price")
        expected_return_pct = best.get("expected_return_pct")
        return_calculation = best.get("return_calculation")
        trade_summary_cn = best.get("trade_summary_cn")
        target_action = best.get("target_action")

    return {
        "bias": bias,
        "bias_stars": bias_stars,
        "driver_type": driver_type,
        "driver": driver,
        "driver_tree": driver_tree,
        "driver_display": " — ".join(x for x in (driver_type, driver) if x) or "—",
        "best_trade": best_trade,
        "primary_trade": primary,
        "confidence": confidence,
        "entry": entry,
        "entry_source": entry_source if primary else None,
        "entry_price": entry_price if primary else best.get("entry_price"),
        "stop": stop,
        "stop_source": stop_source if primary else None,
        "stop_price": stop_price if primary else best.get("stop_price"),
        "target": target,
        "target_source": target_source if primary else None,
        "target_price": target_price if primary else best.get("target_price"),
        "target_action": target_action if primary else best.get("target_action"),
        "direction": (primary or best or {}).get("direction"),
        "expected_return_pct": expected_return_pct if primary else best.get("expected_return_pct"),
        "expected_move": (primary or best or {}).get("expected_move"),
        "return_calculation": return_calculation if primary else best.get("return_calculation"),
        "trade_summary_cn": trade_summary_cn if primary else best.get("trade_summary_cn"),
        "why_chain": why_chain,
        "why_factors": why_factors if primary else [],
        "why_vs_runner_up": why_vs_runner_up if primary else None,
        "edge_type": edge_type if primary else None,
        "win_prob_breakdown": (primary or {}).get("win_prob_breakdown"),
        "level_reasons": (primary or {}).get("level_reasons"),
        "trade_economics": (primary or {}).get("trade_economics"),
        "position_sizing": (primary or {}).get("position_sizing"),
        "rr_display": (primary or {}).get("rr_display"),
        "entry_zone": (primary or {}).get("entry_zone"),
        "entry_status": (primary or {}).get("entry_status"),
        "rank_summary": (primary or {}).get("rank_summary"),
        "win_prob_source": (primary or {}).get("win_prob_source"),
        "similar_days": (primary or {}).get("similar_days"),
        "ev_distribution": (primary or {}).get("ev_distribution"),
        "why_wins_today": (primary or {}).get("why_wins_today") or [],
        "why_not_alternatives": (primary or {}).get("why_not_alternatives") or [],
        "one_liner": one_liner,
        "threshold_message": threshold_msg,
        "index_trade": (best_trades or {}).get("index_trade"),
        "advisory": ADVISORY_TAG,
    }
