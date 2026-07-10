"""Decision transparency — auditable win%, levels, sizing, and alternative comparisons.

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

from typing import Any

DEFAULT_ACCOUNT_SIZE = 10_000.0
DEFAULT_RISK_PCT = 1.0

_WIN_PROB_LABELS: dict[str, str] = {
    "base": "Base",
    "rs": "RS",
    "trend": "Trend",
    "gap": "Gap",
    "volume": "Volume",
    "macro": "Macro",
    "catalyst": "Catalyst",
}

_DIMENSION_LABELS: dict[str, str] = {
    "relative_strength": "Highest Relative Strength",
    "expected_range": "Largest Expected Range",
    "liquidity": "Highest Liquidity",
    "win_prob": "Highest Probability",
    "risk_reward": "Best Risk Reward",
    "catalyst": "Best Catalyst Alignment",
}

_RANK_DIM_SHORT: dict[str, str] = {
    "relative_strength": "weakest RS",
    "expected_range": "largest expected range",
    "liquidity": "acceptable liquidity",
    "win_prob": "highest probability",
    "risk_reward": "best reward/risk",
    "catalyst": "best catalyst alignment",
}


def build_rr_display(
    *,
    reward_per_share: float | None = None,
    risk_per_share: float | None = None,
    reward_pct: float | None = None,
    risk_pct: float | None = None,
    reward_risk_ratio: float | None = None,
) -> dict[str, Any]:
    """Clarify Reward/Risk vs Risk:Reward (legacy field name risk_reward = reward/risk)."""
    rr = reward_risk_ratio
    if rr is None:
        if (
            reward_per_share is not None
            and risk_per_share is not None
            and risk_per_share > 0
        ):
            rr = round(reward_per_share / risk_per_share, 2)
        elif reward_pct is not None and risk_pct is not None and risk_pct > 0:
            rr = round(reward_pct / risk_pct, 2)

    if rr is None or rr <= 0:
        return {}

    risk_over_reward = round(1.0 / rr, 2)
    return {
        "reward_risk_ratio": rr,
        "risk_reward_ratio": risk_over_reward,
        "reward_per_share": reward_per_share,
        "risk_per_share": risk_per_share,
        "label_reward_risk": f"Reward/Risk {rr:.2f}",
        "label_risk_reward": f"Risk:Reward 1:{risk_over_reward:.2f}",
        "display": f"Reward/Risk {rr:.2f} · Risk:Reward 1:{risk_over_reward:.2f}",
        "convention": "reward_risk_ratio = reward ÷ risk; Risk:Reward shows risk per unit reward",
    }


def build_rank_summary(
    primary: dict[str, Any],
    why_wins: list[dict[str, Any]] | None = None,
) -> str:
    """One-liner for why the primary trade ranks #1 today."""
    sym = primary.get("symbol", "—")
    wins = why_wins or primary.get("why_wins_today") or []
    if not wins:
        return f"{sym} ranked #1 as top trade candidate today"
    parts = [
        _RANK_DIM_SHORT.get(w.get("dimension", ""), w.get("label", "").lower())
        for w in wins[:3]
    ]
    return f"{sym} ranked #1 because {' + '.join(parts)}"


def compute_ev_distribution(
    *,
    win_prob: float | None,
    trade_economics: dict[str, Any] | None,
    similar_days: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expected value and win/loss scenarios from trade economics."""
    if not trade_economics:
        return {}

    win_usd = float(trade_economics.get("win_usd") or 0)
    loss_usd = float(trade_economics.get("loss_usd") or 0)
    wp = (win_prob or 50.0) / 100.0
    ev_usd = round(wp * win_usd - (1.0 - wp) * loss_usd, 2)

    out: dict[str, Any] = {
        "expected_value_usd": ev_usd,
        "win_scenario_usd": win_usd,
        "loss_scenario_usd": loss_usd,
        "win_prob_pct": win_prob,
        "display": (
            f"EV ${ev_usd:+.0f} "
            f"(win +${win_usd:.0f} @ {win_prob or '—'}% · loss -${loss_usd:.0f})"
        ),
    }

    if similar_days:
        from src.research.win_rate_calibration import return_distribution_bins

        returns = [
            float(d["return_pct"])
            for d in similar_days
            if d.get("return_pct") is not None
        ]
        bins = return_distribution_bins(returns)
        if bins:
            out["return_distribution"] = bins
            out["similar_days_count"] = len(returns)

    return out


def init_win_prob_breakdown() -> dict[str, float]:
    return {"base": 50.0}


def add_win_prob_delta(breakdown: dict[str, float], key: str, delta: float) -> None:
    if delta:
        breakdown[key] = breakdown.get(key, 0.0) + delta


def finalize_win_prob_breakdown(
    breakdown: dict[str, float],
    *,
    clamped: float,
) -> dict[str, Any]:
    raw_total = sum(breakdown.values())
    components = [
        {
            "factor": _WIN_PROB_LABELS.get(k, k),
            "key": k,
            "points": round(v, 1),
        }
        for k, v in breakdown.items()
        if abs(v) >= 0.05
    ]
    return {
        "components": components,
        "total": round(clamped, 1),
        "raw_total": round(raw_total, 1),
        "was_clamped": abs(raw_total - clamped) > 0.05,
    }


def compute_trade_economics(
    *,
    entry_price: float | None,
    stop_price: float | None,
    target_price: float | None,
    direction: str,
    win_prob: float | None = None,
    trade_action: str = "Small",
    account_size: float = DEFAULT_ACCOUNT_SIZE,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> dict[str, Any]:
    """Dollar win/loss and position sizing from entry/stop/target."""
    if (
        entry_price is None
        or stop_price is None
        or target_price is None
        or entry_price <= 0
        or direction not in ("LONG", "SHORT")
    ):
        return {}

    risk_per_share = abs(entry_price - stop_price)
    if direction == "LONG":
        reward_per_share = abs(target_price - entry_price)
    else:
        reward_per_share = abs(entry_price - target_price)

    if risk_per_share <= 0:
        return {}

    max_loss_usd = account_size * risk_pct / 100.0
    shares = max(1, int(max_loss_usd / risk_per_share))
    position_value = shares * entry_price
    position_pct = min(100.0, position_value / account_size * 100.0)

    win_usd = round(shares * reward_per_share, 2)
    loss_usd = round(shares * risk_per_share, 2)

    size_label = "Small"
    if trade_action == "BUY":
        size_label = "Full"
    elif trade_action == "Pass":
        size_label = "None"

    contracts = 1 if trade_action in ("BUY", "Small") else 0

    rr_display = build_rr_display(
        reward_per_share=round(reward_per_share, 2),
        risk_per_share=round(risk_per_share, 2),
    )
    wp_frac = (win_prob or 50.0) / 100.0
    expected_value_usd = round(wp_frac * win_usd - (1.0 - wp_frac) * loss_usd, 2)

    return {
        "account_size": account_size,
        "risk_pct": risk_pct,
        "position_pct": round(position_pct, 1),
        "shares": shares,
        "contracts": contracts,
        "max_loss_usd": round(loss_usd, 2),
        "win_usd": win_usd,
        "loss_usd": loss_usd,
        "expected_value_usd": expected_value_usd,
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "rr_display": rr_display,
        "size_label": size_label,
        "display": f"${account_size:,.0f} → Win +${win_usd:.0f} / Loss -${loss_usd:.0f}",
    }


def compute_position_sizing(
    economics: dict[str, Any],
    *,
    trade_action: str,
) -> dict[str, Any]:
    if not economics:
        return {}
    return {
        "risk_pct": economics.get("risk_pct"),
        "position_pct": economics.get("position_pct"),
        "contracts": economics.get("contracts"),
        "shares": economics.get("shares"),
        "max_loss_usd": economics.get("max_loss_usd"),
        "size_label": economics.get("size_label") or trade_action,
        "display": (
            f"Risk {economics.get('risk_pct')}% → Position {economics.get('position_pct')}% "
            f"→ {economics.get('contracts') or economics.get('shares')} unit(s) "
            f"→ Max Loss ${economics.get('max_loss_usd')}"
        ),
    }


def _expected_range_pct(row: dict[str, Any]) -> float:
    up = abs(row.get("upside_pct") or 0)
    down = abs(row.get("downside_risk_pct") or 0)
    return up + down


def _liquidity_score(row: dict[str, Any]) -> float:
    fb = row.get("factor_breakdown") or {}
    score = 0.0
    if fb.get("volume_signal"):
        score += 2.0
    sym = row.get("symbol", "")
    if sym in ("QQQ", "SPY", "TQQQ"):
        score += 3.0
    elif sym in ("NVDA", "TSLA"):
        score += 2.0
    elif sym == "SMH":
        score += 1.5
    return score


def _catalyst_score(row: dict[str, Any]) -> float:
    fb = row.get("factor_breakdown") or {}
    return float(fb.get("news_count") or 0) + (1.0 if row.get("edge_type") == "Stock Edge" else 0)


def why_wins_today(primary: dict[str, Any], ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-symbol dimensions where primary wins today."""
    if not primary or not ranked:
        return []

    sym = primary["symbol"]
    others = [r for r in ranked if r["symbol"] != sym]
    if not others:
        return []

    checks: list[tuple[str, str, Any, Any]] = []

    rs = primary.get("relative_strength")
    if rs is not None:
        best_other = max(
            (r.get("relative_strength") for r in others if r.get("relative_strength") is not None),
            default=None,
        )
        if best_other is None or abs(rs) >= abs(best_other):
            checks.append(
                ("relative_strength", "Highest Relative Strength", rs, best_other)
            )

    erng = _expected_range_pct(primary)
    best_erng = max(_expected_range_pct(r) for r in others)
    if erng >= best_erng:
        checks.append(("expected_range", "Largest Expected Range", round(erng, 2), round(best_erng, 2)))

    liq = _liquidity_score(primary)
    best_liq = max(_liquidity_score(r) for r in others)
    if liq >= best_liq:
        checks.append(("liquidity", "Highest Liquidity", liq, best_liq))

    wp = primary.get("win_prob")
    if wp is not None:
        best_wp = max(r.get("win_prob") or 0 for r in others)
        if wp >= best_wp:
            checks.append(("win_prob", "Highest Probability", wp, best_wp))

    rr = primary.get("risk_reward")
    if rr is not None:
        best_rr = max(r.get("risk_reward") or 0 for r in others)
        if rr >= best_rr:
            checks.append(("risk_reward", "Best Risk Reward", rr, best_rr))

    cat = _catalyst_score(primary)
    best_cat = max(_catalyst_score(r) for r in others)
    if cat >= best_cat and cat > 0:
        checks.append(("catalyst", "Best Catalyst Alignment", cat, best_cat))

    out: list[dict[str, Any]] = []
    for i, (key, label, val, runner) in enumerate(checks, start=1):
        out.append(
            {
                "rank": i,
                "dimension": key,
                "label": label,
                "primary_value": val,
                "runner_up_value": runner,
                "wins": True,
            }
        )
    return out


def why_not_alternatives(
    primary: dict[str, Any],
    ranked: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Why primary beat each alternative (not only #2)."""
    if not primary:
        return []

    sym = primary["symbol"]
    alts = [r for r in ranked if r["symbol"] != sym][:limit]
    out: list[dict[str, Any]] = []

    for alt in alts:
        reasons: list[str] = []
        p_wp = primary.get("win_prob")
        a_wp = alt.get("win_prob")
        if p_wp is not None and a_wp is not None and p_wp != a_wp:
            reasons.append(f"Win% {p_wp:.0f} vs {a_wp:.0f}")

        p_er = primary.get("expected_return_pct")
        a_er = alt.get("expected_return_pct")
        if p_er is not None and a_er is not None and p_er != a_er:
            reasons.append(f"ER {p_er:.1f}% vs {a_er:.1f}%")

        p_sc = primary.get("final_score")
        a_sc = alt.get("final_score")
        if p_sc is not None and a_sc is not None and p_sc != a_sc:
            reasons.append(f"Score {p_sc} vs {a_sc}")

        p_rs = primary.get("relative_strength")
        a_rs = alt.get("relative_strength")
        if p_rs is not None and a_rs is not None:
            if (p_rs > 0) != (a_rs > 0) or abs(p_rs - a_rs) > 0.1:
                reasons.append(f"RS {p_rs:+.1f}% vs {a_rs:+.1f}%")

        p_rr = primary.get("risk_reward")
        a_rr = alt.get("risk_reward")
        if p_rr is not None and a_rr is not None and p_rr > a_rr:
            reasons.append(f"R:R {p_rr} vs {a_rr}")

        p_ta = primary.get("trade_action")
        a_ta = alt.get("trade_action")
        if p_ta in ("BUY", "Small") and a_ta == "Pass":
            reasons.append(f"Tradeable ({p_ta}) vs Pass")

        if not reasons:
            reasons.append(f"Rank #{primary.get('rank', 1)} vs #{alt.get('rank')}")

        out.append(
            {
                "symbol": alt["symbol"],
                "rank": alt.get("rank"),
                "reasons": reasons,
                "summary": f"Why {sym} over {alt['symbol']}: " + " · ".join(reasons),
            }
        )
    return out


def index_rejection_reasons(
    *,
    index_trade: str,
    edges: dict[str, Any] | None,
    p16_gate: str,
    ranked: list[dict[str, Any]],
    catalysts: list[dict[str, Any]] | None = None,
    qqq_pct: float | None = None,
    smh_pct: float | None = None,
    total_score: int | None = None,
) -> list[dict[str, str]]:
    if index_trade and index_trade != "NO TRADE":
        return []

    reasons: list[dict[str, str]] = []
    edges = edges or {}

    if p16_gate in ("No Trade", "Wait"):
        reasons.append(
            {
                "factor": "P16 Gate",
                "status": p16_gate,
                "detail": f"指数敞口：{p16_gate}",
            }
        )

    idx_edge = edges.get("index_edge") or {}
    if idx_edge.get("edge") == "NO":
        reasons.append(
            {
                "factor": "Index Edge",
                "status": "NO",
                "detail": idx_edge.get("why") or "指数方向不明",
            }
        )

    index_syms = {"QQQ", "SPY", "TQQQ"}
    index_rows = [r for r in ranked if r["symbol"] in index_syms]
    if index_rows:
        top_idx = index_rows[0]
        if top_idx.get("trade_action") == "Pass":
            reasons.append(
                {
                    "factor": top_idx["symbol"],
                    "status": "Pass",
                    "detail": f"Score {top_idx.get('final_score')} 未达阈值",
                }
            )

    if qqq_pct is not None and qqq_pct < 0:
        reasons.append(
            {
                "factor": "QQQ",
                "status": "Weak",
                "detail": f"QQQ {qqq_pct:+.2f}%",
            }
        )

    if smh_pct is not None and qqq_pct is not None and smh_pct - qqq_pct < 0:
        reasons.append(
            {
                "factor": "Semis",
                "status": "Weak",
                "detail": f"SMH 相对 QQQ 偏弱",
            }
        )

    if catalysts:
        names = [c.get("name") for c in catalysts if c.get("name")]
        if names:
            reasons.append(
                {
                    "factor": "Macro Event",
                    "status": "High uncertainty",
                    "detail": "、".join(names[:3]),
                }
            )

    if total_score is not None and total_score < 0:
        reasons.append(
            {
                "factor": "Breadth",
                "status": "Weak",
                "detail": f"Morning total {total_score}",
            }
        )

    if not reasons:
        reasons.append(
            {
                "factor": "Index",
                "status": "NO TRADE",
                "detail": "无合格指数交易设置",
            }
        )
    return reasons


def day_risks(
    *,
    primary: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
    catalysts: list[dict[str, Any]] | None = None,
    vix_chg: float | None = None,
    total_score: int | None = None,
    gap_pct_market: float | None = None,
) -> dict[str, Any]:
    risks: list[str] = []
    donts: list[str] = []

    if gap_pct_market is not None and abs(gap_pct_market) >= 3.0:
        risks.append(f"High Gap ({gap_pct_market:+.1f}%)")
        donts.append("Don't Chase")

    if primary:
        p_gap = (primary.get("factor_breakdown") or {}).get("gap_pct")
        if p_gap is not None and abs(p_gap) >= 4.0:
            risks.append(f"Extended Gap on {primary['symbol']}")
            donts.append("Don't Buy First Green Candle")

    if total_score is not None and total_score <= 0:
        risks.append("Low Breadth")
        donts.append("Don't Oversize")

    if catalysts:
        names = [c.get("name") for c in catalysts if c.get("name")]
        if names:
            risks.append(f"{' / '.join(names[:2])} Whipsaw")
            donts.append("Don't Trade Before Data Release")

    if vix_chg is not None and vix_chg > 5:
        risks.append("VIX Spike")
        donts.append("Don't Fight Volatility")

    smh_row = next((r for r in ranked if r["symbol"] == "SMH"), None)
    if smh_row:
        rs = smh_row.get("relative_strength")
        if rs is not None and rs < -0.2:
            risks.append("Semiconductor Weakness")

    if not risks:
        risks.append("Normal session risk")

    return {
        "risks": risks[:5],
        "donts": list(dict.fromkeys(donts))[:4],
        "headline": risks[0] if risks else "—",
    }


def _stars_from_win_prob(win_prob: float | None, trade_action: str) -> str:
    if trade_action == "Pass":
        return "★☆☆☆☆"
    if trade_action == "Wait":
        return "★★☆☆☆"
    if win_prob is None:
        return "★★★☆☆"
    filled = min(5, max(1, int(win_prob / 20)))
    return "★" * filled + "☆" * (5 - filled)


def _opportunity_action(row: dict[str, Any], *, index_trade: str | None = None) -> str:
    sym = row.get("symbol", "")
    ta = row.get("trade_action", "Pass")
    if sym in ("QQQ", "SPY", "TQQQ") and index_trade == "NO TRADE":
        return "Avoid" if ta == "Pass" else "Wait"
    if ta == "BUY":
        return "Trade"
    if ta == "Small":
        return "Trade"
    if ta == "Pass":
        return "Avoid"
    return ta


def build_todays_opportunities(
    ranked: list[dict[str, Any]],
    *,
    direction: str,
    index_trade: str | None = None,
    primary_symbol: str | None = None,
    slot_directions: dict[str, str] | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    slot_dirs = slot_directions or {}
    out: list[dict[str, Any]] = []
    for row in ranked[:limit]:
        sym = row["symbol"]
        act = _opportunity_action(row, index_trade=index_trade)
        dir_label = slot_dirs.get(sym, direction)
        if act in ("Avoid", "Wait"):
            dir_label = act

        stars = _stars_from_win_prob(row.get("win_prob"), row.get("trade_action", "Pass"))
        if act == "Avoid":
            stars = "★☆☆☆☆"
        elif act == "Wait":
            stars = "★★☆☆☆"

        label = f"{sym} {dir_label}" if act == "Trade" else f"{sym} {act}"
        why = row.get("why") or " · ".join(row.get("why_factors") or []) or row.get("edge_type")
        out.append(
            {
                "symbol": sym,
                "direction": dir_label if act == "Trade" else act,
                "stars": stars,
                "label": label,
                "win_prob": row.get("win_prob"),
                "expected_return_pct": row.get("expected_return_pct"),
                "instrument": row.get("instrument"),
                "key_reason": why,
                "action": act,
                "trade_action": row.get("trade_action"),
                "final_score": row.get("final_score"),
                "is_primary": sym == primary_symbol,
            }
        )
    return out


def enrich_trade_slot(
    slot: dict[str, Any],
    *,
    ranked: list[dict[str, Any]],
    trade_action: str | None = None,
    exclude_date: str | None = None,
    vix_chg: float | None = None,
    macro_calendar: dict[str, Any] | None = None,
    session_phase: str | None = "premarket",
) -> dict[str, Any]:
    """Attach transparency fields to a trade slot (primary/secondary)."""
    ta = trade_action or slot.get("trade_action", "Small")
    economics = compute_trade_economics(
        entry_price=slot.get("entry_price"),
        stop_price=slot.get("stop_price"),
        target_price=slot.get("target_price"),
        direction=slot.get("direction", ""),
        win_prob=slot.get("win_prob"),
        trade_action=ta,
    )
    if economics:
        slot["trade_economics"] = economics
        slot["position_sizing"] = compute_position_sizing(economics, trade_action=ta)
        if economics.get("rr_display"):
            slot["rr_display"] = economics["rr_display"]

    slot["why_wins_today"] = why_wins_today(slot, ranked)
    slot["why_not_alternatives"] = why_not_alternatives(slot, ranked)

    if slot.get("rank") == 1:
        slot["rank_summary"] = build_rank_summary(slot, slot["why_wins_today"])

    from src.research.win_rate_calibration import (
        build_feature_signature,
        calibrate_win_prob,
        find_similar_days,
    )

    sig = build_feature_signature(
        direction=slot.get("direction", ""),
        rs_vs_qqq=slot.get("relative_strength"),
        gap_pct=slot.get("gap_pct"),
        vix_chg=vix_chg,
        macro_calendar=macro_calendar,
        symbol=slot.get("symbol"),
    )
    rules_wp = float(slot.get("win_prob") or 50.0)
    calibration = calibrate_win_prob(
        sig,
        rules_win_prob=rules_wp,
        exclude_date=exclude_date,
    )
    slot["win_prob_source"] = calibration.get("win_prob_source", "rules")
    slot["calibration"] = calibration
    if calibration.get("source") == "historical":
        slot["win_prob"] = calibration["calibrated_win_prob"]
        slot["calibrated_win_prob"] = calibration["calibrated_win_prob"]

    similar = find_similar_days(sig, exclude_date=exclude_date, limit=5)
    if similar:
        slot["similar_days"] = similar

    if economics:
        slot["ev_distribution"] = compute_ev_distribution(
            win_prob=slot.get("win_prob"),
            trade_economics=economics,
            similar_days=similar or None,
        )

    from src.research.entry_status import attach_entry_status

    # Morning Ideal Entry stays frozen; status is PREMARKET/PLANNED until live recompute
    attach_entry_status(slot, session_phase=session_phase or "premarket")

    return slot


def build_top5_board(top_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact Top 5 trade board for homepage / today view."""
    out: list[dict[str, Any]] = []
    for slot in top_trades[:5]:
        why_factors = slot.get("why_factors") or []
        key_reason = (
            why_factors[0]
            if why_factors
            else slot.get("edge_type") or slot.get("why_chain") or "—"
        )
        out.append(
            {
                "rank": slot.get("rank"),
                "symbol": slot.get("symbol"),
                "direction": slot.get("direction"),
                "instrument": slot.get("instrument"),
                "win_prob": slot.get("win_prob"),
                "expected_return_pct": slot.get("expected_return_pct"),
                "risk_reward": slot.get("risk_reward"),
                "rr_display": slot.get("rr_display"),
                "entry_price": slot.get("entry_price"),
                "entry_zone": slot.get("entry_zone"),
                "entry_status": slot.get("entry_status"),
                "stop_price": slot.get("stop_price"),
                "target_price": slot.get("target_price"),
                "why_today": slot.get("why_today") or slot.get("why_chain"),
                "key_reason": key_reason,
                "rank_summary": slot.get("rank_summary"),
                "catalyst": slot.get("catalyst"),
                "invalidation": slot.get("invalidation"),
                "trade_action": slot.get("trade_action"),
            }
        )
    return out


def build_decision_transparency(
    *,
    ranked: list[dict[str, Any]],
    best_trades: dict[str, Any],
    edges: dict[str, Any] | None,
    p16_gate: str,
    index_trade: str | None,
    direction: str,
    catalysts: list[dict[str, Any]] | None = None,
    vix_chg: float | None = None,
    qqq_pct: float | None = None,
    smh_pct: float | None = None,
    total_score: int | None = None,
    gap_pct_market: float | None = None,
    macro_calendar: dict[str, Any] | None = None,
    trade_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = best_trades.get("primary")
    top_trades = best_trades.get("top_trades") or []
    slot_directions = {s["symbol"]: s["direction"] for s in top_trades if s.get("direction")}
    todays_opps = build_todays_opportunities(
        ranked,
        direction=direction,
        index_trade=index_trade,
        primary_symbol=primary.get("symbol") if primary else None,
        slot_directions=slot_directions,
    )
    if primary and primary.get("rank_summary"):
        for opp in todays_opps:
            if opp.get("is_primary"):
                opp["rank_summary"] = primary["rank_summary"]
                break

    transparency: dict[str, Any] = {
        "top_trades": build_top5_board(top_trades),
        "watchlist": best_trades.get("watchlist_items") or [],
        "macro_calendar": macro_calendar,
        "trade_plan": trade_plan,
        "todays_opportunities": todays_opps,
        "index_rejection_reasons": index_rejection_reasons(
            index_trade=index_trade or "NO TRADE",
            edges=edges,
            p16_gate=p16_gate,
            ranked=ranked,
            catalysts=catalysts,
            qqq_pct=qqq_pct,
            smh_pct=smh_pct,
            total_score=total_score,
        ),
        "day_risks": day_risks(
            primary=primary,
            ranked=ranked,
            catalysts=catalysts,
            vix_chg=vix_chg,
            total_score=total_score,
            gap_pct_market=gap_pct_market,
        ),
    }

    if primary:
        transparency["why_wins_today"] = primary.get("why_wins_today") or why_wins_today(
            primary, ranked
        )
        transparency["why_not_alternatives"] = primary.get("why_not_alternatives") or (
            why_not_alternatives(primary, ranked)
        )
        transparency["win_prob_breakdown"] = primary.get("win_prob_breakdown")
        transparency["level_reasons"] = primary.get("level_reasons")
        transparency["trade_economics"] = primary.get("trade_economics")
        transparency["position_sizing"] = primary.get("position_sizing")
        transparency["rr_display"] = primary.get("rr_display")
        transparency["rank_summary"] = primary.get("rank_summary")
        transparency["similar_days"] = primary.get("similar_days")
        transparency["ev_distribution"] = primary.get("ev_distribution")
        transparency["win_prob_source"] = primary.get("win_prob_source")
        transparency["calibration"] = primary.get("calibration")

    return transparency
