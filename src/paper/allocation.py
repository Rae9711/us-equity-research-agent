"""Capital allocation policy for paper trading.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Ambition (advisory, **not a guarantee**): pursue setups whose *calibrated*
expected R is positive so the book can *plausibly* compound toward ~10%/month
when edge is real — while keeping stop / correlation / edge-pause controls.

Separate (quant-research discipline):
  - FORECAST: win_prob (model/heuristic)
  - GEOMETRY: entry/stop/target → R:R and remaining upside distance
  - RULE: deploy only when expected_R = p×R − (1−p)×1 clears the EV floor
  - EXECUTION / OUTCOME: fills + realised PnL (never claimed as guaranteed)

Sizing math (comment for the ~10%/mo ambition):
  monthly ≈ n_trades × E[R] × risk_pct
  e.g. 8 trades × 0.40 R expectancy × 3.0% risk ≈ 9.6% — *if* edge is real.
  Raise effective risk only when rolling expectancy is positive; otherwise pause /
  min-risk. ``expected_return_pct`` is *target distance*, not EV — do not gate on it alone.
"""

from __future__ import annotations

from typing import Any

from src.paper.account import BOOK_INTRADAY, BOOK_SWING, STARTING_CASH, open_positions
from src.paper.correlation import correlation_scale

# Cash buffer: ~80% of equity can be deployed; reserve band 10–20%.
CASH_RESERVE_MIN_PCT = 10.0
CASH_RESERVE_MAX_PCT = 20.0

# Hard caps (scaled by setup quality + proven edge). Solo high-conviction books
# can deploy more of the ~80% deployable sleeve without dumping it into one ticket.
# When rolling edge is positive, risk can rise toward MONTHLY_TARGET math (≤ MAX).
BASE_RISK_PCT = 1.5
MAX_RISK_PCT = 3.0
MIN_RISK_PCT = 0.35
MAX_POSITION_PCT_INTRADAY = 50.0
MAX_POSITION_PCT_SWING = 70.0
MAX_TOTAL_DEPLOYED_PCT = 90.0  # 100 - CASH_RESERVE_MIN

# Dual-book default split of *deployable* capital (after cash reserve)
SWING_CORE_SHARE = 0.60
INTRADAY_SATELLITE_SHARE = 0.40

# --- Economic gates (EV / expected R is primary; geometric upside is soft) ---
# expected_R = p × reward_R − (1−p) × 1.0   (reward_R = R:R from levels)
MIN_EXPECTED_R = 0.15
# Soft floor on distance-to-target (%). NOT the primary EV gate.
MIN_GEOMETRIC_UPSIDE_PCT = 1.5
# Enforce payoff asymmetry before entry (avg loss ≫ avg win fix starts here).
MIN_RR_TO_DEPLOY = 1.5
MIN_WIN_PROB_TO_DEPLOY = 52.0
# Softened companions when expected_R already clears.
SOFT_WIN_PROB_WHEN_EV_OK = 50.0
SOFT_RR_WHEN_EV_OK = 1.35
# Skip new risk when recent closed expectancy is broken.
EDGE_PAUSE_MIN_TRADES = 5
EDGE_PAUSE_MAX_PF = 0.9
# Reject setups whose stop is so wide that one loss dominates the day.
MAX_STOP_PCT_INTRADAY = 5.0
MAX_STOP_PCT_SWING = 8.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def expected_r(
    *,
    win_prob: float | None,
    rr: float | None,
) -> float | None:
    """Calibrated expected R multiples: p×R − (1−p)×1.

    This is the economic edge estimate. ``expected_return_pct`` (target distance)
    is *not* EV and must not be used as the sole deploy gate.
    """
    if win_prob is None or rr is None:
        return None
    p = float(win_prob) / 100.0
    reward = float(rr)
    if reward <= 0 or p < 0 or p > 1:
        return None
    return round(p * reward - (1.0 - p) * 1.0, 4)


def compute_rr(
    *,
    entry: float | None,
    stop: float | None,
    target: float | None,
    direction: str,
    fallback: float | None = None,
) -> float | None:
    """Reward:risk from levels; fall back to provided R:R if levels incomplete."""
    if fallback is not None and fallback > 0:
        # Prefer explicit if levels missing
        if entry is None or stop is None or target is None:
            return float(fallback)
    if entry is None or stop is None or target is None or entry <= 0:
        return float(fallback) if fallback and fallback > 0 else None
    risk = abs(entry - stop)
    if risk <= 0:
        return float(fallback) if fallback and fallback > 0 else None
    d = (direction or "LONG").upper()
    if d == "SHORT":
        reward = abs(entry - target)
    else:
        reward = abs(target - entry)
    return round(reward / risk, 2)


def remaining_er_pct(
    *,
    price: float | None,
    entry: float | None,
    target: float | None,
    direction: str,
    planned_er: float | None = None,
) -> float | None:
    """Geometric upside still left vs current price (distance to target, not EV)."""
    if price is None or price <= 0 or target is None:
        return planned_er
    d = (direction or "LONG").upper()
    if d == "SHORT":
        rem = (price - target) / price * 100.0
    else:
        rem = (target - price) / price * 100.0
    rem = max(0.0, rem)
    if planned_er is not None and entry is not None and entry > 0:
        # Cap by planned ER so we don't overstate after a gap toward target
        planned = abs(float(planned_er))
        return round(min(rem, planned * 1.15), 2)
    return round(rem, 2)


def setup_confidence(
    *,
    win_prob: float | None,
    rr: float | None,
    remaining_er: float | None,
    entry_status: str | None,
    horizon: str,
    expected_r_val: float | None = None,
) -> float:
    """0–1 confidence used for cash reserve + sizing multipliers."""
    wp = float(win_prob or 50.0)
    score = 0.0
    # Win prob ~35% of score
    score += _clamp((wp - 40.0) / 40.0, 0.0, 1.0) * 0.35
    # Expected R / R:R ~30%
    if expected_r_val is not None:
        score += _clamp((float(expected_r_val) + 0.1) / 0.8, 0.0, 1.0) * 0.30
    elif rr is not None:
        score += _clamp((float(rr) - 0.8) / 2.2, 0.0, 1.0) * 0.25
    else:
        score += 0.06
    # Remaining geometric upside ~20%
    if remaining_er is not None:
        score += _clamp(float(remaining_er) / 4.0, 0.0, 1.0) * 0.20
    else:
        score += 0.06
    # Entry status ~15%
    st = (entry_status or "").upper()
    if st in ("READY", "TRIGGERED"):
        score += 0.15
    elif st == "ACTIVE":
        score += 0.06
    else:
        score += 0.0
    # Slight bump for swing (core thesis) when edge exists
    if "swing" in (horizon or "").lower() and score > 0.35:
        score = min(1.0, score + 0.05)
    return round(_clamp(score, 0.0, 1.0), 3)


def cash_reserve_pct_for_confidence(confidence: float, *, open_book_count: int = 0) -> float:
    """Higher confidence → lower cash floor (still within 10–20%)."""
    # conf 0 → 20% (80% deployable), conf 1 → 10% (90% deployable)
    pct = CASH_RESERVE_MAX_PCT - confidence * (CASH_RESERVE_MAX_PCT - CASH_RESERVE_MIN_PCT)
    # Already in one book: keep a bit more buffer for the second book / slippage
    if open_book_count >= 1:
        pct = min(CASH_RESERVE_MAX_PCT, pct + 2.0)
    return round(_clamp(pct, CASH_RESERVE_MIN_PCT, CASH_RESERVE_MAX_PCT), 1)


def risk_and_cap_pct(
    *,
    confidence: float,
    win_prob: float | None,
    rr: float | None,
    remaining_er: float | None,
    horizon: str,
    base_risk_pct: float = BASE_RISK_PCT,
    base_max_position_pct: float | None = None,
    sole_book: bool = True,
    expected_r_val: float | None = None,
    edge_positive: bool = False,
) -> tuple[float, float, list[str]]:
    """Return (risk_pct, max_position_pct, rationale bullets).

    When ``edge_positive`` (rolling expectancy > 0), allow risk toward MAX so
    high-EV setups can actually use the ~80% deployable sleeve (not stuck at
    1% risk → ~30% notional forever).
    """
    reasons: list[str] = []
    wp = float(win_prob or 50.0)
    is_swing = "swing" in (horizon or "").lower()
    horizon_cap = MAX_POSITION_PCT_SWING if is_swing else MAX_POSITION_PCT_INTRADAY
    risk = float(base_risk_pct)
    risk_cap = MAX_RISK_PCT if edge_positive else min(MAX_RISK_PCT, max(base_risk_pct * 1.25, 2.0))

    # Win-prob scale
    if wp >= 72:
        risk *= 1.35
        reasons.append(f"胜率 {wp:.0f}% → 加仓")
    elif wp >= 62:
        risk *= 1.15
        reasons.append(f"胜率 {wp:.0f}% → 标准偏多")
    elif wp >= 55:
        risk *= 1.0
        reasons.append(f"胜率 {wp:.0f}% → 标准")
    elif wp >= MIN_WIN_PROB_TO_DEPLOY:
        risk *= 0.7
        reasons.append(f"胜率 {wp:.0f}% → 减仓")
    else:
        risk *= 0.4
        reasons.append(f"胜率 {wp:.0f}% 偏低 → 轻仓/保留现金")

    # Expected R (primary economic signal) / R:R
    if expected_r_val is not None:
        if expected_r_val < MIN_EXPECTED_R:
            risk *= 0.35
            reasons.append(f"期望R {expected_r_val:.2f} 过低（需≥{MIN_EXPECTED_R:.2f}）")
        elif expected_r_val >= 0.40:
            risk *= 1.25 if edge_positive else 1.15
            reasons.append(f"期望R {expected_r_val:.2f} 充足")
        else:
            risk *= 1.05
            reasons.append(f"期望R {expected_r_val:.2f} 合格")
    elif rr is not None:
        if rr >= 2.5:
            risk *= 1.15
            reasons.append(f"R:R {rr:.1f} 优秀")
        elif rr >= MIN_RR_TO_DEPLOY:
            risk *= 1.0
            reasons.append(f"R:R {rr:.1f} 合格")
        else:
            risk *= 0.45
            reasons.append(f"R:R {rr:.1f} 偏弱 → 轻仓")

    # Soft geometric upside (distance to target — not EV)
    if remaining_er is not None:
        if remaining_er < MIN_GEOMETRIC_UPSIDE_PCT:
            risk *= 0.40
            reasons.append(f"几何上行 {remaining_er:.2f}% 过小（软门槛≥{MIN_GEOMETRIC_UPSIDE_PCT:.1f}%）")
        elif remaining_er >= 4.0:
            risk *= 1.10
            reasons.append(f"几何上行 {remaining_er:.1f}% 充足")

    if edge_positive:
        risk *= 1.10
        reasons.append("滚动期望为正 → 提高风险预算（朝月10%目标，非保证）")

    # Confidence blend
    risk *= 0.75 + 0.5 * confidence
    risk = _clamp(risk, MIN_RISK_PCT, risk_cap)

    if base_max_position_pct is not None:
        cap = min(float(base_max_position_pct), float(horizon_cap))
    else:
        cap = float(horizon_cap)
    # Scale cap with confidence. Solo book + proven edge → fuller use of ~80% sleeve.
    solo_boost = 1.0 if sole_book else 0.0
    edge_boost = 0.10 if edge_positive else 0.0
    scale = (0.65 + 0.35 * confidence) + (0.15 * solo_boost * confidence) + edge_boost
    cap = cap * _clamp(scale, 0.55, 1.15)
    cap = _clamp(cap, 12.0, horizon_cap)
    if is_swing:
        reasons.append(f"Swing 核心仓上限 {cap:.0f}%")
    else:
        reasons.append(f"Intraday 卫星仓上限 {cap:.0f}%")

    return round(risk, 2), round(cap, 1), reasons


def should_hold_cash(
    *,
    action: str,
    win_prob: float | None,
    rr: float | None,
    remaining_er: float | None,
    entry_status: str | None,
    confidence: float,
    stop_pct: float | None = None,
    horizon: str = "",
    edge_stats: dict[str, Any] | None = None,
    expected_r_val: float | None = None,
) -> tuple[bool, str]:
    """Deliberate cash hold even if Entry Status is enterable.

    Primary economic gate = calibrated expected_R (and hard min R:R).
    Geometric upside (remaining_er / expected_return_pct) is a soft companion only.
    """
    if action in ("wait", "skip"):
        return False, ""
    st = (entry_status or "").upper()
    if st not in ("READY", "TRIGGERED"):
        return False, ""

    # Broken recent edge → sit in cash until expectancy recovers.
    stats = edge_stats or {}
    n = int(stats.get("n") or 0)
    pf = float(stats.get("profit_factor") or 0.0)
    exp = float(stats.get("expectancy") or 0.0)
    if n >= EDGE_PAUSE_MIN_TRADES and (pf < EDGE_PAUSE_MAX_PF or exp < 0):
        return True, (
            f"保留现金：近{n}笔期望为负/PF={pf:.2f}<{EDGE_PAUSE_MAX_PF:.1f}，暂停新开仓"
        )

    # Hard R:R — payoff asymmetry must be in the plan before risking capital.
    if rr is not None and rr < MIN_RR_TO_DEPLOY:
        return True, (
            f"保留现金：R:R {rr:.2f} 低于硬门槛 {MIN_RR_TO_DEPLOY:.1f}"
            f"（需先保证盈亏不对称）"
        )

    # Primary EV gate.
    ev = expected_r_val
    if ev is None:
        ev = expected_r(win_prob=win_prob, rr=rr)
    if ev is not None and ev < MIN_EXPECTED_R:
        return True, (
            f"保留现金：期望R={ev:.2f} < {MIN_EXPECTED_R:.2f}"
            f"（EV=p×R−(1−p)×1，非目标距离ER）"
        )
    if ev is None:
        # Cannot form EV without win_prob + R:R — refuse to size blindly.
        return True, "保留现金：缺少胜率或R:R，无法计算期望R"

    # Soft geometric upside (distance to target).
    if remaining_er is not None and remaining_er < MIN_GEOMETRIC_UPSIDE_PCT:
        return True, (
            f"保留现金：几何上行仅剩 {remaining_er:.2f}%"
            f"（软门槛≥{MIN_GEOMETRIC_UPSIDE_PCT:.1f}%）"
        )

    ev_ok = ev >= MIN_EXPECTED_R
    min_wp = SOFT_WIN_PROB_WHEN_EV_OK if ev_ok else MIN_WIN_PROB_TO_DEPLOY

    wp = float(win_prob or 0.0)
    if wp < min_wp:
        return True, f"保留现金：胜率 {wp:.0f}% 不足 {min_wp:.0f}%"
    if confidence < 0.32:
        return True, f"保留现金：综合信心 {confidence:.0%} 偏低，等更好 setup"

    is_swing = "swing" in (horizon or "").lower()
    max_stop = MAX_STOP_PCT_SWING if is_swing else MAX_STOP_PCT_INTRADAY
    if stop_pct is not None and stop_pct > max_stop:
        return True, (
            f"保留现金：止损过宽 {stop_pct:.1f}% > {max_stop:.0f}%（单笔亏损过大）"
        )
    return False, ""


def book_budget_share(
    book: str,
    *,
    other_book_open: bool,
    both_entering: bool,
) -> float:
    """Fraction of *deployable* capital allocated to this book."""
    if book == BOOK_SWING:
        if both_entering or other_book_open:
            return SWING_CORE_SHARE
        return 1.0
    # intraday
    if both_entering or other_book_open:
        return INTRADAY_SATELLITE_SHARE
    return 1.0


def allocate_for_entry(
    account: dict[str, Any],
    *,
    book: str,
    signal: dict[str, Any],
    quote: float,
    entry_status: dict[str, Any] | str | None = None,
    peer_entering: bool = False,
) -> dict[str, Any]:
    """Compute allocation decision for one potential entry.

    Returns dict with:
      hold_cash, cash_reserve_pct, risk_pct, max_position_pct,
      budget_share, deployable_cash, reason_zh, factors, expected_r, ...
    """
    params = account.get("params") or {}
    equity = float(account.get("equity") or account.get("cash") or STARTING_CASH)
    cash = float(account.get("cash") or 0.0)
    open_rows = open_positions(account)
    open_count = len(open_rows)

    es = entry_status
    if isinstance(es, dict):
        status = es.get("status")
    else:
        status = es
    if status is None and isinstance(signal.get("entry_status"), dict):
        status = signal["entry_status"].get("status")

    direction = str(signal.get("direction") or "LONG").upper()
    horizon = str(signal.get("horizon") or ("Swing" if book == BOOK_SWING else "Intraday"))
    entry = _safe_float(signal.get("entry_price")) or quote
    stop = _safe_float(signal.get("stop_price"))
    target = _safe_float(signal.get("target_price"))
    win_prob = _safe_float(signal.get("win_prob"))
    planned_er = _safe_float(signal.get("expected_return_pct"))
    rr = compute_rr(
        entry=entry,
        stop=stop,
        target=target,
        direction=direction,
        fallback=_safe_float(signal.get("risk_reward")),
    )
    rem_er = remaining_er_pct(
        price=quote,
        entry=entry,
        target=target,
        direction=direction,
        planned_er=planned_er,
    )
    ev_r = expected_r(win_prob=win_prob, rr=rr)
    conf = setup_confidence(
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        entry_status=str(status) if status else None,
        horizon=horizon,
        expected_r_val=ev_r,
    )
    reserve_pct = cash_reserve_pct_for_confidence(conf, open_book_count=open_count)
    # Allow params override bounds
    reserve_pct = _clamp(
        reserve_pct,
        float(params.get("cash_reserve_min_pct") or CASH_RESERVE_MIN_PCT),
        float(params.get("cash_reserve_max_pct") or CASH_RESERVE_MAX_PCT),
    )

    stop_pct = None
    if entry and stop and entry > 0:
        stop_pct = abs(entry - stop) / entry * 100.0

    from src.paper.journal import _closed_trade_pnls, rolling_expectancy

    edge_stats = rolling_expectancy(_closed_trade_pnls(account))
    edge_positive = (
        int(edge_stats.get("n") or 0) >= EDGE_PAUSE_MIN_TRADES
        and float(edge_stats.get("expectancy") or 0) > 0
        and float(edge_stats.get("profit_factor") or 0) >= 1.1
    )

    hold, hold_reason = should_hold_cash(
        action="enter",
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        entry_status=str(status) if status else None,
        confidence=conf,
        stop_pct=stop_pct,
        horizon=horizon,
        edge_stats=edge_stats,
        expected_r_val=ev_r,
    )

    base_risk = float(params.get("risk_pct") or BASE_RISK_PCT)
    base_cap = _safe_float(params.get("max_position_pct"))
    other_book = BOOK_SWING if book == BOOK_INTRADAY else BOOK_INTRADAY
    other_open = any(b == other_book for b, _ in open_rows)
    sole_book = not other_open and not peer_entering
    risk_pct, max_pos_pct, size_reasons = risk_and_cap_pct(
        confidence=conf,
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        horizon=horizon,
        base_risk_pct=base_risk,
        base_max_position_pct=base_cap,
        sole_book=sole_book,
        expected_r_val=ev_r,
        edge_positive=edge_positive,
    )

    share = book_budget_share(
        book, other_book_open=other_open, both_entering=peer_entering
    )

    # Deployable = equity not reserved as cash floor, then book share
    reserved = equity * (reserve_pct / 100.0)
    deployable_total = max(0.0, equity - reserved)
    # Also never deploy more than cash currently available above a soft floor
    cash_above_floor = max(0.0, cash - reserved * (cash / equity if equity > 0 else 1.0))
    book_budget = deployable_total * share
    # Cap by available cash (longs need cash)
    deployable_cash = min(cash, max(cash_above_floor, cash * share), book_budget)
    # Ensure we don't push cash below reserve after entry (approx for LONG)
    min_cash_after = reserved
    max_notional_by_reserve = max(0.0, cash - min_cash_after)
    deployable_cash = min(deployable_cash, max_notional_by_reserve)

    # Correlation / concentration guard vs the other open book.
    corr_note = None
    corr_scale = 1.0
    if bool(params.get("correlation_guard", True)):
        for other_b, other_pos in open_rows:
            if other_b == book:
                continue
            corr_scale, corr_note = correlation_scale(
                new_symbol=str(signal.get("symbol") or ""),
                new_direction=direction,
                open_symbol=str(other_pos.get("symbol") or ""),
                open_direction=str(other_pos.get("direction") or "LONG"),
            )
            if corr_scale < 1.0:
                break
        if corr_scale <= 0.0:
            hold = True
            hold_reason = corr_note or "集中度过高，保留现金"
        elif corr_scale < 1.0:
            deployable_cash = round(deployable_cash * corr_scale, 2)
            max_pos_pct = round(max_pos_pct * corr_scale, 1)

    book_zh = "长线" if book == BOOK_SWING else "短线"
    if hold:
        reason_zh = hold_reason
    else:
        pos_hint = min(max_pos_pct, (deployable_cash / equity * 100.0) if equity else 0)
        ev_txt = f"期望R {ev_r:.2f}" if ev_r is not None else "期望R n/a"
        reason_zh = (
            f"为何用约 {pos_hint:.0f}% 仓位 / 留 {reserve_pct:.0f}% 现金"
            f"（{book_zh}预算 {share:.0%}·信心 {conf:.0%}·{ev_txt}）："
            + "；".join(size_reasons[:4])
        )
        if corr_note:
            reason_zh = f"{reason_zh}；{corr_note}"

    return {
        "hold_cash": hold,
        "cash_reserve_pct": reserve_pct,
        "risk_pct": risk_pct,
        "max_position_pct": max_pos_pct,
        "budget_share": round(share, 2),
        "deployable_cash": round(deployable_cash, 2),
        "deployable_total": round(deployable_total, 2),
        "confidence": conf,
        "win_prob": win_prob,
        "rr": rr,
        "expected_r": ev_r,
        "remaining_er_pct": rem_er,
        "edge_positive": edge_positive,
        "corr_scale": round(corr_scale, 2),
        "corr_note": corr_note,
        "entry_status": status,
        "horizon": horizon,
        "book": book,
        "reason_zh": reason_zh,
        "factors": size_reasons,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "open_books": open_count,
        "advisory": True,
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }


def portfolio_allocation_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    """Cash / position mix for UI."""
    equity = float(account.get("equity") or 0.0) or float(
        account.get("cash") or STARTING_CASH
    )
    cash = float(account.get("cash") or 0.0)
    cash_pct = (cash / equity * 100.0) if equity > 0 else 100.0
    invested = 0.0
    legs: list[dict[str, Any]] = []
    for book, pos in open_positions(account):
        mv = _safe_float(pos.get("market_value"))
        if mv is None:
            shares = float(pos.get("shares") or 0)
            px = _safe_float(pos.get("last_price")) or _safe_float(pos.get("avg_entry")) or 0
            mv = shares * px
        invested += float(mv or 0)
        legs.append(
            {
                "book": book,
                "symbol": pos.get("symbol"),
                "shares": pos.get("shares"),
                "market_value": round(float(mv or 0), 2),
                "pct": round(float(mv or 0) / equity * 100.0, 1) if equity else 0,
                "horizon": pos.get("horizon"),
            }
        )
    position_pct = (invested / equity * 100.0) if equity > 0 else 0.0
    last_alloc = None
    for d in reversed(account.get("decisions") or []):
        if d.get("allocation"):
            last_alloc = d["allocation"]
            break
    return {
        "cash": round(cash, 2),
        "equity": round(equity, 2),
        "cash_pct": round(cash_pct, 1),
        "position_pct": round(position_pct, 1),
        "invested": round(invested, 2),
        "legs": legs,
        "last_allocation": last_alloc,
        "allocation_reason": (last_alloc or {}).get("reason_zh"),
    }
