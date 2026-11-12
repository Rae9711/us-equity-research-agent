"""Capital allocation policy for paper trading.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Each tick the agent chooses:
  - cash reserve floor (5–15% of equity, regime / confidence dependent)
  - position size from win_prob, R:R, horizon, Entry Status, remaining ER
  - whether to hold cash deliberately (SKIP/WAIT · 保留现金)
  - dual-book budget split: swing core + intraday satellite when both clean
"""

from __future__ import annotations

from typing import Any

from src.paper.account import BOOK_INTRADAY, BOOK_SWING, STARTING_CASH, open_positions

# Deployable band: always keep at least this much cash; never exceed this reserve.
CASH_RESERVE_MIN_PCT = 5.0
CASH_RESERVE_MAX_PCT = 15.0

# Hard caps (still soft vs fixed 1%/25% — scaled by setup quality)
BASE_RISK_PCT = 1.5
MAX_RISK_PCT = 3.0
MIN_RISK_PCT = 0.35
MAX_POSITION_PCT_INTRADAY = 40.0
MAX_POSITION_PCT_SWING = 70.0
MAX_TOTAL_DEPLOYED_PCT = 95.0  # 100 - CASH_RESERVE_MIN

# Portfolio-level cap on combined *planned* risk across both books (fix 5):
# the sum of every open position's entry→stop dollars plus a new entry's
# planned risk must stay within this % of equity. This is separate from the
# per-trade 3% clamp in risk_and_cap_pct — one 3% trade plus another 3%
# trade would otherwise put 6% of equity at risk simultaneously.
MAX_PORTFOLIO_PLANNED_RISK_PCT = 3.0

# Dual-book default split of *deployable* capital (after cash reserve)
SWING_CORE_SHARE = 0.60
INTRADAY_SATELLITE_SHARE = 0.40

# Edge thresholds for deliberate cash hold
MIN_WIN_PROB_TO_DEPLOY = 48.0
MIN_RR_TO_DEPLOY = 0.85
MIN_REMAINING_ER_PCT = 0.35


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
    """Upside still left vs current price (or planned ER scaled by path remaining)."""
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
) -> float:
    """0–1 confidence used for cash reserve + sizing multipliers."""
    wp = float(win_prob or 50.0)
    score = 0.0
    # Win prob ~40% of score
    score += _clamp((wp - 40.0) / 40.0, 0.0, 1.0) * 0.40
    # R:R ~25%
    if rr is not None:
        score += _clamp((float(rr) - 0.8) / 2.2, 0.0, 1.0) * 0.25
    else:
        score += 0.08
    # Remaining ER ~20%
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
    """Higher confidence → lower cash floor (still within 5–15%)."""
    # conf 0 → 15%, conf 1 → 5%
    pct = CASH_RESERVE_MAX_PCT - confidence * (CASH_RESERVE_MAX_PCT - CASH_RESERVE_MIN_PCT)
    # If already holding one book, nudge reserve up slightly (leave room / buffer)
    if open_book_count >= 1:
        pct = min(CASH_RESERVE_MAX_PCT, pct + 3.0)
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
) -> tuple[float, float, list[str]]:
    """Return (risk_pct, max_position_pct, rationale bullets)."""
    reasons: list[str] = []
    wp = float(win_prob or 50.0)
    is_swing = "swing" in (horizon or "").lower()
    risk = float(base_risk_pct)

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

    # R:R scale
    if rr is not None:
        if rr >= 2.5:
            risk *= 1.15
            reasons.append(f"R:R {rr:.1f} 优秀")
        elif rr >= 1.5:
            risk *= 1.0
            reasons.append(f"R:R {rr:.1f} 合格")
        elif rr >= MIN_RR_TO_DEPLOY:
            risk *= 0.75
            reasons.append(f"R:R {rr:.1f} 一般 → 减仓")
        else:
            risk *= 0.45
            reasons.append(f"R:R {rr:.1f} 偏弱 → 轻仓")

    # Remaining ER
    if remaining_er is not None:
        if remaining_er < MIN_REMAINING_ER_PCT:
            risk *= 0.35
            reasons.append(f"剩余空间 {remaining_er:.2f}% 过小")
        elif remaining_er < 1.0:
            risk *= 0.7
            reasons.append(f"剩余空间 {remaining_er:.2f}% → 减仓")
        elif remaining_er >= 3.0:
            risk *= 1.05
            reasons.append(f"剩余空间 {remaining_er:.1f}% 充足")

    # Confidence blend
    risk *= 0.75 + 0.5 * confidence
    risk = _clamp(risk, MIN_RISK_PCT, MAX_RISK_PCT)

    if base_max_position_pct is not None:
        cap = float(base_max_position_pct)
    else:
        cap = MAX_POSITION_PCT_SWING if is_swing else MAX_POSITION_PCT_INTRADAY
    # Scale cap with confidence (weaker setups get tighter notional)
    cap = cap * (0.55 + 0.55 * confidence)
    cap = _clamp(
        cap,
        8.0,
        MAX_POSITION_PCT_SWING if is_swing else MAX_POSITION_PCT_INTRADAY,
    )
    if is_swing:
        reasons.append(f"Swing 核心仓上限 {cap:.0f}%")
    else:
        reasons.append(f"Intraday 卫星仓上限 {cap:.0f}%")

    return round(risk, 2), round(cap, 1), reasons


def book_notional_cap_pct(book: str, confidence: float) -> float:
    """Per-book TOTAL-notional cap: 40% intraday / 70% swing x confidence
    factor — shared by all the book's legs (fix 7).

    Deliberately NOT the per-entry ``max_position_pct`` (which the account
    ``max_position_pct`` param further tightens to a 25% base): the book
    cap is the book's own limit, and each new leg must fit BOTH its own
    per-entry cap and the headroom left under this book cap.
    """
    base = MAX_POSITION_PCT_SWING if book == BOOK_SWING else MAX_POSITION_PCT_INTRADAY
    return _clamp(base * (0.55 + 0.55 * float(confidence or 0.0)), 8.0, base)


def should_hold_cash(
    *,
    action: str,
    win_prob: float | None,
    rr: float | None,
    remaining_er: float | None,
    entry_status: str | None,
    confidence: float,
) -> tuple[bool, str]:
    """Deliberate cash hold even if Entry Status is enterable."""
    if action in ("wait", "skip"):
        return False, ""
    st = (entry_status or "").upper()
    if st not in ("READY", "TRIGGERED"):
        return False, ""
    wp = float(win_prob or 0.0)
    if wp < MIN_WIN_PROB_TO_DEPLOY:
        return True, f"保留现金：胜率 {wp:.0f}% 不足 {MIN_WIN_PROB_TO_DEPLOY:.0f}%"
    if rr is not None and rr < MIN_RR_TO_DEPLOY:
        return True, f"保留现金：R:R {rr:.2f} 过低"
    if remaining_er is not None and remaining_er < MIN_REMAINING_ER_PCT:
        return True, f"保留现金：目标空间仅剩 {remaining_er:.2f}%"
    if confidence < 0.28:
        return True, f"保留现金：综合信心 {confidence:.0%} 偏低，等更好 setup"
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
      budget_share, deployable_cash, reason_zh, factors, ...
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
    conf = setup_confidence(
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        entry_status=str(status) if status else None,
        horizon=horizon,
    )
    reserve_pct = cash_reserve_pct_for_confidence(conf, open_book_count=open_count)
    # Allow params override bounds
    reserve_pct = _clamp(
        reserve_pct,
        float(params.get("cash_reserve_min_pct") or CASH_RESERVE_MIN_PCT),
        float(params.get("cash_reserve_max_pct") or CASH_RESERVE_MAX_PCT),
    )

    hold, hold_reason = should_hold_cash(
        action="enter",
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        entry_status=str(status) if status else None,
        confidence=conf,
    )

    base_risk = float(params.get("risk_pct") or BASE_RISK_PCT)
    base_cap = _safe_float(params.get("max_position_pct"))
    risk_pct, max_pos_pct, size_reasons = risk_and_cap_pct(
        confidence=conf,
        win_prob=win_prob,
        rr=rr,
        remaining_er=rem_er,
        horizon=horizon,
        base_risk_pct=base_risk,
        base_max_position_pct=base_cap,
    )

    other_book = BOOK_SWING if book == BOOK_INTRADAY else BOOK_INTRADAY
    other_open = any(b == other_book for b, _ in open_rows)
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

    book_zh = "长线" if book == BOOK_SWING else "短线"
    if hold:
        reason_zh = hold_reason
    else:
        pos_hint = min(max_pos_pct, (deployable_cash / equity * 100.0) if equity else 0)
        reason_zh = (
            f"为何用约 {pos_hint:.0f}% 仓位 / 留 {reserve_pct:.0f}% 现金"
            f"（{book_zh}预算 {share:.0%}·信心 {conf:.0%}）："
            + "；".join(size_reasons[:4])
        )

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
        "remaining_er_pct": rem_er,
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


def planned_risk_dollars(
    *,
    shares: float,
    entry: float | None,
    stop: float | None,
    direction: str = "LONG",
) -> float | None:
    """Dollars lost if the stop is hit (plan risk for one position).

    Mirrors the sizing fallback in ``broker_sim.size_shares``: with no
    usable stop, plan risk is 2% of entry per share.
    """
    entry_f = _safe_float(entry)
    if entry_f is None or entry_f <= 0 or shares is None or float(shares) <= 0:
        return None
    stop_f = _safe_float(stop)
    if stop_f is not None and stop_f > 0:
        per_share = abs(entry_f - stop_f)
    else:
        per_share = entry_f * 0.02
    if per_share <= 0:
        return None
    return round(float(shares) * per_share, 2)


def portfolio_planned_risk(account: dict[str, Any]) -> dict[str, Any]:
    """Combined planned risk of all open positions vs the portfolio cap."""
    params = account.get("params") or {}
    equity = float(account.get("equity") or account.get("cash") or STARTING_CASH)
    cap_pct = float(
        params.get("max_portfolio_planned_risk_pct")
        or MAX_PORTFOLIO_PLANNED_RISK_PCT
    )
    cap_dollars = round(equity * cap_pct / 100.0, 2)
    legs: list[dict[str, Any]] = []
    total = 0.0
    for book, pos in open_positions(account):
        risk = planned_risk_dollars(
            shares=pos.get("shares"),
            entry=pos.get("avg_entry"),
            stop=pos.get("stop"),
            direction=pos.get("direction") or "LONG",
        )
        legs.append(
            {
                "book": book,
                "symbol": pos.get("symbol"),
                "direction": pos.get("direction"),
                "shares": pos.get("shares"),
                "planned_risk": risk,
            }
        )
        if risk is not None:
            total += risk
    total = round(total, 2)
    return {
        "equity": round(equity, 2),
        "cap_pct": cap_pct,
        "cap_dollars": cap_dollars,
        "total_planned_risk": total,
        "headroom": round(max(0.0, cap_dollars - total), 2),
        "legs": legs,
    }


def apply_portfolio_risk_cap(
    account: dict[str, Any],
    *,
    direction: str,
    entry_price: float,
    stop_price: float | None,
    shares: int,
) -> tuple[int, dict[str, Any]]:
    """Reduce ``shares`` so existing + new planned risk stays in the cap.

    Returns ``(shares, detail)``. ``detail["capped"]`` is True when the
    request was reduced; ``shares == 0`` means even one share would breach
    the cap and the entry must be skipped.
    """
    detail = portfolio_planned_risk(account)
    cap_dollars = detail["cap_dollars"]
    existing = detail["total_planned_risk"]
    per_share = planned_risk_dollars(
        shares=1, entry=entry_price, stop=stop_price, direction=direction
    )
    detail["new_per_share_risk"] = per_share
    detail["requested_shares"] = int(shares)
    if per_share is None or per_share <= 0:
        detail["capped"] = False
        detail["reason_zh"] = ""
        return int(shares), detail
    new_risk = round(float(shares) * per_share, 2)
    detail["new_planned_risk"] = new_risk
    detail["combined_planned_risk"] = round(existing + new_risk, 2)
    if existing + new_risk <= cap_dollars + 1e-9:
        detail["capped"] = False
        detail["reason_zh"] = ""
        return int(shares), detail
    headroom = max(0.0, cap_dollars - existing)
    max_shares = int(headroom // per_share)
    detail["capped"] = True
    if max_shares <= 0:
        detail["allowed_shares"] = 0
        detail["reason_zh"] = (
            f"组合计划风险已满：已有持仓计划风险 ${existing:.2f}，"
            f"上限 ${cap_dollars:.2f}（{detail['cap_pct']:g}% equity），无法新开仓"
        )
        return 0, detail
    detail["allowed_shares"] = max_shares
    detail["reason_zh"] = (
        f"组合计划风险上限减仓：已有 ${existing:.2f} + 新开 ${new_risk:.2f} "
        f"> 上限 ${cap_dollars:.2f}，股数 {shares} → {max_shares}"
    )
    return max_shares, detail


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
