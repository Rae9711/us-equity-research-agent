"""Event-driven backtest for the paper-trading *execution* engine.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Unlike ``src/jobs/backtest_replay.py`` (a QQQ-return proxy that never touches the
trader), this harness drives the REAL profitability-determining code path:

  broker_sim (frictions + fills) · exits (scale-out / breakeven / trailing) ·
  allocation (risk sizing + cash reserve) · correlation guard

over point-in-time daily OHLC bars.

Separation of concerns (per quant-research discipline):
  - FACTS: historical OHLC bars (point-in-time, auto-adjusted).
  - FEATURES: SMA / ATR computed only from bars strictly before the entry day.
  - FORECAST: the pluggable ``SignalProvider`` (default is a MOMENTUM-BREAKOUT
    PROXY — an ASSUMPTION standing in for the research pipeline's alpha, which
    cannot be replayed historically without raw data).
  - RULES + EXECUTION: the real paper engine (this is what we are validating).
  - OUTCOMES: realised fills with slippage/commission → equity curve + metrics.

Intraday path is modelled conservatively: within a day the ADVERSE extreme is
assumed to trade before the favourable one, so stops are penalised, not gifted.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Protocol

from src.paper.account import (
    BOOK_INTRADAY,
    BOOK_SWING,
    STARTING_CASH,
    append_equity_point,
    default_account,
    get_position,
    mark_to_market,
    open_positions,
)
from src.paper.allocation import allocate_for_entry
from src.paper.broker_sim import (
    InsufficientCashError,
    NonsensePriceError,
    can_afford,
    execute_entry,
    execute_exit,
)
from src.paper.execution_agent import _stop_hit, _target_hit
from src.paper.exits import plan_eod_exit, plan_exit, runner_target
from src.utils.paths import reports_dir

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA",
    "AMD", "MU", "AVGO", "ARM",
    "SPY", "QQQ", "TQQQ", "DIA", "SMH", "XLK",
]
BENCHMARK = "QQQ"


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class SignalProvider(Protocol):
    def __call__(
        self, symbol: str, history: list[Bar], today: Bar
    ) -> dict[str, Any] | None: ...


# --------------------------------------------------------------------------- #
# Features (point-in-time only)
# --------------------------------------------------------------------------- #
def _sma(bars: list[Bar], period: int) -> float | None:
    if len(bars) < period:
        return None
    return sum(b.close for b in bars[-period:]) / period


def _atr(bars: list[Bar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


class MomentumBreakoutProvider:
    """PROXY alpha: trend + prior-day breakout, ATR-based levels.

    This is a documented ASSUMPTION, not the real research signal. It exists so
    the execution engine can be measured end-to-end. Long-only by default (the
    universe is high-beta; shorting it blindly is not a realistic personal edge).
    """

    def __init__(
        self,
        *,
        sma_period: int = 20,
        atr_period: int = 14,
        stop_atr: float = 1.5,
        t1_atr: float = 3.0,  # 3.0/1.5 = 2.0R clears MIN_RR_TO_DEPLOY
        t2_atr: float = 3.5,
        breakout_lookback: int = 10,
    ) -> None:
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.t1_atr = t1_atr
        self.t2_atr = t2_atr
        self.breakout_lookback = breakout_lookback

    def __call__(
        self, symbol: str, history: list[Bar], today: Bar
    ) -> dict[str, Any] | None:
        if len(history) < max(self.sma_period, self.atr_period) + 1:
            return None
        prior = history[-1]
        sma = _sma(history, self.sma_period)
        atr = _atr(history, self.atr_period)
        if sma is None or atr is None or atr <= 0:
            return None

        # Trend filter + breakout: prior close clears the highs of the window
        # ENDING BEFORE the prior bar (exclude prior's own high → no self-compare).
        window = history[-(self.breakout_lookback + 1):-1]
        if not window:
            return None
        recent_high = max(b.high for b in window)
        uptrend = prior.close > sma
        broke_out = prior.close >= recent_high
        if not (uptrend and broke_out):
            return None

        entry = today.open  # known at entry time
        if entry <= 0:
            return None
        stop = entry - self.stop_atr * atr
        t1 = entry + self.t1_atr * atr
        t2 = entry + self.t2_atr * atr
        if stop >= entry:
            return None

        # Confidence proxy from trend strength (bounded, honest about uncertainty).
        trend_str = (prior.close - sma) / sma
        win_prob = max(52.0, min(68.0, 52.0 + trend_str * 400.0))
        er = (t1 - entry) / entry * 100.0
        rr = round((t1 - entry) / (entry - stop), 2)
        from src.paper.allocation import expected_r

        ev = expected_r(win_prob=win_prob, rr=rr)
        return {
            "symbol": symbol,
            "direction": "LONG",
            "entry_price": round(entry, 4),
            "stop_price": round(stop, 4),
            "target_price": round(t1, 4),
            "target2": round(t2, 4),
            "win_prob": round(win_prob, 1),
            "expected_return_pct": round(er, 2),
            "expected_r": ev,
            "risk_reward": rr,
            "horizon": "Swing",
            "source": "backtest_proxy",
            "entry_status": {"status": "READY"},
            "_edge": round((ev or 0.0) * 100.0 + win_prob * 0.1, 2),
        }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_daily_bars(
    symbols: list[str], start: date, end: date
) -> dict[str, list[Bar]]:
    """Point-in-time daily OHLC via yfinance (auto-adjusted for splits/divs)."""
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - env dependent
        logger.warning("yfinance not installed; cannot fetch bars")
        return {}

    fetch_start = start - timedelta(days=90)  # warm-up for SMA/ATR
    df = yf.download(
        " ".join(symbols),
        start=fetch_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    out: dict[str, list[Bar]] = {}
    if df is None or df.empty:
        return out
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
        except KeyError:
            continue
        bars: list[Bar] = []
        for idx, row in sub.iterrows():
            o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
            if any(v is None or v != v for v in (o, h, l, c)):  # skip NaN rows
                continue
            bars.append(
                Bar(
                    date=idx.strftime("%Y-%m-%d"),
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(row.get("Volume") or 0.0),
                )
            )
        if bars:
            out[sym] = bars
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def _intraday_ticks(bar: Bar, direction: str) -> list[float]:
    """Conservative within-day price path: adverse extreme before favourable."""
    if (direction or "LONG").upper() == "LONG":
        return [bar.open, bar.low, bar.high, bar.close]
    return [bar.open, bar.high, bar.low, bar.close]


def _manage_book_for_day(
    account: dict[str, Any],
    book: str,
    bar: Bar,
    params: dict[str, Any],
    *,
    is_intraday: bool,
) -> list[dict[str, Any]]:
    """Replay one day's OHLC path against an open book using the real exit rules."""
    trades: list[dict[str, Any]] = []
    pos = get_position(account, book)
    if not pos:
        return trades
    direction = (pos.get("direction") or "LONG").upper()

    for px in _intraday_ticks(bar, direction):
        pos = get_position(account, book)
        if not pos:
            break
        plan = plan_exit(pos, px, params)
        if plan.get("action") == "scale_out":
            if plan.get("new_stop") is not None:
                pos["stop"] = plan["new_stop"]
            try:
                t = execute_exit(
                    account, price=px, reason=plan.get("reason") or "scale-out",
                    trading_date=bar.date, book=book, shares=plan.get("shares"),
                )
                trades.append(t)
            except (NonsensePriceError, ValueError):
                pass
            continue
        if plan.get("new_stop") is not None:
            pos["stop"] = plan["new_stop"]

        stop = pos.get("stop")
        target = pos.get("target1") or pos.get("target")
        if pos.get("scaled_out"):
            rt = runner_target(pos, params)
            if rt is not None:
                target = rt
        if _stop_hit(direction, px, stop):
            try:
                trades.append(
                    execute_exit(account, price=px, reason=f"stop @ {px}",
                                 trading_date=bar.date, book=book)
                )
            except (NonsensePriceError, ValueError):
                pass
            break
        if _target_hit(direction, px, target):
            try:
                trades.append(
                    execute_exit(account, price=px, reason=f"target @ {px}",
                                 trading_date=bar.date, book=book)
                )
            except (NonsensePriceError, ValueError):
                pass
            break

    # Soft EOD for the intraday book (mirrors live execution_agent).
    pos = get_position(account, book)
    if pos and is_intraday:
        eod = plan_eod_exit(pos, bar.close, params)
        if eod.get("new_stop") is not None:
            pos["stop"] = eod["new_stop"]
        if eod.get("action") == "scale_out":
            try:
                trades.append(
                    execute_exit(
                        account, price=bar.close,
                        reason=eod.get("reason") or "EOD scale-out",
                        trading_date=bar.date, book=book, shares=eod.get("shares"),
                    )
                )
            except (NonsensePriceError, ValueError):
                pass
        elif eod.get("action") == "exit":
            try:
                trades.append(
                    execute_exit(
                        account, price=bar.close,
                        reason=eod.get("reason") or "EOD flatten",
                        trading_date=bar.date, book=book,
                    )
                )
            except (NonsensePriceError, ValueError):
                pass
    return trades


def _is_swing(sig: dict[str, Any]) -> bool:
    return "swing" in str(sig.get("horizon") or "").lower()


def _assign_books(
    account: dict[str, Any],
    day_signals: list[dict[str, Any]],
    *,
    horizon_aware: bool,
) -> dict[str, dict[str, Any]]:
    """Route ranked signals to the swing / intraday books for empty slots.

    ``horizon_aware`` (report mode): swing book only takes Swing-horizon signals,
    intraday only intraday. Proxy mode ignores horizon (all signals eligible).
    """
    picks: dict[str, dict[str, Any]] = {}
    if not get_position(account, BOOK_SWING):
        cands = [s for s in day_signals if _is_swing(s)] if horizon_aware else day_signals
        if cands:
            picks[BOOK_SWING] = cands[0]
    if not get_position(account, BOOK_INTRADAY):
        for s in day_signals:
            if s is picks.get(BOOK_SWING):
                continue
            if horizon_aware and _is_swing(s):
                continue
            picks[BOOK_INTRADAY] = s
            break
    return picks


def resolve_fill_price(sig: dict[str, Any], bar: Bar) -> float | None:
    """Realistic same-day fill for a planned entry (limit order, no chasing).

    A LONG limit at ``entry`` only fills if the day traded down to it
    (low ≤ entry); it then fills at the better of open/entry. Symmetric for
    SHORT. If ``entry_price`` is absent, treat it as market-on-open. Returns
    ``None`` when the entry level was never touched (a genuine "missed" plan).
    """
    direction = (sig.get("direction") or "LONG").upper()
    entry = sig.get("entry_price")
    try:
        entry = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        entry = None
    if entry is None or entry <= 0:
        return bar.open
    if direction == "LONG":
        if bar.low <= entry:
            return round(min(bar.open, entry), 4)
        return None
    # SHORT
    if bar.high >= entry:
        return round(max(bar.open, entry), 4)
    return None


def _try_enter_book(
    account: dict[str, Any],
    book: str,
    sig: dict[str, Any],
    bar: Bar,
    *,
    peer_entering: bool,
) -> dict[str, Any] | None:
    fill = resolve_fill_price(sig, bar)
    if fill is None or fill <= 0:
        return None  # entry level never traded → no chase
    entry_px = float(fill)
    alloc = allocate_for_entry(
        account, book=book, signal=sig, quote=entry_px,
        entry_status=sig.get("entry_status"), peer_entering=peer_entering,
    )
    if alloc.get("hold_cash"):
        return None
    shares, err = can_afford(
        account, price=entry_px, stop=sig.get("stop_price"),
        direction=sig.get("direction") or "LONG",
        risk_pct=alloc.get("risk_pct"),
        max_position_pct=alloc.get("max_position_pct"),
        max_notional=alloc.get("deployable_cash"),
    )
    if shares <= 0:
        return None
    try:
        return execute_entry(
            account, symbol=sig["symbol"], direction=sig["direction"],
            price=entry_px, shares=shares, stop=sig.get("stop_price"),
            target=sig.get("target_price"), signal=sig,
            reason=f"backtest entry {sig['symbol']}", trading_date=bar.date, book=book,
        )
    except (InsufficientCashError, NonsensePriceError, ValueError):
        return None


def run_paper_backtest(
    *,
    symbols: list[str] | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    bars: dict[str, list[Bar]] | None = None,
    benchmark_symbol: str = BENCHMARK,
    signal_provider: SignalProvider | None = None,
    day_signal_provider: Callable[[str], list[dict[str, Any]]] | None = None,
    params_override: dict[str, Any] | None = None,
    starting_cash: float = STARTING_CASH,
    walk_forward_folds: int = 3,
    write_report: bool = False,
) -> dict[str, Any]:
    """Run the event-driven backtest and return a metrics summary.

    Two signal modes:
      - ``signal_provider`` (per-symbol): the PROXY alpha (default).
      - ``day_signal_provider`` (per-date): returns already-picked signals for a
        date, e.g. the report-driven provider that replays real research output.
        When supplied it takes precedence and enables horizon-aware book routing
        plus limit-touch fill realism (no chasing).
    """
    symbols = symbols or list(DEFAULT_UNIVERSE)
    provider = signal_provider or MomentumBreakoutProvider()

    if bars is None:
        s = date.fromisoformat(start) if isinstance(start, str) else start
        e = date.fromisoformat(end) if isinstance(end, str) else end
        if s is None or e is None:
            raise ValueError("start and end required when bars not supplied")
        need = list(dict.fromkeys([*symbols, benchmark_symbol]))
        bars = fetch_daily_bars(need, s, e)
    if not bars:
        return {"error": "no bars available", "n_days": 0}

    # Union of trading dates across the universe (chronological).
    all_dates = sorted({b.date for sym in symbols for b in bars.get(sym, [])})
    if start is not None:
        s_str = start if isinstance(start, str) else start.isoformat()
        all_dates = [d for d in all_dates if d >= s_str]
    if end is not None:
        e_str = end if isinstance(end, str) else end.isoformat()
        all_dates = [d for d in all_dates if d <= e_str]
    if not all_dates:
        return {"error": "no trading dates in window", "n_days": 0}

    by_sym_date: dict[str, dict[str, Bar]] = {
        sym: {b.date: b for b in blist} for sym, blist in bars.items()
    }
    hist_index: dict[str, dict[str, int]] = {
        sym: {b.date: i for i, b in enumerate(blist)} for sym, blist in bars.items()
    }

    account = default_account()
    account["starting_cash"] = starting_cash
    account["cash"] = starting_cash
    account["equity"] = starting_cash
    if params_override:
        account["params"].update(params_override)
    params = account["params"]

    equity_series: list[tuple[str, float]] = []

    report_mode = day_signal_provider is not None

    for d in all_dates:
        # 1) Entries for empty books. Signals use data through d-1 (no look-ahead).
        held_syms = {p.get("symbol") for _, p in open_positions(account)}
        day_signals: list[dict[str, Any]] = []
        if report_mode:
            for sig in day_signal_provider(d) or []:
                sym = str(sig.get("symbol") or "").upper()
                if not sym or sym in held_syms:
                    continue
                if by_sym_date.get(sym, {}).get(d) is None:
                    continue  # no price bar to fill/manage this name
                day_signals.append(sig)
        else:
            for sym in symbols:
                idx = hist_index.get(sym, {}).get(d)
                if idx is None or idx < 1:
                    continue
                today_bar = bars[sym][idx]
                history = bars[sym][:idx]  # strictly prior — no look-ahead
                sig = provider(sym, history, today_bar)
                if sig and sig["symbol"] not in held_syms:
                    day_signals.append(sig)
        day_signals.sort(key=lambda s: s.get("_edge", 0.0), reverse=True)

        picks = _assign_books(account, day_signals, horizon_aware=report_mode)

        peer = len(picks) >= 2
        # Swing first (mirrors live ordering so the correlation guard sees it).
        if BOOK_SWING in picks:
            _try_enter_book(account, BOOK_SWING, picks[BOOK_SWING],
                            by_sym_date[picks[BOOK_SWING]["symbol"]][d], peer_entering=peer)
        if BOOK_INTRADAY in picks:
            _try_enter_book(account, BOOK_INTRADAY, picks[BOOK_INTRADAY],
                            by_sym_date[picks[BOOK_INTRADAY]["symbol"]][d], peer_entering=peer)

        # 2) Manage every open book across the day's path.
        for book, is_intra in ((BOOK_SWING, False), (BOOK_INTRADAY, True)):
            pos = get_position(account, book)
            if not pos:
                continue
            sym = pos.get("symbol")
            bar = by_sym_date.get(sym, {}).get(d)
            if bar is None:
                continue
            _manage_book_for_day(account, book, bar, params, is_intraday=is_intra)

        # 3) Mark to close and record equity.
        close_map: dict[str, float] = {}
        for _, pos in open_positions(account):
            sym = pos.get("symbol")
            bar = by_sym_date.get(sym, {}).get(d)
            if bar:
                close_map[str(sym).upper()] = bar.close
        mark_to_market(account, price_by_symbol=close_map or None)
        append_equity_point(account, label=f"bt:{d}")
        equity_series.append((d, float(account.get("equity") or starting_cash)))

    metrics = _compute_metrics(
        account, equity_series, bars, benchmark_symbol, all_dates,
        starting_cash, walk_forward_folds,
    )
    metrics["signal_mode"] = "report" if report_mode else "proxy"
    if write_report:
        _write_report(metrics, account, all_dates, params)
    metrics["_account"] = account
    return metrics


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _daily_returns(equity_series: list[tuple[str, float]]) -> list[float]:
    rets: list[float] = []
    for i in range(1, len(equity_series)):
        prev = equity_series[i - 1][1]
        cur = equity_series[i][1]
        if prev > 0:
            rets.append(cur / prev - 1.0)
    return rets


def _sharpe(rets: list[float]) -> float:
    if len(rets) < 2:
        return 0.0
    sd = statistics.stdev(rets)
    if sd == 0:
        return 0.0
    return statistics.mean(rets) / sd * math.sqrt(252)


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _benchmark_curve(
    bars: dict[str, list[Bar]], symbol: str, dates: list[str], starting_cash: float
) -> list[float]:
    by_date = {b.date: b for b in bars.get(symbol, [])}
    curve: list[float] = []
    shares = None
    for d in dates:
        bar = by_date.get(d)
        if bar is None:
            curve.append(curve[-1] if curve else starting_cash)
            continue
        if shares is None:
            shares = starting_cash / bar.close if bar.close > 0 else 0.0
        curve.append(shares * bar.close)
    return curve


def _monthly_returns(equity_series: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Calendar-month returns from equity curve (honest in-sample reporting)."""
    if len(equity_series) < 2:
        return []
    by_month: dict[str, list[tuple[str, float]]] = {}
    for d, eq in equity_series:
        key = d[:7]  # YYYY-MM
        by_month.setdefault(key, []).append((d, eq))
    rows: list[dict[str, Any]] = []
    for month in sorted(by_month):
        pts = by_month[month]
        start_eq = pts[0][1]
        end_eq = pts[-1][1]
        ret = (end_eq / start_eq - 1.0) if start_eq > 0 else 0.0
        rows.append({
            "month": month,
            "start": pts[0][0],
            "end": pts[-1][0],
            "return_pct": round(ret * 100.0, 2),
            "hit_10pct": ret >= 0.10,
        })
    return rows


def _compute_metrics(
    account: dict[str, Any],
    equity_series: list[tuple[str, float]],
    bars: dict[str, list[Bar]],
    benchmark_symbol: str,
    dates: list[str],
    starting_cash: float,
    folds: int,
) -> dict[str, Any]:
    equity_vals = [e for _, e in equity_series]
    rets = _daily_returns(equity_series)
    closed = [
        t for t in account.get("trades") or []
        if t.get("action") in ("EXIT", "SCALE_OUT") and t.get("pnl") is not None
    ]
    pnls = [float(t["pnl"]) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    final_equity = equity_vals[-1] if equity_vals else starting_cash
    agent_cum = final_equity / starting_cash - 1.0

    bench_curve = _benchmark_curve(bars, benchmark_symbol, dates, starting_cash)
    bench_series = list(zip(dates, bench_curve))
    bench_rets = _daily_returns(bench_series)
    bench_cum = (bench_curve[-1] / starting_cash - 1.0) if bench_curve else 0.0

    # Turnover
    entries = [t for t in account.get("trades") or [] if t.get("action") == "ENTRY"]
    turnover_notional = sum(float(t.get("notional") or 0) for t in account.get("trades") or [])
    avg_equity = statistics.mean(equity_vals) if equity_vals else starting_cash

    # Walk-forward stability
    wf: list[dict[str, Any]] = []
    if folds > 1 and len(equity_series) >= folds * 2:
        size = len(equity_series) // folds
        for f in range(folds):
            lo = f * size
            hi = (f + 1) * size if f < folds - 1 else len(equity_series)
            seg = equity_series[lo:hi]
            if len(seg) < 2:
                continue
            seg_ret = seg[-1][1] / seg[0][1] - 1.0
            wf.append({
                "fold": f + 1,
                "start": seg[0][0],
                "end": seg[-1][0],
                "return_pct": round(seg_ret * 100, 2),
                "sharpe": round(_sharpe(_daily_returns(seg)), 2),
            })

    monthly = _monthly_returns(equity_series)
    months_hit = sum(1 for m in monthly if m.get("hit_10pct"))
    avg_monthly = (
        statistics.mean([m["return_pct"] for m in monthly]) if monthly else 0.0
    )
    # OOS = last walk-forward fold when available; else last calendar month.
    oos_hit = None
    oos_label = None
    if wf:
        last = wf[-1]
        oos_label = f"walk_forward_fold_{last['fold']}"
        try:
            days = max(1, (
                date.fromisoformat(str(last["end"])[:10])
                - date.fromisoformat(str(last["start"])[:10])
            ).days)
        except ValueError:
            days = max(1, len(equity_series) // max(folds, 1))
        months_approx = max(days / 21.0, 0.5)  # ~21 trading days / month
        fold_monthly_equiv = (
            (1 + last["return_pct"] / 100.0) ** (1 / months_approx) - 1
        ) * 100
        oos_hit = fold_monthly_equiv >= 10.0
    elif monthly:
        oos_label = f"last_month_{monthly[-1]['month']}"
        oos_hit = bool(monthly[-1].get("hit_10pct"))

    return {
        "n_days": len(dates),
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "starting_cash": round(starting_cash, 2),
        "final_equity": round(final_equity, 2),
        "agent_cumulative_pct": round(agent_cum * 100, 2),
        "agent_sharpe": round(_sharpe(rets), 2),
        "agent_max_drawdown_pct": round(_max_drawdown(equity_vals) * 100, 2),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_cumulative_pct": round(bench_cum * 100, 2),
        "benchmark_sharpe": round(_sharpe(bench_rets), 2),
        "benchmark_max_drawdown_pct": round(_max_drawdown(bench_curve) * 100, 2),
        "n_trades": len(closed),
        "n_entries": len(entries),
        "win_rate": round(len(wins) / len(pnls), 3) if pnls else 0.0,
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win else 0.0),
        "expectancy": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "total_realized_pnl": round(sum(pnls), 2),
        "turnover_x": round(turnover_notional / avg_equity, 2) if avg_equity else 0.0,
        "walk_forward": wf,
        "excess_return_pct": round((agent_cum - bench_cum) * 100, 2),
        "monthly_returns": monthly,
        "avg_monthly_return_pct": round(avg_monthly, 2),
        "months_hit_10pct": months_hit,
        "months_total": len(monthly),
        "in_sample_hit_10pct_mo": bool(monthly) and all(m.get("hit_10pct") for m in monthly),
        "any_month_hit_10pct": months_hit > 0,
        "oos_label": oos_label,
        "oos_hit_10pct_mo_equiv": oos_hit,
        # Honest disclaimer fields
        "target_10pct_mo_guaranteed": False,
        "caveat_zh": "回测≠实盘；10%/月是进取目标而非保证；无前视偷看。",
    }


def _write_report(
    metrics: dict[str, Any],
    account: dict[str, Any],
    dates: list[str],
    params: dict[str, Any],
) -> Path:
    d = reports_dir() / "backtest"
    d.mkdir(parents=True, exist_ok=True)
    mode = metrics.get("signal_mode", "proxy")
    tag = "report" if mode == "report" else "proxy"
    path = d / f"paper-engine-{tag}-{metrics['start']}-to-{metrics['end']}.md"

    def _pct(v: Any) -> str:
        try:
            return f"{float(v):.2f}%"
        except (TypeError, ValueError):
            return "n/a"

    if mode == "report":
        signal_note = (
            "> ADVISORY ONLY · 模拟交易 · 不构成投资建议. Signals are the REAL research\n"
            "> pipeline output replayed from persisted morning.json / step3.json."
        )
    else:
        signal_note = (
            "> ADVISORY ONLY · 模拟交易 · 不构成投资建议. Signal alpha is a documented\n"
            "> momentum-breakout PROXY; execution/exit/sizing is the real engine."
        )

    lines = [
        "# Paper Engine Backtest (real execution path)",
        "",
        f"**Window**: {metrics['start']} → {metrics['end']} ({metrics['n_days']} trading days)",
        f"**Signal source**: {mode}",
        f"**Generated**: {date.today().isoformat()}",
        "",
        signal_note,
        "",
        "## Performance vs Benchmark",
        "",
        "| Metric | Agent | " + metrics["benchmark_symbol"] + " (B&H) |",
        "|---|---:|---:|",
        f"| Cumulative Return | {_pct(metrics['agent_cumulative_pct'])} | {_pct(metrics['benchmark_cumulative_pct'])} |",
        f"| Sharpe (ann.) | {metrics['agent_sharpe']:.2f} | {metrics['benchmark_sharpe']:.2f} |",
        f"| Max Drawdown | {_pct(metrics['agent_max_drawdown_pct'])} | {_pct(metrics['benchmark_max_drawdown_pct'])} |",
        f"| Excess Return | {_pct(metrics['excess_return_pct'])} | — |",
        "",
        "## Trade Stats",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Entries | {metrics['n_entries']} |",
        f"| Closed legs (exits + scale-outs) | {metrics['n_trades']} |",
        f"| Win Rate | {metrics['win_rate']:.1%} |",
        f"| Avg Win / Avg Loss | ${metrics['avg_win']:.2f} / ${metrics['avg_loss']:.2f} |",
        f"| Profit Factor | {metrics['profit_factor']:.2f} |",
        f"| Expectancy / leg | ${metrics['expectancy']:.2f} |",
        f"| Total Realized PnL | ${metrics['total_realized_pnl']:.2f} |",
        f"| Turnover (notional / avg equity) | {metrics['turnover_x']:.2f}x |",
        f"| Avg monthly return | {metrics.get('avg_monthly_return_pct', 0):.2f}% |",
        f"| Months hit ≥10% | {metrics.get('months_hit_10pct', 0)} / {metrics.get('months_total', 0)} |",
        f"| In-sample every month ≥10%? | {'YES' if metrics.get('in_sample_hit_10pct_mo') else 'NO'} |",
        f"| OOS ({metrics.get('oos_label') or 'n/a'}) ≥10%/mo equiv? | "
        f"{'YES' if metrics.get('oos_hit_10pct_mo_equiv') else 'NO' if metrics.get('oos_hit_10pct_mo_equiv') is not None else 'n/a'} |",
        "",
        "> **Not a guarantee**: `target_10pct_mo_guaranteed=False`. Past paper / backtest ≠ future live.",
        "",
        "## Walk-Forward Stability",
        "",
        "| Fold | Window | Return | Sharpe |",
        "|---|---|---:|---:|",
    ]
    for f in metrics.get("walk_forward") or []:
        lines.append(
            f"| {f['fold']} | {f['start']}→{f['end']} | {f['return_pct']:.2f}% | {f['sharpe']:.2f} |"
        )
    if not metrics.get("walk_forward"):
        lines.append("| — | insufficient days | — | — |")

    lines += [
        "",
        "## Execution Assumptions (frictions modelled)",
        "",
        f"- Slippage: {params.get('slippage_bps')} bps adverse on every fill",
        f"- Commission: ${params.get('commission_per_trade')}/trade + ${params.get('commission_per_share')}/share",
        f"- Scale-out {params.get('scale_out_pct')} at T1; breakeven @ +{params.get('breakeven_trigger_r')}R; "
        f"trail {params.get('trail_distance_r')}R after +{params.get('trail_trigger_r')}R",
        "- Within-day path assumes the adverse extreme trades before the favourable one",
        "",
        "## Caveats (read before trusting)",
        "",
    ]
    if mode == "report":
        lines += [
            "- Signals are REAL research output, but replayed on DAILY bars — intraday",
            "  entry-status nuance is approximated by a limit-touch fill rule.",
            f"- Sample = {metrics['n_days']} days; treat as indicative until ≥40 sessions.",
            "- Fills use the planned entry only if the day traded through it (no chase).",
        ]
    else:
        lines += [
            "- Signal alpha is a PROXY, not the research pipeline — treat relative",
            "  metrics (vs B&H, across folds) as more meaningful than absolute return.",
            "- Long-only high-beta universe; results are regime-sensitive.",
            "- Daily bars only: no intrabar sequencing beyond the conservative rule.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Paper engine backtest report → %s", path)
    return path


def _run_from_reports(args: argparse.Namespace) -> dict[str, Any]:
    """Backtest the REAL persisted research signals (morning.json / step3.json)."""
    from src.paper.report_signals import (
        available_report_dates,
        report_signals_for_date,
        report_symbols,
    )

    dates = available_report_dates()
    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    if not dates:
        return {"error": "no persisted reports found (run Step 0/1 daily first)"}

    syms = report_symbols(dates)
    if not syms:
        return {"error": f"{len(dates)} report day(s) found but no valid trade signals"}

    need = list(dict.fromkeys([*syms, BENCHMARK]))
    start = date.fromisoformat(dates[0])
    end = date.fromisoformat(dates[-1])
    logger.info(
        "Report-driven backtest: %d day(s) %s→%s, symbols=%s",
        len(dates), dates[0], dates[-1], ",".join(syms),
    )
    bars = fetch_daily_bars(need, start, end)
    result = run_paper_backtest(
        symbols=syms,
        start=dates[0],
        end=dates[-1],
        bars=bars,
        day_signal_provider=report_signals_for_date,
        walk_forward_folds=args.folds,
        write_report=True,
    )
    result.setdefault("n_report_days", len(dates))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Event-driven paper-engine backtest")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_UNIVERSE))
    p.add_argument("--folds", type=int, default=3)
    p.add_argument(
        "--from-reports",
        action="store_true",
        help="Replay REAL persisted research signals (also auto when reports exist)",
    )
    p.add_argument(
        "--proxy",
        action="store_true",
        help="Force momentum PROXY even when report signals exist",
    )
    args = p.parse_args()

    from src.paper.report_signals import available_report_dates

    report_dates = available_report_dates()
    use_reports = (not args.proxy) and (args.from_reports or bool(report_dates))

    if use_reports:
        if report_dates and not args.from_reports:
            logger.info(
                "Preferring REAL report signals (%d day(s)); pass --proxy for momentum PROXY",
                len(report_dates),
            )
        result = _run_from_reports(args)
    else:
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        start = end - timedelta(days=int(args.days * 1.5) + 90)
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        result = run_paper_backtest(
            symbols=syms, start=start, end=end,
            walk_forward_folds=args.folds, write_report=True,
        )

    if result.get("error"):
        print(f"Backtest error: {result['error']}")
        return
    mode = result.get("signal_mode", "proxy")
    print("\n=== Paper Engine Backtest ===")
    print(
        f"Signal:      {mode}"
        + (
            f" ({result.get('n_report_days')} report days)"
            if mode == "report"
            else " (PROXY — not research alpha)"
        )
    )
    print(f"Window:      {result['start']} → {result['end']} ({result['n_days']} days)")
    print(
        f"Agent:       {result['agent_cumulative_pct']:.2f}%  "
        f"Sharpe {result['agent_sharpe']:.2f}  MDD {result['agent_max_drawdown_pct']:.2f}%"
    )
    print(
        f"{result['benchmark_symbol']} B&H:     {result['benchmark_cumulative_pct']:.2f}%  "
        f"Sharpe {result['benchmark_sharpe']:.2f}"
    )
    print(f"Excess:      {result['excess_return_pct']:.2f}%")
    print(
        f"Trades:      {result['n_trades']} "
        f"(win {result['win_rate']:.0%}, PF {result['profit_factor']:.2f}, "
        f"exp ${result['expectancy']:.2f})"
    )
    print(
        f"Monthly:     avg {result.get('avg_monthly_return_pct', 0):.2f}% · "
        f"hit≥10%: {result.get('months_hit_10pct', 0)}/{result.get('months_total', 0)} · "
        f"in-sample all≥10%: {'YES' if result.get('in_sample_hit_10pct_mo') else 'NO'}"
    )
    oos = result.get("oos_hit_10pct_mo_equiv")
    print(
        f"OOS:         {result.get('oos_label') or 'n/a'} ≥10%/mo equiv: "
        f"{'YES' if oos else 'NO' if oos is not None else 'n/a'}"
    )
    print("NOTE:        10%/mo is an ambition target — NOT guaranteed. Backtest ≠ live.")


if __name__ == "__main__":
    main()
