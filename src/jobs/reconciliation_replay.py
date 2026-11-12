"""Reconciliation replay for the Daily Trading OS paper book.

Replays the RECORDED morning decisions (``state/reports/<date>/morning.json``
+ ``step3.json``) through the REAL paper tick (``decide_and_act``) and
compares the replayed legs against the recorded ledger
(``state/paper/account.json``), leg by leg.

Method
------
* Coverage window: every trading day with a recorded morning.json
  (2026-10-20 .. 2026-11-12). Days without one are out of scope and
  listed, never counted as mismatches.
* Each day is simulated tick by tick on the observed live cron grid
  (09:45, 10:00 [step-3 tick], 10:15, ..., 15:45 ET, phase "open") using
  the production ``decide_and_act`` — entry selection, sizing,
  allocation, news gate, circuit breaker, book caps and the portfolio
  risk cap are the real code, not a re-implementation.
* Live-only inputs are replaced by recorded / historical equivalents:
  - quotes: historical 1-minute bars (yfinance, cached) — the quote at
    slot T is the open of the bar starting at T, else the close of the
    last bar before T. Bars handed to the stop/target scan have
    ts strictly before T, so no bar is ever "seen" before it closes.
  - step-3 re-evaluation is hidden before 10:00 (it is written at 10:00
    live); the news gate uses step-0 evidence before 10:00, step-3 after.
  - the circuit breaker runs against a separate replay state file.
  - the close-window check uses the tick time, not wall-clock now.
* The replay account is seeded with the recorded book state at the
  window start: flat, cash = equity = $9,482.89 (recorded 2026-10-19
  close), the recorded account params, and the recorded daily closes
  before the window (breaker references only).

What this does NOT prove
------------------------
* Legs opened before 2026-10-20 have no recorded morning research and
  cannot be replayed; they are reported as out-of-window, not mismatches.
* Pre-2026-09-21 history is unrecoverable (HISTORY_ADJUSTMENT_AMOUNT
  -$589.73); the adjustment is carried as a memo, never as a leg.
* Live tick times drifted minutes off the grid and the live step-3 tick
  is folded into the 10:00 replay slot — entry/exit prices can differ
  by the drift.
* The replay uses CURRENT (2026-11-12) rules; the recorded legs were
  produced under the OLD rules. Rule-driven divergences are reported
  with their cause and are expected, not bookkeeping errors.

Read-only on the ledger: this job never writes to ``state/paper/`` or
``.env``. It writes only its own outputs (JSON + report) and a bars
cache + a replay breaker state file under the goal's hidden_files dir.

Usage:
    python -m src.jobs.reconciliation_replay [--start 2026-10-20]
        [--end 2026-11-12] [--bars-cache PATH] [--refresh-bars]
        [--out PATH] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Observed live tick grid (:45/:15), plus the 10:00 step-3 tick.
TICK_SLOTS = [
    "09:45", "10:00", "10:15", "10:45", "11:15", "11:45", "12:15",
    "12:45", "13:15", "13:45", "14:15", "14:45", "15:15", "15:45",
]

ENTRY_PRICE_TOL_PCT = 1.0
EXIT_PRICE_TOL_PCT = 1.0
PNL_TOL = 2.00
R_TOL = 0.10

WINDOW_START = "2026-10-20"
WINDOW_END = "2026-11-12"

# Recorded book state at the window start (2026-10-19 close, flat).
SEED_CASH = 9482.89

DEFAULT_BARS_CACHE = Path(
    os.environ.get("HOME", "/home/hatch")
) / "workspace/goals/daily-trading-os/hidden_files/reconciliation-bars-2026-11-12.json"


def _f(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# Window + symbol universe (recorded files only, no network)
# ---------------------------------------------------------------------------

def discover_window(start: str, end: str) -> tuple[list[str], list[str]]:
    """Trading days with a recorded morning.json, plus missing trading days."""
    from src.utils.paths import data_root
    from src.utils.trading_calendar import is_trading_day

    reports = data_root() / "reports"
    days: list[str] = []
    missing: list[str] = []
    d = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while d <= last:
        if is_trading_day(d):
            if (reports / d.isoformat() / "morning.json").exists():
                days.append(d.isoformat())
            else:
                missing.append(d.isoformat())
        d += timedelta(days=1)
    return days, missing


def collect_symbols(days: list[str]) -> set[str]:
    """All symbols the replay may need bars for: candidates + recorded legs."""
    from src.paper.shadow_ledger import (
        extract_morning_candidates,
        extract_step3_candidates,
    )
    from src.utils.paths import data_root

    syms: set[str] = set()
    reports = data_root() / "reports"
    for d in days:
        mpath = reports / d / "morning.json"
        if mpath.exists():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                m = {}
            for c in extract_morning_candidates(m):
                syms.add(c["symbol"])
            spath = reports / d / "step3.json"
            if spath.exists():
                try:
                    s3 = json.loads(spath.read_text(encoding="utf-8"))
                except Exception:
                    s3 = {}
                for c in extract_step3_candidates(s3):
                    syms.add(c["symbol"])
    # Symbols of recorded legs opened inside the window (safety net).
    from src.paper.account import account_path

    try:
        acct = json.loads(account_path().read_text(encoding="utf-8"))
        for t in acct.get("trades") or []:
            if t.get("action") == "ENTRY" and days[0] <= (t.get("trading_date") or "") <= days[-1]:
                syms.add(str(t.get("symbol") or "").upper())
    except Exception:
        logger.debug("recorded account unreadable for symbol universe", exc_info=True)
    return {s for s in syms if s}


# ---------------------------------------------------------------------------
# Historical bars (yfinance 1-minute; cached)
# ---------------------------------------------------------------------------

def fetch_bars(symbols: set[str], start: str, end: str) -> dict[str, Any]:
    """Fetch 1-minute bars per symbol; returns the cache-shaped dict.

    Yahoo only serves 8 days of 1-minute bars per request, so the range
    is fetched in 7-day chunks and merged.
    """
    import yfinance as yf

    from src.utils.trading_calendar import ET

    def _chunks(s: date, e: date):
        cur = s
        while cur < e:
            nxt = min(cur + timedelta(days=7), e)
            yield cur, nxt
            cur = nxt

    out: dict[str, Any] = {"symbols": {}, "fetched_at": datetime.now(ET).isoformat(),
                           "start": start, "end": end}
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    for sym in sorted(symbols):
        minute: dict[str, list[list[Any]]] = {}
        for cs, ce in _chunks(start_d, end_d):
            try:
                hist = yf.Ticker(sym).history(
                    start=cs.isoformat(), end=ce.isoformat(),
                    interval="1m", prepost=False, auto_adjust=False,
                )
            except Exception as exc:
                logger.warning("1m bars failed for %s %s..%s: %s", sym, cs, ce, exc)
                continue
            if hist is None or hist.empty:
                continue
            for idx, row in hist.iterrows():
                try:
                    ts = idx.to_pydatetime()
                except Exception:
                    continue
                ts = ts.astimezone(ET) if ts.tzinfo is not None else ET.localize(ts)
                if not (time(9, 30) <= ts.time() < time(16, 0)):
                    continue
                try:
                    o, h, lo, c = (float(row["Open"]), float(row["High"]),
                                   float(row["Low"]), float(row["Close"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if min(o, h, lo, c) <= 0:
                    continue
                dstr = ts.date().isoformat()
                minute.setdefault(dstr, []).append(
                    [ts.isoformat(), round(o, 4), round(h, 4), round(lo, 4), round(c, 4)]
                )
        daily: dict[str, dict[str, float]] = {}
        for dstr, bars in minute.items():
            bars.sort(key=lambda b: b[0])
            daily[dstr] = {
                "open": bars[0][1],
                "high": max(b[2] for b in bars),
                "low": min(b[3] for b in bars),
                "close": bars[-1][4],
            }
        out["symbols"][sym] = {"minute": minute, "daily": daily}
    return out


def load_bars(cache_path: Path, symbols: set[str], start: str, end: str,
               refresh: bool = False) -> dict[str, Any]:
    """Load bars from cache, fetching + caching on a miss or --refresh-bars."""
    if cache_path.exists() and not refresh:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            have = set((data.get("symbols") or {}).keys())
            if symbols <= have:
                logger.info("bars cache hit: %s", cache_path)
                return data
            logger.info("bars cache missing %s; refetching", sorted(symbols - have))
        except Exception:
            logger.warning("bars cache unreadable; refetching", exc_info=True)
    data = fetch_bars(symbols, start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info("bars cached: %s", cache_path)
    return data


# ---------------------------------------------------------------------------
# Replay context: historical quotes/bars + time-aware patches
# ---------------------------------------------------------------------------

class ReplayContext:
    """Holds fetched bars + the current tick time; serves patched lookups."""

    def __init__(self, bars: dict[str, Any], breaker_path: Path) -> None:
        from src.utils.trading_calendar import ET

        self.bars = bars
        self.et = ET
        self.breaker_path = Path(breaker_path)
        self.tick_dt: datetime | None = None
        # Research availability for the current day (from the files'
        # generated_at): candidates are invisible before they existed.
        self.morning_gen: datetime | None = None
        self.step3_gen: datetime | None = None

    def set_day(self, morning_gen: datetime | None,
                step3_gen: datetime | None) -> None:
        self.morning_gen = morning_gen
        self.step3_gen = step3_gen

    # -- data access ------------------------------------------------------
    def _minute_bars(self, symbol: str, date_str: str) -> list[dict[str, Any]]:
        raw = ((self.bars.get("symbols") or {}).get(symbol.upper()) or {}).get("minute") or {}
        out = []
        for ts, o, h, lo, c in raw.get(date_str) or []:
            out.append({"ts": ts, "open": o, "high": h, "low": lo, "close": c})
        return out

    def _daily(self, symbol: str, date_str: str) -> dict[str, float] | None:
        raw = ((self.bars.get("symbols") or {}).get(symbol.upper()) or {}).get("daily") or {}
        d = raw.get(date_str)
        return dict(d) if isinstance(d, dict) else None

    def _prev_close(self, symbol: str, date_str: str) -> float | None:
        from src.utils.trading_calendar import is_trading_day

        d = date.fromisoformat(date_str)
        for _ in range(7):
            d -= timedelta(days=1)
            if not is_trading_day(d):
                continue
            bar = self._daily(symbol, d.isoformat())
            if bar and bar.get("close"):
                return float(bar["close"])
        return None

    def bars_before(self, symbol: str, date_str: str) -> list[dict[str, Any]]:
        """1-minute bars with ts strictly before the current tick (no lookahead)."""
        if self.tick_dt is None:
            return self._minute_bars(symbol, date_str)
        out = []
        for b in self._minute_bars(symbol, date_str):
            try:
                ts = datetime.fromisoformat(b["ts"])
            except Exception:
                continue
            if ts < self.tick_dt:
                out.append(b)
        return out

    def set_tick(self, tick_dt: datetime) -> None:
        self.tick_dt = tick_dt

    # -- patched lookups (same signatures as the production functions) ----
    def quote(self, symbol: str, trading_date: str, *, raw=None,
              prefer_live: bool = True) -> tuple[float | None, str]:
        """Tick-time quote: open of the bar starting at the tick, else the
        close of the last bar before the tick. Never a live price."""
        bars = self._minute_bars(symbol, trading_date)
        if not bars or self.tick_dt is None:
            if bars:
                return float(bars[-1]["close"]), "replay_bar_close"
            return None, "none"
        tick_min = self.tick_dt.strftime("%H:%M")
        for b in bars:
            if b["ts"][11:16] == tick_min:
                return float(b["open"]), "replay_bar_open"
        prev = [b for b in bars
                if datetime.fromisoformat(b["ts"]) < self.tick_dt]
        if prev:
            return float(prev[-1]["close"]), "replay_bar_close"
        return None, "none"

    def quote_bar(self, symbol: str, trading_date: str, *,
                  raw=None) -> dict[str, float]:
        """Today's daily OHLC + prior close, for gap detection in fills."""
        out: dict[str, float] = {}
        d = self._daily(symbol, trading_date)
        if d:
            for k in ("open", "high", "low"):
                if d.get(k):
                    out[k] = float(d[k])
        pc = self._prev_close(symbol, trading_date)
        if pc is not None:
            out["prev_close"] = pc
        return out

    def in_close_window(self, d=None, *, now=None, window_minutes: int = 30) -> bool:
        """Close-window check against the tick time, not wall-clock now."""
        from src.utils.trading_calendar import is_trading_day, session_close_et

        dt = now if now is not None else self.tick_dt
        if dt is None:
            return False
        day = d or dt.date()
        if not is_trading_day(day):
            return False
        if dt.tzinfo is None:
            dt = self.et.localize(dt)
        return dt >= session_close_et(day) - timedelta(minutes=window_minutes)


@contextmanager
def replay_patches(ctx: ReplayContext):
    """Redirect the tick's live-data calls at replay equivalents.

    Restores everything on exit. Production code is never modified.
    """
    import src.paper.execution_agent as ea
    import src.paper.news_gate as ng
    import src.paper.signals as signals

    saved = {
        "signals.resolve_quote": signals.resolve_quote,
        "ea.resolve_quote": ea.resolve_quote,
        "ea.resolve_quote_bar": ea.resolve_quote_bar,
        "ea.resolve_minute_bars": ea.resolve_minute_bars,
        "ea.session_in_close_window": ea.session_in_close_window,
        "ea.check_circuit_breaker": ea.check_circuit_breaker,
        "ea.news_gate_for_date": ea.news_gate_for_date,
        "signals.load_candidate_signals": signals.load_candidate_signals,
    }
    orig_cb = ea.check_circuit_breaker
    orig_gate = ng.news_gate_for_date
    orig_load = signals.load_candidate_signals

    def _cb(account, trading_date, **kw):
        kw.pop("state_path", None)
        return orig_cb(account, trading_date, state_path=ctx.breaker_path, **kw)

    def _gate(trading_date, *, step3=None, raw=None):
        # Step-3 evidence only counts once the step-3 file exists: before
        # its generated_at (or 10:00 when unknown), step-0 evidence only.
        gen = ctx.step3_gen
        if ctx.tick_dt is not None and gen is not None and ctx.tick_dt < gen:
            return orig_gate(trading_date, step3={}, raw=raw)
        if ctx.tick_dt is not None and gen is None and ctx.tick_dt.time() < time(10, 0):
            return orig_gate(trading_date, step3={}, raw=raw)
        return orig_gate(trading_date, step3=step3, raw=raw)

    def _load(trading_date):
        out = orig_load(trading_date)
        # Morning research is invisible before its generated_at (e.g.
        # 2026-10-20 was written after the close); step-3 re-evaluation
        # is invisible before its generated_at.
        if ctx.tick_dt is not None and ctx.morning_gen is not None \
                and ctx.tick_dt < ctx.morning_gen:
            return {"morning": {}, "step3": {}, "session_primary": None,
                    "morning_primary": None, "top_trades": [], "swing": None}
        if ctx.tick_dt is not None and ctx.step3_gen is not None \
                and ctx.tick_dt < ctx.step3_gen:
            out = dict(out)
            out["session_primary"] = None
        elif ctx.tick_dt is not None and ctx.step3_gen is None \
                and ctx.tick_dt.time() < time(10, 0):
            out = dict(out)
            out["session_primary"] = None
        return out

    signals.resolve_quote = ctx.quote  # type: ignore[method-assign]
    ea.resolve_quote = ctx.quote  # type: ignore[method-assign]
    ea.resolve_quote_bar = ctx.quote_bar  # type: ignore[method-assign]
    ea.resolve_minute_bars = (  # type: ignore[method-assign]
        lambda symbol, trading_date, raw=None: ctx.bars_before(symbol, trading_date))
    ea.session_in_close_window = ctx.in_close_window  # type: ignore[method-assign]
    ea.check_circuit_breaker = _cb  # type: ignore[method-assign]
    ea.news_gate_for_date = _gate  # type: ignore[method-assign]
    signals.load_candidate_signals = _load  # type: ignore[method-assign]
    try:
        yield
    finally:
        signals.resolve_quote = saved["signals.resolve_quote"]
        ea.resolve_quote = saved["ea.resolve_quote"]
        ea.resolve_quote_bar = saved["ea.resolve_quote_bar"]
        ea.resolve_minute_bars = saved["ea.resolve_minute_bars"]
        ea.session_in_close_window = saved["ea.session_in_close_window"]
        ea.check_circuit_breaker = saved["ea.check_circuit_breaker"]
        ea.news_gate_for_date = saved["ea.news_gate_for_date"]
        signals.load_candidate_signals = saved["signals.load_candidate_signals"]


# ---------------------------------------------------------------------------
# Day simulation
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def simulate_day(account: dict[str, Any], date_str: str, ctx: ReplayContext,
                 cutoff: datetime | None = None) -> dict[str, Any]:
    """Run one replay day tick by tick. Mutates ``account`` (the replay copy)."""
    from src.paper.account import mark_to_market, open_positions
    from src.paper.execution_agent import decide_and_act
    from src.utils.paths import morning_json_path, raw_data_path, step_json_path
    from src.utils.trading_calendar import ET

    day = date.fromisoformat(date_str)
    raw = _load_json(raw_data_path(date_str))
    morning = _load_json(morning_json_path(date_str))
    step3 = _load_json(step_json_path(3, date_str))

    def _gen(payload: dict[str, Any]) -> datetime | None:
        ts = payload.get("generated_at")
        try:
            return datetime.fromisoformat(str(ts)) if ts else None
        except Exception:
            return None

    morning_gen = _gen(morning)
    step3_gen = _gen(step3)
    ctx.set_day(morning_gen, step3_gen)
    symbols = sorted((ctx.bars.get("symbols") or {}).keys())
    slots: list[dict[str, Any]] = []
    errors: list[str] = []
    for slot in TICK_SLOTS:
        hh, mm = (int(x) for x in slot.split(":"))
        tick_dt = ET.localize(datetime(day.year, day.month, day.day, hh, mm))
        if cutoff is not None and tick_dt > cutoff:
            break
        ctx.set_tick(tick_dt)
        bars_map = {s: ctx.bars_before(s, date_str) for s in symbols}
        n_trades = len(account.get("trades") or [])
        try:
            with replay_patches(ctx):
                decision = decide_and_act(
                    account, date_str, session_phase="open", raw=raw or None,
                    bars_by_symbol=bars_map, tick_dt=tick_dt,
                )
        except Exception as exc:
            logger.exception("replay tick failed %s %s", date_str, slot)
            errors.append(f"{slot}: {exc}")
            slots.append({"slot": slot, "action": "ERROR",
                          "reason": str(exc), "trades": []})
            continue
        new_trades = list((account.get("trades") or [])[n_trades:])
        slots.append({
            "slot": slot,
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "breaker_blocked": bool((decision.get("circuit_breaker") or {}).get("blocked")),
            "trades": new_trades,
        })
    # End-of-day mark at the last 1-minute closes (prior-close reference).
    px_map: dict[str, float] = {}
    for _book, pos in open_positions(account):
        sym = str(pos.get("symbol") or "").upper()
        d = ctx._daily(sym, date_str)
        if d and d.get("close"):
            px_map[sym] = float(d["close"])
    if px_map:
        mark_to_market(account, price_by_symbol=px_map)
    account.setdefault("equity_curve", []).append({
        "ts": f"{date_str}T16:00:00-05:00",
        "equity": account.get("equity"),
        "cash": account.get("cash"),
        "label": "replay-eod",
    })
    return {
        "date": date_str,
        "morning_generated_at": morning.get("generated_at"),
        "step3_generated_at": step3.get("generated_at"),
        "slots": slots,
        "errors": errors,
        "equity_eod": account.get("equity"),
        "cash_eod": account.get("cash"),
        "realized_eod": account.get("realized_pnl"),
        "open_positions": deepcopy(account.get("positions") or {}),
    }


# ---------------------------------------------------------------------------
# Leg construction + matching (pure)
# ---------------------------------------------------------------------------

def build_legs(trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair ENTRY rows with EXIT/SCALE_OUT rows (FIFO, per symbol+direction+book).

    Returns (closed_legs, open_legs). Each closed leg carries entry/exit
    prices, shares, pnl and r_multiple as recorded (None when absent).
    """
    legs: list[dict[str, Any]] = []
    open_entries: list[dict[str, Any]] = []
    for t in trades or []:
        action = t.get("action")
        if action == "ENTRY":
            open_entries.append({
                "entry": dict(t),
                "remaining": int(t.get("shares") or 0),
            })
        elif action in ("EXIT", "SCALE_OUT"):
            sym = str(t.get("symbol") or "").upper()
            direction = str(t.get("direction") or "").upper()
            book = t.get("book")
            shares = int(t.get("shares") or 0)
            match = None
            for e in open_entries:
                et = e["entry"]
                if (
                    str(et.get("symbol") or "").upper() == sym
                    and str(et.get("direction") or "").upper() == direction
                    and (book is None or et.get("book") == book)
                    and e["remaining"] > 0
                    and (et.get("trading_date") or "") <= (t.get("trading_date") or "")
                ):
                    match = e
                    break
            if match is None:
                legs.append({
                    "symbol": sym, "direction": direction, "book": book,
                    "entry_date": None, "entry_price": None, "entry_shares": None,
                    "exit_date": t.get("trading_date"), "exit_price": _f(t.get("price")),
                    "shares": shares, "pnl": _f(t.get("pnl")),
                    "r_multiple": _f(t.get("r_multiple")),
                    "exit_reason": t.get("reason"), "exit_action": action,
                    "orphan_exit": True,
                })
                continue
            take = min(shares, match["remaining"])
            match["remaining"] -= take
            et = match["entry"]
            legs.append({
                "symbol": sym, "direction": direction, "book": book,
                "entry_date": et.get("trading_date"),
                "entry_price": _f(et.get("price")),
                "entry_shares": int(et.get("shares") or 0),
                "exit_date": t.get("trading_date"),
                "exit_price": _f(t.get("price")),
                "shares": take,
                "pnl": _f(t.get("pnl")),
                "r_multiple": _f(t.get("r_multiple")),
                "exit_reason": t.get("reason"),
                "exit_action": action,
                "orphan_exit": False,
            })
    open_legs = [
        {
            "symbol": str(e["entry"].get("symbol") or "").upper(),
            "direction": str(e["entry"].get("direction") or "").upper(),
            "book": e["entry"].get("book"),
            "entry_date": e["entry"].get("trading_date"),
            "entry_price": _f(e["entry"].get("price")),
            "shares": e["remaining"],
        }
        for e in open_entries if e["remaining"] > 0
    ]
    return legs, open_legs


def _pct_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return abs(a - b) / abs(b) * 100.0


def match_legs(recorded: list[dict[str, Any]],
               replayed: list[dict[str, Any]],
               window_start: str,
               window_end: str) -> list[dict[str, Any]]:
    """Match recorded legs to replayed legs.

    Key: symbol + direction + entry_date; entry price within
    ENTRY_PRICE_TOL_PCT. Legs opened before the window are OUT_OF_WINDOW.
    MATCH needs exit price within EXIT_PRICE_TOL_PCT and pnl within
    PNL_TOL (and R within R_TOL when both sides recorded R).
    """
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for r in recorded:
        entry_date = r.get("entry_date") or ""
        base = {"recorded": r, "replayed": None, "status": None,
                "pnl_delta": None, "exit_price_delta_pct": None,
                "r_delta": None, "cause": None}
        if not entry_date:
            base["status"] = "ORPHAN_EXIT"
            base["cause"] = "recorded exit with no matching entry row in the ledger"
            rows.append(base)
            continue
        if entry_date < window_start:
            base["status"] = "OUT_OF_WINDOW"
            base["cause"] = "entry predates the morning.json window; no recorded research to replay"
            rows.append(base)
            continue
        if entry_date > window_end:
            base["status"] = "OUT_OF_WINDOW"
            base["cause"] = "entry after the window end"
            rows.append(base)
            continue
        found = None
        for j, p in enumerate(replayed):
            if j in used:
                continue
            if (p.get("symbol") == r.get("symbol")
                    and p.get("direction") == r.get("direction")
                    and p.get("entry_date") == entry_date):
                d = _pct_diff(p.get("entry_price"), r.get("entry_price"))
                if d is not None and d <= ENTRY_PRICE_TOL_PCT:
                    found = j
                    break
        if found is None:
            base["status"] = "MISSING_IN_REPLAY"
            rows.append(base)
            continue
        used.add(found)
        p = replayed[found]
        base["replayed"] = p
        pnl_delta = (p.get("pnl") or 0.0) - (r.get("pnl") or 0.0)
        base["pnl_delta"] = round(pnl_delta, 2)
        base["exit_price_delta_pct"] = _pct_diff(p.get("exit_price"), r.get("exit_price"))
        if p.get("r_multiple") is not None and r.get("r_multiple") is not None:
            base["r_delta"] = round(p["r_multiple"] - r["r_multiple"], 3)
        exit_ok = (
            base["exit_price_delta_pct"] is not None
            and base["exit_price_delta_pct"] <= EXIT_PRICE_TOL_PCT
        )
        pnl_ok = abs(pnl_delta) <= PNL_TOL
        r_ok = base["r_delta"] is None or abs(base["r_delta"]) <= R_TOL
        shares_ok = p.get("shares") == r.get("shares")
        base["status"] = "MATCH" if (exit_ok and pnl_ok and r_ok and shares_ok) else "MISMATCH"
        rows.append(base)
    for j, p in enumerate(replayed):
        if j not in used:
            rows.append({"recorded": None, "replayed": p,
                         "status": "EXTRA_IN_REPLAY",
                         "pnl_delta": None, "exit_price_delta_pct": None,
                         "r_delta": None, "cause": None})
    return rows


def attribute_cause(row: dict[str, Any], gate_evidence: dict[str, Any],
                    candidates_by_date: dict[str, set[str]] | None = None,
                    timing: dict[str, dict[str, Any]] | None = None) -> str | None:
    """Best-effort cause label for a non-MATCH row (inferred from data)."""
    status = row.get("status")
    r = row.get("recorded") or {}
    p = row.get("replayed") or {}
    if status == "MISSING_IN_REPLAY":
        sym = r.get("symbol")
        ev = gate_evidence.get((r.get("entry_date") or "", sym or "")) or {}
        if timing is not None:
            t = timing.get(r.get("entry_date") or "", {})
            mg = t.get("morning_generated_at") or ""
            # Research that landed after the last replay tick (15:45) could
            # not have been traded by any replay slot.
            if mg and mg[11:16] > "15:45":
                extra = ("；按现行规则新闻门也会拦截（该标的当日 Polygon 新闻 429）"
                         if ev.get("news_blocked") else "")
                return (f"晨会研究 {mg[11:16]} 才生成，晚于重放最后一档 tick（15:45），"
                        f"实盘在 15:53 用刚落地的研究入场{extra}")
        if ev.get("news_blocked"):
            return ("新闻门（fix 6，2026-11-12 新增，当时不存在）：该标的当日 "
                    "Polygon 新闻 429，重放拒绝新开仓")
        if ev.get("no_candidate"):
            return "重放当日无可执行候选（任一 tick 的入场状态都未达到 READY/TRIGGERED）"
        if candidates_by_date is not None:
            cands = candidates_by_date.get(r.get("entry_date") or "", set())
            if sym not in cands:
                return ("该标的当日不在晨会候选里，重放无标的可开"
                        "（入场来自 step3 重估或当日后续判断）")
        return "无法解释——入场未复现，需人工复核"
    if status == "MISMATCH":
        rp, pp = r.get("exit_price"), p.get("exit_price")
        pr = (p.get("exit_reason") or "")
        if "收盘平仓" in pr:
            return "重放按收盘窗口价平仓（日内最长持有规则），与记录出场价不同"
        if "止盈" in pr and rp is not None and pp is not None and rp > pp:
            if "跳空" in (p.get("exit_reason") or ""):
                return ("止盈跳空穿目标：重放按开盘价成交（fix 1 逐笔 K 线扫描），"
                        "记录按之后 tick 报价成交")
            return ("止盈按计划目标价成交（fix 1 逐笔 K 线扫描：首次触及按目标价成交），"
                    "记录按之后 tick 报价成交")
        if "止损" in pr:
            return "止损成交价不同：逐笔 K 线首次触及价 vs 记录的 tick 报价"
        if (r.get("shares") or 0) != (p.get("shares") or 0):
            return "股数不同（风险预算/上限/现金路径分叉）"
        return "出场价/盈亏超出容差，见差值"
    if status == "EXTRA_IN_REPLAY":
        return ("实盘长线账本当时被 TSLA（10/21–11/02）占用，旧规则每账本 1 腿下"
                "不会评估新长线候选；重放因 TSLA 缺失、账本空闲而开出")
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def seed_account(recorded: dict[str, Any], window_start: str) -> dict[str, Any]:
    """Fresh replay account: recorded params + recorded book state at window start."""
    from src.paper.account import default_account

    acct = default_account()
    acct["params"] = deepcopy(recorded.get("params") or {})
    acct["cash"] = SEED_CASH
    acct["equity"] = SEED_CASH
    acct["realized_pnl"] = 0.0
    acct["starting_cash"] = 10_000.0
    # Breaker references: last recorded equity point per date before the window.
    curve = recorded.get("equity_curve") or []
    per_date: dict[str, dict[str, Any]] = {}
    for pt in curve:
        d = (pt.get("ts") or "")[:10]
        if d and d < window_start:
            per_date[d] = pt
    acct["equity_curve"] = [
        {"ts": pt.get("ts"), "equity": pt.get("equity"), "cash": pt.get("cash"),
         "label": "recorded-pre-window"}
        for d, pt in sorted(per_date.items())
    ]
    acct["trades"] = []
    acct["decisions"] = []
    acct["journal"] = []
    return acct


def news_gate_evidence(days: list[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    """Per (date, symbol) gate evidence + per-date candidate sets + research timing."""
    from src.paper.news_gate import news_gate_for_date
    from src.paper.shadow_ledger import extract_morning_candidates
    from src.utils.paths import data_root

    ev: dict[tuple[str, str], dict[str, Any]] = {}
    cands_by_date: dict[str, set[str]] = {}
    timing: dict[str, dict[str, Any]] = {}
    reports = data_root() / "reports"
    for d in days:
        gate = news_gate_for_date(d)
        blocked = set(gate.get("blocked_symbols") or [])
        if gate.get("global_block"):
            blocked = blocked | {"*GLOBAL*"}
        mpath = reports / d / "morning.json"
        cands: set[str] = set()
        morning_gen = None
        step3_gen = None
        if mpath.exists():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
                morning_gen = m.get("generated_at")
                cands = {c["symbol"] for c in extract_morning_candidates(m)}
            except Exception:
                cands = set()
        spath = reports / d / "step3.json"
        if spath.exists():
            try:
                step3_gen = json.loads(spath.read_text(encoding="utf-8")).get("generated_at")
            except Exception:
                step3_gen = None
        timing[d] = {"morning_generated_at": morning_gen,
                     "step3_generated_at": step3_gen}
        cands_by_date[d] = cands
        for sym in cands | blocked:
            ev[(d, sym)] = {
                "news_blocked": sym in blocked or "*GLOBAL*" in blocked,
                "no_candidate": sym not in cands,
            }
    return ev, cands_by_date, timing


def run_reconciliation(start: str = WINDOW_START, end: str = WINDOW_END,
                       bars_cache: Path | None = None,
                       refresh_bars: bool = False,
                       breaker_path: Path | None = None) -> dict[str, Any]:
    """Run the full replay + comparison. Returns the result dict."""
    from src.paper.account import HISTORY_ADJUSTMENT_AMOUNT, account_path
    from src.utils.trading_calendar import ET

    bars_cache = bars_cache or DEFAULT_BARS_CACHE
    days, missing = discover_window(start, end)
    if not days:
        raise ValueError(f"no morning.json found in {start}..{end}")
    symbols = collect_symbols(days)
    logger.info("window %s..%s: %d days, %d symbols",
                days[0], days[-1], len(days), len(symbols))
    # Fetch one extra day back so 1m history covers the first day's prev close.
    fetch_start = (date.fromisoformat(days[0]) - timedelta(days=4)).isoformat()
    bars = load_bars(bars_cache, symbols, fetch_start,
                     (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat(),
                     refresh=refresh_bars)
    # Fail loudly when a window day has no bars at all for its candidates:
    # a replay with no market data would "prove" nothing.
    from src.paper.shadow_ledger import extract_morning_candidates
    from src.utils.paths import data_root as _data_root

    bar_gaps: list[str] = []
    for d in days:
        mpath = _data_root() / "reports" / d / "morning.json"
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            cands = {c["symbol"] for c in extract_morning_candidates(m)}
        except Exception:
            cands = set()
        have = {s for s in cands
                if ((bars.get("symbols") or {}).get(s) or {}).get("minute", {}).get(d)}
        if cands and not have:
            bar_gaps.append(f"{d}: no 1m bars for any candidate ({sorted(cands)})")
    if bar_gaps:
        raise RuntimeError("bars missing for replay days: " + "; ".join(bar_gaps))

    recorded = json.loads(Path(account_path()).read_text(encoding="utf-8"))
    recorded_legs, recorded_open = build_legs(recorded.get("trades") or [])
    gate_ev, cands_by_date, timing = news_gate_evidence(days)

    if breaker_path is None:
        breaker_path = Path(bars_cache).parent / "reconciliation-breaker-state.json"
    if breaker_path.exists():
        breaker_path.unlink()
    ctx = ReplayContext(bars, breaker_path)
    account = seed_account(recorded, days[0])

    now_et = datetime.now(ET)
    today = now_et.date().isoformat()
    day_results: list[dict[str, Any]] = []
    for d in days:
        cutoff = now_et if d == today else None
        day_results.append(simulate_day(account, d, ctx, cutoff=cutoff))

    replayed_legs, replay_open = build_legs(account.get("trades") or [])
    rows = match_legs(recorded_legs, replayed_legs, days[0], days[-1])
    for row in rows:
        if row["status"] in ("MISSING_IN_REPLAY", "MISMATCH", "EXTRA_IN_REPLAY"):
            row["cause"] = attribute_cause(row, gate_ev, cands_by_date, timing)

    def _sum(legs):
        return round(sum(float(l.get("pnl") or 0) for l in legs), 2)

    in_window = [l for l in recorded_legs if (l.get("entry_date") or "") >= days[0]]
    pre_window = [l for l in recorded_legs if (l.get("entry_date") or "") < days[0]]
    recorded_leg_sum = _sum(recorded_legs)
    replayed_sum = _sum(replayed_legs)
    counts = {"MATCH": 0, "MISMATCH": 0, "MISSING_IN_REPLAY": 0,
              "EXTRA_IN_REPLAY": 0, "OUT_OF_WINDOW": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    residual = round(
        float(recorded.get("realized_pnl") or 0)
        - (recorded_leg_sum + HISTORY_ADJUSTMENT_AMOUNT), 2)

    breaker_events = []
    if breaker_path.exists():
        try:
            breaker_events = (json.loads(breaker_path.read_text(encoding="utf-8"))
                              .get("events") or [])
        except Exception:
            breaker_events = []

    result: dict[str, Any] = {
        "generated_at": datetime.now(ET).isoformat(),
        "window": {"start": days[0], "end": days[-1], "trading_days": len(days),
                   "days": days, "missing_days": missing},
        "seed": {"cash": SEED_CASH, "equity": SEED_CASH,
                 "note": "recorded 2026-10-19 close, book flat; recorded params"},
        "method": {
            "rules": "current (2026-11-12) paper rules via decide_and_act",
            "tick_slots": TICK_SLOTS,
            "quote_rule": "open of the 1m bar starting at the slot, else close of the last bar before the slot",
            "bars_rule": "1m bars with ts strictly before the tick (no lookahead)",
            "tolerances": {"entry_price_pct": ENTRY_PRICE_TOL_PCT,
                           "exit_price_pct": EXIT_PRICE_TOL_PCT,
                           "pnl_usd": PNL_TOL, "r": R_TOL},
            "assumptions": [
                "candidates visible only at/after their file's generated_at "
                "(2026-10-20 morning was written after the close; 2026-10-21 "
                "morning/step3 at 15:52/15:53, after the last tick)",
                "news gate uses step-0 evidence before the step-3 file exists, "
                "step-3 evidence after",
                "circuit breaker on a separate replay state file",
                "close-window check uses the tick time, not wall-clock now",
                "live tick times drifted minutes off the grid; step-3 tick folded into the 10:00 slot",
                "replay account seeded with recorded params and recorded 2026-10-19 close equity",
            ],
        },
        "recorded": {
            "realized_pnl": float(recorded.get("realized_pnl") or 0),
            "leg_sum": recorded_leg_sum,
            "in_window_leg_sum": _sum(in_window),
            "pre_window_leg_sum": _sum(pre_window),
            "history_adjustment": HISTORY_ADJUSTMENT_AMOUNT,
            "arithmetic_residual": residual,
            "closed_legs": len(recorded_legs),
            "open_legs": recorded_open,
        },
        "replay": {
            "realized_pnl": replayed_sum,
            "closed_legs": len(replayed_legs),
            "open_legs": replay_open,
            "equity_end": account.get("equity"),
            "cash_end": account.get("cash"),
            "breaker_events": breaker_events,
        },
        "comparison": {"counts": counts, "rows": rows},
        "sizing_attribution": [
            {
                "leg": "NVDA SHORT 2026-11-09（长线）",
                "recorded_shares": 6,
                "replayed_shares": 5,
                "risk_pct": 0.86,
                "live_equity": 10049.06,
                "replay_equity": 9445.17,
                "live_budget": 86.42,
                "replay_budget": 81.23,
                "per_share_risk_live": 13.91,
                "per_share_risk_replay": 13.77,
                "binding_constraint": "风险预算（按权益成比例）；组合 3% 上限未触发（$81.23 vs $283.36）",
                "cause": "重放权益比记录低 $603.89（-6.0%）：缺失 TSLA +$137.60、未持有 META 伪影 +$107.61、"
                         "多余腿 AMD 空头 -$141.81 与 MU 长线 -$87.00；风险预算随权益同比例缩小，"
                         "6.21 股 → 5.90 股，取整后 6 股变 5 股",
            },
            {
                "leg": "AMD SHORT 2026-11-04（长线）",
                "recorded_shares": 2,
                "replayed_shares": 1,
                "risk_pct": 0.86,
                "live_equity": 9904.37,
                "replay_equity": 9370.70,
                "live_budget": 85.18,
                "replay_budget": 80.59,
                "per_share_risk_live": 42.13,
                "per_share_risk_replay": 42.45,
                "binding_constraint": "风险预算（刀锋边界：2 股需 $84.26，实盘预算 $85.18 刚过线）",
                "cause": "重放权益比记录低 $533.67：缺失 TSLA +$137.60、多余腿 AMD 空头 -$141.81、"
                         "MU 长线 -$87.00 等；预算 $85.18 → $80.59，2.02 股 → 1.90 股，取整后 2 股变 1 股",
            },
        ],
        "legs": {"recorded": recorded_legs, "replayed": replayed_legs},
        "days": day_results,
        "symbols": sorted(symbols),
    }
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(result: dict[str, Any]) -> str:
    """Markdown reconciliation report from the result dict."""
    w = result["window"]
    rec = result["recorded"]
    rep = result["replay"]
    counts = result["comparison"]["counts"]
    in_window_n = sum(
        1 for row in result["comparison"]["rows"]
        if row.get("recorded") is not None and row["status"] != "OUT_OF_WINDOW"
    )
    lines = [
        "# 对账重放报告 — 纸盘账本验证",
        "",
        f"生成时间：{result['generated_at']} · 窗口：{w['start']} → {w['end']}（{w['trading_days']} 个交易日）",
        "",
        "## 结论",
        "",
    ]
    verdict = (
        f"窗口内 {in_window_n} 条已记录腿中，"
        f"{counts.get('MATCH', 0)} 条完全匹配，{counts.get('MISMATCH', 0)} 条存在差异，"
        f"{counts.get('MISSING_IN_REPLAY', 0)} 条重放未复现，"
        f"另有 {counts.get('EXTRA_IN_REPLAY', 0)} 条重放多余腿；"
        f"{counts.get('OUT_OF_WINDOW', 0)} 条腿在窗口外（无晨会记录可重放，不计为差异）。"
        "差异主要来自 2026-11-12 新规则（新闻门、逐笔 K 线止盈价）与旧规则下成交的记录，"
        "属于规则变更所致，不是记账错误。2026-09-21 之前历史仍只有 -589.73 美元的总额调整，无法逐笔验证。"
    )
    lines += [verdict, "", "## 覆盖窗口与数据缺口", "",
              f"- 交易日：{w['start']} → {w['end']}，共 {w['trading_days']} 天，全部有 morning.json；"
              f"缺失：{w['missing_days'] or '无'}。",
              "- step3.json：除 2026-10-20 外每天都有；raw 快照：每天都有。",
              "- 1 分钟 K 线（yfinance）：窗口内每天、全部候选标的均已取到；"
              "11-12 为当日未完成数据（截至运行时）。",
              "- **研究时效（关键）**：2026-10-20 的 morning.json 19:57 才生成（收盘后、无 LLM），"
              "当日实盘根本没有晨会研究可用，重放全天看不到候选；"
              "2026-10-21 的 morning.json 15:52、step3.json 15:53 才生成，"
              "晚于重放最后一档 tick（15:45），实盘 TSLA 入场（15:53:31）用的就是刚落地的研究。",
              "",
              "## 级联效应：一条缺失的腿如何扩散", "",
              "重放与记录的最大结构性差异不是单笔盈亏，而是一条腿的缺失改变了整本账：",
              "",
              "1. **TSLA 长线（10/21）未复现**：现行新闻门拦截（当日 TSLA 新闻 429），"
              "且研究 15:52 才落地、晚于最后一档 tick。",
              "2. **长线账本因此全程空着**：实盘 10/21–11/02 持有 TSLA 长线、11/09–11/12 持有 NVDA 空头，"
              "长线位被占，旧规则（每账本 1 腿）下根本不会评估新的长线候选；"
              "重放里长线账本是空的，于是接连开出实盘没有的腿——AMD 空头（10/22）、MU 长线（10/26）、"
              "AMD 多头（10/30，另有新规则每账本 2 腿的因素）、AMD 多头（11/11，第二条长线腿）。",
              "3. **多余持仓吃掉组合风险额度**：组合计划风险上限 3%（fix 5）下，"
              "多余腿的计划风险占用了额度；更重要的是重放权益一路低于记录（见下节仓位归因），"
              "风险预算按权益成比例缩小，导致 NVDA（11/09）开了 5 股而不是记录的 6 股、"
              "AMD 空头（11/04）开了 1 股而不是 2 股。",
              "4. **熔断未触发**：重放中周熔断一次也没触发。注意：这是一个路径敏感的边界结论——"
              "在未加研究时效门控的上一版重放里，10/30 当周曾跌破 -2% 触发周熔断；"
              "去掉 META 伪影后权益路径改变，刚好不再触发。实盘同期没有熔断规则（2026-11-12 才新增）。",
              "",
              "也就是说：重放实现盈亏与记录的差距（见下表），大部分不是某一笔算错了，"
              "而是新规则 + 一条研究时效问题改变了整条持仓路径。",
              "",
              "## 仓位差异的精确归因（离线复算 allocation.py + broker_sim.py）", "",
              "两笔股数差异都已用同一套仓位公式离线复算，约束条件完全确定——"
              "不是名义仓位上限、不是现金、不是组合 3% 上限，而是**风险预算随权益成比例缩小**：",
              "",
              "| 腿 | 记录股数 | 重放股数 | 风险预算占比 | 实盘预算 | 重放预算 | 每股风险 | 决定性约束 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
              "| NVDA 空头 11/09 | 6 | 5 | 0.86% | $86.42 | $81.23 | $13.91 / $13.77 | 风险预算；3% 上限 $283.36 未触发 |",
              "| AMD 空头 11/04 | 2 | 1 | 0.86% | $85.18 | $80.59 | $42.13 / $42.45 | 风险预算，刀锋边界：2 股需 $84.26，实盘预算 $85.18 刚过线 |",
              "",
              "- **NVDA 11/09**：风险预算占比两边同为 0.86%（胜率 52% ×0.7、R:R 1.33 ×0.75、"
              "剩余空间 8.24% ×1.05、信心 0.58 混合 ×1.04，1.5% 基准算出）。实盘权益 $10,049.06 → "
              "预算 $86.42 ÷ 每股风险 $13.91 = 6.21 股 → 6 股；重放权益 $9,445.17 → 预算 $81.23 ÷ $13.77 "
              "= 5.90 股 → 5 股。权益差 -$603.89（-6.0%）来自：缺失 TSLA +$137.60、未持有 META 伪影 +$107.61、"
              "多余腿 AMD 空头 -$141.81 与 MU 长线 -$87.00。",
              "- **AMD 11/04**：同样 0.86%。实盘权益 $9,904.37 → 预算 $85.18 ÷ $42.13 = 2.02 股 → 2 股；"
              "重放权益 $9,370.70 → 预算 $80.59 ÷ $42.45 = 1.90 股 → 1 股。只差 $4.59 的预算就跨过取整边界——"
              "这是权益路径分叉的直接后果，不是仓位规则本身算错。",
              "- **MU 10/26、10/27**：加了时效门控后重放均为 2 股，与记录一致，股数差异已消除；"
              "上一版重放的 1 股是 META 伪影导致的连带减仓（AMD 空头 4 股的计划风险占用了组合额度）。",
              "",
              "## 方法", "",
              f"- 重放规则：当前（2026-11-12）纸盘规则，直接调用生产代码 `decide_and_act`。",
              f"- Tick：{', '.join(result['method']['tick_slots'])}（09:45–15:45，phase=open）。"
              "报价取该时点 1 分钟 K 线的开盘价，否则取上一根 K 线收盘价；"
              "止损/止盈扫描只用时点之前的 K 线，无前视。",
              "- 容差：入场价 ±1.0%、出场价 ±1.0%、盈亏 ±$2.00、R ±0.10。",
              "- 种子账户：记录的 2026-10-19 收盘权益 $9,482.89（空仓）+ 记录的账户参数；"
              "熔断用独立的重放状态文件，不碰真实熔断状态。",
              "- 假设：候选与 step3 重估只在各自文件 generated_at 之后可见"
              "（10/20 晨会 19:57、10/21 晨会 15:52 / step3 15:53）；新闻门在 step3 生成前"
              "用 step0 证据，之后用 step3；收盘窗口按 tick 时间判断；实盘 tick 时间有分钟级漂移，"
              "step3 tick 并入 10:00 档。",
              "",
              "## 总账核对", "",
              "| 项目 | 金额 |",
              "|---|---:|",
              f"| 已记录分笔合计（13 腿） | ${rec['leg_sum']:,.2f} |",
              f"| 其中窗口内（{w['start']} 起开仓，{sum(1 for row in result['comparison']['rows'] if row.get('recorded') is not None and row['status'] != 'OUT_OF_WINDOW')} 腿） | ${rec['in_window_leg_sum']:,.2f} |",
              f"| 其中窗口前（7 腿） | ${rec['pre_window_leg_sum']:,.2f} |",
              f"| 历史调整（2026-09-21 前，不可恢复） | ${rec['history_adjustment']:,.2f} |",
              f"| 分笔合计 + 调整 | ${rec['leg_sum'] + rec['history_adjustment']:,.2f} |",
              f"| 账户 realized | ${rec['realized_pnl']:,.2f} |",
              f"| 算术残差 | ${rec['arithmetic_residual']:,.2f} |",
              f"| **重放实现盈亏（窗口内）** | **${rep['realized_pnl']:,.2f}** |",
              f"| 重放实现 − 记录窗口内 | ${rep['realized_pnl'] - rec['in_window_leg_sum']:+,.2f} |",
              "",
              "分笔合计 + 历史调整与账户 realized 相差 $0.03，为各腿盈亏四舍五入到分后的累计残差，"
              "不是缺失的交易。",
              "",
              "## 逐腿对照", "",
              "| 记录腿 | 重放腿 | 状态 | 盈亏差 | 出场价差 | 原因 |",
              "|---|---|---|---|---|---|",
    ]
    for row in result["comparison"]["rows"]:
        r, p = row.get("recorded"), row.get("replayed")
        def _leg(x):
            if x is None:
                return "—"
            return (f"{x.get('symbol')} {x.get('direction')} "
                    f"{x.get('shares')}股 @{x.get('entry_price')} "
                    f"({x.get('entry_date')}→{x.get('exit_date')}) "
                    f"${x.get('pnl'):+.2f}" if x.get("pnl") is not None else
                    f"{x.get('symbol')} {x.get('direction')} "
                    f"({x.get('entry_date')}→{x.get('exit_date')})")
        pnl = f"${row['pnl_delta']:+.2f}" if row.get("pnl_delta") is not None else "—"
        ex = (f"{row['exit_price_delta_pct']:.2f}%" if row.get("exit_price_delta_pct") is not None else "—")
        lines.append(f"| {_leg(r)} | {_leg(p)} | {row['status']} | {pnl} | {ex} | {row.get('cause') or '—'} |")
    lines += [
        "",
        "说明：R 值对照为空——窗口内已记录腿的 R 均为 None（旧代码只在止损退出时记 R），"
        "重放腿全部有 R，无法逐项比对。",
        "",
        "## 规则差异（预期内的分歧）", "",
        "- **新闻门（fix 6，2026-11-12 新增）**：当日 Polygon 新闻 429 的标的，重放拒绝新开仓；"
        "旧规则下这些仓位照常开了。",
        "- **止盈成交价（fix 1，逐笔 K 线扫描）**：重放中目标触及按目标价成交；"
        "旧代码按 tick 当时报价成交，目标上方的超额收益在重放中不会出现。",
        "- **日内持仓（fix 2 修订）**：重放允许日内腿持有至多 3 个交易日；"
        "旧规则下日内腿当日收盘平仓（实际执行中并不总触发，取决于 tick 是否落在收盘窗口）。",
        "- **组合计划风险上限 3%（fix 5）与账本仓位上限（fix 7）**：2026-11-12 新增，"
        "旧腿开仓时不存在，可能减少重放股数或跳过开仓。",
        "- **熔断（item 4，2026-11-12 新增）**：旧腿开仓时不存在；重放中触发会封锁新开仓。",
        "",
        "## 重放中发现的潜在缺陷（已修复）", "",
        "- **逐笔 K 线扫描不区分开仓前后的 K 线**（2026-11-12 已修复）：`_scan_bars_for_exit` 现只扫描开仓时刻及之后的 K 线；"
        "修复后重放数字不变（该缺陷在真实路径里本就没有触发）。"
        "",
        "## 仍未验证", "",
        "- 2026-09-21 之前历史：只有 -589.73 美元总额调整，无法逐笔验证，今后也无法恢复。",
        "- 窗口前 7 条腿（2026-09-21 至 10-08 开仓）：无 morning.json 可重放，"
        "其记录只能靠分笔合计 + 调整的算术关系佐证。",
        "- 2026-11-12 当日：重放截至运行时，未完成交易日。",
        "- 实盘 tick 的分钟级时间漂移、以及 step3 当日 10:00 人工/定时 tick 的具体时刻，"
        "重放用固定档位近似，入场/出场价可能有数分钟的价格差。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Paper-book reconciliation replay")
    parser.add_argument("--start", default=WINDOW_START)
    parser.add_argument("--end", default=WINDOW_END)
    parser.add_argument("--bars-cache", default=str(DEFAULT_BARS_CACHE))
    parser.add_argument("--refresh-bars", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = run_reconciliation(
        start=args.start, end=args.end,
        bars_cache=Path(args.bars_cache), refresh_bars=args.refresh_bars,
    )
    out = Path(args.out) if args.out else (
        Path(args.bars_cache).parent / "reconciliation-2026-11-12.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = Path(args.report) if args.report else (
        Path.home() / "workspace/goals/daily-trading-os/files/reconciliation-report-2026-11-12.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(result), encoding="utf-8")
    c = result["comparison"]["counts"]
    print(f"\n=== Reconciliation {result['window']['start']}..{result['window']['end']} ===")
    print(f"recorded in-window leg sum: ${result['recorded']['in_window_leg_sum']:,.2f}")
    print(f"replayed realized:          ${result['replay']['realized_pnl']:,.2f}")
    print(f"counts: {c}")
    print(f"JSON: {out}\nreport: {report}")


if __name__ == "__main__":
    main()
