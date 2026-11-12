"""Daily shadow-ledger update job.

Ingests morning.json trade candidates (+ step3 re-evaluations) for every
report date not yet processed, then advances all open/awaiting shadow
positions through daily bars. Idempotent: re-running reproduces the same
ledger and summary.

Usage (env first, as with every job in this repo):
    set -a; . ~/workspace/trading-os/.env; set +a
    export no_proxy="localhost,127.0.0.1,198.19.0.1,198.19.0.2" NO_PROXY="$no_proxy"
    export DATA_ROOT=~/workspace/trading-os/state
    export TMPDIR=~/workspace/.tmp
    ~/workspace/.venvs/trading/bin/python -m src.jobs.shadow_ledger_update [--as-of YYYY-MM-DD] [--from-date YYYY-MM-DD]

Bars come from yfinance daily history. Only fully closed bars are ever
used: as-of defaults to today when run at/after 16:05 ET, else yesterday.
Exit code 0 on success; per-date / per-symbol failures are logged to
stderr and skipped, never fatal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.paper.shadow_ledger import (  # noqa: E402
    BACKFILL_FROM,
    VERSION,
    compute_summary,
    extract_morning_candidates,
    extract_step3_candidates,
    make_record,
    update_record,
)

ET = ZoneInfo("America/New_York")
CLOSE_CUTOFF = time(16, 5)

DEFAULT_STATE = Path.home() / "workspace" / "trading-os" / "state"


def resolve_state_dir() -> Path:
    root = os.environ.get("DATA_ROOT")
    return Path(root) if root else DEFAULT_STATE


def resolve_as_of(now: datetime | None = None) -> date:
    """Latest closed trading date: today if run after the close, else yesterday."""
    now = now or datetime.now(ET)
    today = now.date()
    if now.timetz().replace(tzinfo=None) >= CLOSE_CUTOFF:
        return today
    return today - timedelta(days=1)


def _iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_daily_bars(symbol: str, start: date, end: date) -> list[dict]:
    """Daily OHLC bars for one symbol, sorted ascending. Empty on failure."""
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - import failure
        print(f"[shadow] yfinance import failed: {exc}", file=sys.stderr)
        return []
    try:
        hist = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
        )
    except Exception as exc:
        print(f"[shadow] fetch failed for {symbol}: {exc}", file=sys.stderr)
        return []
    if hist is None or hist.empty:
        return []
    bars: list[dict] = []
    for ts, row in hist.iterrows():
        try:
            bars.append({
                "date": ts.date().isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            })
        except Exception:
            continue
    bars.sort(key=lambda b: b["date"])
    return bars


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            state = json.loads(path.read_text())
            if isinstance(state, dict) and "records" in state:
                return state
        except Exception as exc:
            print(f"[shadow] corrupt state {path}: {exc}; starting fresh", file=sys.stderr)
    return {"version": VERSION,
            "records": [],
            "processed_dates": {"morning": [], "step3": []},
            "updated_at": None}


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_json(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
        return doc if isinstance(doc, dict) else None
    except Exception as exc:
        print(f"[shadow] cannot read {path}: {exc}", file=sys.stderr)
        return None


def run(state_dir: Path, as_of: date, from_date: date) -> dict:
    reports_dir = state_dir / "reports"
    paper_dir = state_dir / "paper"
    ledger_path = paper_dir / "shadow_ledger.json"
    summary_path = paper_dir / "shadow_summary.json"

    state = load_state(ledger_path)
    records: list[dict] = state.get("records") or []
    by_id = {r["id"]: r for r in records if isinstance(r, dict) and r.get("id")}
    processed = state.get("processed_dates") or {}
    done_morning = set(processed.get("morning") or [])
    done_step3 = set(processed.get("step3") or [])
    today = datetime.now(ET).date()

    stats = {"dates_ingested": [], "new_records": 0, "entered": 0,
             "exited": 0, "skipped": 0, "symbols_fetched": 0}

    # 1) Ingest unprocessed report dates.
    for d in _iter_dates(from_date, as_of):
        ds = d.isoformat()
        day_dir = reports_dir / ds
        if ds not in done_morning:
            doc = _load_json(day_dir / "morning.json")
            if doc is not None:
                for cand in extract_morning_candidates(doc):
                    rec = make_record(ds, "morning", cand)
                    if rec["id"] not in by_id:
                        by_id[rec["id"]] = rec
                        records.append(rec)
                        stats["new_records"] += 1
                done_morning.add(ds)
                stats["dates_ingested"].append(ds)
            elif d < today:
                # A past date with no morning.json will never appear.
                done_morning.add(ds)
        if ds not in done_step3:
            doc = _load_json(day_dir / "step3.json")
            if doc is not None:
                for cand in extract_step3_candidates(doc):
                    rec = make_record(ds, "step3", cand)
                    if rec["id"] not in by_id:
                        by_id[rec["id"]] = rec
                        records.append(rec)
                        stats["new_records"] += 1
                done_step3.add(ds)
            elif d < today:
                done_step3.add(ds)

    # 2) Advance open / awaiting records through closed bars.
    need_symbols = sorted({r["symbol"] for r in records
                           if r.get("status") in ("awaiting-entry", "open")})
    earliest = min(
        [from_date] + [date.fromisoformat(r["candidate_date"]) for r in records
                       if r.get("status") in ("awaiting-entry", "open")
                       and r.get("candidate_date")],
        default=from_date,
    )
    bars_by_symbol: dict[str, list[dict]] = {}
    for sym in need_symbols:
        bars = fetch_daily_bars(sym, earliest, as_of)
        stats["symbols_fetched"] += 1
        if bars:
            bars_by_symbol[sym] = bars

    as_of_s = as_of.isoformat()
    for rec in records:
        if rec.get("status") not in ("awaiting-entry", "open"):
            continue
        bars = bars_by_symbol.get(rec["symbol"])
        if not bars:
            continue
        before = rec["status"]
        update_record(rec, bars, as_of_s)
        if before == "awaiting-entry" and rec["status"] in ("open", "exited"):
            stats["entered"] += 1
        if before in ("awaiting-entry", "open") and rec["status"] == "exited":
            stats["exited"] += 1
        if before == "awaiting-entry" and rec["status"] == "skipped":
            stats["skipped"] += 1

    # 3) Persist ledger + summary.
    state["version"] = VERSION
    state["records"] = records
    state["processed_dates"] = {"morning": sorted(done_morning),
                                "step3": sorted(done_step3)}
    state["updated_at"] = datetime.now(ET).isoformat()
    atomic_write(ledger_path, state)

    summary = compute_summary(records, as_of_date=as_of_s)
    summary["generated_at"] = datetime.now(ET).isoformat()
    summary["run_stats"] = stats
    atomic_write(summary_path, summary)

    return {"as_of": as_of_s, "ledger": str(ledger_path),
            "summary": str(summary_path), **stats,
            "totals": summary["totals"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Update the shadow ledger.")
    ap.add_argument("--as-of", default=None, help="Latest closed date YYYY-MM-DD")
    ap.add_argument("--from-date", default=BACKFILL_FROM,
                    help="First report date to ingest (default 2026-10-20)")
    ap.add_argument("--state-dir", default=None)
    args = ap.parse_args(argv)

    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else resolve_as_of()
        from_date = date.fromisoformat(args.from_date)
    except ValueError as exc:
        print(f"[shadow] bad date: {exc}", file=sys.stderr)
        return 2
    state_dir = Path(args.state_dir) if args.state_dir else resolve_state_dir()
    result = run(state_dir, as_of, from_date)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
