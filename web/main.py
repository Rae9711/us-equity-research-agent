from __future__ import annotations

import json
import os
from datetime import date as date_type

import markdown
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.db import ConclusionRecord, MarketCase, PlaybookCase, init_db
from src.db.session import get_session
from src.research.parts_meta import part_about, part_label
from src.steps.meta import STEPS, STEP_BY_NUM, step_about, step_label
from src.utils.paths import (
    morning_json_path,
    morning_report_path,
    raw_data_path,
    step_json_path,
    step_report_path,
)
from src.utils.trading_calendar import today_et
from src.web.history import list_trading_days
from src.web.labels import unified_about, unified_label
from src.web.stats import accuracy_by_date, accuracy_for_date, global_accuracy
from src.web.steps_status import step_available, steps_status
from src.web.verify_order import conclusion_sort_key, sort_conclusions
from src.web.verify_service import VerifyItem, VerifyPayload, save_verifications
from web.auth import optional_basic_auth

app = FastAPI(title="Daily Trading OS", version="0.1.0")
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

_AUTH = [Depends(optional_basic_auth)]


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "daily-trading-os",
            "base_url": os.environ.get("APP_BASE_URL", ""),
        }
    )


@app.get("/", response_class=HTMLResponse, dependencies=_AUTH)
def index(request: Request, date: str | None = None) -> HTMLResponse:
    trading_date = date or today_et().isoformat()
    days = list_trading_days()
    if not days:
        days = [{"date": trading_date, "steps": steps_status(trading_date)}]
    status = steps_status(trading_date)
    timeline = []
    for s in STEPS:
        timeline.append(
            {
                "num": s.num,
                "time": s.time_et,
                "title": s.title,
                "subtitle": s.subtitle,
                "step_id": s.step_id,
                "available": status.get(s.num, False),
                "url": f"/step/{s.num}?date={trading_date}",
            }
        )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active": "home",
            "trading_date": trading_date,
            "days": list_trading_days(),
            "timeline": timeline,
            "stats": global_accuracy(),
        },
    )


@app.get("/history", response_class=HTMLResponse, dependencies=_AUTH)
def history_page(request: Request) -> HTMLResponse:
    acc = accuracy_by_date()
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "active": "history",
            "days": list_trading_days(),
            "date_accuracy": {row["date"]: row for row in acc},
            "stats": global_accuracy(),
        },
    )


def _load_conclusions(trading_date: str) -> list[ConclusionRecord]:
    day = date_type.fromisoformat(trading_date)
    session = get_session()
    try:
        rows = (
            session.query(ConclusionRecord)
            .filter(ConclusionRecord.trading_date == day)
            .all()
        )
    finally:
        session.close()
    return sort_conclusions(rows)


@app.get("/verify", response_class=HTMLResponse, dependencies=_AUTH)
def verify_form(
    request: Request,
    date: str | None = None,
    saved: int | None = None,
) -> HTMLResponse:
    trading_date = date or today_et().isoformat()
    rows = _load_conclusions(trading_date)
    day_stats = accuracy_for_date(date_type.fromisoformat(trading_date))

    return templates.TemplateResponse(
        request,
        "verify.html",
        {
            "active": "verify",
            "rows": rows,
            "today": trading_date,
            "days": list_trading_days(),
            "part_label": unified_label,
            "part_about": unified_about,
            "saved": saved == 1,
            "day_stats": day_stats,
        },
    )


@app.post("/verify", dependencies=_AUTH)
async def verify_submit(request: Request) -> RedirectResponse:
    form = await request.form()
    trading_date = str(form.get("trading_date") or today_et().isoformat())
    day = date_type.fromisoformat(trading_date)

    part_ids: set[str] = set()
    for key in form.keys():
        if str(key).startswith("v_"):
            part_ids.add(str(key)[2:])

    items: list[VerifyItem] = []
    for pid in sorted(part_ids, key=conclusion_sort_key):
        ver = str(form.get(f"v_{pid}") or "").strip()
        if ver and ver not in ("对", "错", "部分对"):
            continue
        items.append(
            VerifyItem(
                part_id=pid,
                verification=ver or None,
                user_judgment=str(form.get(f"uj_{pid}") or "").strip() or None,
                notes=str(form.get(f"n_{pid}") or "").strip() or None,
            )
        )

    save_verifications(day, items)
    return RedirectResponse(
        url=f"/verify?date={trading_date}&saved=1",
        status_code=303,
    )


@app.post("/api/verify", dependencies=_AUTH)
def api_verify_submit(payload: VerifyPayload) -> JSONResponse:
    day = date_type.fromisoformat(payload.trading_date)
    updated = save_verifications(day, payload.items)
    return JSONResponse({"ok": True, "updated": updated})


def _load_raw(trading_date: str) -> dict:
    path = raw_data_path(trading_date)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No raw data for {trading_date}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/raw/{trading_date}", dependencies=_AUTH)
def api_raw(trading_date: str) -> JSONResponse:
    return JSONResponse(_load_raw(trading_date))


@app.get("/api/raw", dependencies=_AUTH)
def api_raw_latest() -> JSONResponse:
    return JSONResponse(_load_raw(today_et().isoformat()))


@app.get("/step0", response_class=HTMLResponse, dependencies=_AUTH)
def step0_view(request: Request, date: str | None = None) -> HTMLResponse:
    trading_date = date or today_et().isoformat()
    try:
        payload = _load_raw(trading_date)
    except HTTPException:
        return templates.TemplateResponse(
            request,
            "raw.html",
            {
                "active": "step0",
                "trading_date": trading_date,
                "collected_at": "—",
                "data_ready": False,
                "checklist": {},
                "missing": ["尚未采集，等待 7:45 ET 或手动运行 step0"],
                "conclusion": None,
            },
            status_code=404,
        )

    return templates.TemplateResponse(
        request,
        "raw.html",
        {
            "active": "step0",
            "trading_date": payload.get("trading_date", trading_date),
            "collected_at": payload.get("collected_at", "—"),
            "data_ready": payload.get("data_ready", False),
            "checklist": payload.get("checklist", {}),
            "missing": payload.get("missing", []),
            "conclusion": payload.get("conclusion"),
        },
    )


@app.get("/step/{step_num}", response_class=HTMLResponse, dependencies=_AUTH)
def step_page(request: Request, step_num: int, date: str | None = None) -> HTMLResponse:
    if step_num < 0 or step_num > 8:
        raise HTTPException(status_code=404, detail="Invalid step")
    trading_date = date or today_et().isoformat()

    if step_num == 0:
        return step0_view(request, date=trading_date)
    if step_num == 1:
        return today_report(request, date=trading_date)

    step_def = STEP_BY_NUM[step_num]
    json_path = step_json_path(step_num, trading_date)
    report_path = step_report_path(step_num, trading_date)

    if not report_path.exists():
        return templates.TemplateResponse(
            request,
            "step.html",
            {
                "active": "home",
                "step_num": step_num,
                "step_def": step_def,
                "trading_date": trading_date,
                "days": list_trading_days(),
                "report_html": (
                    f"<p class='empty'>Step {step_num} 尚未生成。"
                    f"等待 {step_def.time_et} ET 自动运行，或在 VPS 执行 "
                    f"<code>step{step_num}_*</code>。</p>"
                ),
                "meta": None,
                "conclusion": None,
            },
            status_code=404,
        )

    meta = None
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))

    report_html = markdown.markdown(
        report_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    return templates.TemplateResponse(
        request,
        "step.html",
        {
            "active": "home",
            "step_num": step_num,
            "step_def": step_def,
            "trading_date": trading_date,
            "days": list_trading_days(),
            "report_html": report_html,
            "meta": meta,
            "conclusion": (meta or {}).get("conclusion"),
        },
    )


@app.get("/today", response_class=HTMLResponse, dependencies=_AUTH)
def today_report(request: Request, date: str | None = None) -> HTMLResponse:
    trading_date = date or today_et().isoformat()
    report_path = morning_report_path(trading_date)
    json_path = morning_json_path(trading_date)

    if not report_path.exists():
        return templates.TemplateResponse(
            request,
            "today.html",
            {
                "active": "today",
                "trading_date": trading_date,
                "report_html": "<p class='empty'>该日 Morning Research 尚未生成。<a href='/history'>查看历史</a> 或等待 8:00 ET 自动运行。</p>",
                "meta": None,
            },
            status_code=404,
        )

    report_html = markdown.markdown(
        report_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    meta = None
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))

    return templates.TemplateResponse(
        request,
        "today.html",
        {
            "active": "today",
            "trading_date": trading_date,
            "report_html": report_html,
            "meta": meta,
        },
    )


@app.get("/api/step/{step_num}/{trading_date}", dependencies=_AUTH)
def api_step(step_num: int, trading_date: str) -> JSONResponse:
    path = step_json_path(step_num, trading_date)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No step {step_num} for {trading_date}")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/morning/{trading_date}", dependencies=_AUTH)
def api_morning(trading_date: str) -> JSONResponse:
    path = morning_json_path(trading_date)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No morning report for {trading_date}")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/history", dependencies=_AUTH)
def api_history() -> JSONResponse:
    return JSONResponse(list_trading_days())


@app.get("/cases", response_class=HTMLResponse, dependencies=_AUTH)
def cases_page(request: Request) -> HTMLResponse:
    """Read-only Market Case browser."""
    session = get_session()
    try:
        rows = session.query(MarketCase).order_by(MarketCase.date.desc()).limit(90).all()
        cases = []
        for row in rows:
            try:
                data = json.loads(row.case_json)
                cases.append({
                    "date": row.date,
                    "regime": (data.get("regime") or {}).get("label", "N/A"),
                    "actual_driver": (data.get("labels") or {}).get("actual_driver", "N/A"),
                    "hypothesis_correct": (data.get("labels") or {}).get("hypothesis_correct", "N/A"),
                    "surprise": data.get("surprise", ""),
                    "attribution": data.get("attribution", {}),
                })
            except Exception:
                cases.append({"date": row.date, "regime": "Error", "actual_driver": "N/A",
                               "hypothesis_correct": "N/A", "surprise": "", "attribution": {}})
    finally:
        session.close()

    return templates.TemplateResponse(
        request,
        "cases.html",
        {"active": "cases", "cases": cases},
    )


@app.get("/playbook", response_class=HTMLResponse, dependencies=_AUTH)
def playbook_page(request: Request) -> HTMLResponse:
    """Read-only Playbook Case browser."""
    session = get_session()
    try:
        rows = session.query(PlaybookCase).order_by(PlaybookCase.created_at.desc()).limit(100).all()
        playbook = []
        for row in rows:
            try:
                pattern = json.loads(row.pattern_json)
            except Exception:
                pattern = {}
            playbook.append({
                "case_id": row.case_id,
                "date": row.trading_date or "N/A",
                "regime": pattern.get("regime", "N/A"),
                "actual_driver": pattern.get("actual_driver", "N/A"),
                "lesson": row.lesson or "",
                "surprise": row.surprise or "",
            })
    finally:
        session.close()

    return templates.TemplateResponse(
        request,
        "playbook.html",
        {"active": "playbook", "playbook": playbook},
    )


@app.get("/api/cases", dependencies=_AUTH)
def api_cases() -> JSONResponse:
    session = get_session()
    try:
        rows = session.query(MarketCase).order_by(MarketCase.date.desc()).limit(90).all()
        return JSONResponse([{"date": r.date, "case": json.loads(r.case_json)} for r in rows])
    finally:
        session.close()


@app.get("/api/cases/{trading_date}", dependencies=_AUTH)
def api_case_detail(trading_date: str) -> JSONResponse:
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == trading_date).first()
        if not row:
            raise HTTPException(status_code=404, detail=f"No case for {trading_date}")
        return JSONResponse(json.loads(row.case_json))
    finally:
        session.close()


@app.get("/api/stats", dependencies=_AUTH)
def api_stats() -> JSONResponse:
    g = global_accuracy()
    return JSONResponse(
        {
            "global": {
                "verified": g.verified,
                "correct": g.correct,
                "wrong": g.wrong,
                "partial": g.partial,
                "accuracy_pct": g.accuracy_pct,
            },
            "by_date": accuracy_by_date(),
        }
    )
