# AGENTS.md

## Cursor Cloud specific instructions

Daily Trading OS is a single Python 3.12 product (FastAPI web UI + APScheduler runner) backed by SQLite. There is no Node/JS frontend build — templates/static under `web/` are served directly by FastAPI.

### Environment
- Dependencies live in a virtualenv at `.venv` (created by the startup update script from `requirements.txt`). Activate it with `source .venv/bin/activate` before running anything.
- There are currently no automated tests and no lint/type-check config in the repo, so "lint" and "test" are N/A. If you add tests, prefer `pytest`.

### Required env vars for local dev
The code defaults `DATA_ROOT` to `/data` (production Docker volume), which is not writable locally. For local dev you MUST override paths, e.g.:
```bash
export DATA_ROOT=./data
export DATABASE_URL=sqlite:///./data/trading_os.db
```
Without `DATA_ROOT=./data`, collectors/report writers fail trying to write to `/data`.

### Running services
- Web app (required): `uvicorn web.main:app --reload --host 0.0.0.0 --port 8020` then check `curl http://127.0.0.1:8020/health`. The DB and its tables auto-create on startup via `init_db()`.
- Runner (scheduler): `python -m src.runner.main` runs a blocking APScheduler on a US/Eastern trading-calendar cron — it will not do anything immediately outside scheduled windows. To exercise the pipeline on demand, run individual jobs instead: `python -m src.jobs.step0_collect [--date YYYY-MM-DD]` (and `step1_morning`, `step2_open`, `step3_update`, `step4_decision`).

### API keys / offline behavior
- External data/LLM keys (`POLYGON_API_KEY`, `FRED_API_KEY`, `ANTHROPIC_API_KEY`) are optional for basic dev. `step0_collect` runs without them: it collects free RSS news, records missing sources in a checklist, still writes `data/raw/<date>.json`, and persists a `Step0` conclusion to SQLite. Steps needing paid data (Polygon options/intraday) or Claude (morning research) require the corresponding key.

### Hello-world / smoke check
Run `python -m src.jobs.step0_collect` to create a `Step0` conclusion, then in the web UI go to `/verify`, mark it 对/错/部分对, save, and confirm accuracy updates on `/history`. This exercises the collector → SQLite → UI feedback loop without any API keys.
