# Daily Trading OS v2 — BUILD REPORT

**Date**: 2026-07-02  
**Branch**: `cursor/bc-543e269a-34d0-4a2b-af00-c96b796d47b1-d18b`  
**Trigger**: `automation/trading-os-v2-build` push  
**Baseline**: Phase 0–4 complete (collectors, Morning P1–P16, Web, S2–S4)

---

## Phase Completion Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 5a | Market Case DB | ✅ COMPLETE | `market_cases`, `playbook_cases`, `training_rows` tables + Pydantic models |
| 5b | Feature Store | ✅ COMPLETE | `src/features/build.py` — 10 features from raw data |
| 5c | R0 Regime Engine | ✅ COMPLETE | `src/engines/regime.py` + `config/rules/regime.yaml` |
| 5d | P17 Hypothesis | ✅ COMPLETE | `src/engines/hypothesis.py` — integrated into morning research |
| 5e | S7 Attribution | ✅ COMPLETE | `src/engines/attribution.py` + `src/steps/s07_evening.py` |
| 5f | S8 Learning | ✅ COMPLETE | `src/engines/bayesian.py` + `src/engines/playbook.py` + `src/jobs/step8_learning.py` |
| 6 | S5/S6 + Full Schedule | ✅ COMPLETE | `s05_midday.py`, `s06_afternoon.py` + runner wired + `/cases` `/playbook` |
| 7 | S4 EV + Calibration | ✅ COMPLETE | `src/engines/calibration.py` + EV fields in step4.json |
| 8 | Weekly ML | ✅ COMPLETE | `src/engines/ml/train.py` — XGBoost+SHAP, graceful skip if <10 rows |
| 60d Replay | Backtest | ✅ COMPLETE | `src/jobs/backtest_replay.py` + `60d-summary.md` |

---

## Deliverable Checklist

- [x] Phase 5a–8 code + tests
- [x] `data/reports/backtest/60d-summary.md`
- [x] `data/reports/TODAY/validation-today.md`
- [x] `docs/BUILD-REPORT.md` (this file)
- [ ] VPS `/health` — requires `./deploy/hetzner-ship.sh` (API keys on VPS only)

---

## 60-Day Backtest Summary

**Period**: 2026-04-10 → 2026-07-02 (58 NYSE trading days)

| Metric | Agent | B&H |
|--------|------:|----:|
| Cumulative Return | 0.00% | 17.13% |
| Sharpe (annualized) | 0.00 | 3.00 |
| Max Drawdown | 0.00% | 7.03% |
| Win Rate | 0.0% | N/A |
| Agent Trades | 0 | N/A |

**Important context**: Agent made 0 trades because `data/raw/` was empty at build time — no API keys in CI environment. Features could not be computed, so the signal was never triggered. The B&H 17.13% return represents actual QQQ performance over the period (yfinance historical). This is the cold-start expected state, not a fabrication.

---

## Verdict

```markdown
## Verdict: NOT READY (Cold-Start State)

### 证据
- Agent 0 trades: cold-start, no raw data in CI → features all null → signal never fires
- B&H +17.13% over 60 days confirms the market was favorable
- All core engines (Regime, Hypothesis, Attribution, Bayesian, Playbook) code-complete and tested
- DB schema: market_cases, playbook_cases, training_rows tables created
- LLM boundary verified: no anthropic import in engines/

### 与「赚钱 Trading Agent」的差距
- 需要 ≥60 天真实 raw data 积累（Step 0 每日运行）
- Hypothesis correct rate 和 attribution accuracy 需收盘核对数据才能统计
- Bayesian 后验需 ≥10 天迭代才能脱离 prior
- XGBoost/SHAP 需 ≥30 行训练数据才能产出可靠 feature importance
- 当前 S4 EV 使用 heuristic（无历史 win rate）；需 ≥5 行 labeled 数据切换为 historical

### 优化建议（按优先级 P0/P1/P2）
- P0: 部署到 VPS 并启动日常 Step 0–8 数据积累（`./deploy/hetzner-ship.sh`）
- P0: Regime 规则当前是 3条 if-then；30天后考虑用聚类或 HMM 替代
- P0: Attribution 当前是启发式权重；30天后用 OLS 回归替代
- P1: 自动 pseudo-label：用 `actual_driver` 自动填写 training_rows，不依赖手动 /verify
- P1: EV calibration：当 labeled rows ≥30 时，用历史 win rate 替代 heuristic
- P1: 60日报告应在真实数据积累后重跑（使用实际 raw data 而非 yfinance fallback）
- P2: Polygon 付费层补全 breadth/sector 数据
- P2: Playbook 相似度提升：加入 regime + sector + macro event 多维度检索
- P2: Postgres 迁移（当前 SQLite 适合单机）
```

---

## Architecture Verification

### LLM Boundary (ADVISORY_ONLY)
```
grep -r "anthropic" src/engines/ src/features/ src/steps/ src/jobs/
→ 0 matches

LLM used only in:
- src/llm/anthropic_client.py
- src/research/morning.py (report narrative + Hypothesis statement polish)
```

### No Future-Function Leakage
- `src/features/build.py`: all features use `trading_date` or `prior_trading_day(trading_date)` data
- `src/engines/attribution.py`: uses `trading_date` raw data only
- `src/jobs/backtest_replay.py`: each day uses only T-day features and T-day prices

### ADVISORY_ONLY Confirmation
- All S4 outputs include `disclaimer: "仅供参考，非交易指令 · ADVISORY_ONLY"`
- No broker API integration anywhere in codebase
- Web `/verify` used for label entry only, not for order placement

---

## Blocking Items for VPS Deploy

1. **API Keys**: Ensure `/opt/trading-os/.env` contains:
   - `ANTHROPIC_API_KEY`
   - `POLYGON_API_KEY`
   - `FRED_API_KEY`

2. **Deploy command**:
   ```bash
   cd /opt/trading-os/repo
   git pull origin cursor/bc-543e269a-34d0-4a2b-af00-c96b796d47b1-d18b
   ./deploy/hetzner-ship.sh
   ```

3. **Smoke test**:
   ```bash
   curl https://rae-trading.com/health
   # Expected: {"status":"ok","service":"daily-trading-os"}
   ```

4. **Cold-start data collection** (run once manually):
   ```bash
   docker exec trading_runner python -m src.jobs.step0_collect
   docker exec trading_runner python -m src.jobs.step1_morning
   ```

---

## Files Added in Phase 5a–8

```
src/db/models.py              — +MarketCase, PlaybookCase, TrainingRow tables
src/db/__init__.py            — export new models
src/db/market_case_service.py — dual-write service (file + DB)
src/schemas/market_case.py    — Pydantic models for Market Case
src/features/__init__.py      — feature engineering module
src/features/build.py         — build_features() from raw data
src/engines/__init__.py       — engines module
src/engines/regime.py         — R0 Regime Engine (rule-based)
src/engines/hypothesis.py     — P17 Hypothesis Engine (rule-based)
src/engines/attribution.py    — S7 Attribution Engine (heuristic)
src/engines/bayesian.py       — L4 Bayesian Driver update
src/engines/playbook.py       — L3 Playbook Case store+retrieve
src/engines/calibration.py    — S4 EV calibration
src/engines/ml/__init__.py    — ML module
src/engines/ml/train.py       — XGBoost + SHAP weekly trainer
src/steps/s05_midday.py       — Step 5 Midday Review
src/steps/s06_afternoon.py    — Step 6 Afternoon Review
src/steps/s07_evening.py      — Step 7 Evening Review (Attribution)
src/jobs/step5_midday.py      — Job wrapper S5
src/jobs/step6_afternoon.py   — Job wrapper S6
src/jobs/step7_evening.py     — Job wrapper S7
src/jobs/step8_learning.py    — Step 8 Learning job
src/jobs/backtest_replay.py   — 60-day backtest replay script
src/runner/main.py            — +S5,S6,S7,S8,weekly_ml jobs wired
web/main.py                   — +/cases, /playbook, /api/cases routes
web/templates/cases.html      — Market Case browser
web/templates/playbook.html   — Playbook browser
web/templates/nav.html        — +Cases/Playbook nav links
config/rules/regime.yaml      — Regime classification rules
requirements.txt              — +numpy, xgboost, shap
data/reports/backtest/60d-summary.md  — Backtest results
data/reports/TODAY/validation-today.md — Today's validation
docs/BUILD-REPORT.md          — This file
```

---

*Build Agent: Cursor Automation · Daily Trading OS v2 · 2026-07-02*
