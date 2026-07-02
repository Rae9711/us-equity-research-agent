# Cursor Automation — Daily Trading OS v2 完整构建与验证

> 本文件供 **Cursor Automation**（Cloud Agent）单次长跑使用。  
> 规范来源：`WORKFLOW.md` v2.0 · `IMPLEMENTATION-PLAN.md` v2.0

---

## 0. 运行前检查

1. 仓库已 push 到 GitHub（Cloud Agent 需要 `gitConfig`）。
2. `.env` **不入 git**；VPS 上 `/opt/trading-os/.env` 已配置 API keys。
3. 当前基线：**Phase 0–4 已实现**（collectors、P1–P16、Web、S2–S4）。
4. 禁止：伪造数据、跳过验收、声称「能自动下单赚钱」（系统为 `ADVISORY_ONLY`）。

---

## 1. 实施顺序（必须串行，每 Phase 验收后再进下一 Phase）

### Phase 5a — Market Case DB

- [ ] `market_cases` 表（`date` PK、`case_json` TEXT、`created_at`）
- [ ] `playbook_cases` 表（`case_id`、`pattern_json`、`lesson`、`surprise`）
- [ ] `training_rows` 表（`date`、`features_json`、`labels_json`）
- [ ] `src/schemas/market_case.py` Pydantic 模型（对齐 IMPLEMENTATION-PLAN § Market Case）
- [ ] 每日 Job 结束双写：`data/reports/YYYY-MM-DD/case.json` + DB
- **验收：** `python -c` 或 pytest 插入/读取一条 case；Web 可读（见 Phase 6 UI）

### Phase 5b — Feature Store

- [ ] `src/features/build.py`：从 `data/raw/{date}.json` + 历史 OHLCV 生成 `features` 字典
- [ ] 字段至少：`dgs10`, `vix`, `vix_chg`, `qqq_chg`, `smh_chg`, `nvda_chg`, `spy_chg`, `oil_chg`, `dxy_chg`, `breadth_proxy`
- **验收：** 对 `2026-07-02`（或最近有 raw 的日期）产出非空 features

### Phase 5c — R0 Regime Engine

- [ ] `src/engines/regime.py` + `config/rules/regime.yaml`
- [ ] 规则优先（非 LLM）：输入 features → `Regime` 枚举 + confidence
- [ ] 接入 `morning_research`：**先于 P1** 运行，结论 ID `R0`，写入 case
- **验收：** morning.json 含 `r0`；R0 与 P12 不矛盾

### Phase 5d — P17 Hypothesis

- [ ] `src/engines/hypothesis.py`
- [ ] 输出：`id`, `statement`, `evidence[]`, `counter_evidence[]`, `confidence`
- [ ] Morning 末尾生成；S2/S3 更新 hypothesis 状态（支持/削弱/推翻）
- **验收：** `morning.json` 含 `hypothesis`；S2 结论引用 hypothesis

### Phase 5e — S7 Attribution

- [ ] `src/engines/attribution.py` + `src/steps/s07_evening.py` + `src/jobs/step7_evening.py`
- [ ] 分解 QQQ 日涨跌 → `ai`, `bond`, `macro`, `oil`, `other`（合计 1.0）
- [ ] 产出 `surprise`, `lesson`, `labels.actual_driver`, `labels.hypothesis_correct`
- **验收：** `step7.json` 含 attribution；case.json 字段完整

### Phase 5f — S8 Learning

- [ ] `src/engines/bayesian.py`：Driver 后验每日更新，持久化 `data/bayesian_drivers.json`
- [ ] `src/engines/playbook.py`：模式匹配 + Case 入库（`Case-{n}`）
- [ ] `src/jobs/step8_learning.py`：写 training_row + 更新 bayesian + playbook
- **验收：** 跑完 S8 后 bayesian 权重变化可观测；playbook_cases 有记录

### Phase 6 — S5/S6 + 全调度 + Web

- [ ] `s05_midday.py`, `s06_afternoon.py` + jobs
- [ ] `src/runner/main.py` 注册 S5–S8 cron（对齐 IMPLEMENTATION-PLAN 调度表）
- [ ] Web：`/cases`, `/playbook` 只读页
- **验收：** 本地或 Docker `curl /health`；history 矩阵显示 S5–S8

### Phase 7 — S4 EV + Calibration

- [ ] 扩展 `s04_decision.py`：`expected_return`, `expected_loss`, `risk_reward`
- [ ] `src/engines/calibration.py`：用历史 `/verify` + attribution 校准 confidence
- **验收：** step4.json 含 EV 字段；向后兼容 YES/NO

### Phase 8 — Weekly ML

- [ ] `src/engines/ml/train.py`：XGBoost + SHAP（需 ≥30 训练行，目标 60）
- [ ] 周日 job `weekly_ml`；报告 `data/reports/ml/YYYY-MM-DD-shap.md`
- **验收：** 有数据时产出 feature importance；无数据时优雅 skip

### Deploy

- [ ] `./deploy/hetzner-ship.sh` 部署到 VPS
- [ ] `ssh root@5.161.58.191 "docker exec trading_runner python -m ..."` 冒烟

---

## 2. 60 交易日回放（Backtest / Replay）

实现 `src/jobs/backtest_replay.py`（或 `scripts/replay_60d.py`）：

```bash
python -m src.jobs.backtest_replay --days 60 --end-date TODAY
```

**对每个交易日 T（仅 NYSE 开市日）：**

| Step | 回放动作 | 检验点 |
|------|----------|--------|
| 0 | 拉 raw（或缓存 yfinance 历史补全） | `data_ready` |
| 1a–c | R0 → Morning → P17 | case 草稿 |
| 2–4 | S2–S4（用 T 日 OHLCV 模拟盘中） | hypothesis 验证 |
| 5–6 | S5–S6 | driver 持续性 |
| 7–8 | S7 attribution + S8 learning | bayesian/playbook 累积 |

**Learning 专项指标（必须输出报告 `data/reports/backtest/60d-summary.md`）：**

1. **Hypothesis 命中率**：S7 `hypothesis_correct` 分布（对/错/部分）
2. **Driver 归因误差**：Morning P10 vs S7 `actual_driver` 一致率
3. **Bayesian 稳定性**：后验是否剧烈震荡（日际 Δ>20% 计数）
4. **Playbook 召回**：相似日检索 Top-3 是否含同 regime 历史日
5. **S4 虚拟 PnL**（见 §3）：60 日累计

**注意：** 历史新闻/期权可能不全 → 记录 `data_gaps[]`，不得编造 Polygon 新闻。

---

## 3. 今日实盘检验（TODAY）

对**今天**跑完整 pipeline（非回放）：

```bash
python -m src.jobs.step0_collect
python -m src.jobs.step1_morning   # 含 R0/P17
# ... S2–S8 按时间或手动触发
```

产出 `data/reports/TODAY/validation-today.md`：

- 逐步核对 WORKFLOW 结论索引（R0, P1–P17, S0–S8）
- Market Case JSON 字段完整性 checklist
- Web UI 截图路径或 curl 检查列表

---

## 4. 「能否赚钱」评估框架（诚实边界）

系统为 **ADVISORY_ONLY**，评估的是 **决策质量**，不是实盘下单。

### 4.1 虚拟回测规则（60 日）

- **信号：** S4 `should_trade == YES` 且 `confidence >= 0.65`
- **标的：** step4 `instrument` 默认映射 QQQ 当日收益（Call 简化：QQQ 涨则正收益）
- **成本：** 每笔扣 0.05% 滑点 + 0.1% 期权近似损耗
- **基准：** 买入持有 QQQ（B&H）

**输出表：**

| 指标 | Agent | B&H |
|------|-------|-----|
| 累计收益 | | |
| 夏普（年化） | | |
| 最大回撤 | | |
| 胜率 | | |
| 盈亏比 | | |

### 4.2 必须通过的红线

- [ ] 60 日虚拟夏普 **不显著低于** B&H - 0.3（否则决策层无效）
- [ ] Learning：Bayesian 后验随 attribution 方向一致更新（抽检 10 日）
- [ ] 无「未来函数」：特征仅用 T 日及之前数据
- [ ] LLM 调用仅报告层（grep 确认 engines 无 anthropic import）

### 4.3 最终结论模板

```markdown
## Verdict: {NOT READY / PROMISING / NEEDS WORK}

### 证据
- ...

### 与「赚钱 Trading Agent」的差距
- ...

### 优化建议（按优先级 P0/P1/P2）
- P0: ...
```

---

## 5. 优化建议方向（Agent 需自行验证后填写）

- **P0** Regime 规则是否过粗？是否需 HMM/聚类？
- **P0** Attribution 是否用回归/SHAP 替代手工比例？
- **P1** 60 日标签：自动 pseudo-label vs 人工 `/verify`
- **P1** EV 校准：confidence 是否 well-calibrated（可靠性图）
- **P2** Polygon 付费层、 breadth 真数据、Mag7 以外 leader
- **P2** Postgres 迁移、监控告警

---

## 6. 交付物清单

1. 全部 Phase 5a–8 代码 + 测试
2. `data/reports/backtest/60d-summary.md`
3. `data/reports/TODAY/validation-today.md`
4. `docs/BUILD-REPORT.md`（Verdict + 优化建议）
5. VPS 部署确认（`curl https://rae-trading.com/health`）
6. 更新 `IMPLEMENTATION-PLAN.md` 各 Phase 状态为 ✅ 或注明阻塞项

---

*Automation runbook v1 · 2026-07-02*
