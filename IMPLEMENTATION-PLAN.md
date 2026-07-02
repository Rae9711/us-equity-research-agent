# Daily Trading OS — 实现计划

> 配套文档：[WORKFLOW.md](WORKFLOW.md)  
> 状态：**v2.0 — Learning-First 架构**（Phase 0–4 已落地，Phase 5+ 为 Learning 飞轮）

---

## 产品定位（v2）

| v1 定位 | v2 定位 |
|---------|---------|
| Morning Research Agent + 日报 | **可积累 Alpha 的 Trading OS** |
| 核心产出 = Markdown 报告 | 核心产出 = **Market Case**（结构化、可训练、可检索） |
| LLM 驱动各 Part 判断 | **LLM ≈ 10%**（叙事与解释）；决策与学习由 **规则 + 统计 + Case + Bayesian** |

**壁垒不在「ChatGPT + 新闻总结」，而在 Learning Engine 与 Playbook 数据库。**

---

## 已锁定决策（保留）

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **数据源** | yfinance + FRED + **Polygon News** + RSS | Polygon 主新闻源；RSS 补充 |
| **运行形态** | 常驻 Runner（APScheduler） | Step 0–8 美东 cron |
| **交易执行** | **建议 only** | 无券商 API；`ADVISORY_ONLY` |
| **核对闭环** | Web `/verify` | 对/错/部分对 → DB（**即 Label 入口**） |
| **部署** | Hetzner VPS · **`rae-trading.com`** | Docker + Caddy |
| **LLM** | Anthropic Claude | **仅**报告叙事、Hypothesis 表述、SHAP/Case 解释 |
| **推送** | 无 | Web UI |

---

## 一、总体架构（v2）

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        Automation Runner (APScheduler)                   │
│                        Step 0–8 · America/New_York                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Layer   │     │  Feature Store   │     │  Web UI          │
│  collectors   │────►│  每日特征向量     │     │  流程 Step 0–8   │
│  Step 0       │     │  (from raw)      │     │  /verify 核对    │
└───────┬───────┘     └────────┬────────┘     │  /history 日历   │
        │                      │                └─────────────────┘
        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Decision Stack（非 LLM 为主）                          │
│  ① Regime Engine → ② Morning P1–P16 → ③ P17 Hypothesis                  │
│  → ④ 盘中 S2–S6 → ⑤ S4 Decision（EV，非单纯 YES/NO）                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Learning Stack（系统核心价值）                         │
│  ⑥ S7 Attribution → ⑦ Bayesian Driver 更新 → ⑧ Playbook Case 入库        │
│  + Rules（YAML）+ 周度 XGBoost/SHAP（数据够后）                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Market Case（每日一行 JSON + DB）← 真相来源；Markdown 报告 = 视图        │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LLM Layer（~10%）                                                       │
│  Morning Report · Evening Review · 解释 SHAP · 解释相似 Case · surprise  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 五模块学习（基金经理模型）

```text
① Bayesian Layer      — 每日更新「各 Driver 有多重要」的后验信念
② Statistical (XGB)   — 周度：哪些 Feature 预测收益 / 主导 Driver（SHAP 解释）
③ Playbook Retrieval  — 检索历史最相似 Market Case（Recall，非 Predict）
④ Rule Engine           — Regime 绑定的 if-then（冷启动与可解释兜底）
⑤ LLM                   — 把 ①–④ 的输出翻译成人能读的报告
```

**LLM 不参与：** 权重更新、Attribution 数值、XGBoost 训练、Playbook 写入逻辑。

---

## 二、Market Case（核心数据模型）

每天收盘后（S7/S8）写入一条 **Market Case**，而非仅追加 Markdown。

```json
{
  "date": "2026-07-02",
  "features": {
    "dgs10": 4.44, "vix_chg": -0.5, "nvda_chg": 2.1, "smh_chg": 1.8,
    "qqq_gap": 0.12, "breadth": 0.65, "oil_chg": -1.2
  },
  "regime": { "label": "AI Expansion", "confidence": 0.82 },
  "hypothesis": {
    "id": "H-2026-07-02-001",
    "statement": "AI dominates macro today",
    "evidence": ["NVDA +2%", "SOXX +1.8%", "VIX↓"],
    "counter_evidence": ["10Y↑", "Consumer weak"],
    "confidence": 0.74
  },
  "morning": { "bias": "Neutral", "total": -1 },
  "intraday": { "s2": "...", "s3": "...", "s4": "..." },
  "trade_plan": { "scenarios": ["A", "B", "C"], "advisory_only": true },
  "actual": { "qqq_return": 0.021 },
  "attribution": {
    "ai": 0.70, "bond": 0.10, "oil": 0.05, "macro": 0.10, "other": 0.05
  },
  "labels": {
    "actual_driver": "AI",
    "hypothesis_correct": "部分对"
  },
  "surprise": "Bond mattered less than geo risk fade",
  "lesson": "When geo risk falls, AI leadership strengthens even if yields elevated",
  "playbook_refs": ["Case-127"],
  "bayesian_drivers": { "AI": 0.38, "Bond": 0.22, "Macro": 0.18 }
}
```

**实现表：** `market_cases` · `playbook_cases` · `training_rows` · 现有 `conclusions`（过渡）

---

## 三、数据源（保留 + 补充）

| 数据类型 | 主源 | 备源 |
|----------|------|------|
| 指数/ETF/个股 | yfinance | Polygon |
| 利率 | FRED | ^TNX intraday |
| 新闻 | **Polygon News** | RSS |
| 期权 | Polygon / yfinance | — |
| 盘中 | yfinance 1m / Polygon | — |

Collector 映射不变：`src/collectors/*` → Step 0 `raw_data.json`。

**Feature Engineering：** `src/features/` 从 raw + 盘中快照生成 `features` 字典，写入 Market Case。

---

## 四、Step 1 内部结构（v2）

对外仍为 **Step 1 @ 8:00**，对内三段：

| 子步 | ID | 引擎 | 说明 |
|------|-----|------|------|
| 1a | **R0** | Regime Engine | **最先**判断当前市场时代（规则→ML） |
| 1b | P1–P16 | Research | 在 Regime 约束下做 Morning Research |
| 1c | **P17** | Hypothesis | 可验证假设（Evidence / Counter / ID） |

P12 Regime **保留**为叙事 Part，但 **R0 为决策级 Regime**，P12 与之对齐、不矛盾。

---

## 五、Automation 调度（保留）

| Cron (ET) | Step | Job ID | v2 要点 |
|-----------|------|--------|---------|
| 7:45 | Step 0 | `collect_raw` | ✅ 已实现 |
| 8:00 | Step 1 | `morning_research` | R0→P1–16→P17；写 `morning.json` + Case 草稿 |
| 9:30 | Step 2 | `open_report` | ✅ 验证 **Hypothesis**，非散文 |
| 10:00 | Step 3 | `market_update` | ✅ Driver 切换；**实时 Learning 层** |
| 10:15 | Step 4 | `trade_decision` | ✅ → 升级 **EV + R:R** |
| 12:00 | Step 5 | `midday_review` | 待实现 |
| 14:00 | Step 6 | `afternoon_review` | 待实现 |
| 16:10 | Step 7 | `evening_review` | **Attribution** + Decision Log |
| 20:00 | Step 8 | `learning` | Bayesian + Playbook + `training_rows` |
| 周日 | — | `weekly_ml` | XGBoost + SHAP（可选，≥60 天数据） |

每次 Job 结束：写 `reports/YYYY-MM-DD/stepN.*` + 更新 `market_cases` + `conclusions` + `daily_runs`。

---

## 六、Decision Engine（v2）

### S4 目标 Schema（升级中）

```json
{
  "should_trade": "YES",
  "instrument": "QQQ Call",
  "expected_return": 0.032,
  "expected_loss": -0.011,
  "risk_reward": 2.9,
  "confidence": 0.73,
  "disclaimer": "ADVISORY_ONLY",
  "inputs": {
    "regime": "AI Expansion",
    "hypothesis_id": "H-2026-07-02-001",
    "playbook_similar": ["Case-127"],
    "bayesian_ai_weight": 0.38
  }
}
```

当前 MVP 仍为 YES/NO；EV 字段在 Phase 7 补齐。

### LLM 使用边界

| 用 LLM | 不用 LLM |
|--------|----------|
| Morning/Evening 叙事段落 | Regime 分类（规则优先） |
| Hypothesis `statement` 润色 | P4–P7、P9、P11 打分 |
| 解释 SHAP / 相似 Case | Attribution 分解 |
| `surprise` / `lesson` 一句话 | Bayesian 更新、Playbook 匹配分 |

---

## 七、Learning Engine（四层）

| 层 | 方法 | 何时 | 模块 |
|----|------|------|------|
| **L1 Rules** | YAML if-then，绑定 Regime | 日初 | `engines/rules/` |
| **L2 Statistical** | XGBoost + SHAP | 每周 | `engines/ml/` |
| **L3 Case Memory** | Playbook 相似检索 | 日初+日终 | `engines/playbook/` |
| **L4 Bayesian** | Driver 后验每日更新 | 日终 S8 | `engines/bayesian/` |

**Learning 学什么：** 不是「预测 QQQ 涨跌」本身，而是 **哪些 Driver 在何种 Regime 下对结果贡献多大**。

**Label 来源：** `/verify` 核对 + S7 `actual_driver` + Attribution 结果。

**核心日终问句：** `What surprised me today?` → 写入 `market_cases.surprise`。

---

## 八、Web UI（已实现 + 规划）

| 路径 | 状态 | 功能 |
|------|------|------|
| `/` | ✅ | Step 0–8 时间线 + 日期选择 |
| `/step/{0-8}` | ✅ | 各 Step 报告 |
| `/history` | ✅ | 日历矩阵 S0–S8 + 准确率 |
| `/verify` | ✅ | 核对表单 → **Labels** |
| `/cases` | 规划 | Market Case 浏览 |
| `/playbook` | 规划 | Case 检索与相似日 |

---

## 九、项目结构（目标 v2）

```text
us-equity-research-agent/
├── WORKFLOW.md
├── IMPLEMENTATION-PLAN.md
├── config/
│   ├── symbols.yaml
│   ├── scoring_rules.yaml
│   ├── rules/                    # Regime 绑定规则
│   └── schedule.yaml
├── src/
│   ├── collectors/               # Step 0 ✅
│   ├── features/                 # Feature Engineering
│   ├── engines/
│   │   ├── regime.py             # R0
│   │   ├── hypothesis.py           # P17
│   │   ├── attribution.py        # S7
│   │   ├── bayesian.py
│   │   ├── playbook.py
│   │   ├── rules.py
│   │   └── ml/                   # XGBoost weekly
│   ├── research/                 # Morning P1–P16 ✅
│   ├── steps/                    # S2–S8
│   ├── llm/                      # 仅报告/解释 ✅
│   ├── runner/
│   └── db/
│       ├── models.py             # + market_cases, playbook_cases
│       └── ...
├── web/
└── data/
    ├── raw/
    ├── reports/
    └── trading_os.db
```

---

## 十、分 Phase 实施（v2 路线图）

### 已完成

| Phase | 交付物 | 状态 |
|-------|--------|------|
| **0** | VPS + Docker + SQLite + Schema | ✅ |
| **1** | collectors → `raw_data.json` | ✅ |
| **2** | P1–P16 + `morning_research` | ✅ |
| **3** | Web 流程首页 + `/verify` + `/history` | ✅ |
| **4** | S2–S4 盘中 jobs | ✅（S4 待升 EV） |

### 已完成（Learning 飞轮 — Phase 5a–8）

| Phase | 交付物 | 状态 |
|-------|--------|------|
| **5a** | `market_cases` / `playbook_cases` / `training_rows` 表 + Pydantic schema + 日终双写 | ✅ 2026-07-02 |
| **5b** | `src/features/build.py` 自动特征（10字段） | ✅ 2026-07-02 |
| **5c** | R0 Regime Engine `engines/regime.py` + `config/rules/regime.yaml` | ✅ 2026-07-02 |
| **5d** | P17 Hypothesis `engines/hypothesis.py`，集成进 morning research | ✅ 2026-07-02 |
| **5e** | S7 Attribution `engines/attribution.py` + `steps/s07_evening.py` | ✅ 2026-07-02 |
| **5f** | S8 Bayesian `engines/bayesian.py` + Playbook `engines/playbook.py` + `jobs/step8_learning.py` | ✅ 2026-07-02 |
| **6** | S5/S6 步骤 + 全 cron wired + `/cases` `/playbook` 页面 | ✅ 2026-07-02 |
| **7** | S4 EV + Calibration `engines/calibration.py`，step4.json 含 EV 字段 | ✅ 2026-07-02 |
| **8** | Weekly ML `engines/ml/train.py`（XGBoost+SHAP；< 10 行优雅 skip） | ✅ 2026-07-02 |

### 阻塞项（VPS 部署待确认）

- [ ] VPS `./deploy/hetzner-ship.sh` 部署（需 API keys）
- [ ] `curl https://rae-trading.com/health` 冒烟
- [ ] 真实 raw data 积累后重跑 60d 回放

---

## 十一、部署与运维（保留）

| 项 | 值 |
|----|-----|
| 域名 | `https://rae-trading.com` |
| 目录 | `/opt/trading-os/repo` |
| Docker | `docker compose -p trading-os` |
| 端口 | `127.0.0.1:8020` |
| 发布 | `./deploy/hetzner-ship.sh` |

```text
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5
POLYGON_API_KEY=
FRED_API_KEY=
WEB_BASIC_AUTH_USER=          # 可选
WEB_BASIC_AUTH_PASS=
TZ=America/New_York
```

详见 [deploy/DEPLOY.md](deploy/DEPLOY.md)、[docs/API-SETUP.md](docs/API-SETUP.md)。

---

## 十二、与 WORKFLOW 的对应（v2）

| WORKFLOW 概念 | 实现落点 |
|---------------|----------|
| Market Case | `db/market_cases` + `reports/.../case.json` |
| R0 Regime | `engines/regime.py` · Step 1a |
| P17 Hypothesis | `engines/hypothesis.py` |
| P1–P16 | `research/morning.py`（逐步减 LLM） |
| 结论块 + 核对 | `conclusions` + `/verify` → **Labels** |
| S7 Attribution | `engines/attribution.py` |
| S8 Bayesian + Playbook | `engines/bayesian.py` + `engines/playbook.py` |
| 周度 ML | `engines/ml/train.py` |
| LLM 叙事 | `llm/` 仅报告渲染 |
| 建议 only | S4 `ADVISORY_ONLY` |

---

## 十三、成本粗估（月）

| 项目 | 估算 |
|------|------|
| Polygon | ~$29–199/月 |
| FRED / yfinance | $0 |
| Hetzner VPS | $0 增量（共享） |
| Anthropic | **降本方向**：LLM 仅报告层，约 $0.3–1/天 |

---

*版本：v2.0 · Learning-First · rae-trading.com · Phase 0–4 已落地*
