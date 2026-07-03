# Daily Trading Operating System（Daily Trading OS）

> 无论是交易员本人，还是 Agent，都走**完全一样的流程**。  
> 区别只是：你手动看信息、做判断；Agent 自动收集信息、给出判断。

本文档是整套系统的**唯一 workflow 规范**。  
实现细节见 [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md)（v2 Learning-First）。

---

## 设计原则

### 1. 不叫 Morning Agent

Morning Research 只是全天流程中的一个 Step（8:00 AM），不是系统名字。完整系统叫 **Daily Trading OS**。

### 2. 核心是 Market Case + Learning，不是报告字数

每天最重要的不是写了多少分析，而是沉淀一条 **Market Case**（结构化 JSON），并回答四件事：

1. **今天市场最大的 Driver 是什么？（只能选一个）**
2. **我为什么决定交易或不交易？（含 Hypothesis 与证据/反证）**
3. **收盘后，涨跌由谁贡献？（Attribution）**
4. **今天什么出乎我意料？（surprise → Playbook）**

连续几个月坚持，得到的不是交易日记，而是**可训练、可检索、经市场验证的 Case 数据库**。

### 3. 六层流水线（人走流程时不感知分层）

```text
                         Daily Trading OS v2

                ┌────────────────────┐
                │   Data Collector   │  Step 0：Raw Data
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Feature Engineering │  特征向量 → 写入 Market Case
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Regime Engine R0  │  Step 1a：先定「市场时代」
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Research + Hypothesis│ Step 1b P1–P16 · Step 1c P17
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Decision Engine   │  Step 2–4：验证 → EV 决策（建议 only）
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Attribution Engine │  Step 7：谁真正推动了涨跌
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Learning Engine   │  Step 8：Bayesian · Playbook · ML
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  LLM（~10%）        │  仅：报告叙事 · 解释 SHAP/Case
                └────────────────────┘
```

**同一根 10Y 涨跌，在 AI Expansion 与 Macro Fear 两种 Regime 下含义不同——故 R0 必须在 Morning 叙事之前。**

### 4. Market Preference Engine（市场偏好引擎）

机构还会问一个问题：

> **今天市场有没有「忽略」某个消息？**

例如：消费者信心弱于预期（按理偏空）、债券收益率略升（按理偏空），市场照样涨。  
说明当下**偏好函数**是：AI > 宏观。

**Market Preference Engine**（P14）每天回答一句话：

> **市场今天更愿意交易什么，而不是理论上应该交易什么。**

### 5. Hypothesis 优先于散文（P17）

Morning 不能只输出「市场可能涨」类模糊判断。Step 1 结束前必须形成 **可验证 Hypothesis**：

```text
Hypothesis ID：H-2026-07-02-001
Statement：AI dominates macro today
Evidence：NVDA +2% · SOXX +1.8% · VIX↓
Counter-evidence：10Y↑ · Consumer weak
Confidence：74%
```

盘中 Step 2–4 的核心任务是 **验证或推翻 Hypothesis**，而非重复写新闻摘要。

### 6. LLM 边界

| LLM 负责 | 非 LLM 负责 |
|----------|-------------|
| Morning / Evening 报告段落 | Regime（R0）、Scoring（P11） |
| Hypothesis 表述润色 | Attribution 分解 |
| 解释 SHAP、相似 Playbook Case | Bayesian 权重、Playbook 入库 |
| `surprise` / `lesson` 一句话 | XGBoost 训练 |

### 7. 每个 Part 必须有结论（可核对）

Agent 每完成一个 Part / Step，**必须**输出结构化 **结论**块，不能只写分析过程。

目的：

- 你一眼看到 Agent **判断了什么**
- 你对照自己的判断，收盘后标记 **对 / 错 / 部分对**（**即 Learning 的 Label**）
- 积累成可统计的准确率，而不只是长文日记

格式见下文 **[结论输出规范](#结论输出规范)**。

---

## 全天时间线（Overview）

| 时间 | Step | 名称 | 核心问题 |
|------|------|------|----------|
| 7:45 AM | Step 0 | 数据更新 | 所有 Raw Data 是否就绪？ |
| 8:00 AM | Step 1 | Morning Research | 昨天发生了什么？今天 Driver 是什么？怎么打分？怎么计划？ |
| 9:30 AM | Step 2 | Open | 开盘是否验证 Morning？**不交易，只观察** |
| 10:00 AM | Step 3 | Market Update | Driver 有没有变？Score 要不要重算？ |
| 10:15 AM | Step 4 | Trading Decision | Should we trade? Why? |
| 12:00 PM | Step 5 | Midday Review | 今天最大的 Driver 还是 Morning 那个吗？ |
| 2:00 PM | Step 6 | Afternoon Review | Trade still valid? |
| 4:10 PM | Step 7 | Evening Review | Morning 哪里错了？哪个 Indicator 有用/没用？ |
| 8:00 PM | Step 8 | Learning | Bayesian 更新 · Playbook 入库 · 训练行写入 |

> **Step 1 对内三段：** 1a **R0 Regime** → 1b **P1–P16** → 1c **P17 Hypothesis**（对外仍显示为 8:00 Morning）。

---

## 核心资产：Market Case

每天最终落库的是一条 **Market Case**（JSON），Markdown 报告只是其人类可读视图。

| 字段块 | 来源 Step | 用途 |
|--------|-----------|------|
| `features` | Step 0 + 特征工程 | ML / 相似日检索 |
| `regime` | R0 | 规则与 Bayesian 的条件变量 |
| `hypothesis` | P17 | 盘中验证对象 |
| `morning` / `intraday` | P1–S6 | 决策轨迹 |
| `attribution` | S7 | 谁真正推动了涨跌 |
| `labels` | `/verify` + S7 | 监督学习 Label（**verify 优先于 S7 自动**） |
| `surprise` / `lesson` | S7–S8 + `/verify` | Playbook 入库（核对纠正写入 `lesson`） |
| `bayesian_drivers` | S8 | 次日 Morning 先验 |

完整 Schema 见 [IMPLEMENTATION-PLAN.md § Market Case](IMPLEMENTATION-PLAN.md#二market-case核心数据模型)。

---

## Learning 四层（Step 8 与周度）

| 层 | 方法 | 何时 |
|----|------|------|
| **L1 Rules** | Regime 绑定的 if-then（YAML） | 每日晨会 |
| **L2 Statistical** | XGBoost + SHAP | 每周（数据 ≥60 天） |
| **L3 Case Memory** | Playbook 相似 Market Case 检索 | 每日晨会 + 日终 |
| **L4 Bayesian** | Driver 后验每日更新 | Step 8（S7 核对为错且无纠正 Driver 时跳过） |

**`/verify` 保存后**：立即将 P17/P15/S7 核对写入 Market Case `labels`，提取核对纠正为 Playbook `lesson`，并重跑 Step 8（Bayesian · Playbook · `training_rows`）。

**Learning 学什么：** 在何种 Regime 下，哪些 Driver 对结果贡献多大——不是单纯「猜明天涨跌」。

---

## 结论输出规范

### 统一结构

每个 Part / Step 的分析正文之后，Agent **必须**追加以下两块（核对块由你填写，Agent 留空）：

```text
┌─ 结论（Agent 必填）────────────────────────────┐
│ 判断：    {本 Part 的核心结论，尽量用枚举或短语}      │
│ 置信度：  {0–100% 或 N/A}                        │
│ 一句话：  {≤ 30 字，PM 晨会怎么讲就怎么写}          │
└────────────────────────────────────────────────┘

┌─ 核对（你填写，用于验收 Agent）──────────────────┐
│ Agent 对吗？  ☐ 对  ☐ 错  ☐ 部分对               │
│ 你的判断：                                       │
│ 备注：                                           │
└────────────────────────────────────────────────┘
```

**规则**：

1. **判断** 必须是可验证的陈述（收盘后能判定对错），不能是模糊废话
2. **一句话** 必须能单独念出来，不依赖上文
3. Scoring / Scenario 等 Part，**判断** 用文档规定的枚举值（见下表）
4. Evening Review 时，回头填 **核对**，写入 Decision Log

### 每日结论索引（Agent 在 Step 1 结束后汇总）

Agent 在 Morning Research 全部 Part 完成后，输出一张 **结论总表**，便于你对照：

| ID | Part / Step | 判断（Agent） | 置信度 | 一句话 |
|----|-------------|---------------|--------|--------|
| R0 | Regime Engine | | | |
| P1 | Part 1 | | | |
| P2 | Part 2 | | | |
| … | … | | | |
| P17 | Hypothesis | | | |
| S4 | Step 4 交易 | | | |

收盘后你在 **核对** 列填 ✓ / ✗ / ~。

### 各 Part 结论字段定义

| ID | 判断（枚举 / 格式） | 一句话示例 |
|----|---------------------|------------|
| **Step 0** | `数据就绪：YES / NO` | 全部 Raw Data 已更新 |
| **R0** | `Regime：{AI Expansion / Macro Fear / …}` | 当前处于 AI 扩张期，Bond 次要 |
| **P1** | `昨日定性：Risk-on / Risk-off / Neutral / 分化` | 周五无新宏观利好也无新利空 |
| **P2** | `昨日主 Driver：{一个词}` | 昨天是 Risk Premium 下降推动 |
| **P3** | `本周首要：{Event}` | 市场进入 Labor Week |
| **P4** | `Growth：Bullish / Neutral / Bearish` + Score | Bond 中性偏空 Growth |
| **P5** | `对 QQQ：Positive / Neutral / Negative` + Score | Dollar 略强，轻微利空 QQQ |
| **P6** | `恐慌：Low / Medium / High` + Score | VIX 降，无恐慌 |
| **P7** | `最强板块：{Sector}` + `轮动：{一句话}` | 科技开始分化 |
| **P8** | `AI：Bullish / Neutral / Bearish` + Confidence | AI 偏多但估值存疑 |
| **P9** | `买期权：Yes / No` · `0DTE：Yes / No` | IV 中等可买，不做 0DTE |
| **P10** | `今日 Driver：{一个词}` | 今天等就业，不是昨天 Risk-on |
| **P11** | `Bias：Strong Bull / Bull / Neutral / Bear / Strong Bear` + Total | Bullish Bias，Total +5 |
| **P12** | `Regime：Trending / Range / Unclear` | 今天不是趋势确认日 |
| **P13** | `Edge：YES / NO` | 今天无明显 Edge |
| **P14** | `偏好函数：{X > Y}` | 市场更愿意交易 AI 而非宏观 |
| **P15** | `最可能 Scenario：A / B / C` | 基准情景为震荡观望 |
| **P16** | `计划：Trade / Wait / No Trade` | 等 NVDA+QQQ 突破再 Call |
| **P17** | `Hypothesis：{ID}` · `成立/待验证` | AI 主导宏观，等开盘验证 |
| **S2** | `开盘：Healthy / Weak / Mixed` · `Hypothesis：支持/削弱/中性` | 开盘验证 Risk-on |
| **S3** | `Driver 变了吗：YES / NO` · 若变 `{新 Driver}` | Driver 切到 Risk Appetite |
| **S4** | `Should trade：YES / NO` · 若 YES `{标的}` · `EV/R:R`（目标） | 买 QQQ Call，R:R 2.9 |
| **S5** | `Driver 仍成立：YES / NO` | Midday Driver 仍为 AI |
| **S6** | `Trade valid：YES / NO` | Thesis 仍成立，Hold |
| **S7** | `真正 Driver：{一个词}` · `Attribution：{AI 70%…}` · `Hypothesis：对/错/部分` · `Surprise：{一句}` | 实际是半导体；Bond 影响被高估 |
| **S8** | `Bayesian 最大调整：{Driver ±%}` · `Playbook：新增/命中 Case` | AI 后验 +10%；入库 Case-127 |

---

## Step 0 — 7:45 AM 数据更新

**执行者**：Agent 自动（人：确认 checklist 打勾）

**输出**：Raw Data（原始数据包，不做解读）

```text
Macro
✓ Fed
✓ Economic Calendar
✓ PCE
✓ Nonfarm countdown

Market
✓ SPY · QQQ · TQQQ · VIX · DXY · 10Y Treasury

Sector
✓ XLK · SMH · XLF · XLE

Stocks
✓ NVDA · MSFT · AAPL · AMZN · META · GOOGL · TSLA

News
✓ Reuters · Bloomberg · WSJ

Options
✓ QQQ Option Chain · IV · Put/Call Ratio · Open Interest
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 数据就绪：`YES` / `NO`（若有缺失项列出） |
| 置信度 | N/A |
| 一句话 | 例：全部 Raw Data 已更新，可进入 Morning Research |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 1 — 8:00 AM Morning Research

整个系统最重要的 Step。结构固定，每日填满。

**对内顺序（必须遵守）：**

```text
Step 1a  R0 Regime Engine     → 先定市场时代（决策级）
Step 1b  Part 1–16            → Morning Research（在 Regime 约束下）
Step 1c  Part 17 Hypothesis   → 可验证假设（Evidence / Counter）
```

---

### Step 1a — R0 Regime Engine（先于 Part 1）

**核心问题：** 当前市场处于什么「时代」？同一宏观信号在不同 Regime 下权重不同。

**输入：** Step 0 特征（10Y、VIX、SMH/QQQ 相对强度、广度、新闻主题等）

**输出示例：**

```text
Regime：AI Expansion
Confidence：82%
Regime 含义：Bond 对科技股约束减弱；半导体领导力优先于就业数据
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Regime：`{枚举，如 AI Expansion / Macro Fear / Liquidity Driven / Range}` |
| 置信度 | {如 82%} |
| 一句话 | 例：AI 扩张期，Bond 为次要变量 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

> **Part 12（Market Regime）** 保留为叙事与 Trend/Range 描述；**R0 为引擎级 Regime**，二者应对齐、不矛盾。

---

### Part 1 — 昨天发生了什么？

先总结**上一个交易日**（周五 / 周一等），再写一句话结论。

**输出要素**：
- 指数表现表（SPY、QQQ、Dow、VIX 等）
- 关键宏观/新闻事实（如：PCE 符合预期、Fed 路径未变）
- **一句话总结**（可含解读；与旧版「纯 Fact Part 1」不同，此处允许 Morning 层面的叙事）

**边界**：区分「昨天的事实」与「对昨天的解释」——解释放在 Part 2。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 昨日定性：`Risk-on` / `Risk-off` / `Neutral` / `分化` |
| 置信度 | N/A（本 Part 偏事实汇总） |
| 一句话 | 例：周五没有新的宏观利好，也没有新的宏观利空 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 2 — 昨天为什么涨/跌？

回答：**昨天最大的 Driver 是什么？**

格式建议：

```text
昨天最大的 Driver：{X}

因果链：
{X} → {Y} → {Z} → 市场反应

Morning Note（一句话）：
{例如：昨天是 Risk Premium 下降推动，而不是宏观数据推动。}
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 昨日主 Driver：`{一个词，如 Middle East / Macro / AI}` |
| 置信度 | {如 80%} |
| 一句话 | 例：昨天是 Risk Premium 下降推动，不是宏观数据 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 3 — 今天市场最关心什么？

列出今日/本周重要 Event，标 Importance（⭐ 1–5）。

| Event | Importance |
|-------|----------:|
| … | ⭐⭐⭐⭐⭐ |

**结论**：Today's Driver = `{一个词或短语}`

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 本周/今日首要关注：`{Event，如 Employment / Fed}` |
| 置信度 | {Importance 最高项的 ⭐ 数对应主观置信，如 85%} |
| 一句话 | 例：市场进入 Labor Week，Nonfarm 是本周核心 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 4 — Bond

```text
2Y：{High / Flat / Low}
10Y：{如 4.39%}
Direction：{Up / Flat / Down}
```

**结论**：Growth → {Bullish / Neutral / Bearish}

**Morning Score**：Bond → `{如 -1}`（附简短理由）

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Growth：`Bullish` / `Neutral` / `Bearish` · Score：`{如 -1}` |
| 置信度 | {如 70%} |
| 一句话 | 例：10Y 小涨，Growth 中性偏空 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 5 — Dollar

```text
Dollar：{Slightly Down / Up / Flat}
```

**结论**：对 Risk / QQQ → {Positive / Negative / Neutral}

**Morning Score**：Dollar → `{分数}`

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 对 QQQ：`Positive` / `Neutral` / `Negative` · Score：`{分数}` |
| 置信度 | {如 65%} |
| 一句话 | 例：Dollar 略强，轻微利空 QQQ |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 6 — VIX

```text
VIX：{Up / Down}
情绪：{No Panic / Elevated / …}
```

**Morning Score**：VIX → `{分数}`

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 恐慌：`Low` / `Medium` / `High` · Score：`{分数}` |
| 置信度 | {如 75%} |
| 一句话 | 例：VIX 下降，无恐慌，Risk-on 可信 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 7 — Sector Rotation

昨日板块相对表现（箭头或分数）：

```text
Technology      ↓
Semiconductor   ↓
Energy          ↓
Healthcare      ↑
Financial       ↑
```

**一句话**：{如：科技开始分化。}

**Sector Score 表**（可选）：

| Sector | Score |
|--------|------:|
| Semiconductor | +3 |
| … | … |

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 最强板块：`{Sector}` · 轮动：`{分化 / 普涨 / 普跌 / 防御轮动}` |
| 置信度 | {如 70%} |
| 一句话 | 例：科技开始分化，半导体相对最强 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 8 — AI Theme

```text
AI：{Bullish / Neutral / Bearish}
Confidence：{如 75%}

理由：
· …
· …

但是：{风险或不确定点，如估值受到质疑}
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | AI：`Bullish` / `Neutral` / `Bearish` |
| 置信度 | {如 75%} |
| 一句话 | 例：AI 偏多，但需区分反弹还是真回流 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 9 — Options

```text
QQQ IV：{Low / Medium / High}
是否适合买期权：{Yes / No}
是否做 0DTE：{通常本周有重大数据则 No}
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 买期权：`Yes` / `No` · 0DTE：`Yes` / `No` |
| 置信度 | N/A |
| 一句话 | 例：IV 中等可买，本周有非农不做 0DTE |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 10 — Today's Driver

**与 Part 3 呼应，但更聚焦「今天」**：

> 今天市场真正等待的是 {X}，而不是昨天的 {Y}。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 今日 Driver：`{一个词，全天只能有一个}` |
| 置信度 | {如 80%} |
| 一句话 | 例：今天等就业，不是昨天的 Risk-on |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 11 — Scoring

| Factor | Score |
|--------|------:|
| Macro | |
| Fed | |
| Bond | |
| Dollar | |
| VIX | |
| AI | |
| Earnings | |
| Breadth | |
| Momentum | |
| … | |

**Total**：`{如 +5}`

**Morning 结论**：{如 Bullish Bias — 不是 Strong Bullish，因为还有就业数据}

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Bias：`Strong Bull` / `Bull` / `Neutral` / `Bear` / `Strong Bear` · Total：`{如 +5}` |
| 置信度 | {如 72%} |
| 一句话 | 例：Bullish Bias，但非 Strong Bull |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 12 — Market Regime

> **与 R0 的关系：** R0（Step 1a）给出决策级 Regime（如 AI Expansion）；本 Part 描述 Trend/Range/Vol 等**战术层面**状态，须与 R0 一致、不矛盾。

```text
Trend：{Up / Down / Neutral}
Volatility：{Low / Medium / High}
Risk：{Low / Medium / High}
Liquidity：{Good / Tight / …}
```

**一句话**：{如：今天不是趋势确认日。}

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Regime：`Trending Up` / `Trending Down` / `Range` / `Unclear` |
| 置信度 | {如 65%} |
| 一句话 | 例：今天不是趋势确认日 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 13 — Edge

回答系统里最重要的问题之一：

```text
Do we have an edge today?

YES / NO

Why?
```

若无重大 Catalyst（无 CPI / PCE / FOMC 等），Edge 往往为 **NO**。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Edge：`YES` / `NO` |
| 置信度 | {若 YES，如 60%；若 NO 可 N/A} |
| 一句话 | 例：今天无 CPI/PCE/FOMC，无明显 Edge |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 14 — Market Preference（市场偏好引擎）

回答：

> **市场今天更愿意交易什么，而不是理论上应该交易什么？**

格式：

```text
理论应交易：{如 弱消费者信心 → 避险}
实际在交易：{如 AI / Semiconductor}
偏好函数：{如 AI > Macro}
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 偏好函数：`{X > Y}`（如 `AI > Macro`） |
| 置信度 | {如 70%} |
| 一句话 | 例：市场忽略弱数据，继续买科技 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 15 — Today's Scenario

至少三个 Scenario：

**Scenario A（偏多）**

```text
触发条件 → 市场反应 → 行动（如买 QQQ Call）
```

**Scenario B（偏空 / 放弃）**

```text
触发条件 → 市场反应 → 行动（如放弃、减仓）
```

**Scenario C（震荡 / 不交易）**

```text
Range → No Trade
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 最可能 Scenario：`A` / `B` / `C` |
| 置信度 | {如 A=50%, B=30%, C=20% 或只写最可能项置信度} |
| 一句话 | 例：基准情景 C 震荡，等突破 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 16 — Trading Plan

把 Scenario 落成可执行的 IF-THEN：

```text
Scenario A：若 NVDA 领涨 + QQQ 突破昨日高点 → 买 QQQ Call
Scenario B：若 10Y 继续涨 + QQQ 跌回昨日开盘 → 放弃
Scenario C：若 Semiconductor 回落 → 不追
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 计划：`Trade` / `Wait` / `No Trade` |
| 置信度 | {如 68%} |
| 一句话 | 例：等 NVDA 领涨+QQQ 破高再 Call |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Part 17 — Hypothesis（Step 1c，Morning 最后一步）

在 P1–P16 之后，把当日判断收敛为 **一条可验证假设**，供 Step 2–4 检验。

**格式：**

```text
Hypothesis ID：H-YYYY-MM-DD-001
Statement：{一句话，可证伪}
Evidence：{支持点 1} · {支持点 2} · …
Counter-evidence：{反证 1} · {反证 2} · …
Confidence：{0–100%}
```

**示例：**

```text
Hypothesis ID：H-2026-06-30-001
Statement：AI / Semiconductor 主导指数，Macro（就业）为次要变量
Evidence：SOXX 强于 QQQ · NVDA 领涨 · VIX 下行
Counter-evidence：10Y 略升 · 消费者信心偏弱
Confidence：74%
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Hypothesis：`{ID}` · 状态：`待验证`（盘中更新为 支持/削弱/推翻） |
| 置信度 | {如 74%} |
| 一句话 | 例：芯片主导，就业数据当日非主驱动 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

### Morning Research 结束：输出结论总表

R0 + Part 1–17 全部完成后，Agent **必须**输出 [每日结论索引](#每日结论索引agent-在-step-1-结束后汇总) 中 **R0 + P1–P17** 一行汇总，写入 **Market Case 草稿**，再进入 Step 2。

---

## Step 2 — 9:30 AM Open

**原则：不交易，只观察。**

确认 **P17 Hypothesis** 是否被开盘验证（不只验证 Morning 散文叙事）。

| 观察项 | 看什么 |
|--------|--------|
| ① Gap | QQQ Gap Up 还是 Gap Down？ |
| ② Breadth | 上涨家数 vs 下跌家数；QQQ 涨但 80% 股票跌 → 异常 |
| ③ Leader | 谁带队？（如 NVDA） |
| ④ Bond | 10Y 是否突然跳？ |
| ⑤ Hypothesis | Evidence / Counter 哪边在开盘得到印证？ |

**交叉验证**：

- QQQ 涨 + SOXX 跌 → Morning Risk-on **可能为假**

**输出**：

```text
Opening Report

Market：Healthy / Weak / Mixed
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 开盘：`Healthy` / `Weak` / `Mixed` · Hypothesis：`支持` / `削弱` / `中性` |
| 置信度 | {如 75%} |
| 一句话 | 例：Gap up + SOXX 同步，Hypothesis 获支持 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 3 — 10:00 AM Market Update

Morning Research **没有**的能力：盘中 Driver 切换与 **Hypothesis 实时修订**。

当重大新闻出现（例：Reuters — Middle East 停火 → Oil 跌 → Risk On）：

```text
Driver Changed

Old：Nonfarm
New：Risk Appetite
```

**必须**：重新计算 Score，并更新 P17 Hypothesis 的 Evidence / Counter 权重（**实时 Learning 层**）。

示例：

```text
         昨日          更新后
Macro    -2            -2
AI       +2            +3
Risk      0            +2
─────────────────────────
Total              →  +3（说明可以买）
```

**Confidence 更新**（若开盘验证 Morning）：

```text
Confidence：70% → 85%
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Driver 变了吗：`YES` / `NO` · 若变，新 Driver：`{词}` · 新 Total：`{分数}` |
| 置信度 | {更新后整体置信，如 80%} |
| 一句话 | 例：Driver 切到 Risk Appetite，Total 升至 +3 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 4 — 10:15 AM Trading Decision

Agent（或人）**不直接说 Buy**，先回答：

```text
Should we trade?

YES / NO
```

**目标形态（Phase 7）：** 在 YES/NO 之外输出 **期望值 EV** 与 **风险回报比 R:R**，而非仅凭感觉。

```text
Expected Return：+3.2%
Expected Loss：-1.1%
Risk/Reward：2.9
Confidence：73%
```

若 **YES**，再回答 Why（须引用 R0 Regime、P17 Hypothesis、Playbook 相似 Case 若存在）：

```text
Why?

· Risk ↓
· AI ↑
· 10Y Stable
· Breadth Healthy
· QQQ Breakout
· Hypothesis H-… 仍成立
```

**输出示例（当前 MVP）：**

```text
Trade：QQQ Call
Confidence：72%
Disclaimer：ADVISORY_ONLY（仅供参考，非交易指令）
```

若 **NO**，写明原因（无 Edge、Scenario 未触发、Hypothesis 被削弱、Preference 与理论矛盾等）。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Should trade：`YES` / `NO` · 若 YES：`{标的，如 QQQ Call}` |
| 置信度 | {如 72%} |
| 一句话 | 例：Risk↓ AI↑ Breadth 健康，买 QQQ Call |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 5 — 12:00 PM Midday Review

**核心问题**：今天最大的 Driver 还是 Morning 那个吗？

- 若 **没变** → Hold / 维持计划
- 若 **变了** → 更新 Driver，必要时调整仓位或撤销 Thesis

```text
Morning Driver：Employment
Midday Driver：AI（未变 / 已变为 …）
```

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Driver 仍成立：`YES` / `NO` · 行动：`Hold` / `Adjust` / `Exit` |
| 置信度 | {如 70%} |
| 一句话 | 例：Midday Driver 仍为 AI，继续 Hold |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 6 — 2:00 PM Afternoon Review

**只有一个问题**：

```text
Trade still valid?
```

例：Morning 买 Call，午后 NVDA 转跌 → Thesis 破坏 → **减仓或平仓**。

另：监控 Bond — 若 10Y 突然飙升 → 提醒「科技估值压力增加」。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Trade valid：`YES` / `NO` · 若 NO，行动：`减仓` / `平仓` / `观望` |
| 置信度 | {如 65%} |
| 一句话 | 例：NVDA 转弱，Thesis 破坏，应减仓 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## Step 7 — 4:10 PM Evening Review（Attribution Engine）

**不是写散文复盘，是产出 Market Case 的 `attribution` + `labels` + `surprise`。**

### 7.1 Outcome Attribution（涨跌由谁贡献）

将当日 QQQ（或主交易标的）涨跌分解为 Driver 贡献（合计 100%）：

```text
AI / Semiconductor：70%
Bond / Rates：10%
Oil / Geo：5%
Macro Data：10%
Other：5%
```

### 7.2 今天真正的 Driver 是谁？（只能选一个）

```text
Morning / Hypothesis 认为：Employment
实际主导：Semiconductor / AI
Hypothesis 结果：部分对（方向对，Driver 标签错）
```

### 7.3 What surprised me today?（Playbook 入库关键）

```text
Surprise：Bond 影响小于预期；地缘缓和后 AI 领导力加强
Lesson：当 geo risk 回落时，即使 10Y 偏高，Chip 仍可主导指数
```

### 7.4 Morning 哪里错了？

```text
低估：Risk Appetite / AI 强度
高估：Macro 就业叙事对当日的约束力
```

### 7.5 哪个 Indicator 最有用 / 最没用？

```text
最有用：SOXX / Chip 领涨
最没用：Dollar（当日）
```

### 7.6 Decision Log（必填）

| 字段 | 内容 |
|------|------|
| 今日唯一 Driver | |
| Hypothesis ID + 结果 | |
| Attribution 摘要 | |
| Surprise | |
| 交易决定 | Trade / No Trade |
| 收盘验证 | 哪些对 / 哪些错 |

**Step 7 结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | 真正 Driver：`{词}` · Attribution：`{AI 70%…}` · Hypothesis：`对/错/部分` · Surprise：`{一句}` |
| 置信度 | N/A（复盘项） |
| 一句话 | 例：芯片主导 70%；Bond 被高估；Hypothesis 部分对 |

**核对（你）**：逐项对照 R0 + P1–P17 + S2–S6 结论总表，填 ✓ / ✗ / ~ → 写入 **Labels**

---

## Step 8 — 8:00 PM Learning

**不是写日报，是更新模型与 Case 库。**

### 8.1 Bayesian Driver 更新（L4）

用 S7 Attribution + 历史先验，更新各 Driver **后验权重**（供次日 R0 / Morning 使用）：

| Driver | 晨间先验 | 日终后验 | 为什么调整 |
|--------|--------:|--------:|------------|
| AI / 半导体 | 35% | **45%** | Attribution 70%；Hypothesis 部分验证 |
| Macro（就业） | 25% | **20%** | 重要但当日未兑现 |
| Fed | 15% | 15% | 无新信息 |
| Bond | 15% | **10%** | 收益率未主导走势 |
| 地缘政治 | 10% | 10% | 余波弱于前日 |

### 8.2 Playbook Case 入库（L3）

当出现可复现模式时写入 **Playbook**，供未来 R0 / S4 检索相似 Case：

```text
Case ID：Case-127
IF：Middle East 缓和 + Oil 跌 + 科技股超跌 + Regime=AI Expansion
THEN：提高 Risk Appetite；Driver 可能从 Macro 切到 Risk-on
Surprise：Bond 次要
Lesson：{来自 S7}
```

### 8.3 训练行写入（L2 输入）

将当日 `features` + `labels` + `attribution` 追加到 `training_rows`，供周度 XGBoost/SHAP。

### 8.4 Preference 学习

记录当日「理论 vs 实际」偏好，累积成 **Market Preference Engine** 训练集。

### 8.5 日终 Market Case 定稿

合并 Step 0–S7 全部字段，写入 `market_cases` 表；Markdown Evening 报告仅为视图。

**结论（Agent 必填）**

| 字段 | 内容 |
|------|------|
| 判断 | Bayesian 最大调整：`{Driver} {±%}` · Playbook：`新增 Case-…` / `命中 Case-…` |
| 置信度 | N/A |
| 一句话 | 例：AI 后验 +10%；入库 Case-127 |

**核对（你）**：Agent 对吗？ · 你的判断 · 备注

---

## 参考日：Monday 2026/6/29（Day 1）

> 交易员晨会视角 — 非事后解释。

### Morning Research 摘要

**Part 1 — 昨天（周五）**

```text
PCE 符合预期 · Fed 路径未变 · 科技股连续调整 · 市场等待新 Catalyst

一句话：周五没有新的宏观利好，也没有新的宏观利空。
```

**Part 3 / 10 — Today's Driver**

| Event | Importance |
|-------|----------:|
| Nonfarm 本周四 | ⭐⭐⭐⭐⭐ |
| Fed | ⭐⭐⭐⭐ |
| AI | ⭐⭐⭐⭐ |
| Earnings | ⭐⭐ |
| Oil | ⭐⭐ |

→ **Employment Expectations**

**Part 12 — Regime**

```text
Trend：Neutral · Volatility：Medium · Risk：Medium · Liquidity：Good
→ 今天不是趋势确认日。
```

**Part 4–6** — Bond Flat / Growth Neutral · Dollar Slightly Down / Risk Positive · VIX Down / No Panic

**Part 7 — Sector**

```text
Technology ↓ · Semiconductor ↓ · Energy ↓ · Healthcare ↑ · Financial ↑
→ 科技开始分化。
```

**Part 8 — AI** — Bullish 75%，但估值受质疑

**Part 13 — Edge** — **NO**（无 CPI / PCE / FOMC）

**Part 15 — Scenario**

- A：NVDA Strong + QQQ 破周五高点 → Bullish
- B：10Y Spike + QQQ 破周五低点 → Bearish
- C：Range → No Trade

### 盘中（简述）

- **9:30 Open**：观察 Gap / Breadth / Leader / Bond → Opening Report
- **10:00**：若 Middle East 停火 → Driver 从 Nonfarm 切到 **Risk Appetite** → Score 重算
- **10:15**：Should we trade? → 若 Total 转正 + 条件满足 → QQQ Call
- **12:00 / 2:00**：Driver 是否切换 · Trade still valid?

### Evening / Learning

- 真正 Driver 可能是 Risk Appetite，而非 Morning 的 Employment
- 更新权重矩阵与 Playbook

---

## 参考日：Tuesday 2026/6/30（Day 2）

> 完全按交易员工作流 — PM 8:00 晨会怎么讲。

### Part 1 — 昨天（周一）发生了什么？

| 指数 | 表现 | 解读 |
|------|------|------|
| SPY | +1% 左右 | Risk-on 恢复 |
| QQQ | +2% 左右 | 科技明显强于大盘 |
| Dow | 创新高 | 资金风险偏好恢复 |
| VIX | 下跌 | 恐慌下降 |

> **昨天不是普通反弹，而是 Risk Appetite 回来了。**

### Part 2 — 昨天为什么涨？

最大 Driver：**Middle East 风险缓和** → Oil 跌 → 通胀预期下降 → Risk On → 科技股反弹  
（辅以 AI 板块技术性反弹）

> **昨天是 Risk Premium 下降推动，而不是宏观数据推动。**

### Part 3 — 今天最关心什么？

进入 **Labor Week**（ADP、Initial Claims、Nonfarm）。

**Today's Driver**：⭐⭐⭐⭐⭐ **Employment**

### Part 4 — Bond

10Y 昨天略涨（Job Openings 仍强 → 市场重估 Fed 担忧）

**Bond Score：-1**（不是 -3，涨幅不大）

### Part 5–6 — Dollar 略强（轻微利空 QQQ）· VIX 续降（**Score +1**）

### Part 7 — Sector

**最强：Semiconductor**（NVDA、AMD、Intel 齐涨）

```text
Semiconductor +3 · Software +1 · Financial +1 · Energy -2 · Healthcare 0
```

### Part 8 — AI

重新成为 Leader；但需区分 Short Covering vs 真资金回流 → **Confidence 70%**

### Part 9 — Options

IV 中等，可买期权；因有 Nonfarm，**不做 0DTE**

### Part 10 — Today's Driver

> 今天市场真正等待的是**就业数据**，而不是昨天的 Risk On。

### Part 11 — Scoring

| Factor | Score |
|--------|------:|
| Macro | 0 |
| Fed | -2 |
| Bond | -1 |
| Dollar | -1 |
| VIX | +1 |
| AI | +3 |
| Earnings | +1 |
| Breadth | +2 |
| Momentum | +2 |
| **Total** | **+5** |

**结论：Bullish Bias**（非 Strong Bullish，因有就业）

### Part 14 — Market Preference

- 理论：弱消费者信心 + 收益率升 → 偏空
- 实际：市场继续买 Technology
- **偏好函数：AI > Macro**

### Part 16 — Trading Plan

- A：NVDA 领涨 + QQQ 破昨日高点 → QQQ Call
- B：10Y 续涨 + QQQ 跌回昨日开盘 → 放弃
- C：Semiconductor 回落 → 不追

### 盘中

| 时间 | 动作 |
|------|------|
| 9:30 | 确认 Risk-on 真假：看 NVDA、SOXX、Breadth、Bond |
| 10:00 | 若 QQQ+SOXX+Breadth 健康 → Confidence 70%→85%，可考虑 QQQ |
| 12:00 | Driver 仍为 AI → Hold |
| 2:00 | 监控 10Y；无飙升 → Hold |
| 4:00 | 实际：Nasdaq +1.5%，S&P +0.8%，Chip 领涨 |

### Evening Review

| 问题 | 答案 |
|------|------|
| Morning Driver | Employment |
| **实际 Driver** | **Semiconductor / AI** |
| Morning 错在哪 | 高估 Macro 约束，低估 Chip 主导 |
| 最有用 Indicator | Chip / SOXX |
| 最没用 Indicator | Dollar（当日） |

### Learning — Driver 权重调整

| Driver | Morning | Evening | 调整原因 |
|--------|--------:|--------:|----------|
| AI / 半导体 | 35% | **45%** | 芯片主导，忽略弱消费者信心 |
| Macro（就业） | 25% | **20%** | 重要但当日未主导 |
| Fed | 15% | 15% | 无新信息 |
| Bond | 15% | **10%** | 未主导走势 |
| 地缘政治 | 10% | 10% | 余波减弱 |

---

## 附录 A：结论输出速查（Agent 打印用）

```text
每个 Part 结尾必有三行：
  判断：{枚举}
  置信度：{% 或 N/A}
  一句话：{≤30字}

Morning 结束 → 输出 R0 + P1–P17 结论总表 + Market Case 草稿
收盘后 → 你填 核对（对/错/部分）→ Labels → Learning
```

## 附录 B：Morning Research Checklist（打印用）

```text
□ R0    Regime Engine（先于 Part 1）  → 结论：Regime 枚举
□ Part 1  昨天发生了什么？     → 结论：昨日定性 Risk-on/off/Neutral/分化
□ Part 2  昨天为什么涨/跌？    → 结论：昨日主 Driver
□ Part 3  今天最关心什么？     → 结论：本周/今日首要关注
□ Part 4  Bond                 → 结论：Growth + Score
□ Part 5  Dollar               → 结论：对 QQQ + Score
□ Part 6  VIX                  → 结论：恐慌等级 + Score
□ Part 7  Sector Rotation      → 结论：最强板块 + 轮动
□ Part 8  AI Theme             → 结论：AI Bull/Neutral/Bear + Confidence
□ Part 9  Options              → 结论：买期权 Yes/No · 0DTE Yes/No
□ Part 10 Today's Driver       → 结论：今日 Driver（一个词）
□ Part 11 Scoring              → 结论：Bias + Total
□ Part 12 Market Regime        → 结论：Regime Trending/Range/Unclear（与 R0 对齐）
□ Part 13 Edge                 → 结论：Edge YES/NO
□ Part 14 Market Preference    → 结论：偏好函数 X > Y
□ Part 15 Scenario A/B/C       → 结论：最可能 Scenario
□ Part 16 Trading Plan         → 结论：计划 Trade/Wait/No Trade
□ Part 17 Hypothesis           → 结论：ID + Evidence/Counter + 待验证
□ ─── Morning 结论总表 R0 + P1–P17 ───
```

```text
□ Step 0  数据更新             → 结论：数据就绪 YES/NO
□ Step 2  Open                 → 结论：Healthy/Weak/Mixed + Hypothesis 支持度
□ Step 3  Market Update        → 结论：Driver 变了吗 + 新 Total + Hypothesis 修订
□ Step 4  Trade Decision       → 结论：Should trade YES/NO（目标含 EV/R:R）
□ Step 5  Midday               → 结论：Driver 仍成立 + Hold/Adjust/Exit
□ Step 6  Afternoon            → 结论：Trade valid YES/NO
□ Step 7  Evening              → 结论：Attribution + Surprise + Hypothesis 对错
□ Step 8  Learning             → 结论：Bayesian 调整 + Playbook Case
□ ─── 收盘核对：对照结论总表填 ✓ ✗ ~ → Labels ───
```

---

*文档版本：Daily Trading OS v2.0 · Learning-First · 含 R0/P17/Market Case · 配套 IMPLEMENTATION-PLAN v2*
