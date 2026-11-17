# 逐标的研判（Per-Symbol Research）— 设计，2026-11-17

## 现状与证据

`compute_trade_decision`（`src/research/trade_candidates.py`）先由市场 bias/total
算出**一个**方向（`_pick_direction`），再把这个方向盖在全部 11 个候选行上
（`row["direction"] = direction`），`_score_candidate_v2` 也在该单一方向下给每个
标的打分。判断质量面板的实测结果：77 次方向翻转在 11 个标的上同一天同步发生——
系统每天实际只做一次市场判断，独立判断约等于 17 天，不是 250 个。

## 改动 1：逐标的方向（规则引擎——真正驱动进场的那个）

- 每个标的算**两次** `_score_candidate_v2`（LONG 一次、SHORT 一次），取
  `final_score` 高的方向为该标的的方向；平局时用市场方向（bias）做 tiebreak。
- 每行记录 `direction`、`direction_source="per-symbol"`、
  `alt_direction` / `alt_direction_final_score` / `alt_direction_win_prob`
 （另一方向的分数，供透明度和复核）。
- 市场 bias 降级为三样东西：tiebreak、指数交易标签（`_index_trade_label`，
  改用排名第一的指数行自己的方向）、swing 的 `bias_direction` 输入
  （`compute_swing_opportunity` 里的 `_swing_direction` 本来就是逐标的，
  保持不动）。
- `_decision_tree._make_slot` 改用 `row["direction"]` 建 slot（horizon、
  `derive_trade_levels` 都按行方向），不再用单一 `direction` 参数。
- 不新发明模型、不新增数据源：打分器里逐标的证据已经全了（相对强度 RS vs
  QQQ/SMH、昨日涨跌、gap、动量、新闻数、VIX、driver、geo penalty）。双向打分
  只是把现成的证据在两个方向各算一遍。
- LLM 部分（P1–P18 结构不动）：`build_research_context` 增加一个 `per_symbol`
  块——每个候选标的的 {symbol, close, change_pct, 该标的自己的头条标题}。
  P12（focus stocks）据此逐标的判断，不再只继承市场判断。Anthropic 和 Gemini
  拿到同一份上下文，所以 `model_comparison.py` 的 P3/P12 对比不受影响。

## 改动 2：按校准后期望值排序（接 Item 2 的字段）

- 排序键改为三档（`_calibrated_rank_key`）：
  1. 校准期望值 ≥ 0 的行，按期望值降序；
  2. 无校准数据（map 缺失 / 桶 n=0）的行，按 `final_score` 降序——和改动前完全一致；
  3. 校准期望值 < 0 的行，按期望值降序垫底（进场过滤会在 tick 时再拦一次）。
- `_decision_tree` 里 `tradeable` 的二次排序用同一把键。
- `final_score` 保留为档内 tiebreak 和无数据时的排序依据。

## 与 book 上限 / 3% 组合计划风险上限的配合

- Book 上限（intraday 3 / swing 2 / total 5）**不动**，仍由
  `decide_and_act` / `_try_entry` 在进场时强制执行。排序只决定**先试谁**：
  `pick_intraday_signal` / `pick_swing_signal` 按
  session_primary → morning_primary → top_trades 的排名顺序取，每 tick 每本书
  最多 1 条新腿。“每本书只取前几名” = 排名即优先级，上限仍在下游截断。
- 3% 组合计划风险上限（`apply_portfolio_risk_cap`，进场时在 allocation 内执行）
  **不动**。逐标的方向可能让同一天出现日内 LONG + 摆动 SHORT，两条腿的计划风险
  在上限下**相加**（保守处理，和以前一样），不会因为对冲而互相抵消。
- 止损冷却仍按（日期，book，标的）执行，不动。
- Swing 仍由 `compute_swing_opportunity` 从 ranked 里挑一条；逐标的方向让它
  可以在日内偏多的日子里挑一条走弱标的的 SHORT（以前两者都被市场方向绑住）。

## 不变的东西

- Parts 结构 P1–P18、morning.json / step3.json 的字段形状（只新增字段）。
- 仓位参数与 `allocation.py`（冻结）、cron 定时、Alpaca mirror、go-live 门槛。
- 影子账本摄入：只新增字段，不改结果。`model_comparison.py`：对比的 parts 键不变。
- `research_is_fresh`、新闻门、熔断行为。

## 测试计划

- 同一天两个标的方向不同：TSLA 强（RS+、昨日 +6%）→ LONG；MU 弱（RS−、
  昨日 −3%）→ SHORT（或至少置信度显著不同）。
- 有 map 时排名：校准期望值高的行排在 `final_score` 更高的行前面；无 map 时
  排名与改动前一致（`final_score` 降序）。
- `test_trade_candidates_direction_from_bias` 按新语义更新（旧断言“所有行跟
  bias”即本次要废除的行为）。
- 全量测试必须全部通过。
