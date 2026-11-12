# Alpaca 模拟盘接入指南（Paper Trading）

目的：把 Daily Trading OS 的信号放到 Alpaca 模拟盘上跑真实的 bracket
挂单（入场 + 止损 + 止盈），验证实盘执行（真实成交、滑点、部分成交），
再考虑任何真钱。**只用 paper 端点，绝不动真钱。**

适配器：`src/broker/alpaca_client.py`。官方文档：
- 下单与 bracket：https://docs.alpaca.markets/us/docs/orders-at-alpaca
- 订单操作示例：https://docs.alpaca.markets/us/docs/working-with-orders

## 1. 创建 Alpaca paper 账户（你自己做）

1. 打开 https://app.alpaca.markets/paper/dashboard/overview ，用邮箱注册 paper 账户（免费，模拟资金）。
2. 进 paper dashboard → API Keys，生成一对 **paper** API key。
3. 注意：paper key 和 live key 不能混用，live key 调 paper 端点会报 401。

## 2. 把 key 放进 .env（不要贴到聊天里）

在 `~/workspace/trading-os/.env` 里加两行（文件权限 600）：

```text
APCA_API_KEY_ID=PKxxxxxxxxxxxxxxxx
APCA_API_SECRET_KEY=xxxxxxxxxxxxxxxx
```

Key 只活在 `.env` 和进程环境变量里，适配器不写盘、不打日志。

## 3. 验证连通性（先 dry-run，再真读）

每次运行前先加载环境（和交易任务同一套）：

```bash
set -a; . ~/workspace/trading-os/.env; set +a
export DATA_ROOT=~/workspace/trading-os/state
export no_proxy="localhost,127.0.0.1,198.19.0.1,198.19.0.2"
export TMPDIR=~/workspace/.tmp
```

不联网的 dry-run（只验证信号到 bracket 的映射）：

```bash
~/workspace/.venvs/trading/bin/python -m src.broker.alpaca_client dry-run \
  --signal '{"symbol":"AMD","direction":"LONG","shares":2,"stop":493.4,"target":643.27,"horizon":"Intraday"}'
```

真读 paper 账户（只读，不下单），同时刷新状态文件：

```bash
~/workspace/.venvs/trading/bin/python -m src.broker.alpaca_client sync
```

成功时打印 `connected: true` 和账户/持仓/未结订单；失败时退出码为 1，
并在状态文件里写清原因。

## 4. 状态文件：日报的唯一接口

`~/workspace/trading-os/state/paper/alpaca_status.json`，每次适配器运行
（`sync` / `account` / `positions` / `orders`）后都会刷新。结构：

```json
{
  "last_sync": "2026-11-12T10:33:02-05:00",
  "connected": true,
  "reason": null,
  "account": {"equity": 10194.6, "cash": 9000.0, "buying_power": 18000.0, "status": "ACTIVE"},
  "positions": [{"symbol": "AMD", "side": "long", "direction": "LONG", "qty": 2.0, "avg_entry_price": 557.495, "unrealized_pl": 9.76}],
  "open_orders": [{"id": "order-123", "symbol": "AMD", "side": "buy", "direction": "LONG", "type": "market", "qty": 2.0, "status": "accepted", "order_class": "bracket"}]
}
```

- 没有 key 或连不通时，文件**仍然存在**，`connected` 为 `false`，
  `reason` 写清原因（`no_keys` / `auth_failed` / `request_failed: HTTP ...`），
  数字留空（`account` 为 `null`，列表为空），绝不编造。
- 每天 17:24 的日报只读这个文件。读不到文件、或 `connected` 为 `false`，
  日报直接报“Alpaca 未接入”并附 `reason`。
- 日报跑之前先跑一次 `sync`（由日报流程调用，不由本适配器定时）。

## 5. 信号到 bracket 的映射

`build_bracket_order(signal)` / `submit_bracket_order(signal)`：

- `symbol`、`direction`（LONG→buy，SHORT→sell）、`shares`
- 主单是市价单；`take_profit.limit_price = target`，
  `stop_loss.stop_price = stop`（纯止损单，不带 limit，保证触发后成交）
- TIF：短线（Intraday）→ `day`，长线（Swing）→ `gtc`
- 校验：LONG 要求 stop < entry < target，SHORT 要求 target < entry < stop，
  否则拒绝下单；Alpaca 还要求 bracket 的 TP 在 SL 之上（买入方向）
- `client_order_id` 默认 `tos-<symbol>-<随机>`，也可由调用方传入

## 6. 安全与限制

- 适配器只认 `https://paper-api.alpaca.markets`，传 live 端点会直接拒绝。
- Alpaca paper 不模拟：市场冲击、延迟滑点、排队位置、价格改善、规费、分红。
  所以 paper 成交仍比实盘乐观，结论要打折。
- 限速：每账户每分钟 200 次请求；`sync` 一次发 3 个请求（账户/持仓/订单）。
- 下单是真实写操作（paper 环境内），只在确认信号无误后调用
  `submit_bracket_order`；日常验证用 `dry-run` 和 `sync` 就够了。

## 7. 排错

- `401`：key 错了，或把 live key 用在了 paper 端点。重新生成 paper key。
- `403`：可能是 wash-trade 拒绝（自成交风险），检查是否有反向挂单。
- `connected: false, reason: no_keys`：`.env` 里没有那两行，或运行前没 source `.env`。
- 状态文件缺失：跑一次 `sync` 就会生成（有 key 时连通，无 key 时写 `no_keys`）。
