# API 密钥申请与配置

Trading OS 需要三个 API Key，全部写在 VPS 的 **`/opt/trading-os/.env`** 里（不要提交到 git）。

---

## 总览

| 变量 | 用途 | 费用 | 必填？ |
|------|------|------|--------|
| `ANTHROPIC_API_KEY` | LLM 解读、Driver、Morning Research 叙事 | 按量付费 | **是**（Phase 2+） |
| `POLYGON_API_KEY` | 行情、盘中、期权链、新闻 | 免费档有限；期权建议付费 | **是**（期权/盘中） |
| `FRED_API_KEY` | 2Y/10Y 国债等宏观利率 | **免费** | **是** |

Phase 0 只跑 `/health` 不需要 Key；**Phase 1 起**需要 FRED + Polygon；**Phase 2 起**需要 Anthropic。

---

## 1. Anthropic（Claude）

### 申请

1. 打开 [https://console.anthropic.com](https://console.anthropic.com)
2. 注册 / 登录
3. 左侧 **API Keys** → **Create Key**
4. 复制密钥（形如 `sk-ant-api03-...`），**只显示一次**

### 计费

- 按 token 计费；Morning Research 粗估 **$0.5–2/天**（视 Part 数量与模型）
- 需绑定支付方式（Settings → Billing）

### 写入 `.env`

```text
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxx
```

### 测试（本机或 VPS）

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
```

返回 JSON 含 `content` 即成功（不要用 `echo` 打印完整 key 到终端历史）。

---

## 2. FRED（美联储经济数据）

### 申请

1. 打开 [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. 登录或注册（免费）
3. **Request API Key** → 邮件或页面获得 32 位 key

### 费用

完全免费，有速率限制（够用）。

### 写入 `.env`

```text
FRED_API_KEY=your32characterfredapikeyhere
```

### 测试

**不要把文档里的 `你的FRED_KEY` 原样粘贴进命令** — 必须换成真实 Key，或用下面从 `.env` 读取的方式。

```bash
# 在 VPS 上：从 .env 加载（勿把 key 贴在命令里）
set -a && source /opt/trading-os/.env && set +a

# FRED — 10年期国债（key 应为 32 位小写字母数字）
curl -sS "https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=${FRED_API_KEY}&file_type=json&limit=1&sort_order=desc" | head -c 200
echo ""

# Polygon — SPY 昨收
curl -sS "https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=${POLYGON_API_KEY}" | head -c 300
echo ""
```

成功时 FRED 返回含 `"observations"` 的 JSON；Polygon 返回 `"status":"OK"`。

---

## 3. Polygon（行情 + 期权 + 新闻）

> Polygon 已 rebranded 为 Massive，API Key 通用：[https://massive.com](https://massive.com)

### 申请

1. 打开 [https://polygon.io](https://polygon.io) 或 [https://massive.com](https://massive.com)
2. 注册账号
3. Dashboard → **API Keys** → 复制 Key

### 选哪个套餐？

| 需求 | 建议 |
|------|------|
| 仅测通、日线 | **Stocks Basic（免费）** 可先试 |
| **期权链 / IV**（Part 9） | 需 **Options** 权限，通常 **Starter $29/月** 起 |
| 盘中实时（Step 2–6） | Starter 或更高 |
| 新闻 | 含在 Stocks 套餐内（`/v2/reference/news`） | **主新闻源**，无需单独 News API |

**新闻说明**：Step 0 / Morning Research 以 **Polygon News** 为主（按 SPY/QQQ/Mag7 等 ticker + 全市场 broad feed）。Bloomberg/WSJ RSS 仅作补充；Reuters 官方 RSS 已不可用，**不需要**额外申请 Reuters API。

### 测试新闻

```bash
set -a && source /opt/trading-os/.env && set +a
curl -sS "https://api.polygon.io/v2/reference/news?limit=3&order=desc&apiKey=${POLYGON_API_KEY}" | head -c 400
```

免费档限制：5 次/分钟、期权数据可能不可用——Phase 1 可先拉日线，Phase 4 前建议升级。

### 写入 `.env`

```text
POLYGON_API_KEY=your_polygon_key_here
```

### 测试

```bash
set -a && source /opt/trading-os/.env && set +a
curl -sS "https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=${POLYGON_API_KEY}"
```

返回 `"status":"OK"` 或含 `results` 即成功。若 `Unknown API Key`：检查 `.env` 里是否填了真实 Polygon Key（不是占位文字）。

---

## 在 VPS 上配置（最终步骤）

### 1. SSH 登录

```bash
ssh root@5.161.58.191
```

### 2. 编辑环境文件

```bash
nano /opt/trading-os/.env
```

完整示例：

```text
ANTHROPIC_API_KEY=sk-ant-api03-你的密钥
ANTHROPIC_MODEL=claude-sonnet-5
POLYGON_API_KEY=你的Polygon密钥
FRED_API_KEY=你的FRED密钥
TZ=America/New_York
APP_BASE_URL=https://rae-trading.com
```

保存：`Ctrl+O` → Enter → `Ctrl+X`

### 3. 权限（仅 root 可读）

```bash
chmod 600 /opt/trading-os/.env
```

### 4. 重启容器使配置生效

```bash
cd /opt/trading-os/repo
docker compose -p trading-os -f deploy/docker-compose.yml up -d
```

### 5. 确认容器读到了变量（不打印 key 值）

```bash
docker exec trading_web printenv | grep -E '^(ANTHROPIC|POLYGON|FRED)_API_KEY=' | sed 's/=.*/=***/'
```

应看到三行 `***` 占位，说明已注入。

---

## 安全注意

- **不要**把 `.env` 提交到 GitHub
- **不要**在聊天 / 截图里发完整 Key
- Key 泄露 → 到对应控制台 **立即 Revoke** 并新建
- VPS 上只保留一份 `/opt/trading-os/.env`

---

## 申请顺序建议

```text
1. FRED（5 分钟，免费）     → 先测宏观利率
2. Polygon（注册 + 选套餐） → 行情 / 期权 / 新闻
3. Anthropic（绑卡）        → Phase 2 Morning LLM 再用
```

---

## 常见问题

**Q：三个都要一次配齐吗？**  
Phase 0 不用。按 Phase 逐步加：Phase 1 要 FRED + Polygon；Phase 2 再加 Anthropic。

**Q：Polygon 免费够用吗？**  
日线 + 新闻试跑可以；**期权 Part 9 和盘中实时** 建议付费档。

**Q：新闻能用 Polygon 吗？要改 API 吗？**  
可以，且**已默认使用 Polygon News**。只需现有 `POLYGON_API_KEY`，无需单独新闻 API。在 `config/symbols.yaml` 的 `news.polygon_tickers` 可调整抓取标的。

**Q：本机开发怎么配？**  
复制 `.env.example` 为项目根目录 `.env`（已在 `.gitignore`），本地 `docker compose` 或 `uvicorn` 同样读取。

---

*配置完成后：`docker exec trading_runner python -m src.jobs.step0_collect` 与 `step1_morning` 测试。*
