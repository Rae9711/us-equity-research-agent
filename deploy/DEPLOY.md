# 部署指南 — rae-trading.com

域名：**https://rae-trading.com**（Cloudflare DNS → VPS `5.161.58.191`）

与 Alfred / agenttech.com **无任何关系**，仅共用 VPS 物理机。

---

## 当前状态

| 项 | 状态 |
|----|------|
| DNS | `rae-trading.com` → `5.161.58.191`（Cloudflare） |
| Caddy | `rae-trading.com` → `127.0.0.1:8020` |
| HTTPS | Let's Encrypt 自动证书 |
| 应用 | `/opt/trading-os/` Docker Compose |

验证：

```bash
curl -sS https://rae-trading.com/health
```

---

## DNS（Cloudflare，已配置）

若需修改，登录 [Cloudflare Dashboard](https://dash.cloudflare.com) → **rae-trading.com** → **DNS**：

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | `@` | `5.161.58.191` | DNS only（灰云） |
| A | `www` | `5.161.58.191` | 灰云（可选） |

```bash
dig +short rae-trading.com
# 5.161.58.191
```

---

## Caddy（VPS）

配置在 `/etc/caddy/Caddyfile`（与 `alfredaitech.com` 并列）：

```caddy
rae-trading.com {
	encode gzip
	reverse_proxy 127.0.0.1:8020
}
```

重载：

```bash
ssh root@5.161.58.191
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
```

可选 Basic Auth：见仓库 `deploy/Caddyfile.snippet`。

---

## 部署 / 更新代码

本机：

```bash
cd ~/Projects/us-equity-research-agent
export HETZNER_HOST=root@5.161.58.191
./deploy/hetzner-ship.sh
```

VPS 环境变量 `/opt/trading-os/.env`：

见 **[docs/API-SETUP.md](../docs/API-SETUP.md)**（申请链接、测试命令、安全说明）

```text
ANTHROPIC_API_KEY=sk-ant-...
POLYGON_API_KEY=...
FRED_API_KEY=...          # 可选但推荐；无 key 时宏观轮询降级（rate-limited）
BLS_API_KEY=              # 可选；NFP/失业率 BLS 兜底
TZ=America/New_York
APP_BASE_URL=https://rae-trading.com
```

---

## 访问

- 首页：https://rae-trading.com/
- 健康检查：https://rae-trading.com/health
- 结论核对：https://rae-trading.com/verify

---

## 故障排查

| 现象 | 处理 |
|------|------|
| `dig` 无解析 | Cloudflare 检查 A 记录、灰云 |
| HTTPS 证书失败 | 确认 80/443 开放；DNS 灰云 |
| 502 | `docker ps` 看 `trading_web`；`curl 127.0.0.1:8020/health` |

---

*Phase 0 完成 · 下一步 Phase 1：数据 collectors*
