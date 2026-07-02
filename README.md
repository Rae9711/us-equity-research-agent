# Daily Trading OS

美股研究 **Daily Trading OS** — workflow 见 [WORKFLOW.md](WORKFLOW.md)，实现计划见 [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md)。

## Automation 构建（Cloud Agent）

推送到分支 **`automation/trading-os-v2-build`** 会触发 Cursor Automation，按 [docs/CURSOR-AUTOMATION-BUILD.md](docs/CURSOR-AUTOMATION-BUILD.md) 完成 Phase 5a–8 与 60 日验证。

```bash
git checkout automation/trading-os-v2-build
# ... 改动 ...
git push origin automation/trading-os-v2-build
```

---

## Phase 0 状态

- Docker：`trading_web` + `trading_runner`
- Web：`/health`、`/`、`/verify`（表单骨架）
- DB：SQLite + `conclusions` / `daily_runs` 表
- 部署：https://rae-trading.com

## 部署

见 **[deploy/DEPLOY.md](deploy/DEPLOY.md)**（DNS → Caddy → ship）

```bash
export HETZNER_HOST=root@YOUR_VPS_IP
./deploy/hetzner-ship.sh
```

## API 密钥

申请与 VPS 配置步骤：**[docs/API-SETUP.md](docs/API-SETUP.md)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=sqlite:///./data/trading_os.db
uvicorn web.main:app --reload --port 8020
curl http://127.0.0.1:8020/health
```
