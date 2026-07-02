#!/usr/bin/env bash
# Ship Daily Trading OS to the VPS.
#
#   export HETZNER_HOST=root@YOUR_VPS_IP
#   ./deploy/hetzner-ship.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${HETZNER_HOST:?Set HETZNER_HOST=root@your-vps-ip}"

REMOTE_DIR="${TRADING_OS_REMOTE_DIR:-/opt/trading-os/repo}"
ENV_FILE="${TRADING_OS_ENV_FILE:-/opt/trading-os/.env}"
TAG="${TRADING_OS_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"

ARCHIVE="/tmp/trading-os-src.tar.gz"
echo "→ packaging $TAG"
if git rev-parse HEAD >/dev/null 2>&1; then
  git archive --format=tar.gz -o "$ARCHIVE" HEAD
else
  tar -czf "$ARCHIVE" \
    --exclude='.git' --exclude='.venv' --exclude='data/trading_os.db' \
    -C "$ROOT" .
fi

echo "→ uploading to $HETZNER_HOST"
ssh "$HETZNER_HOST" "mkdir -p '$REMOTE_DIR' '$(dirname "$ENV_FILE")'"
scp "$ARCHIVE" "$HETZNER_HOST:/tmp/trading-os-src.tar.gz"
rm -f "$ARCHIVE"

ssh "$HETZNER_HOST" "set -euo pipefail
REMOTE_DIR='$REMOTE_DIR'
ENV_FILE='$ENV_FILE'
TAG='$TAG'

mkdir -p \"\$(dirname \"\$REMOTE_DIR\")\"
rm -rf \"\$REMOTE_DIR\"
mkdir -p \"\$REMOTE_DIR\"
tar -xzf /tmp/trading-os-src.tar.gz -C \"\$REMOTE_DIR\"
rm -f /tmp/trading-os-src.tar.gz

if [[ ! -f \"\$ENV_FILE\" ]]; then
  cp \"\$REMOTE_DIR/.env.example\" \"\$ENV_FILE\"
  echo 'APP_BASE_URL=https://rae-trading.com' >> \"\$ENV_FILE\"
  chmod 600 \"\$ENV_FILE\"
  echo '⚠ Created '\$ENV_FILE' — edit API keys before relying on jobs'
fi

cd \"\$REMOTE_DIR\"
export TRADING_OS_TAG=\"\$TAG\"
docker compose -p trading-os -f deploy/docker-compose.yml build
docker compose -p trading-os -f deploy/docker-compose.yml up -d

echo ''
docker compose -p trading-os -f deploy/docker-compose.yml ps
echo ''
curl -fsS http://127.0.0.1:8020/health && echo ''
echo '✓ deployed trading-os @' \"\$TAG\"
"

echo "✓ ship complete — configure Caddy if not done: deploy/DEPLOY.md"
