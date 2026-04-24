#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/webapp/frontend/public/config.js"

cat > "${CONFIG_PATH}" <<EOF
window.__TAIKO_CONFIG__ = {
  apiBaseUrl: "${API_BASE_URL}"
};
EOF

echo "Updated frontend runtime API URL:"
echo "  ${API_BASE_URL}"
echo "Config file: ${CONFIG_PATH}"
