#!/usr/bin/env bash
set -euo pipefail

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTEND_ROOT="${REPO_ROOT}/webapp/frontend"
CONFIG_PATH="${FRONTEND_ROOT}/public/config.js"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

cat > "${CONFIG_PATH}" <<EOF
window.__TAIKO_CONFIG__ = {
  apiBaseUrl: "${BACKEND_URL}"
};
EOF

pushd "${FRONTEND_ROOT}" >/dev/null
npm run build
popd >/dev/null

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "${REPO_ROOT}"
  export TAIKO_WEBAPP_ALLOW_ORIGINS="*"
  python -m uvicorn webapp.backend.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

(
  cd "${REPO_ROOT}"
  python -m http.server "${FRONTEND_PORT}" --bind "${FRONTEND_HOST}" --directory "webapp/frontend/out"
) &
FRONTEND_PID=$!

echo "Backend:  ${BACKEND_URL}"
echo "Frontend: ${FRONTEND_URL}"
echo "Press Ctrl+C to stop both services."

wait "${BACKEND_PID}" "${FRONTEND_PID}"
