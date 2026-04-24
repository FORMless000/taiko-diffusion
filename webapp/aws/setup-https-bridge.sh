#!/usr/bin/env bash
set -euo pipefail

PUBLIC_HOST="${PUBLIC_HOST:-ec2-18-117-249-161.us-east-2.compute.amazonaws.com}"
FRPS_PORT="${FRPS_PORT:-7000}"
BACKEND_PROXY_PORT="${BACKEND_PROXY_PORT:-12205}"
CADDYFILE_PATH="${CADDYFILE_PATH:-/etc/caddy/Caddyfile}"
MARKER_BEGIN="# BEGIN taiko-diffusion https bridge"
MARKER_END="# END taiko-diffusion https bridge"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script as root with sudo."
    exit 1
  fi
}

ensure_commands() {
  local missing=()
  for cmd in apt-get curl gpg python3 ss systemctl; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      missing+=("${cmd}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing required commands: ${missing[*]}"
    exit 1
  fi
}

discover_frps_service() {
  mapfile -t FRPS_UNITS < <(systemctl list-units --type=service --all --no-legend | awk '{print $1}' | grep -E '^frps.*\.service$' || true)
  if (( ${#FRPS_UNITS[@]} > 0 )); then
    FRPS_SERVICE="${FRPS_UNITS[0]}"
    if ! systemctl is-active --quiet "${FRPS_SERVICE}"; then
      echo "Detected ${FRPS_SERVICE}, but it is not active."
      exit 1
    fi
    echo "Detected active FRP server service: ${FRPS_SERVICE}"
    return
  fi

  if pgrep -af '(^|/)frps($| )' >/dev/null; then
    echo "Detected frps as a running process."
    pgrep -af '(^|/)frps($| )'
    return
  fi

  echo "Could not find frps as either a systemd service or a running process."
  exit 1
}

port_listener_summary() {
  local port="$1"
  ss -ltnp "( sport = :${port} )" 2>/dev/null | tail -n +2 || true
}

ensure_port_safe() {
  local port="$1"
  local allowed_pattern="${2:-}"
  local listeners
  listeners="$(port_listener_summary "${port}")"
  if [[ -z "${listeners}" ]]; then
    echo "Port ${port} is free."
    return
  fi
  echo "Port ${port} listeners:"
  echo "${listeners}"
  if [[ -n "${allowed_pattern}" ]] && echo "${listeners}" | grep -Eq "${allowed_pattern}"; then
    echo "Port ${port} is already owned by an allowed process."
    return
  fi
  echo "Port ${port} is already in use by another service. Refusing to continue."
  exit 1
}

install_caddy() {
  if command -v caddy >/dev/null 2>&1; then
    echo "Caddy already installed."
    return
  fi

  apt-get update
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /etc/apt/keyrings/caddy-stable-archive-keyring.gpg
  echo "deb [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update
  apt-get install -y caddy
}

write_caddyfile() {
  mkdir -p "$(dirname "${CADDYFILE_PATH}")"
  local block
  block="$(cat <<EOF
${MARKER_BEGIN}
${PUBLIC_HOST} {
    encode zstd gzip
    reverse_proxy 127.0.0.1:${BACKEND_PROXY_PORT}
}
${MARKER_END}
EOF
)"

  if [[ -f "${CADDYFILE_PATH}" ]] && grep -Fq "${MARKER_BEGIN}" "${CADDYFILE_PATH}"; then
    python3 - <<'PY' "${CADDYFILE_PATH}" "${MARKER_BEGIN}" "${MARKER_END}" "${block}"
from pathlib import Path
import sys

path = Path(sys.argv[1])
begin = sys.argv[2]
end = sys.argv[3]
block = sys.argv[4]
text = path.read_text(encoding="utf-8")
start = text.index(begin)
finish = text.index(end) + len(end)
new_text = text[:start].rstrip() + "\n" + block + "\n"
if finish < len(text):
    remainder = text[finish:].lstrip("\n")
    if remainder:
        new_text += "\n" + remainder
path.write_text(new_text, encoding="utf-8")
PY
  elif [[ -f "${CADDYFILE_PATH}" && -s "${CADDYFILE_PATH}" ]]; then
    cp "${CADDYFILE_PATH}" "${CADDYFILE_PATH}.bak.$(date +%Y%m%d%H%M%S)"
    printf "\n%s\n" "${block}" >> "${CADDYFILE_PATH}"
  else
    printf "%s\n" "${block}" > "${CADDYFILE_PATH}"
  fi
}

configure_firewall() {
  if ! command -v ufw >/dev/null 2>&1; then
    echo "ufw not installed; skipping firewall automation."
    return
  fi
  if ! ufw status | grep -q "Status: active"; then
    echo "ufw not active; skipping firewall automation."
    return
  fi

  ufw allow 443/tcp
  ufw allow "${FRPS_PORT}/tcp"
  ufw deny "${BACKEND_PROXY_PORT}/tcp" || true
}

restart_caddy() {
  systemctl enable caddy
  systemctl restart caddy
  systemctl --no-pager --full status caddy
}

verify_backend_reachable() {
  if ! curl -fsS "http://127.0.0.1:${BACKEND_PROXY_PORT}/api/models" >/dev/null; then
    echo "Could not reach the FRP-published backend at http://127.0.0.1:${BACKEND_PROXY_PORT}/api/models"
    echo "Make sure your local backend and frpc are running before rerunning this script."
    exit 1
  fi
}

print_next_steps() {
  cat <<EOF

HTTPS bridge is configured.

Verification commands:
  curl -v https://${PUBLIC_HOST}/api/models
  sudo journalctl -u caddy -n 50 --no-pager

Frontend runtime config should point to:
  https://${PUBLIC_HOST}

Make sure DNS for ${PUBLIC_HOST} resolves to this EC2 instance and that ports 80/443 are open in the AWS security group.
EOF
}

main() {
  require_root
  ensure_commands
  discover_frps_service
  ensure_port_safe 443 "(caddy)"
  ensure_port_safe 80 "(caddy)"
  install_caddy
  verify_backend_reachable
  write_caddyfile
  configure_firewall
  restart_caddy
  print_next_steps
}

main "$@"
