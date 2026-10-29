#!/bin/sh
# Install the Jade Fleet broker on Debian (the OptiPlex). Run from the repository: sudo deploy/install.sh
#   --binary PATH   use a jade-fleet binary built elsewhere instead of building here
set -eu
cd "$(dirname "$0")/.."
BIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --binary) BIN="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

echo "== packages"
apt-get update -qq
apt-get install -y -qq build-essential pkg-config curl ca-certificates git python3 python3-pip python3-venv nodejs npm >/dev/null

if [ -z "$BIN" ]; then
  echo "== building (Rust)"
  SUDO_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
  CARGO="$SUDO_HOME/.cargo/bin/cargo"
  if [ ! -x "$CARGO" ]; then
    su - "${SUDO_USER:-root}" -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
  fi
  su "${SUDO_USER:-root}" -c "cd '$PWD' && '$CARGO' build --release -p jade-fleet"
  BIN=target/release/jade-fleet
fi
install -m 755 "$BIN" /usr/local/bin/jade-fleet

echo "== user, folders, settings"
id jade >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/jade-fleet --shell /bin/bash jade
install -d -o jade -g jade -m 750 /var/lib/jade-fleet /var/lib/jade-fleet/vault
install -d -m 755 /etc/jade-fleet /usr/share/jade-fleet/web
[ -f /etc/jade-fleet/fleet.env ] || install -m 640 -g jade deploy/fleet.env /etc/jade-fleet/fleet.env
if [ -d web ] && [ -n "$(ls -A web 2>/dev/null)" ]; then cp -r web/. /usr/share/jade-fleet/web/; fi

install -m 644 deploy/jade-fleet.service /etc/systemd/system/jade-fleet.service
systemctl daemon-reload
systemctl enable --now jade-fleet
sleep 1
echo
echo "Jade Fleet is running: $(systemctl is-active jade-fleet)"
echo "Token for phones and browsers: $(sudo -u jade env $(grep -v '^#' /etc/jade-fleet/fleet.env | xargs) /usr/local/bin/jade-fleet token)"
echo "Settings: /etc/jade-fleet/fleet.env (set the Xavier's address there)   Logs: journalctl -u jade-fleet -f"
