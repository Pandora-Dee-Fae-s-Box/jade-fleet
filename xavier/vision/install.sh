#!/bin/sh
# Install the vision service on the Xavier (after export_models.py and build_engines.sh): sudo ./install.sh
set -eu
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
ls engines/*.engine >/dev/null 2>&1 || { echo "no engines yet: run ./build_engines.sh first"; exit 1; }
id jade >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin jade
usermod -aG video jade
install -d /opt/jade-vision /etc/jade-fleet
cp -r jade_vision.py engines models /opt/jade-vision/
[ -f /etc/jade-fleet/vision.env ] || install -m 644 jade-vision.env /etc/jade-fleet/vision.env
install -m 644 jade-vision.service /etc/systemd/system/jade-vision.service
systemctl daemon-reload
systemctl enable --now jade-vision
sleep 3
systemctl --no-pager status jade-vision | head -5
curl -s http://127.0.0.1:8090/v1/vision/models || true
