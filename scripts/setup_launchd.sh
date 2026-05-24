#!/usr/bin/env bash
# Install (or reload) the launchd job for daily pipeline runs.
# Usage: bash scripts/setup_launchd.sh [uninstall]

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.ai-bubble-index.daily.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.ai-bubble-index.daily.plist"
LABEL="com.ai-bubble-index.daily"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
  rm -f "$PLIST_DEST"
  echo "Uninstalled $LABEL"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"

# Stop existing job if running
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

cp "$PLIST_SRC" "$PLIST_DEST"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"

echo "✅ Installed: $LABEL"
echo "   실행 시각: 평일 06:30 KST"
echo "   로그:      $(dirname "$PLIST_SRC")/../logs/launchd.log"
echo ""
echo "수동 즉시 실행: launchctl kickstart -k gui/\$(id -u)/$LABEL"
echo "제거:          bash scripts/setup_launchd.sh uninstall"
