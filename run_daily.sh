#!/usr/bin/env bash
# Daily pipeline: collect → score → validate
# Logs to logs/YYYY-MM-DD.log

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python"
LOG_DIR="$PROJECT/logs"
TODAY="$(date '+%Y-%m-%d')"
LOG="$LOG_DIR/$TODAY.log"

mkdir -p "$LOG_DIR"

# Append all output to today's log file and also print to stdout
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo " AI Bubble Index — daily run $TODAY"
echo "========================================"

cd "$PROJECT"

echo "[1/3] Collecting data..."
"$PYTHON" -m src.run_collect

echo "[2/3] Scoring..."
"$PYTHON" -m src.run_score

echo "[3/3] Validating..."
"$PYTHON" -m src.run_validate

echo "========================================"
echo " Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Remove logs older than 30 days
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
