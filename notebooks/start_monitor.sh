#!/bin/bash
# Start Colab training monitor in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/colab_monitor.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "🚀 Starting Colab training monitor..."
echo "📝 Log file: $LOG_FILE"
echo ""
echo "Monitor will:"
echo "  - Check progress every 10 minutes"
echo "  - Send updates to Telegram"
echo "  - Stop when training completes (or after 24h)"
echo ""
echo "To stop monitor:"
echo "  pkill -f monitor_colab_training.py"
echo ""

# Run in background
nohup python3 "$SCRIPT_DIR/monitor_colab_training.py" \
  --interval 600 \
  --max-hours 24 \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Monitor started (PID: $PID)"
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
