#!/bin/bash
# Milestone save script - for significant checkpoints
# Usage: ./scripts/milestone_save.sh "milestone name" "description"

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MILESTONE="${1:-}"
DESCRIPTION="${2:-Milestone checkpoint}"

if [[ -z "$MILESTONE" ]]; then
    echo "❌ Error: Milestone name required"
    echo "Usage: ./scripts/milestone_save.sh \"milestone-name\" \"description\""
    exit 1
fi

# Create milestone tag (e.g., "week1-day1-complete")
TAG="milestone-$(echo "$MILESTONE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"

# Generate commit message
COMMIT_MSG="🎯 Milestone: $MILESTONE

$DESCRIPTION

Time: $(date '+%Y-%m-%d %H:%M:%S %Z')
"

echo "🎯 Saving Milestone: $MILESTONE"
echo "================================"
echo ""
echo "Description: $DESCRIPTION"
echo "Tag: $TAG"
echo ""

# Use auto_commit script
"$PROJECT_ROOT/scripts/auto_commit.sh" "$COMMIT_MSG" "$TAG"

# Log milestone
MILESTONE_LOG="$PROJECT_ROOT/MILESTONES.md"
if [[ ! -f "$MILESTONE_LOG" ]]; then
    cat > "$MILESTONE_LOG" <<EOF
# Project Milestones

This file tracks major milestones in the medical robotics simulation project.

---

EOF
fi

cat >> "$MILESTONE_LOG" <<EOF
## $(date '+%Y-%m-%d %H:%M') - $MILESTONE

**Tag**: \`$TAG\`

$DESCRIPTION

---

EOF

echo ""
echo "✅ Milestone saved!"
echo "   View: https://github.com/zhuangzard/medical-robotics-sim/releases/tag/$TAG"
