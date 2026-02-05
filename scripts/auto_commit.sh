#!/bin/bash
# Auto-commit and push script for medical-robotics-sim
# Usage: ./scripts/auto_commit.sh "commit message" [tag]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get commit message
COMMIT_MSG="${1:-Auto-save: $(date '+%Y-%m-%d %H:%M:%S')}"
TAG="${2:-}"

echo -e "${YELLOW}📝 Medical Robotics Sim - Auto Commit${NC}"
echo "======================================"
echo ""

# Check if there are changes
if [[ -z $(git status -s) ]]; then
    echo -e "${GREEN}✅ No changes to commit${NC}"
    exit 0
fi

# Show what will be committed
echo -e "${YELLOW}📊 Changes to commit:${NC}"
git status -s
echo ""

# Add all changes
echo -e "${YELLOW}➕ Adding changes...${NC}"
git add -A

# Commit
echo -e "${YELLOW}💾 Committing...${NC}"
git commit -m "$COMMIT_MSG"

# Push
echo -e "${YELLOW}🚀 Pushing to GitHub...${NC}"
git push origin main

# Create tag if provided
if [[ -n "$TAG" ]]; then
    echo -e "${YELLOW}🏷️  Creating tag: $TAG${NC}"
    git tag -a "$TAG" -m "$COMMIT_MSG"
    git push origin "$TAG"
fi

echo ""
echo -e "${GREEN}✅ Successfully saved to GitHub!${NC}"
echo -e "   Repository: https://github.com/zhuangzard/medical-robotics-sim"
echo -e "   Commit: $COMMIT_MSG"
if [[ -n "$TAG" ]]; then
    echo -e "   Tag: $TAG"
fi
echo ""
