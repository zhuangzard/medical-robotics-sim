#!/bin/bash
# Week 1 Complete Setup and Execution Script
# Run this to install dependencies and execute experiments

set -e  # Exit on error

echo "======================================================================"
echo "🔬 Week 1: PushBox Experiments - Setup & Execution"
echo "======================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project root
cd ~/.openclaw/workspace/medical-robotics-sim

# Step 1: Check Python
echo -e "\n${YELLOW}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 not found!${NC}"
    exit 1
fi

# Step 2: Create virtual environment (optional but recommended)
echo -e "\n${YELLOW}[2/5] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo -e "\n${YELLOW}[3/5] Installing dependencies...${NC}"
echo "This may take 5-10 minutes..."

pip install --upgrade pip

# Install PyTorch (CPU version for Mac)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install other dependencies
echo "Installing other packages..."
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 4: Validate setup
echo -e "\n${YELLOW}[4/5] Validating setup...${NC}"
python3 experiments/week1_push_box/quick_test.py

# Step 5: Ask about training
echo -e "\n${YELLOW}[5/5] Ready to train!${NC}"
echo ""
echo "Choose training mode:"
echo "  1) Quick test (10 minutes, 10K/5K/2K steps)"
echo "  2) Full training (8-12 hours, 200K/80K/16K steps)"
echo "  3) Skip for now"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}🚀 Starting quick test training...${NC}"
        python3 training/train.py \
            --ppo-steps 10000 \
            --gns-steps 5000 \
            --physrobot-steps 2000 \
            --n-envs 2
        ;;
    2)
        echo -e "\n${GREEN}🚀 Starting full training...${NC}"
        echo "This will take 8-12 hours. Consider using screen/tmux!"
        python3 training/train.py \
            --ppo-steps 200000 \
            --gns-steps 80000 \
            --physrobot-steps 16000 \
            --n-envs 4
        ;;
    3)
        echo -e "\n${YELLOW}Skipping training. Run manually with:${NC}"
        echo "  python3 training/train.py"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# If training completed, run evaluation
if [ $choice -eq 1 ] || [ $choice -eq 2 ]; then
    echo -e "\n${GREEN}📊 Training complete! Running evaluation...${NC}"
    
    # OOD test
    python3 training/eval.py --ood-test
    
    # Conservation validation
    python3 training/eval.py --validate-physics
    
    # Generate figures and report
    python3 experiments/week1_push_box/analyze_results.py
    
    echo -e "\n${GREEN}✅ All experiments complete!${NC}"
    echo ""
    echo "Results available at:"
    echo "  📄 Report: results/WEEK1_FINAL_REPORT.md"
    echo "  📊 Figure 2: results/figures/ood_generalization.png"
    echo "  📋 Table 1: results/tables/sample_efficiency.md"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}✨ Week 1 Setup Complete!${NC}"
echo "======================================================================"
