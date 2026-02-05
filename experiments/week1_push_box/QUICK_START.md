# Week 1 Quick Start Guide

**⏱️ 30 seconds to understand | 5 minutes to start training**

---

## 🚀 Fastest Path to Results

```bash
cd ~/.openclaw/workspace/medical-robotics-sim
./experiments/week1_push_box/setup_and_run.sh
```

**That's it!** The script will:
1. Install dependencies
2. Validate setup
3. Ask if you want quick test (10 min) or full training (8-12 hours)
4. Run everything and generate paper figures

---

## 📋 What Gets Generated

After training completes, you'll have:

```
results/
├── WEEK1_FINAL_REPORT.md           ← Comprehensive report
├── figures/
│   ├── ood_generalization.png      ← Figure 2 for paper (300 DPI)
│   └── conservation_validation.png ← Supplementary figure
└── tables/
    ├── sample_efficiency.md        ← Table 1 (Markdown)
    └── sample_efficiency.tex       ← Table 1 (LaTeX)

data/
├── week1_training_results.json     ← Raw training data
├── ood_generalization.json         ← OOD test results
└── conservation_validation.json    ← Physics validation

models/
├── pure_ppo_final.zip              ← Trained Pure PPO
├── gns_final.zip                   ← Trained GNS
└── physrobot_final.zip             ← Trained PhysRobot (our method)
```

---

## 🎯 Three Commands, Three Results

### Table 1: Sample Efficiency
```bash
python3 training/train.py
```
→ Trains all three methods  
→ Generates `data/week1_training_results.json`

### Figure 2: OOD Generalization
```bash
python3 training/eval.py --ood-test
```
→ Tests on different box masses  
→ Generates `data/ood_generalization.json`

### Final Report & Figures
```bash
python3 experiments/week1_push_box/analyze_results.py
```
→ Creates all paper-ready outputs  
→ Saves to `results/`

---

## ⚡ Quick Test (10 minutes)

Want to verify everything works before full training?

```bash
python3 training/train.py \
  --ppo-steps 10000 \
  --gns-steps 5000 \
  --physrobot-steps 2000 \
  --n-envs 2
```

This won't match paper results, but confirms the pipeline works!

---

## 🔧 One-Line Install

```bash
pip install torch stable-baselines3 gymnasium mujoco torch-geometric matplotlib pandas seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## 📊 Expected Timeline

| Stage | Duration | Can Run Overnight? |
|-------|----------|-------------------|
| Setup & Install | 5-10 min | No |
| Quick Test | 10-15 min | No |
| Full Training | 8-12 hours | ✅ Yes |
| Evaluation | 30 min | No |
| Analysis | 5 min | No |

**Total**: ~9-13 hours (mostly unattended training)

---

## 🎓 What You're Running

### Pure PPO (Baseline)
Standard reinforcement learning. No physics knowledge.

### GNS (Baseline)
Graph neural networks that learn physics, but don't enforce conservation laws.

### PhysRobot (Our Method)
Hybrid approach: PPO policy + Dynami-CAL physics core.  
**Key innovation**: Antisymmetric edge frames guarantee momentum conservation.

---

## 📈 Expected Results

After full training, PhysRobot should achieve:

- ✅ **12.5x sample efficiency** vs Pure PPO (400 episodes vs 5000)
- ✅ **>95% success rate** on out-of-distribution box masses
- ✅ **<0.1% momentum conservation error**

---

## 🆘 If Something Breaks

### Error: "mujoco not found"
```bash
pip install mujoco>=3.0.0
```

### Error: "torch_geometric not found"
```bash
pip install torch-geometric
```

### Error: Training too slow
Reduce parallel environments:
```bash
python3 training/train.py --n-envs 1
```

### Error: Out of memory
Use smaller batch size (edit `baselines/*.py`, set `batch_size=32`)

---

## 📁 File Guide

**Need to modify?**

- **Environment** → `environments/push_box_env.py`
- **Pure PPO** → `baselines/ppo_baseline.py`
- **GNS** → `baselines/gns_baseline.py`
- **PhysRobot** → `baselines/physics_informed.py`
- **Training** → `training/train.py`
- **Evaluation** → `training/eval.py`
- **Visualization** → `experiments/week1_push_box/analyze_results.py`

---

## 🎬 The Full Pipeline (Manual)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Validate (optional but recommended)
python3 experiments/week1_push_box/quick_test.py

# 3. Train
python3 training/train.py

# 4. Evaluate
python3 training/eval.py --ood-test
python3 training/eval.py --validate-physics

# 5. Generate paper outputs
python3 experiments/week1_push_box/analyze_results.py

# 6. View results
open results/WEEK1_FINAL_REPORT.md
open results/figures/ood_generalization.png
```

---

## ✅ Success Criteria

Training succeeded if:

1. All three methods finish training without errors
2. Models saved to `models/` directory
3. `results/WEEK1_FINAL_REPORT.md` shows:
   - PhysRobot episodes to success < 500
   - PhysRobot improvement > 10x
   - OOD success rate > 80%
   - Conservation error < 0.1%

---

## 🚀 Ready?

**Automated** (recommended):
```bash
./experiments/week1_push_box/setup_and_run.sh
```

**Manual** (step-by-step):
```bash
pip install -r requirements.txt
python3 training/train.py
python3 training/eval.py --ood-test --validate-physics
python3 experiments/week1_push_box/analyze_results.py
```

**Quick Test** (verify first):
```bash
python3 experiments/week1_push_box/quick_test.py
```

---

**Questions?** See `README.md` or `IMPLEMENTATION_COMPLETE.md`

**Let's generate some paper results! 🎉**
