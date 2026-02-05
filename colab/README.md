# Google Colab Training

**Purpose**: Run Week 1 training on Google Colab Pro GPU (V100/A100)

---

## Quick Start

### 1. Generate Notebook

```bash
cd medical-robotics-sim/colab
python3 generate_training_notebook.py
```

**Output**: `notebooks/week1_training_colab.ipynb` (10.8 KB, 8 cells)

### 2. Upload to Drive

1. Open https://drive.google.com
2. Upload `notebooks/week1_training_colab.ipynb`
3. Right-click → **Open with → Google Colaboratory**

### 3. Configure and Run

1. **Runtime → Change runtime type → GPU** (select V100)
2. **Runtime → Run all**
3. Wait ~8-10 hours

---

## Progress Monitoring

The notebook automatically tracks progress in your Drive:

**Progress file**: `/MyDrive/medical-robotics-progress/training_progress.json`

**Check status**:
```bash
# From MacBook (if Drive synced)
cat ~/Google\ Drive/medical-robotics-progress/training_progress.json
```

**Progress updates**:
```json
{
  "timestamp": "2026-02-05T14:45:00",
  "status": "training",
  "gpu": "Tesla V100-SXM2-16GB",
  "message": "Training in progress",
  "eta_hours": 8
}
```

Status values:
- `started` - Initialization complete
- `training` - Training in progress
- `complete` - Training finished
- `saved` - Results saved to Drive
- `error` - Something went wrong

---

## Results Location

After training completes, results auto-save to:

**Drive path**: `/MyDrive/medical-robotics-results/YYYYMMDD_HHMMSS/`

**Contents**:
- `results/` - Table 1, Figure 2, reports
- `models/` - Trained models (PPO, GNS, PhysRobot)
- `data/` - Experiment data
- `summary.json` - Training summary

---

## Expected Timeline

| Phase | Duration |
|-------|----------|
| Setup + Clone | 5 min |
| Dependencies | 10 min |
| **Training** | **8-10h** (V100) |
| Results Save | 5 min |
| **Total** | **~9h** |

---

## Features

✅ **Auto GPU Detection** - Configures batch size for T4/V100/A100  
✅ **Progress Tracking** - Real-time status updates to Drive  
✅ **Auto Checkpointing** - Saves every 100 episodes  
✅ **Error Handling** - Retries on OOM, logs errors  
✅ **Result Backup** - Auto-saves to Drive on completion  

---

## Troubleshooting

### No GPU detected
- Runtime → Change runtime type → GPU → Save

### Training interrupted
- Notebook auto-saves checkpoints
- Re-run will resume from last checkpoint

### Out of memory
- Notebook auto-retries with smaller batch
- Or select A100 GPU (Pro+ only)

---

**Generated**: 2026-02-05  
**Project**: https://github.com/zhuangzard/medical-robotics-sim
