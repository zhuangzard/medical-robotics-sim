# Week 1: PushBox 实验

**目标**: 验证 PhysRobot 的样本效率和 OOD 泛化能力

---

## 📁 文件结构

```
week1_push_box/
├── notebooks/
│   └── train_colab.ipynb     ✅ Colab 训练 notebook
├── analyze_results.py         分析实验结果
├── quick_test.py              快速测试
├── setup_and_run.sh           完整训练脚本
├── results/                   实验结果（训练后生成）
└── README.md                  本文件
```

---

## 🚀 运行训练

### 方案 A: Colab 训练（推荐）⭐

**一键打开**:
```
https://colab.research.google.com/github/zhuangzard/medical-robotics-sim/blob/main/experiments/week1_push_box/notebooks/train_colab.ipynb
```

**步骤**:
1. Runtime → Change runtime type → GPU (V100)
2. Runtime → Run all
3. 等待 8-10 小时

**优势**:
- ✅ 免费 V100/A100 GPU
- ✅ 可以关机
- ✅ 自动保存到 Drive
- ✅ 无需本地环境

---

### 方案 B: 本地训练

**前提**: 需要安装 conda 环境

```bash
cd medical-robotics-sim

# 1. 创建环境
conda env create -f environment.yml
conda activate physics-robot

# 2. 快速测试（10 分钟）
cd experiments/week1_push_box
python quick_test.py

# 3. 完整训练（8-12 小时）
bash setup_and_run.sh
```

---

## 🧪 测试代码

### Level 1: 单元测试（30 秒）

```bash
# 测试核心模块
cd medical-robotics-sim
pytest physics_core/tests/ -v

# 预期: 所有测试通过
# EdgeFrame antisymmetry < 1e-5
# Conservation errors < 0.1%
```

### Level 2: 环境测试（2 分钟）

```bash
# 测试 PushBox 环境
python environments/test_push_box.py

# 预期: 6/6 tests passed
# - Environment initialization
# - Random policy
# - Mass variation (OOD)
# - Rendering
# - Data collection
# - Success condition
```

### Level 3: 快速训练（10 分钟）

```bash
cd experiments/week1_push_box
python quick_test.py

# 预期:
# - 训练 10 episodes
# - 验证数据流
# - 生成简单报告
```

---

## 📊 预期结果

### Table 1: Sample Efficiency Comparison

| Method | Episodes to Success | Improvement |
|--------|---------------------|-------------|
| Pure PPO | ~5000 | 1.0x |
| GNS | ~2000 | 2.5x |
| **PhysRobot** | **~400** | **12.5x** ✅ |

### Figure 2: OOD Generalization

- X 轴: Box mass (0.5x → 2.0x)
- Y 轴: Success rate
- PhysRobot: >95% across all masses
- Pure PPO: Drops to ~40% at 2.0x

---

## 📁 结果位置

### Colab 训练

**Drive 路径**:
```
/MyDrive/medical-robotics-results/YYYYMMDD_HHMMSS/
├── results/
│   ├── tables/
│   │   ├── sample_efficiency.md
│   │   └── sample_efficiency.tex
│   ├── figures/
│   │   ├── ood_generalization.png
│   │   └── conservation_validation.png
│   └── WEEK1_FINAL_REPORT.md
├── models/
│   ├── pure_ppo_final.zip
│   ├── gns_final.zip
│   └── physrobot_final.zip
├── data/
└── summary.json
```

### 本地训练

**项目路径**:
```
medical-robotics-sim/experiments/week1_push_box/results/
```

---

## 🎓 学习要点

### 从 `physics_core/` 学到:
- 反对称 EdgeFrame 如何保证动量守恒
- GNN 在物理系统中的应用
- Symplectic 积分器 vs 普通积分器

### 从 `environments/` 学到:
- MuJoCo 物理引擎使用
- Gymnasium 环境设计
- OOD 测试方法

### 从 `training/` 学到:
- PPO 训练流程
- Baseline 对比实验设计
- 论文数据生成

---

## 🐛 常见问题

### Q: Colab "mount failed" 错误

**A**: Notebook 已修复，会自动处理:
- 检测 Drive 是否已挂载
- 失败时使用本地存储
- 不影响训练

### Q: 本地训练 OOM

**A**: 减小 batch size:
```bash
# 修改 training/config.yaml
batch_size: 32  # 改为 16 或 8
```

### Q: 测试失败

**A**: 检查依赖:
```bash
conda activate physics-robot
pip install -r requirements.txt
```

---

**创建时间**: 2026-02-05  
**预计训练时间**: 8-10 小时 (Colab Pro V100)  
**目标会议**: ICRA 2027 / CoRL 2026
