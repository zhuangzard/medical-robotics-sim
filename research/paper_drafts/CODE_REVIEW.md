# PhysRobot Code Review & Reproducibility Report

**审计人**: paper-code subagent  
**日期**: 2026-02-06  
**范围**: physics_core/, training/, environments/, baselines/, colab/  
**状态**: 🔴 存在多个严重问题需要修复

---

## 一、代码结构总览

```
medical-robotics-sim/
├── physics_core/           # 物理核心 (Edge Frame, GNN, Integrators)
│   ├── __init__.py         # Clean exports ✅
│   ├── edge_frame.py       # EdgeFrame + construct_edge_features + fully_connected_edges
│   ├── dynamical_gnn.py    # DynamicalGNN + PhysicsMessagePassing
│   ├── integrators.py      # SymplecticIntegrator + RK4Integrator
│   └── tests/              # 2 test files (edge_frame, conservation)
├── environments/
│   ├── push_box.py         # PushBoxEnv (MuJoCo, 10-dim obs)
│   ├── push_box_env.py     # PushBoxEnv (MuJoCo, 16-dim obs) ← 重复！
│   └── test_push_box.py    # 环境测试
├── baselines/
│   ├── ppo_baseline.py     # Pure PPO agent
│   ├── gns_baseline.py     # GNS + PPO agent
│   ├── physics_informed.py # PhysRobot agent (完整版)
│   └── simple_controller.py
├── training/
│   ├── train.py            # 完整训练管线
│   ├── eval.py             # OOD + conservation 评估
│   ├── train_ppo.py        # 独立 PPO 训练
│   ├── train_all.py        # 三方法顺序训练
│   └── config.yaml         # 超参数配置
├── colab/
│   ├── build_full_notebook.py
│   ├── week1_full_training_v2.ipynb
│   └── ...
└── scripts/
    ├── auto_commit.sh
    └── milestone_save.sh
```

---

## 二、严重 Bug 与问题 (Critical)

### 🔴 BUG-1: 两个 PushBoxEnv 定义冲突（观测空间不一致）

**文件**: `environments/push_box.py` vs `environments/push_box_env.py`

| 属性 | push_box.py | push_box_env.py |
|------|-------------|-----------------|
| 观测维度 | **10** (joint_pos, joint_vel, box_pos, box_vel, goal_pos) | **16** (+ ee_pos, box_pos 3D, box_vel 3D, goal_pos 3D) |
| make_push_box_env | 返回 `lambda` (工厂函数) | 返回 `PushBoxEnv` 实例 |
| 子步数 | 10 substeps | 5 substeps |
| 成功阈值 | 0.05m, 10步 | 0.1m, 立即 |
| Reward | distance + contact + control cost | 0.5*r1 + r2 + 100*success |

**影响**: 
- `baselines/ppo_baseline.py` 导入 `push_box_env`（16-dim）
- `training/train_ppo.py` 导入 `push_box`（10-dim）
- `environments/__init__.py` 导入 `push_box`（10-dim）
- eval.py 试图同时使用两者！

**修复**: 合并为单一实现，统一 16-dim 观测空间。10-dim 版本丢失了关键信息（end-effector 位置）。

### 🔴 BUG-2: `check_antisymmetry()` 逻辑错误 — 永远不可能通过

**文件**: `physics_core/edge_frame.py` L79-95

```python
def check_antisymmetry(self, positions, velocities, edge_index):
    e_ij = self(positions, velocities, edge_index)          # 通过 edge_encoder
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
    e_ji = self(positions, velocities, edge_index_rev)
    antisym_error = torch.max(torch.abs(e_ij + e_ji))
    return antisym_error.item()
```

**问题**: `edge_encoder` 是一个含 ReLU 的 MLP。对于非线性函数 f，`f(x) + f(-x) ≠ 0`。只有当输入是反对称时，输出通过非线性层后不会保持反对称性。

**数学证明**: 
- raw features: `r_ij = -r_ji` ✅（反对称）
- `||r_ij|| = ||r_ji||` ✅（对称，不是反对称）
- 通过 `ReLU(Linear(...))` 后，`f(r_ij, ||r_ij||) + f(r_ji, ||r_ji||) ≠ 0`

这是一个**概念性错误**：raw edge features 是反对称的，但编码后的 hidden features 不是。论文声称的 "antisymmetry by construction" 在当前实现中**不成立**。

**修复方案**:
1. 使用 odd activation (如 tanh 代替 ReLU) + 反对称权重矩阵
2. 或者将 encoder 改为: `f(e_ij) = g(e_ij) - g(e_ji)` (显式反对称化)
3. 或者在 DynamicalGNN 层面强制反对称（如 Dynami-CAL 原论文的方式）

### 🔴 BUG-3: PhysRobot Colab 版本失去所有物理约束

**文件**: `colab/build_full_notebook.py`（已在 PHYSROBOT_DIAGNOSIS.md 中详述）

Colab notebook 中的 PhysRobotFeaturesExtractor 被简化为纯 MLP：
```python
class PhysRobotFeaturesExtractor(BaseFeaturesExtractor):
    def forward(self, observations):
        return self.fusion(self.policy_stream(observations))  # 纯 MLP！
```

**影响**: 所有 Colab 实验结果不代表 PhysRobot 的真实性能。论文中引用的 "12.5x sample efficiency" 无法复现。

### 🔴 BUG-4: PhysRobot 训练步数硬编码为 16K（仅 2 次 PPO 迭代）

**文件**: `baselines/physics_informed.py` L290, `colab/build_full_notebook.py`

```python
'physrobot_timesteps': 16000  # 仅产生 2 次 PPO 更新
```

**修复**: 所有方法统一 200K timesteps 进行公平对比。

---

## 三、中等问题 (Medium)

### 🟡 ISSUE-5: `DynamicalGNN.compute_energy()` 中的 kinetic energy 计算错误

**文件**: `physics_core/dynamical_gnn.py` L129-133

```python
kinetic = 0.5 * masses.unsqueeze(-1) * (velocities ** 2).sum(dim=-1)
```

**问题**: `masses.unsqueeze(-1)` 是 `(N,1)`，而 `(velocities**2).sum(dim=-1)` 是 `(N,)`。虽然 broadcasting 会工作，但结果是 `(N,1)` 而非 `(N,)`。后续的 `kinetic.sum()` 结果正确，但中间形状不一致。

更重要的是：这个函数假设**重力势能** `PE = m*g*h`，但实际环境是 2D 桌面推箱子，z 坐标基本不变。应该使用**弹性势能**或者直接移除不适用的势能项。

### 🟡 ISSUE-6: GNS baseline 图构建不对称

**文件**: `baselines/gns_baseline.py` L126

```python
edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=obs.device)
```

只有单向边 (ee→box)，没有反向边。对于消息传递 GNN，这意味着 box 节点收不到来自 ee 的消息。应该用双向边：

```python
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=obs.device)
```

### 🟡 ISSUE-7: `PhysRobotAgent.evaluate()` 使用错误 API

**文件**: `baselines/physics_informed.py` L264-280

```python
obs, reward, terminated, truncated, info = env.step(action)  # Gymnasium API
```

但 `train.py` 中的 `DetailedTrackingCallback._evaluate()` 使用：
```python
obs, reward, done, info = self.eval_env.step(action)  # Old Gym API
```

**问题**: VecEnv 返回 `(obs, reward, dones, infos)` — 4 值。非 VecEnv 返回 `(obs, reward, terminated, truncated, info)` — 5 值。代码混用两种 API。

### 🟡 ISSUE-8: `_edge_frame()` 的退化情况未处理

**文件**: `baselines/physics_informed.py` L62-76

当两个节点在 z 轴方向对齐时，`cross(e1, [0,0,1])` 为零向量，导致 `e2 = 0/ε` → 数值不稳定。

**修复**: 添加 fallback 方向：
```python
if torch.norm(e2) < 1e-6:
    up = torch.tensor([1., 0., 0.])
    e2 = torch.cross(e1, up)
```

### 🟡 ISSUE-9: eval.py 中的 `collect_trajectory()` 硬编码观测索引

**文件**: `training/eval.py` L158-159

```python
box_pos = obs[0][7:10].copy()   # 假设 16-dim 观测
box_vel = obs[0][10:13].copy()
```

对 10-dim 环境（push_box.py），box_pos 在索引 4:6，box_vel 在 6:8。这会静默地提取错误数据。

---

## 四、代码风格与一致性问题 (Minor)

### 🟢 STYLE-1: `sys.path.append` 反模式
多处使用 `sys.path.append(...)` 而非 proper package setup。应创建 `setup.py` 或 `pyproject.toml`。

### 🟢 STYLE-2: 类型注解不一致
`physics_core/` 有类型注解 ✅，`baselines/` 和 `training/` 大部分没有 ❌。

### 🟢 STYLE-3: 缺少 docstring
`baselines/__init__.py` 有 exports 但部分函数缺少 docstring。

### 🟢 STYLE-4: 重复的环境工厂函数
`make_push_box_env` 在 `push_box.py` 和 `push_box_env.py` 中各有一个，签名不同。

### 🟢 STYLE-5: 硬编码常量
物理常量（g=9.81, dt=0.01）散布在多个文件中，应集中定义。

---

## 五、单元测试覆盖率审计

### 现有测试

| 文件 | 测试数 | 覆盖范围 | 状态 |
|------|--------|----------|------|
| `physics_core/tests/test_edge_frame.py` | 6 | EdgeFrame 反对称、平移不变、输出形状 | ⚠️ 反对称测试基于错误假设 |
| `physics_core/tests/test_conservation.py` | 5 | 能量/动量守恒、辛积分器比较 | ⚠️ 依赖随机初始化的 GNN |
| `environments/test_push_box.py` | 6 | 初始化、随机策略、质量变化、渲染 | ✅ 较完善 |
| `test_reward.py` (root) | ? | Reward 函数 | 未审查 |

### ❌ 缺失的单元测试

**physics_core 缺失测试**:
1. `test_dynamical_gnn.py` — DynamicalGNN 前向传播、梯度流、参数计数
2. `test_integrators.py` — SymplecticIntegrator/RK4Integrator 单独的正式 pytest
3. `test_batch_processing.py` — 批量图处理（当前标记为 placeholder）
4. `test_gradient_flow.py` — 确保梯度可以从 loss 流回所有参数
5. `test_edge_frame_antisymmetry_raw.py` — 测试 raw features 的反对称性（绕过 encoder）
6. `test_message_passing.py` — PhysicsMessagePassing 的消息聚合正确性
7. `test_fully_connected_edges.py` — 含/不含自环的边索引正确性（独立 fixture）

**baselines 缺失测试**:
8. `test_ppo_baseline.py` — Agent 创建、predict 输出形状、save/load 往返
9. `test_gns_baseline.py` — GNSNetwork 前向传播、obs_to_graph 正确性
10. `test_physics_informed.py` — DynamiCALGraphNet 前向传播、edge_frame 构建
11. `test_fusion_module.py` — FusionModule 维度正确性

**training 缺失测试**:
12. `test_train_config.py` — config.yaml 加载和验证
13. `test_eval_metrics.py` — momentum_drift / energy_drift 计算正确性

**environments 缺失测试**:
14. `test_push_box_env_16dim.py` — 16-dim 环境的同等测试
15. `test_env_consistency.py` — 两个环境实现之间的行为一致性

**集成测试**:
16. `test_end_to_end.py` — 从环境创建到训练 1 step 的完整流程
17. `test_colab_notebook.py` — Notebook 内代码可执行性验证

---

## 六、与论文算法描述的一致性审查

### 论文声称 vs 代码实现

| 论文 Section | 声称 | 代码实现 | 一致性 |
|-------------|------|---------|-------|
| §3.3 Theorem 1 | 反对称边框架保证动量守恒 | `edge_encoder` 使用 ReLU → **破坏反对称性** | ❌ 不一致 |
| §3.3 Physics Core | Scalarization → GNN → Vectorization | 实现中没有 scalarize/vectorize 步骤 | ❌ 不一致 |
| §3.4 Fusion | Cross-Attention (Q=vision, KV=physics) | 实际用的是 `concat → Linear → ReLU` | ❌ 不一致 |
| §4.1 Sample Efficiency | 12.5x (400 vs 5000 episodes) | PhysRobot 只训练 2 次迭代 | ❌ 无法验证 |
| §3.2 Architecture | Vision-Language + Physics 双流 | 无 Vision-Language 编码器 | ⚠️ 简化版 |
| §3.5 Physics Pre-training | Stage 1 离线物理预训练 | **完全没有实现** | ❌ 缺失 |
| §3.3 Edge Frame | `F_ij = f1*e1 + f2*e2 + f3*e3` | `baselines/physics_informed.py` 有此实现 ✅ | ✅ 一致 |
| §3.3 MessagePassing | Sum aggregation, antisymmetric | `PhysicsMessagePassing` aggr='add' ✅ | ✅ 一致 |
| §4.1 Integrator | Symplectic (Verlet) | `SymplecticIntegrator` 正确实现 | ✅ 一致 |

### 关键不一致总结

1. **反对称性保证**: 论文的核心理论贡献（Theorem 1）在代码中**未被正确实现**
2. **Fusion 模块**: 论文描述 cross-attention，代码实现为简单 concatenation + MLP
3. **训练流程**: 论文描述两阶段训练（Stage 1 physics pre-training + Stage 2 RL），代码仅实现 Stage 2
4. **Sample efficiency 证据**: 因训练步数设置错误，无法产生有效数据

---

## 七、代码结构重构建议

### 建议 1: 统一环境实现

```python
# environments/push_box.py — 保留为单一实现
class PushBoxEnv(gym.Env):
    """16-dim 观测空间版本（包含 ee_pos, 3D box）"""
    def __init__(self, obs_mode='full'):
        if obs_mode == 'full':
            self.obs_dim = 16  # 完整版
        elif obs_mode == 'compact':
            self.obs_dim = 10  # 兼容旧代码

# 删除 environments/push_box_env.py
```

### 建议 2: 正确实现反对称 EdgeFrame

```python
class AntisymmetricEdgeFrame(nn.Module):
    """保证 f(e_ij) = -f(e_ji) 的编码器"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.Tanh(),  # 奇函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
    
    def forward(self, positions, velocities, edge_index):
        raw_features = construct_edge_features(positions, velocities, edge_index)
        # 显式反对称化: f(x) = g(x) - g(-x)
        encoded = self.encoder(raw_features)
        encoded_neg = self.encoder(-raw_features)  # 反向边等价于 -features
        return (encoded - encoded_neg) / 2  # 保证反对称
```

### 建议 3: 实现 Cross-Attention Fusion

```python
class CrossAttentionFusion(nn.Module):
    """与论文 §3.4 一致的 cross-attention fusion"""
    
    def __init__(self, policy_dim, physics_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=policy_dim,
            num_heads=n_heads,
            kdim=physics_dim,
            vdim=physics_dim,
        )
    
    def forward(self, policy_features, physics_features):
        # Q = policy (what to do), K/V = physics (what's possible)
        fused, attn_weights = self.attn(
            query=policy_features.unsqueeze(0),
            key=physics_features.unsqueeze(0),
            value=physics_features.unsqueeze(0),
        )
        return fused.squeeze(0), attn_weights
```

### 建议 4: 添加 Stage 1 Physics Pre-training

```python
# training/pretrain_physics.py
def pretrain_physics_core(physics_core, trajectory_dataset, epochs=100):
    """
    Stage 1: 离线物理预训练
    Loss = MSE(predicted_next_state, actual_next_state) 
         + λ1 * momentum_violation 
         + λ2 * energy_violation
    """
    optimizer = torch.optim.Adam(physics_core.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for batch in trajectory_dataset:
            pred_acc = physics_core(batch.graph)
            # Position prediction via integration
            pred_pos = batch.pos + batch.vel * dt + 0.5 * pred_acc * dt**2
            loss_mse = F.mse_loss(pred_pos, batch.next_pos)
            loss_momentum = compute_momentum_violation(pred_acc, batch.masses)
            loss_energy = compute_energy_violation(...)
            loss = loss_mse + 0.1 * loss_momentum + 0.1 * loss_energy
            loss.backward()
            optimizer.step()
```

### 建议 5: 创建 proper Python package

```toml
# pyproject.toml
[project]
name = "physrobot"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "torch-geometric>=2.3",
    "gymnasium>=0.29",
    "mujoco>=3.0",
    "stable-baselines3>=2.0",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "black", "flake8"]
```

---

## 八、README 和文档改进建议

### 现有 README 问题
- 缺少安装步骤（特别是 MuJoCo + torch-geometric 的安装顺序）
- 缺少快速复现指南（"run experiment X to reproduce Table Y"）
- 缺少 Colab notebook 使用说明的链接

### 建议的 README 结构

```markdown
# PhysRobot: Physics-Informed Foundation Models for Robotic Manipulation

## Quick Start (Reproduce Paper Results)
```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run ALL experiments (Table 1 + Table 2)
python scripts/run_ablation.py --mode full

# 3. Generate figures
python scripts/generate_figures.py

# 4. Or use Colab
# Open colab/week1_full_training_v2.ipynb
```

## Project Structure
...

## Experiment Reproduction
### Table 1: Sample Efficiency
### Table 2: OOD Generalization
### Table 3: Ablation Study

## Citation
```

---

## 九、优先修复清单

| 优先级 | 编号 | 描述 | 工作量 |
|--------|------|------|--------|
| 🔴 P0 | BUG-1 | 合并两个 PushBoxEnv 实现 | 2h |
| 🔴 P0 | BUG-2 | 修复 EdgeFrame 反对称性实现 | 4h |
| 🔴 P0 | BUG-3 | 恢复 Colab notebook 中的完整物理核心 | 3h |
| 🔴 P0 | BUG-4 | 修复 PhysRobot 训练步数 | 10min |
| 🟡 P1 | ISSUE-5 | 修复 compute_energy 计算 | 30min |
| 🟡 P1 | ISSUE-6 | GNS baseline 添加双向边 | 15min |
| 🟡 P1 | ISSUE-7 | 统一 Gymnasium/VecEnv API 使用 | 1h |
| 🟡 P1 | ISSUE-8 | 处理 edge_frame 退化情况 | 30min |
| 🟡 P1 | ISSUE-9 | eval.py 观测索引适配 | 30min |
| 🟢 P2 | STYLE-1 | 创建 pyproject.toml | 30min |
| 🟢 P2 | STYLE-2-5 | 代码风格统一 | 2h |
| 🟢 P2 | 缺失测试 | 编写 17 个缺失的测试 | 8h |

**总计估算**: ~22 小时

---

## 十、结论

### 积极方面 ✅
1. `physics_core/` 架构设计清晰，模块化良好
2. `integrators.py` 的 Symplectic/RK4 实现正确且有 self-test
3. `baselines/physics_informed.py` 的 DynamiCAL 力分解实现与论文一致
4. `environments/push_box.py` 的 MuJoCo 集成完整，有 OOD 质量变化支持
5. 代码注释和 docstring 质量较高（特别是 physics_core）

### 需要改进 ❌
1. **核心理论实现与论文不一致**（反对称性、fusion、pre-training）
2. **环境定义冲突**导致不同训练/评估代码使用不同环境
3. **Colab notebook 简化过度**，实验结果不可复现
4. **训练配置错误**（16K steps）
5. **单元测试严重不足**（覆盖率 <30%）

### 建议：在提交论文前必须修复 P0 问题，强烈建议修复 P1 问题。
