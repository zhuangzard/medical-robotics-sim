# PhysRobot 训练时间极短 — 诊断报告

**日期**: 2026-02-06  
**调查人**: Sub-agent (physrobot-diagnosis)  
**状态**: 🔴 发现明确根因

---

## 1. 现象复述

| 方法 | timesteps | 训练时间 | 迭代次数 |
|------|-----------|----------|----------|
| Pure PPO | 200,000 | 4.2 min | ~97 |
| GNS | 80,000 | 1.7 min | ~39 |
| **PhysRobot** | **16,000** | **0.4 min** | **2** |

PhysRobot 日志：`iterations: 2, explained_variance: 0.00077, value_loss: 8.19`

---

## 2. 根因分析

### 🔴 根因 #1（主因）：`physrobot_timesteps` 硬编码为 16,000 — 远低于有效训练量

**文件**: `colab/build_full_notebook.py` — CONFIG 字典（约第 240 行生成的 cell）

```python
CONFIG = {
    'ppo_timesteps': 200000,
    'gns_timesteps': 80000,
    'physrobot_timesteps': 16000,   # ← 问题所在！仅 16K
    'n_envs': 4,
    'box_mass': 0.5,
    'eval_episodes': 50
}
```

**影响**：
- 使用 4 个并行环境、`n_steps=2048`，每次 rollout 收集 `4 × 2048 = 8,192` 步
- 16,000 总步 ÷ 8,192 步/迭代 = **1.95 → 仅 2 次 PPO 更新**
- PPO 至少需要 ~50 次迭代才能看到有意义的学习信号
- `explained_variance: 0.00077` 证实 value function 几乎没学到任何东西

**同时**, `baselines/physics_informed.py` 的 `main()` 函数中默认值也是 16,000：

```python
# baselines/physics_informed.py 约第 290 行
parser.add_argument('--total-timesteps', type=int, default=16000,
                    help='Total timesteps for training (default: 16000)')
```

对比其他两个 baseline 的默认值：
- `baselines/ppo_baseline.py`: `default=200000` ✓
- `baselines/gns_baseline.py`: `default=80000` ✓

**结论**: 16K 是一个调试级别的值，不是正式训练的配置。这是整个问题的直接原因。

---

### 🟡 根因 #2（次要）：Colab Notebook 中的 PhysRobotFeaturesExtractor 被简化为纯 MLP，失去了物理约束

**文件**: `colab/build_full_notebook.py` — agents_code 字符串中的 PhysRobotFeaturesExtractor（约第 190-200 行）

**实际在 Colab 运行的代码**（生成进 notebook 的简化版本）：

```python
class PhysRobotFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.policy_stream = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, features_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())
    
    def forward(self, observations):
        policy_features = self.policy_stream(observations)
        return self.fusion(policy_features)
```

**对比 `baselines/physics_informed.py` 中的完整版本**：
- ✅ 完整版有 `PhysicsCore`（DynamiCAL GraphNet）做物理预测
- ✅ 完整版有 `_obs_to_graph()` 把 16-dim observation 转为 PyG 图
- ✅ 完整版把 physics 预测和 policy stream 融合
- ❌ **Colab 简化版完全没有物理核心，只是一个普通 MLP**

这意味着 Colab 中的 "PhysRobot" 本质上和 GNS 一样只是个带自定义 feature extractor 的 PPO，**没有任何物理约束/归纳偏置**。

---

### 🟡 根因 #3（次要）：Colab Notebook 中的 GNSFeaturesExtractor 也被过度简化

```python
class GNSFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_proj = nn.Sequential(nn.Linear(16, features_dim), nn.ReLU())
    
    def forward(self, observations):
        return self.feature_proj(observations)
```

这只是一个 `Linear(16, 128) → ReLU`，**完全没有图网络**。三个方法在 Colab 中实质上都是不同大小的 MLP + PPO，唯一的差异就是 timestep 数量。

---

## 3. 需要修改的具体代码行

### 修改 1：提高 PhysRobot 训练步数（最关键）

**文件**: `colab/build_full_notebook.py`  
**位置**: CONFIG 字典生成的 cell（搜索 `'physrobot_timesteps': 16000`）

```python
# 修改前
'physrobot_timesteps': 16000,

# 修改后（推荐）
'physrobot_timesteps': 200000,
```

**同时修改** `baselines/physics_informed.py` 第 290 行：

```python
# 修改前
parser.add_argument('--total-timesteps', type=int, default=16000,

# 修改后
parser.add_argument('--total-timesteps', type=int, default=200000,
```

### 修改 2：恢复 PhysRobotFeaturesExtractor 的物理核心（重要但可选）

**文件**: `colab/build_full_notebook.py` — `agents_code` 字符串  
**建议**: 把完整的 `PhysRobotFeaturesExtractor`（包括 `PhysicsCore` + `_obs_to_graph()`）从 `baselines/physics_informed.py` 移植到 notebook 内联代码中。

这需要确保 `torch_geometric` 在 Colab 中正确安装，并将 `DynamiCALGraphNet`, `PhysicsCore` 类也内联。

### 修改 3（可选）：恢复 GNS 的图网络

类似地，Colab notebook 中的 GNSFeaturesExtractor 应该使用实际的 `GNSNetwork` + `_obs_to_graph()`。

---

## 4. 推荐的训练步数

| 方法 | 推荐 timesteps | 理由 |
|------|----------------|------|
| Pure PPO | 200,000 | 当前设置合理，约 97 次迭代 |
| GNS | 200,000 | 应与 PPO 相同以公平对比 |
| PhysRobot | 200,000 | **必须**与其他方法相同 |

**为什么要相同的 timesteps？**

- 论文中的核心主张是"**sample efficiency**" — 即 PhysRobot 在相同 timestep 数下学到更好的 policy
- 如果 PhysRobot 用更少的 timestep 就能达到相同的 success rate，那才是真正的 sample efficiency
- **不应该**通过给不同方法不同的训练时间来"制造"差异

**如果想展示 sample efficiency**：
- 所有方法训练 200K timesteps
- 比较在 20K, 40K, 80K, 200K 各个检查点的 success rate
- PhysRobot 应在较少的 timestep 就达到高 success rate → 这是真正的 sample efficiency 证据

---

## 5. 完整版 PhysRobotFeaturesExtractor 的潜在问题

虽然当前 Colab 中的简化版没有这个问题，但如果恢复完整版，需要注意：

### 5.1 `_obs_to_graph()` 中的循环效率

```python
for i in range(batch_size):   # 逐样本循环 → batch 大时很慢
    o = obs[i]
    ...
    graphs.append(graph)
return Batch.from_data_list(graphs)
```

**风险**: `batch_size=64` 时需要创建 64 个 PyG Data 对象再 batch，这在 GPU 上效率不高。  
**建议**: 用批量 tensor 操作代替 Python 循环。

### 5.2 `_edge_frame()` 的退化情况

```python
up = torch.tensor([0., 0., 1.], device=e1.device).unsqueeze(0)
e2 = torch.cross(e1, up.expand_as(e1))
```

当 `e1` 方向接近 `[0, 0, 1]` 时（即两个节点在垂直方向对齐），`cross(e1, up)` 会接近零向量，导致 `e2` 不稳定，可能造成梯度爆炸或 NaN。

### 5.3 整体网络太深/太大

完整版 PhysRobot 的参数量远大于 PPO baseline：
- 3 层 DynamiCALGraphNet（每层含 scalar_mlp + vector_mlp + node_update）
- PhysicsCore 的 encoder + decoder
- policy_stream (3 层 MLP)
- fusion layer

对于仅 2 个节点（end-effector + box）的简单图来说，这个网络**严重过参数化**。

---

## 6. 总结与优先级

| 优先级 | 修改 | 影响 |
|--------|------|------|
| 🔴 P0 | `physrobot_timesteps: 16000 → 200000` | 直接解决"训练时间极短"问题 |
| 🟡 P1 | 恢复 PhysRobot 物理核心到 Colab notebook | 让实验真正测试物理约束的价值 |
| 🟡 P1 | 恢复 GNS 图网络到 Colab notebook | 公平对比 |
| 🟢 P2 | 三个方法都用 200K timesteps 公平对比 | 科学严谨性 |
| 🟢 P2 | 添加中间检查点评估（learning curve） | 展示 sample efficiency |
| ⚪ P3 | 优化 `_obs_to_graph()` 的批量处理 | 性能优化 |

---

## 7. 快速验证方案

修改后，预期 PhysRobot 训练应该：
- 200K timesteps → 约 97 次 PPO 迭代（与 Pure PPO 相同）
- 训练时间: 约 4-6 min（因为 PhysRobot 有更复杂的 feature extractor，可能比 PPO 稍慢）
- `explained_variance` 应逐渐从 0 升到 0.3-0.8
- `value_loss` 应逐渐下降
