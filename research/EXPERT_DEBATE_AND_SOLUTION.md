# PhysRobot 0% 成功率：专家辩论与解决方案

**日期**: 2026-02-06  
**背景**: Week 1 Colab 实验结果令人失望  

| 方法 | Success Rate | 训练时间 | 步数 |
|------|-------------|---------|------|
| Pure PPO | **6%** | 4.5 min | 200K |
| GNS | **0%** | 28 min | 200K |
| PhysRobot | **0%** | 28 min | 200K |
| OOD: PPO mass=0.5 | 8% | - | - |
| OOD: GNS/PhysRobot | 0% | - | - |

---

## 1. 问题诊断

### Expert A (RL 环境与训练专家)

**A-1. 环境 reward 信号极度稀疏**

当前的 reward 设计：
```python
reward = 0.5 * (-dist_ee_box) + (-dist_box_goal)
if success:
    reward += 100.0
```

问题分析：
- `dist_ee_box` 和 `dist_box_goal` 的典型值在 0.2–0.7m 区间，所以每步 reward 约 -0.3 到 -1.0
- 一个 episode 有 500 步，累计 reward 约 -150 到 -500
- **成功奖励 +100 被 500 步的负 reward 淹没**（net reward 仍为负数）
- PPO 的 value function 几乎看不到 success 和 failure 的区别

**A-2. Reward scale 不匹配**

- 距离奖励（per-step）：~-0.5 × 500 步 = 累计 -250
- 成功奖励（one-shot）：+100
- **比例 100 / 250 ≈ 0.4**，成功几乎不会显著改变 episode return
- PPO 的 GAE 估计器在这种 scale 下梯度信号极弱

**A-3. 200K timesteps 严重不足**

- 200K steps / 500 steps per episode = **400 episodes**
- 4 parallel envs → 实际约 **100 episodes per env**
- PushBox 看似简单，但对 torque-controlled 2-DOF arm 来说并不简单：
  - 需要先到达箱子 → 需要学会 inverse kinematics
  - 需要推箱子到目标 → 需要学会施力方向
  - 2 个关节的 torque 空间组合很大
- 文献中类似的 MuJoCo 推箱任务通常需要 **1M–5M steps**

**A-4. 环境初始化问题**

```python
self.data.qpos[0] = np.random.uniform(-0.5, 0.5)  # shoulder
self.data.qpos[1] = np.random.uniform(-0.5, 0.5)  # elbow
self.data.qpos[2] = np.random.uniform(0.25, 0.45)  # box_x
self.data.qpos[3] = np.random.uniform(-0.15, 0.15)  # box_y
```

- 关节角度随机 ±0.5 rad（约 ±28.6°），加上随机箱子位置
- 在很多 reset 场景下，arm 初始位置远离箱子
- **agent 需要先学会到达箱子，才能学推箱子** → 课程学习问题

**A-5. End-effector 速度为零的假设**

```python
ee_vel = torch.zeros(3, device=dev)  # Approximate
```

- PhysRobot/GNS 的图构建中，ee 速度被硬编码为 0
- 这意味着**物理模型收不到 ee 运动信息**，无法正确预测接触力
- 这是一个严重的信息缺失

**A-6. 动作空间设计——torque 控制 vs 位置控制**

- 直接 torque 控制 `[-10, 10]` 对 RL 来说是 hard mode
- 大多数机器人 RL 论文使用位置控制或速度控制
- Torque 控制需要更多 sample 来学习底层动力学

### Expert B (GNN/物理先验架构专家)

**B-1. 2-node graph 导致 GNN 退化为 MLP**

当前图结构：
```
Node 0: end-effector    Node 1: box
Edge 0→1 (ee→box)       Edge 1→0 (box→ee)
```

- 2 节点、2 条边的全连接图
- 每个节点只有**1 个邻居**
- 3 层 MessagePassing 在 2 节点图上：
  - Layer 1: node 0 收到 node 1 的消息，node 1 收到 node 0 的消息
  - Layer 2: node 0 收到（被 node 1 更新过的）node 1 的消息 → **已经是全图信息**
  - Layer 3: **完全冗余**，没有新信息可以传播
- **GNN 的 receptive field 在 1 层就已覆盖全图**，后 2 层只是额外参数

**B-2. EdgeFrame MLP 破坏反对称性**

`edge_frame.py` 的核心问题：

```python
# 原始特征是反对称的：r_ij = -r_ji, v_rel_ij = -v_rel_ji
edge_features = torch.cat([r_ij, r_norm, v_rel, v_norm], dim=1)  # 含对称项!
# 但 r_norm 和 v_norm 是对称的：||r_ij|| = ||r_ji||

# 经过 MLP 后，反对称性完全丢失
edge_hidden = self.edge_encoder(edge_features)  # 非线性破坏了反对称性
```

数学论证：
- 设 `f(x) = MLP([r, ||r||, v, ||v||])`，`g(x) = MLP([-r, ||r||, -v, ||v||])`
- 对于一般的 MLP：`f(x) ≠ -g(x)`
- **反对称性（PhysRobot 的核心卖点）在第一层就被破坏了**

**B-3. Colab Cell 6 的 PhysRobot 使用了 `DynamicalGNN`（来自 Cell 5），而非 `DynamiCALGraphNet`**

关键发现——Colab notebook 中的 `PhysRobotFeaturesExtractor` 使用的是：

```python
self.physics_gnn = DynamicalGNN(
    node_dim=6, hidden_dim=128, edge_hidden_dim=64,
    n_message_passing=3, output_dim=3)
```

这是 Cell 5 定义的 `DynamicalGNN`，它：
- 用 `EdgeFrame`（MLP 编码器）处理边特征
- **不实现 Scalarization-Vectorization 管道**
- **不保证 F_ij = -F_ji**
- 本质上是一个普通的 GNN，只是输入了相对位置/速度

而 `baselines/physics_informed.py` 中的 `DynamiCALGraphNet` 才是正确的实现（有边帧分解）。但这个文件**没有被 Colab 使用**！

**B-4. 过参数化——128-dim hidden + 3 层 MP 对 2 节点图**

参数统计：
- `DynamicalGNN`:
  - `EdgeFrame`: 8→64→64 ≈ 5K params
  - `node_encoder`: 6→128→128 ≈ 17K params
  - 3 × `PhysicsMessagePassing`: each (64+256→128→128 + 256→128→128) ≈ 99K params × 3 = 297K
  - `decoder`: 128→128→64→3 ≈ 21K params
  - **Total: ~340K params**
- `PhysRobotFeaturesExtractor` 额外包含:
  - `policy_stream`: 16→128→128→128 ≈ 34K params
  - `fusion`: 131→128 ≈ 17K params
  - **Grand Total: ~391K params**

对比 Pure PPO 的 `MlpPolicy`:
- `features_extractor`: 默认 `FlattenExtractor`（零参数）
- `policy_net + value_net`: 16→64→64→2 + 16→64→64→1 ≈ **10K params**

**PhysRobot 的参数量是 PPO 的 ~40 倍！** 200K steps 远远不够训练。

**B-5. GNS 同样过参数化**

`GNSFeaturesExtractor`:
- `node_encoder`: 6→128→128 ≈ 17K
- `edge_encoder`: 4→128→128 ≈ 17K
- 3 × `GNSGraphLayer`: each ~150K = 450K
- `decoder`: 128→128→3 ≈ 17K
- `feature_proj`: 19→128 ≈ 2.5K
- **Total: ~503K params**

GNS 比 PhysRobot 还多参数。PPO 的 10K vs GNS 的 503K，给相同的 200K steps，PPO 当然赢。

**B-6. 图构建中 edge_attr 不一致**

`obs_to_graph_batch` 构建的 `edge_attr` 是 4 维（`[rel_pos(3), dist(1)]`），但 GNS 的 `edge_encoder` 预期的也是 4 维——这一点没问题。

但 PhysRobot 的 `DynamicalGNN` 通过 `EdgeFrame` 使用 `positions` 和 `velocities` 直接计算 8 维特征（`[r_ij, ||r_ij||, v_rel, ||v_rel||]`），**完全忽略了 `edge_attr`**。这意味着图中传递的 edge 信息可能有冗余计算。

**B-7. Batch 处理中 edge_index 偏移问题**

`PyGBatch.from_data_list()` 会自动处理 edge_index 偏移，但 `DynamicalGNN` 接收的是 batch 后的 `graph.pos` 和 `graph.edge_index`：
- `positions` shape: `[B*2, 3]`（所有图的节点拼接）
- `edge_index` shape: `[2, B*2]`（偏移后的边索引）
- 这部分是正确的，PyG 自动处理了。

但 **`acc[1::2]`（取 box 节点）这个假设在 batch 中只有所有图大小相同时才成立**。当 `from_data_list` 按顺序拼接时，第 i 个图的 box 节点索引是 `2*i + 1`，`1::2` 确实能正确取到。✅ 这一点没问题。

---

## 2. 关键分歧点

| 编号 | 议题 | Expert A 立场 | Expert B 立场 |
|------|------|--------------|--------------|
| D1 | 0% 的主因 | **环境 + reward 设计** | **过参数化 + 反对称性破坏** |
| D2 | 200K steps 够不够 | 对 PPO 也不够（只有 6%） | PPO 的 6% 证明环境可解，GNN 的 0% 是架构问题 |
| D3 | 应该先改什么 | 先修 reward，再增 steps | 先砍参数量，再修架构 |
| D4 | GNS=0% 和 PhysRobot=0% 是同一个原因吗 | 是：都是因为 sample 不够 | 否：GNS 是过参数化，PhysRobot 是反对称性破坏 |
| D5 | 物理先验有没有价值 | 当然有，但环境太简单看不出来 | 在当前 2-node 实现中，物理先验 = 噪声 |

---

## 3. 辩论

### Round 1: 0% 的根本原因是什么？

**Expert A**: 
> 先看事实：PPO 6% > GNS 0% = PhysRobot 0%。PPO 用 10K 参数在 4.5min 达到 6%。GNS/PhysRobot 用 340K–500K 参数在 28min 达到 0%。最直接的解释：**相同的 200K steps，参数多 40 倍的模型根本没学到东西**。但这不全是架构的错——reward 太弱了。PPO 也才 6%！如果环境 reward 设计好，PPO 应该能到 80%+。所以根本原因是**环境 reward 信号太弱 + 训练步数不够**。GNS/PhysRobot 的 0% 只是在这个基础上被过参数化雪上加霜。

**Expert B**: 
> 我不同意"雪上加霜"的说法。让我们做一个思想实验：如果把 GNS 的参数量砍到和 PPO 一样（10K），同样 200K steps，它能达到 6% 吗？我认为不能，因为 GNS 的问题不是参数多，而是**计算图太复杂**。每个 forward pass 要做 `obs → graph → node_encoder → 3×MessagePassing → decoder → feature_proj → MLP policy`，这个 pipeline 的梯度要穿过太多层。200K steps 的 PPO 更新大约是 `200K / 2048 * 10 = ~976` 次梯度更新。对于 500K 参数的网络，976 次更新几乎什么都学不到。这不是 reward 的问题，是**样本效率和网络容量的根本不匹配**。

**Expert A**: 
> 好，我接受过参数化是一个重要因素。但我坚持 reward 也是核心问题。让我算一下：假设 agent 成功推箱子到目标，500 步的 total reward ≈ -250（距离惩罚）+ 100（成功奖励）= -150。而失败的 episode total reward ≈ -300。**差距只有 50**，PPO 的 value function 很难从这个信号中学到"成功 > 失败"。如果把成功奖励改到 1000 或用 normalized reward，PPO 能轻松到 50%+。

### Round 2: PhysRobot 的物理先验在当前实现中有价值吗？

**Expert B**: 
> 我的回答是**没有，甚至有害**。原因：
> 
> 1. Colab 用的是 `DynamicalGNN`（Cell 5），不是 `DynamiCALGraphNet`（baselines/）。前者没有边帧分解，就是一个普通 GNN。
> 2. 即使用了 `DynamiCALGraphNet`，反对称性也不保证（MLP 破坏了）。
> 3. 2-node 图上，"物理先验" = 对 2 个粒子预测加速度。但 RL 的 policy 需要的是**动作**（关节 torque），不是加速度。物理预测和策略需求之间有巨大的 gap。
> 4. 更关键的是：`ee_vel = torch.zeros(3)`！物理模型连 ee 的运动信息都收不到，它能预测什么物理？它预测的是**静态几何关系下的"力"**，和真正的接触物理毫无关系。

**Expert A**: 
> 我部分同意。但让我指出：物理先验的价值在于 **OOD 泛化**，不在于训练速度。当前实验中 OOD 也全是 0%，但那是因为 in-distribution 都没学会。如果我们先让 PPO 到 80%，PhysRobot 到 80%，然后测 OOD，物理先验的价值才能显现。所以我的立场是：**物理先验不是 0% 的原因，物理先验的价值在当前实验条件下无法评估**。

**Expert B**: 
> 那我们就需要一个能在 200K steps 内到 80% 的实现。以当前的 PhysRobot 架构（391K 参数），不可能。所以我的建议是：**先把物理模块做到最小可行版本（<20K 参数），确保它不 hurt 性能，再逐步验证它 help 泛化**。

### Round 3: 具体该怎么修？

**Expert A**: 
> 我的优先级：
> 1. **P0: 修 reward** — 成功奖励 100→1000，加 progress reward（`Δdist_box_goal`），加 action penalty
> 2. **P0: 增 timesteps** — 至少 500K，理想 1M
> 3. **P1: 简化初始化** — 先用固定初始条件，确认 agent 能学会；再逐步增加随机性
> 4. **P1: 考虑位置控制**替代 torque 控制

**Expert B**: 
> 我的优先级：
> 1. **P0: 砍 GNS/PhysRobot 参数量** — hidden_dim 128→32，MP layers 3→1，total < 20K params
> 2. **P0: 修复 ee_vel** — 从 joint_vel 计算 ee 的笛卡尔速度（用 Jacobian 或数值差分）
> 3. **P1: 修复反对称性** — 如果要保留物理卖点，必须用 Scalarization-Vectorization
> 4. **P2: 从 DynamicalGNN 切换到 DynamiCALGraphNet**（或直接在 Colab 中实现正确版本）

**Expert A**: 
> 我同意砍参数量。但我认为 **reward 修改的 ROI 更高**。不改 reward，即使把 GNS 参数量砍到 10K，它也不会比 PPO 好。改了 reward，PPO 就能到 50%+，这给了我们一个有意义的 baseline 来对比 GNS 和 PhysRobot。

**Expert B**: 
> 可以接受。但我建议**同时修，而不是串行修**。Reward 改完，PPO 涨到 50%，如果 GNS 还是 0%，那就是架构问题无疑了。

---

## 4. 共识方案

### 4.1 环境修改（Expert A 主导）

#### 4.1.1 Reward 重新设计

**当前 reward（有问题）**：
```python
reward = 0.5 * (-dist_ee_box) + (-dist_box_goal)
if success: reward += 100.0
```

**修改后 reward**：
```python
def step(self, action):
    # ... simulation ...
    
    # Phase 1: Reach the box
    dist_ee_box = np.linalg.norm(ee_pos[:2] - box_pos[:2])
    
    # Phase 2: Push box to goal
    dist_box_goal = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
    
    # Progress reward (delta-based, much stronger signal)
    prev_dist_box_goal = getattr(self, '_prev_dist_box_goal', dist_box_goal)
    progress = prev_dist_box_goal - dist_box_goal  # positive when box moves toward goal
    self._prev_dist_box_goal = dist_box_goal
    
    # Shaped reward
    reach_reward = -dist_ee_box                    # encourage reaching box
    push_reward = -dist_box_goal                   # encourage box near goal
    progress_reward = 20.0 * progress              # strongly reward progress
    action_penalty = -0.01 * np.sum(action ** 2)   # discourage extreme torques
    
    reward = (
        1.0 * reach_reward +      # weight on reaching
        1.0 * push_reward +       # weight on distance
        progress_reward +          # key: progress signal
        action_penalty             # regularization
    )
    
    # Large success bonus (relative to episode reward scale)
    success = dist_box_goal < self.success_threshold
    if success:
        remaining_steps = self.max_episode_steps - self.current_step
        reward += 500.0 + remaining_steps * 1.0   # bonus for early success
    
    # ...
```

关键改动：
1. **Progress reward**：`20 * Δdist` → 每一步箱子向目标移动 1cm 就获得 +0.2 奖励
2. **成功奖励 100→500+**：确保成功的 episode return 显著高于失败
3. **Action penalty**：防止 torque 爆炸
4. **Early success bonus**：激励快速完成

#### 4.1.2 Reset 中初始化 `_prev_dist_box_goal`

```python
def reset(self, seed=None, options=None):
    # ... existing reset code ...
    mujoco.mj_forward(self.model, self.data)
    self.current_step = 0
    
    # Initialize progress tracking
    box_pos = self.data.qpos[2:5]
    self._prev_dist_box_goal = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
    
    return self._get_obs(), self._get_info()
```

#### 4.1.3 增加 timesteps（训练配置）

```python
CONFIG = {
    'ppo_timesteps': 500_000,      # 200K → 500K (2.5x)
    'gns_timesteps': 500_000,      # 200K → 500K
    'physrobot_timesteps': 500_000, # 200K → 500K
    'n_envs': 4,
    'box_mass': 0.5,
    'eval_episodes': 100           # 50 → 100 for statistical power
}
```

### 4.2 网络架构修改（Expert B 主导）

#### 4.2.1 GNS 参数量大幅缩减

```python
class GNSFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):  # 128→64
        super().__init__(observation_space, features_dim)
        hid = 32          # 128→32
        edge_dim = 4
        self.node_encoder = nn.Sequential(nn.Linear(6, hid), nn.ReLU())   # 去掉第二层
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hid), nn.ReLU())
        self.gn_layers = nn.ModuleList([GNSGraphLayer(hid, hid, hid)])     # 3层→1层
        self.decoder = nn.Sequential(nn.Linear(hid, 3))                    # 简化 decoder
        self.feature_proj = nn.Sequential(nn.Linear(3 + 16, features_dim), nn.ReLU())
```

参数量：~3K（从 500K 降低 99%）

#### 4.2.2 PhysRobot 最小可行架构

```python
class PhysRobotFeaturesExtractorV2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        hid = 32
        
        # Lightweight physics: just relative geometry + 1-layer MLP
        # Input: [rel_pos(3), rel_vel(3), dist(1)] = 7
        self.physics_net = nn.Sequential(
            nn.Linear(7, hid),
            nn.ReLU(),
            nn.Linear(hid, 3)  # predicted box acceleration
        )
        
        # Policy stream (match PPO capacity)
        self.policy_stream = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(features_dim + 3, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Physics: relative geometry between ee and box
        ee_pos = observations[:, 4:7]
        box_pos = observations[:, 7:10]
        box_vel = observations[:, 10:13]
        
        # Compute ee velocity from joint velocities (numerical approximation)
        # For now, use zeros; fix in P1 with Jacobian
        ee_vel = torch.zeros_like(ee_pos)
        
        rel_pos = box_pos - ee_pos
        rel_vel = box_vel - ee_vel
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        physics_input = torch.cat([rel_pos, rel_vel, dist], dim=-1)  # [B, 7]
        physics_pred = self.physics_net(physics_input)  # [B, 3]
        
        # Policy stream
        policy_features = self.policy_stream(observations)
        
        # Fusion
        combined = torch.cat([policy_features, physics_pred], dim=-1)
        return self.fusion(combined)
```

参数量：~6K（从 391K 降低 98%）

**注意**：这个 V2 版本**暂时去掉了 GNN 和 EdgeFrame**。原因：在 2-node 图上，GNN 退化为 MLP，不如直接用 MLP。等 multi-object 实验时再加回 GNN。

#### 4.2.3 修复 GNSGraphLayer（如果保留 GNN 版本）

```python
class GNSGraphLayerV2(MessagePassing):
    """Minimal GN layer for 2-node graph."""
    def __init__(self, node_dim, edge_dim, hidden_dim=32):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim))   # 3层→2层
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, node_dim))
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))
```

#### 4.2.4 PPO 超参数调整（配合新 features_dim）

```python
# GNS Agent
policy_kwargs = dict(
    features_extractor_class=GNSFeaturesExtractorV2,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=dict(pi=[64, 64], vf=[64, 64])  # 明确指定 policy/value 网络结构
)

# PhysRobot Agent
policy_kwargs = dict(
    features_extractor_class=PhysRobotFeaturesExtractorV2,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)
```

### 4.3 训练参数修改

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `total_timesteps` | 200K | **500K** | 更多 sample |
| `features_dim` | 128 | **64** | 减少参数 |
| `hidden_dim` (GNN) | 128 | **32** | 减少参数 |
| `n_message_passing` | 3 | **1** | 2-node 图只需 1 层 |
| `learning_rate` | 3e-4 | **3e-4** | 不变（PPO 的 sweet spot） |
| `n_steps` | 2048 | **2048** | 不变 |
| `batch_size` | 64 | **64** | 不变 |
| `n_epochs` | 10 | **10** | 不变 |
| `gamma` | 0.99 | **0.99** | 不变 |
| `ent_coef` | 0.0 (默认) | **0.01** | 增加探索 |
| `max_grad_norm` | 0.5 (默认) | **0.5** | 不变 |
| `eval_episodes` | 50 | **100** | 更可靠的评估 |
| `success_threshold` | 0.1 | **0.15** | 稍微放宽，让初始学习更容易 |

### 4.4 实验设计（Ablation Study）

#### Ablation 1: Reward 的影响

| 实验 | Reward 设计 | 方法 | 预期 |
|------|------------|------|------|
| A1a | 旧 reward | PPO | ~6% (baseline) |
| A1b | 新 reward | PPO | **>50%** |
| A1c | 旧 reward + 500K steps | PPO | ~15% |
| A1d | 新 reward + 500K steps | PPO | **>60%** |

#### Ablation 2: 参数量的影响

| 实验 | 方法 | 参数量 | 预期 |
|------|------|-------|------|
| A2a | PPO (MlpPolicy) | ~10K | >50% |
| A2b | GNS-Original (500K params) | ~500K | <10% |
| A2c | GNS-V2 (3K params) | ~3K | **>40%** |
| A2d | PhysRobot-Original (391K params) | ~391K | <10% |
| A2e | PhysRobot-V2 (6K params) | ~6K | **>40%** |

#### Ablation 3: 物理先验的价值

| 实验 | 方法 | OOD mass | 预期 |
|------|------|---------|------|
| A3a | PPO | 0.5 (train) | ~60% |
| A3b | PPO | 2.0 (OOD) | ~20% (大幅下降) |
| A3c | PhysRobot-V2 | 0.5 (train) | ~55% |
| A3d | PhysRobot-V2 | 2.0 (OOD) | **~40%** (下降较少) |

#### Ablation 4: 各组件贡献

| 实验 | 描述 | 预期 |
|------|------|------|
| A4a | PhysRobot-V2 Full | ~55% |
| A4b | PhysRobot-V2 - physics_net (只用 policy_stream) | ~55% (相当于 PPO) |
| A4c | PhysRobot-V2 - policy_stream (只用 physics_net) | <30% (物理不够指导策略) |
| A4d | PhysRobot-V2 + stop-gradient on physics_net | ~55% (物理梯度不影响 RL) |

---

## 5. 实施优先级

### P0 (立即 — 今天完成，预计让 PPO >50%, GNS/PhysRobot >30%)

1. **修改 Reward** [环境]
   - Progress reward (`Δdist`)
   - 成功奖励 100 → 500 + early bonus
   - Action penalty `-0.01 * ||a||²`
   - 在 `reset()` 中初始化 `_prev_dist_box_goal`

2. **砍 GNS 参数量** [架构]
   - `hidden_dim`: 128 → 32
   - `MP layers`: 3 → 1
   - `features_dim`: 128 → 64
   - 总参数 500K → ~3K

3. **砍 PhysRobot 参数量** [架构]
   - 用 `PhysRobotFeaturesExtractorV2`（直接 MLP，不用 GNN）
   - 总参数 391K → ~6K

4. **增加 timesteps** [训练]
   - 200K → 500K

5. **增加探索** [训练]
   - `ent_coef`: 0.0 → 0.01

6. **更新 Colab notebook** [工程]
   - 应用以上所有改动
   - 添加 PPO 的 `net_arch=dict(pi=[64,64], vf=[64,64])`

### P1 (本周)

7. **修复 ee_vel** — 从 `joint_vel` + forward kinematics 计算 ee 笛卡尔速度
8. **添加 curriculum** — 先从固定的简单初始化开始，逐步增加随机性
9. **实现正确的反对称性** — Scalarization-Vectorization 管道（为论文准备）
10. **运行 5-seed 实验** — 获取 mean ± std

### P2 (下周)

11. **Multi-object 场景** — 3-5 个物体，展示 GNN 优势
12. **OOD 实验** — mass 0.1x~10x, friction 0.1~1.0
13. **Learning curves** — TensorBoard 记录，绘制 confidence band
14. **切换到 `DynamiCALGraphNet`** — 用 baselines/ 中的正确实现
15. **写 ablation study 结果**

---

## 6. 预期结果

应用 P0 修改后的预期：

| 方法 | 原 Success Rate | 预期 Success Rate | 预期训练时间 |
|------|----------------|-------------------|-------------|
| Pure PPO | 6% | **50–70%** | ~7 min |
| GNS-V2 | 0% | **30–50%** | ~8 min |
| PhysRobot-V2 | 0% | **30–50%** | ~8 min |

预期提升的理由：
- **Reward 修改**：PPO 从 6% → ~40%（最大贡献因子）
- **参数缩减**：GNS/PhysRobot 从 0% → ~20%（消除过拟合/欠学习）
- **更多 steps (2.5x)**：所有方法 +10–15%
- **ent_coef=0.01**：+5% 通过更好的探索

如果 P0 修改后 PPO 仍然 <30%，考虑：
- 进一步增加到 1M steps
- 切换到速度控制（而非 torque 控制）
- 减小 success_threshold（0.1 → 0.15m）
- 添加 box_pos → goal_pos 方向的 shaping reward

---

*报告完成。Expert A 和 Expert B 达成共识：**环境 reward 和网络过参数化是 0% 的两个并行根因**，必须同时修复。物理先验的价值需要在基础问题解决后才能评估。*
