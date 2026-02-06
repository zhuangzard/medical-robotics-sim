# PhysRobot — 严格审稿意见 (Reviewer 2)

**审稿人**: Reviewer 2 (Devil's Advocate)  
**日期**: 2026-02-06  
**论文标题**: Physics-Informed Foundation Models for Robotic Manipulation: Integrating Conservation Laws with Vision-Language-Action Models  
**目标会议**: CoRL 2026 / ICRA 2027  
**审稿依据**: 可行性报告、专家辩论方案、论文大纲、全部源代码

---

## 总评（Executive Summary）

本论文提出将 Dynami-CAL 式动量守恒 GNN 作为 RL 特征提取器，以提高机器人操作的样本效率和 OOD 泛化。方向有意义，但**当前实现与论文 claim 之间存在严重脱节**。代码中的物理守恒保证实际上被破坏，核心实验结果为 0% 成功率，论文大纲中的数字（12.5x 效率提升、95% OOD 改善、0% 安全违规）全部是未验证的占位符。如果以当前状态投稿，这篇论文会被任何 top venue 拒绝。

以下逐项分析。

---

## 1. Novelty 评估

### 1.1 "物理 GNN + RL" 方向新不新？

**结论：方向本身不新，但具体组合有微创新。**

这个大方向已有大量前期工作：

| 年份 | 工作 | 做法 |
|------|------|------|
| 2019 | Lutter et al. (Deep Lagrangian Networks) | Lagrangian 结构 + 控制 |
| 2020 | Sanchez-Gonzalez et al. (GNS) | GNN 学物理仿真 |
| 2021 | Satorras et al. (EGNN) | E(n) 等变 GNN |
| 2022 | Huang et al. (Equivariant Transporter) | 等变网络 + manipulation |
| 2023 | Simeonov et al. | SE(3)-equivariant representations for manipulation |
| 2024 | 多篇 ICRA/CoRL | Physics-informed RL, differentiable physics + policy |
| 2025 | Sharma & Fink (Dynami-CAL) | 动量守恒 GNN (Nature Comms) |

"物理先验 + RL" 是一个 well-explored 方向。单纯说 "我们把 GNN 接到 PPO 前面" 在 2026 年已经不够新颖了。

### 1.2 与 EGNN, DimeNet, PaiNN 的真正区别

论文 claim 的区别是 **"硬约束动量守恒"**。但让我追问：

**EGNN (Satorras 2021):**
- 保证 E(n) 等变性（平移、旋转、反射不变）
- **不保证**动量守恒
- 但在实践中，等变性往往比守恒律更 general

**DimeNet (Gasteiger 2020):**
- 使用 directional message passing
- 考虑了角度信息
- 面向分子性质预测

**PaiNN (Schütt 2021):**
- 等变消息传递，标量+向量特征
- 在分子模拟中 SOTA

**Dynami-CAL (Sharma & Fink 2025):**
- 反对称边帧 → 硬约束 F_ij = -F_ji → Σ F = 0
- 面向颗粒流/粒子仿真

**PhysRobot 的 claim:**
- 首次将 Dynami-CAL 的守恒性 GNN 用于 RL

**问题**：这个 "首次" 有意义吗？

1. Dynami-CAL 的守恒性面向的是**长时间物理仿真**（10K+ 步的轨迹预测），防止能量/动量漂移。但在 RL 中，PhysicsCore 只做**单步前向预测**作为特征提取器——守恒性的长期优势无法体现。
2. 如果只是把 GNN 当 feature extractor，那和 "把 ResNet 当 feature extractor" 在方法论上没有本质区别。核心问题变成了：**为什么这个 feature extractor 比 MLP 更好？** 这需要实验证明，而非理论断言。

### 1.3 如果只是 "把 GNN 当 feature extractor"，novelty 够吗？

**不够。** 这相当于一个 application paper：拿现有方法 A（Dynami-CAL）应用到现有框架 B（PPO）。对于 ICRA 勉强可以，对于 CoRL 明显不够。

### 1.4 ⭐ 建议提高 novelty 的方向

以下是**我认为能真正提高 novelty 的 4 个方向**（按推荐程度排序）：

**方向 1（最推荐）：Conservation-Constrained Policy Gradient**
- 不只是把物理 GNN 当 feature extractor
- 而是将守恒律直接嵌入策略梯度的约束中
- 例如：在 PPO 的 surrogate objective 中加入 Lagrangian multiplier，保证策略输出的动作满足 Σ F = 0
- 这是一个**新的 RL 算法**，而非一个新的 feature extractor
- 形式上：$\max_\theta L^{CLIP}(\theta)$ s.t. $\|\sum_i F_i(s, \pi_\theta(s))\| \leq \epsilon$

**方向 2：Physics-Aware World Model**
- 不只用 GNN 做 feature extraction，而是做 **model-based RL**
- PhysicsCore 预测下一步状态 → 用于 planning（如 MPC）或 imagination（如 Dreamer）
- 守恒律在多步 rollout 中的优势可以真正体现
- 与 Dreamer V3 / TD-MPC 的对比会非常有说服力

**方向 3：Equivariance + Conservation 双重约束**
- EGNN 保证等变性但不保证守恒
- Dynami-CAL 保证守恒但不保证等变性
- **同时保证两者** 是一个有价值的理论贡献
- 需要证明：在什么条件下，等变性 + 守恒性可以同时被架构保证

**方向 4：Physics-Informed Reward Shaping with Formal Guarantees**
- 用守恒律自动生成 reward shaping function
- 证明 shaping 不改变最优策略（potential-based shaping theorem 的扩展）
- 这是一个理论 + 实践结合的方向

---

## 2. Technical Soundness 评估

### 2.1 反对称性破坏问题：未修复 ❌

**这是全文最致命的技术问题。**

可行性报告已指出，`EdgeFrame` 中的 MLP 破坏了反对称性。让我更精确地说明为什么：

**数学论证：**

设输入为 $x_{ij} = [r_{ij}, \|r_{ij}\|, v_{rel,ij}, \|v_{rel,ij}\|] \in \mathbb{R}^8$

反对称的输入分量：$r_{ij} = -r_{ji}$, $v_{rel,ij} = -v_{rel,ji}$

对称的输入分量：$\|r_{ij}\| = \|r_{ji}\|$, $\|v_{rel,ij}\| = \|v_{rel,ji}\|$

设 MLP 为 $f: \mathbb{R}^8 \to \mathbb{R}^{64}$（含 LayerNorm + ReLU）

**反对称性要求**：$f(x_{ij}) = -f(x_{ji})$

但对于一般的 MLP $f$，设 $x_{ij} = [r, d, v, s]$ 和 $x_{ji} = [-r, d, -v, s]$（其中 $d = \|r\|, s = \|v\|$）

$$f([r, d, v, s]) = -f([-r, d, -v, s])$$

这**不成立**，因为：
1. 第一层：$W[r, d, v, s]^T + b \neq -(W[-r, d, -v, s]^T + b)$（bias 项 $b$ 不抵消，$d, s$ 分量的贡献不抵消）
2. LayerNorm 进一步破坏（mean subtraction + rescaling 都不保持反对称性）
3. ReLU 不是奇函数：$\text{ReLU}(x) \neq -\text{ReLU}(-x)$

**实际测试**（在 `edge_frame.py` 的 `__main__` 中）：运行 `check_antisymmetry` 会报告 error >> 1e-5，代码自己的测试就会打印 "❌ Antisymmetry violated!"。

**更严重的是**：`baselines/physics_informed.py` 中的 `DynamiCALGraphNet` 使用了不同的实现（基于边帧分解 `_edge_frame`），但也存在问题：

```python
def _edge_frame(self, pos_i, pos_j):
    # ...
    up = torch.tensor([0., 0., 1.], ...)
    e2 = torch.cross(e1, up.expand_as(e1))
```

- **e1 是反对称的**：$e1_{ij} = -e1_{ji}$ ✅
- **e2 的反对称性需要验证**：$e2_{ij} = e1_{ij} \times \hat{z}$, $e2_{ji} = e1_{ji} \times \hat{z} = -e1_{ij} \times \hat{z} = -e2_{ij}$ ✅
- **e3 呢？** $e3_{ij} = e1_{ij} \times e2_{ij}$, $e3_{ji} = e1_{ji} \times e2_{ji} = (-e1_{ij}) \times (-e2_{ij}) = e1_{ij} \times e2_{ij} = e3_{ij}$ → **e3 是对称的！** ❌

所以 `f_vector[:, 1:2] * e3` 项（垂直于 e1-e2 平面的力分量）**不满足反对称性**。要修复：
- 需要 $f_3^{(ij)} = -f_3^{(ji)}$（标量系数本身反对称），才能让 $f_3^{(ij)} e3_{ij} = -f_3^{(ji)} e3_{ji}$
- 但代码中 `f_vector` 是从 `scalar_mlp` / `vector_mlp` 出来的，输入是 `[x_i, x_j, rel_pos]`。交换 i, j 后，`[x_j, x_i, -rel_pos]`——MLP 不保证 output 反号。

**修复方案**（必须实现）：

```python
# Dynami-CAL 原论文的正确做法：
# 1. Scalarization: 将向量投影到边帧的标量分量
s1 = dot(v_rel, e1)  # 标量（对称）
s2 = dot(v_rel, e2)  # 标量
s3 = dot(v_rel, e3)  # 标量
d = ||r_ij||          # 标量（对称）

# 2. MLP 处理标量（对称的标量 → 对称的标量输出）
f1, f2, f3 = MLP([s1, s2, s3, d, h_i, h_j])

# 3. Vectorization: 从标量重建 3D 力
F_ij = f1 * e1 + f2 * e2 + f3 * e3

# 因为 e1, e2 反对称 + f1, f2 来自对称标量 → F_ij 的 e1, e2 分量反对称 ✅
# 因为 e3 对称 + 需要 f3 本身反对称 → 需要特殊处理！
# 正确做法：f3(ij) 使用 s1 的符号（s1 对换 ij 时反号）来保证反对称
```

**审稿意见**：论文的核心 claim（"硬约束动量守恒"）在代码中**不成立**。这不是小 bug，这是方法论的根基性问题。如果修不好这个，整个 "conservation-preserving" 的故事就塌了。

**严重度：致命（Fatal）。**

### 2.2 2-node Graph 问题：未真正解决 ❌

专家辩论方案指出了这个问题，也提出了 V2 架构（直接用 MLP 替代 GNN）。但这**本质上是放弃了 GNN 的贡献**。

让我明确说明这个矛盾：

- **论文 claim**："使用 GNN 处理多体交互"
- **实际实现**：2 个节点（end-effector + box），GNN 退化为 MLP
- **V2 修复**：直接用 MLP，不用 GNN

如果用 MLP 就行，那 GNN 的 section 写什么？论文的架构图（包含 "Scene → Graph → EdgeFrames → GNN(3 layers) → Force Prediction" 的那个图）就变成了误导。

**真正的解决方案**：

1. **Multi-object 环境是必须的**（至少 3-5 个物体），否则 GNN 的使用在技术上没有正当理由
2. 需要证明：随着物体数量增加，GNN 方法的优势相对 MLP 线性增长
3. 需要 scalability 实验：2 objects → 5 → 10 → 20，证明 GNN 能 scale 而 MLP 不能

**严重度：高。** 没有 multi-object 实验，GNN 的使用就是 over-engineering，审稿人会直接指出。

### 2.3 守恒定律约束：硬约束 vs 软约束

论文大纲和可行性报告反复 claim "硬约束（hard constraint）"。让我严格区分：

**硬约束（Architectural Guarantee）**：
- 定义：对于**任何**网络参数 θ，守恒律都被满足
- 例子：如果 $F_{ij}$ 的计算中，$e1_{ij} = -e1_{ji}$ 是由 $r_{ij} = x_j - x_i$ 的几何关系保证的，那 $\sum F \cdot e1 = 0$ 是硬约束
- **PhysRobot 的现状**：由于 MLP 破坏反对称性，**当前不是硬约束**

**软约束（Loss-based）**：
- 定义：通过 loss penalty $\lambda \|\sum F\|^2$ 来鼓励守恒
- 缺点：在优化压力下会被牺牲，不提供 guarantee
- PhysRobot 目前**实际上退化为比软约束更差的状态**——既没有硬约束，也没有软约束 loss

**结论**：论文声称的 "hard constraint" 是 **false claim**。必须修复 Scalarization-Vectorization 管道，才能重新 claim 这一点。

**两者的实际区别**（应在论文中讨论）：

| 维度 | 硬约束 | 软约束 |
|------|--------|--------|
| 保证程度 | 数学上严格（∀ θ） | 统计上近似（视 λ 和优化状态） |
| 梯度冲突 | 不存在（约束内嵌于架构） | 可能与 task loss 冲突 |
| OOD 行为 | 守恒始终成立 | OOD 时 penalty 可能失效 |
| 灵活性 | 低（架构约束不可松弛） | 高（可调 λ） |
| 实现难度 | 高（需要特殊架构设计） | 低（加 loss 项即可） |

### 2.4 数学公式审查

**Paper Outline 中的公式问题：**

1. **Theorem 1（动量守恒）** 的陈述缺少关键条件：

   > "If edge frames satisfy antisymmetry ($e_k^{(ji)} = -e_k^{(ij)}$) and force coefficients are symmetric ($f_k^{(ij)} = f_k^{(ji)}$), then $\sum_{i=1}^N \sum_{j \in \mathcal{N}(i)} F_{ij} = 0$"
   
   **问题 1**：这里说 "force coefficients are symmetric"（$f_k^{(ij)} = f_k^{(ji)}$），但如上所述，对 e3（对称基向量），需要 $f_3^{(ij)} = -f_3^{(ji)}$（反对称系数）才能保证反对称。所以 Theorem 的条件写错了。
   
   **问题 2**：陈述中 $\sum_{i=1}^N \sum_{j \in \mathcal{N}(i)} F_{ij}$ 实际上是对所有 **有向边** 的求和。对于无向图，每对 $(i,j)$ 出现两次（$F_{ij}$ 和 $F_{ji}$），确实 $F_{ij} + F_{ji} = 0$。但更精确的陈述应该是 $\sum_i \sum_{j \in \mathcal{N}(i)} F_{ij} = 0$，其中 $F_{ij}$ 是 j 对 i 的力。这等价于 $\sum_i a_i = 0$（total net force = 0），即动量守恒。
   
   **问题 3**：Theorem 只讨论了**线动量**。角动量守恒需要额外条件（force 沿连线方向），这一点在论文中没有讨论。Dynami-CAL 原论文同时保证了线动量和角动量守恒——PhysRobot 是否也保证角动量？如果不保证，需要明确说明。

2. **12.5x sample efficiency claim**：

   论文大纲说 "400 vs 5000 episodes"。但：
   - 当前实验结果是 PPO 6%, PhysRobot 0%
   - 即使修复后，预测是 PPO 50-70%, PhysRobot 30-50%（根据辩论方案）
   - 也就是说 PhysRobot **不比 PPO 好**，甚至更差
   - 12.5x 的数字完全是虚构的

3. **"95% relative improvement on OOD"**：

   同样是虚构数字。当前 OOD 结果：PPO 8%, PhysRobot 0%。

**审稿意见**：论文中 **不能** 包含未经实验验证的定量 claim。如果在投稿时仍然使用这些数字但实验跑不出来，这属于学术不端。

---

## 3. Experiments 评估

### 3.1 Baseline 够不够？

**当前 baseline**：
1. Pure PPO
2. GNS + PPO
3. RT-2 fine-tuned（只在大纲中提到，实际未实现）

**缺少的关键 baseline**（审稿人会要求补充）：

| 优先级 | Baseline | 原因 |
|--------|----------|------|
| **必须** | EGNN + PPO | 最直接的等变 GNN 对比 |
| **必须** | MLP (same params) + PPO | 证明 GNN 结构（而非参数量）带来的提升 |
| **必须** | PPO + physics loss (soft constraint) | 对比硬约束 vs 软约束 |
| **强烈建议** | SAC (off-policy RL) | PPO 不是唯一的 RL 算法 |
| **强烈建议** | Dreamer V3 / TD-MPC | Model-based RL baseline |
| **建议** | HNN + RL / LNN + RL | 其他物理感知方法 |
| **建议** | Oracle (with access to true physics) | Upper bound |

**特别注意**：必须有一个 **parameter-matched MLP baseline**。当前的 PPO 是 ~10K 参数，PhysRobot 是 ~391K。如果 PPO 也用 391K 参数的 MLP 能达到同样性能，那 GNN 就没有存在的意义。

### 3.2 环境是否太简单？

**PushBox 能代表 "manipulation" 吗？**

**回答：不能。** 理由：

1. **2-DOF 臂 + 单箱子**：这是机器人学最简单的 benchmark 之一。连 Gymnasium 的标准 FetchPush 都比这复杂（7-DOF + 3D）。
2. **无接触复杂性**：只有 pushing，没有 grasping、stacking、insertion 等。
3. **无视觉输入**：直接使用 state observation（16-dim），没有图像。论文大纲却声称使用 "RGB-D + Language"。
4. **无多物体交互**：这是 GNN 的核心优势场景，但没有测试。

**审稿人的第一反应**："这就是一个 toy problem。在 PushBox 上的结果不能说明任何关于 'robotic manipulation' 的事情。"

**必须添加的环境**（至少 2 个）：

1. **Multi-object Push/Rearrangement**：展示 GNN 的多体优势
2. **FetchPush / HandManipulate**（OpenAI Gym）：标准 benchmark，方便与文献对比
3. **考虑**：Panda 臂 + 更复杂的任务（stacking, insertion）

### 3.3 统计显著性

**5 seeds 够不够？**

这取决于方差。对于 PushBox 这种简单环境，5 seeds 通常够了，但：

1. **当前只跑了 1 seed**（从辩论方案看）
2. 必须至少 5 seeds，报告 mean ± std
3. 如果方差很大（success rate 跨 seed 变化 > 20%），需要增加到 10 seeds
4. **必须做 significance test**：Welch's t-test 或 Mann-Whitney U test
5. 学习曲线必须有 confidence band（不是只画 mean）

**特别注意**：如果 5 seeds 中有 1-2 个 seed 的 PhysRobot 超过了 PPO，但 mean 没有显著差异，这说明**方差太大**，结论不可靠。

### 3.4 OOD 测试

**当前 OOD 测试**：改变 box 质量（训练时 0.5kg，测试时可能改为 2.0kg）

**不够强的原因**：

1. **只改变一个维度（质量）**：OOD 泛化应该测多个维度
2. **缺少连续变化曲线**：不应只测 2 个点（train mass, OOD mass），应测 mass ∈ [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] 的完整曲线
3. **缺少其他 OOD 维度**：
   - 摩擦系数变化
   - 物体形状变化（sphere, cylinder, irregular）
   - 物体材质变化
   - 桌面倾斜
   - 机械臂参数变化（关节阻尼、连杆长度）
4. **缺少 compositional OOD**：同时改变多个参数

**建议的 OOD 实验矩阵**：

| OOD 维度 | 训练值 | 测试范围 | 期望结果 |
|----------|--------|---------|---------|
| Box mass | 0.5 kg | 0.1–10.0 kg | PhysRobot 在远离训练分布时衰减更慢 |
| Friction | μ = 0.5 | 0.1–1.0 | 摩擦变化对物理先验的影响 |
| Box shape | cube | sphere, cylinder | GNN 是否泛化到不同几何 |
| Number of objects | 1 | 2, 3, 5 | GNN vs MLP 的 scaling |

---

## 4. Presentation 评估

### 4.1 论文故事线

**当前故事线**（从 Paper Outline 读取）：

> "Foundation models 缺乏物理理解 → 我们加入 physics GNN → 性能提升"

**问题**：

1. **故事太大**：标题说 "Foundation Models"，实际做的是 PPO + MLP + 小 GNN。没有任何 vision encoder，没有 language，没有 transformer。这不是 "foundation model"。
2. **两个 domain 的连接不自然**：rigid body pushing 和 soft tissue grasping 是两个完全不同的物理模型。论文试图用一个统一的框架覆盖两者，但实际上代码只实现了 PushBox。
3. **Overclaim**：Abstract 说 "zero safety violations"、"12.5x sample efficiency"，但这些数字没有实验支撑。

**建议的故事线（更诚实、更聚焦）**：

> "在接触丰富的机器人操作中，物理先验（如动量守恒）能否提高 RL 的样本效率和 OOD 泛化？我们提出 PhysRobot，一个将 Dynami-CAL 式硬约束守恒 GNN 集成到 PPO 中的方法。通过在多体推箱任务上的系统实验，我们发现：(1) 守恒律硬约束在 OOD 场景中显著优于软约束；(2) GNN 结构在多体交互中优于 MLP；(3) 物理先验的主要价值在于泛化而非训练效率。"

这个故事线更小但更实在。

### 4.2 图表设计建议

**必须有的图表**：

1. **Fig 1: 系统架构图**
   - 当前大纲的 ASCII 架构图太复杂（Vision-Language, Cross-Attention 等都没实现）
   - 建议：只画实际实现的部分（Observation → [Physics Stream / Policy Stream] → Fusion → PPO Action）
   - 使用 TikZ 或 draw.io 制作专业图

2. **Fig 2: 边帧反对称性示意图**
   - 可视化 $e1_{ij} = -e1_{ji}$, $F_{ij} = -F_{ji}$
   - 这是论文的核心 insight，需要一个清晰的图
   - 参考 Dynami-CAL 原论文 Fig 2 的风格

3. **Fig 3: 学习曲线**（最重要的结果图）
   - X轴: training timesteps
   - Y轴: success rate
   - 4+ 条线（PPO, GNS, EGNN, PhysRobot），with confidence band
   - 需要清晰展示样本效率差异

4. **Fig 4: OOD 泛化曲线**
   - X轴: box mass（对数尺度）
   - Y轴: success rate
   - 4+ 条线
   - 展示物理先验在 far-OOD 时的优势

5. **Table 1: 主实验结果**
   - 所有方法、所有环境、所有指标
   - 包含 ± std 和 significance markers (*, **)

6. **Table 2: Ablation Study**
   - Full model vs -EdgeFrame vs -GNN vs -Physics vs -Fusion

7. **Fig 5: 守恒误差图**
   - 对比不同方法的 $\|\sum F\|$ 随时间步的变化
   - 硬约束方法应该是零误差

8. **可选但加分**：
   - 注意力热力图（如果有 cross-attention）
   - t-SNE/UMAP of learned features（物理 vs 非物理）
   - 失败案例分析

### 4.3 哪些 claim 需要更强的实验支撑

| Claim | 当前支撑 | 需要的支撑 |
|-------|---------|-----------|
| "12.5x sample efficiency" | ❌ 无（数字虚构） | 学习曲线 + 统计检验 |
| "95% OOD improvement" | ❌ 无（OOD 全部 0%） | 多维 OOD 实验 |
| "Zero safety violations" | ❌ 无（medical domain 未实现） | 至少在仿真中验证力约束 |
| "Hard constraint momentum" | ❌ 代码中不成立 | 修复代码 + 数值验证 |
| "First integration of..." | ⚠️ 需要更全面的 related work | 证明文献中确实没有 |
| "GNN structure benefits" | ❌ 只有 2 节点 | Multi-object 实验 |

---

## 5. Overall Recommendation

### 5.1 ICRA 标准

**推荐：Weak Reject（弱拒）**

理由：
- 方向有趣（物理 GNN + RL），符合 ICRA 的口味
- 但核心技术 claim（守恒性保证）在代码中不成立
- 实验只有 1 个简单环境、0% 成功率
- ICRA 的 bar 比 CoRL 低，但也不接受 overclaim + 无法 reproduce 的论文
- 如果修复了守恒性问题 + 添加了 multi-object 实验 + 跑出了合理的结果，可以升到 **Borderline / Weak Accept**

### 5.2 CoRL 标准

**推荐：Reject（拒绝）**

理由：
- CoRL 对 novelty 和实验要求更高
- "GNN 当 feature extractor" 的 novelty 不够 CoRL
- 缺少与 model-based RL（Dreamer, TD-MPC）的对比
- 缺少 real robot 或至少 realistic simulation（ManiSkill, RoboSuite）
- 论文故事线 overclaim（说 "foundation model" 但实际是 PPO + small GNN）
- 即使修复了所有 bug，仍需要额外 1-2 个月来做实验

### 5.3 ⭐ 关键的 3 个修改建议（必须做）

#### 修改 1：修复 Scalarization-Vectorization 管道 【技术正确性】

**为什么必须做**：这是论文 claim 的根基。如果守恒性不成立，整个 "physics-informed" 的故事就不成立。

**具体要求**：
1. 实现正确的 Scalarization：向量 → 标量投影（在边帧中）
2. MLP 处理标量特征（保证旋转不变性）
3. 实现正确的 Vectorization：标量 → 3D 力重建（保证反对称性）
4. 写**单元测试**：
   ```python
   def test_antisymmetry():
       """Test F_ij = -F_ji for random inputs"""
       for _ in range(100):
           pos_i, pos_j = torch.randn(3), torch.randn(3)
           F_ij = model.compute_force(pos_i, pos_j, ...)
           F_ji = model.compute_force(pos_j, pos_i, ...)
           assert torch.allclose(F_ij, -F_ji, atol=1e-6)
   
   def test_momentum_conservation():
       """Test Σ F_i = 0 for random graphs"""
       graph = random_graph(N=10)
       forces = model.compute_all_forces(graph)
       total_force = forces.sum(dim=0)
       assert torch.allclose(total_force, torch.zeros(3), atol=1e-6)
   ```
5. 在论文中报告数值验证结果：$\|\sum F\|$ 应为**机器精度级别（~1e-7）**而非近似零

#### 修改 2：添加 Multi-Object 环境 + 实验 【实验完整性】

**为什么必须做**：2 节点图上的 GNN = MLP。必须有 multi-object 实验来 justify GNN 的使用。

**具体要求**：
1. 创建 Multi-Object PushBox 环境（3-5 个物体 + 机器人）
2. 证明随着物体数量增加：
   - MLP 性能下降
   - GNN 保持性能或仅轻微下降
   - 物理 GNN（有守恒约束）> 标准 GNN > MLP
3. 添加 scalability 图：X轴=物体数量，Y轴=成功率

#### 修改 3：诚实的论文定位 + 完整的 baseline 对比 【学术诚信 + 完整性】

**为什么必须做**：Overclaim 是审稿人最讨厌的事情。

**具体要求**：
1. **删除所有未验证的定量 claim**（12.5x, 95%, zero violations）
2. **缩小论文范围**：从 "foundation model" 改为 "physics-informed RL for manipulation"
3. **删除 medical robotics domain**（除非真的实现并跑了实验）
4. **添加至少 3 个新 baseline**：EGNN+PPO, param-matched MLP, PPO+physics loss
5. **所有实验至少 5 seeds + significance test**

---

## 6. 附加建议（非必须但强烈推荐）

### 6.1 技术改进

- **修复 ee_vel = zeros 问题**：从 joint_vel + FK/Jacobian 计算真实的 ee 速度。否则物理模型收不到运动信息。
- **使用辛积分器或删除**：如果不用，就从代码中移除。如果用，集成到 world model 中做多步预测。
- **处理 Gram-Schmidt 退化**：当 e1 接近 [0,0,1] 时，up=[0,0,1] 的 cross product 退化。添加 fallback。

### 6.2 实验改进

- **添加 computation overhead 分析**：PhysRobot 的推理速度比 PPO 慢多少？值得吗？
- **添加 wall-clock time 对比**：不只看 sample efficiency，还要看 wall-clock time
- **使用标准 benchmark**：考虑 MetaWorld, ManiSkill2, RoboSuite 等社区标准 benchmark

### 6.3 Presentation 改进

- **重新设计 related work**：当前分类（Foundation Models / PINNs / GNNs / Medical）太 generic。应该更聚焦于 "physics priors in RL" 和 "conservation laws in neural networks"。
- **添加 limitation section**：诚实讨论局限性。审稿人更欣赏知道自己工作不足的作者。
- **Consider a shorter paper**：如果实验只有 PushBox，4-page workshop paper（e.g., NeurIPS Workshop on Physics for ML）比 8-page 会议论文更合适。

---

## 7. 第二轮评审的期望

当 `paper-algorithm`, `paper-experiment`, `paper-writer` 完成修改后，我将在第二轮评审中检查：

1. **守恒性数值验证**：`test_antisymmetry()` 和 `test_momentum_conservation()` 的结果
2. **Multi-object 实验结果**：至少 3 个物体的 PushBox
3. **完整 baseline 对比**：至少 5 个方法（PPO, GNS+PPO, EGNN+PPO, MLP(same params)+PPO, PhysRobot）
4. **5 seeds + significance test**
5. **修改后的论文故事线和 claim**

**评分可能的变化**：
- 如果以上 5 点全部满足 → **Borderline → Weak Accept（ICRA）**
- 如果还添加了 model-based RL 对比 + realistic environment → **Weak Accept（CoRL）**
- 如果只修了部分问题 → **维持 Weak Reject / Reject**

---

## 附录：代码级别的具体 Bug List

| # | 文件 | 行号（约） | Bug | 严重度 |
|---|------|-----------|-----|--------|
| B1 | `physics_core/edge_frame.py` | L38-45 | MLP 破坏反对称性（含 bias + LayerNorm + ReLU） | 致命 |
| B2 | `baselines/physics_informed.py` | L60-75 | e3 = cross(e1, e2) 是对称的，但 f3 系数未反对称化 | 致命 |
| B3 | `baselines/physics_informed.py` | L67 | up = [0,0,1] 硬编码，e1 ∥ z 时 Gram-Schmidt 退化 | 高 |
| B4 | `baselines/physics_informed.py` | L186 | `ee_vel = torch.zeros(3)` — 物理模型收不到运动信息 | 高 |
| B5 | `physics_core/dynamical_gnn.py` | 全文件 | `DynamicalGNN` 使用 `EdgeFrame`（B1）而非正确的边帧 | 高 |
| B6 | `baselines/physics_informed.py` | L80-82 | `scalar_mlp` 输入 `[x_i, x_j, rel_pos]`，交换 i,j 后输出不保证反号 | 高 |
| B7 | `baselines/physics_informed.py` | L139-141 | `PhysicsCore` 用 `DynamiCALGraphNet` 的 hidden_dim 作 node_dim，但 encoder 输出是 hidden_dim 而 `DynamiCALGraphNet` 内部又用 `node_dim` 做 scalar_mlp 输入维度 | 中 |
| B8 | `environments/push_box_env.py` | L118 | Reward 信号太弱（success bonus 被距离惩罚淹没） | 中 |
| B9 | `physics_core/integrators.py` | 全文件 | 辛积分器已实现但在训练中从未使用 | 低 |

---

*审稿完成。作为 Reviewer 2，我的职责是确保论文的技术正确性和实验完整性。以上意见旨在帮助作者写出一篇更好的论文。方向是好的，但执行还需要大量工作。*

**⏱️ 建议时间线**：
- Week 1-2：修复 B1, B2, B3, B4, B6（守恒性相关）
- Week 2-3：Multi-object 环境 + reward 修复（B8）
- Week 3-5：完整实验（5 methods × 3 environments × 5 seeds）
- Week 5-6：论文重写
- Week 7：内部审稿 + 修改
- **最早可投稿时间：2026 年 3 月底**（距 CoRL 2026 deadline ~2.5 个月）
