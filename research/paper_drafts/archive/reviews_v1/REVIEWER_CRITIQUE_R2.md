# PhysRobot — 第二轮审稿意见 (Reviewer 2, Round 2)

**审稿人**: Reviewer 2 (Devil's Advocate)  
**日期**: 2026-02-06  
**审稿对象**:
1. `ALGORITHM_DESIGN.md` (算法组, 27KB)
2. `EXPERIMENT_DESIGN.md` (实验组, 849 行)

**背景**: 第一轮审稿 (`REVIEWER_CRITIQUE.md`) 给出了 ICRA Weak Reject / CoRL Reject。本轮评审两个团队的修订方案。

---

## 总评（Round 2 Summary）

**显著进步。** 算法组正面回应了最致命的反对称性问题，提出了 SV-pipeline 修复方案，数学上基本正确。实验组设计了有层次感的实验方案（7 variants × 5 seeds × 3 environments × 6 baselines），结构远超第一轮。

**但仍有关键漏洞需要堵住才能投稿。** 以下逐项评审。

---

## Part A：算法设计评审

### A1. §2.4 Conservation Proof — 严密性审查

**结论：proof sketch 正确，但表述需要补两个环节。**

#### ✅ 正确的部分

证明的核心逻辑链是对的：

1. $\mathbf{e}_1^{ij} = -\mathbf{e}_1^{ji}$ ← 来自 $\mathbf{r}_{ij} = -\mathbf{r}_{ji}$ ✅
2. $\mathbf{e}_2^{ij} = -\mathbf{e}_2^{ji}$ ← 来自 $\mathbf{v}_{ij}^{\perp} = -\mathbf{v}_{ji}^{\perp}$ ✅
3. $\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$ ← 来自 $(-) \times (-) = (+)$ ✅
4. $\alpha_{1,2}^{ij} = \alpha_{1,2}^{ji}$ ← 相同 MLP 作用于相同对称标量输入 ✅
5. $\alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2$ 项在对偶边上抵消 ✅
6. $\alpha_3^{ij} = v_r^{ij} \cdot g(\boldsymbol{\sigma}^{sym})$，$v_r$ 反对称 → $\alpha_3$ 反对称 → $\alpha_3 \mathbf{e}_3$ 抵消 ✅

总计 $\mathbf{F}_{ij} + \mathbf{F}_{ji} = \mathbf{0}$，对所有边求和得 $\sum_i \mathbf{F}_i = \mathbf{0}$ ✅

#### ⚠️ 需要补充的两个环节

**Gap 1：$\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$ 需要更严格的论证**

文档 claim "$\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$（标量特征对称）"。让我逐项验证 §2.3.1 中的 5 个标量分量：

| 标量分量 | 表达式 | $ij \to ji$ 变化 | 对称？ |
|---------|--------|-----------------|--------|
| $\|\mathbf{r}\|$ | $\|\mathbf{x}_j - \mathbf{x}_i\|$ | 不变 | ✅ |
| $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ | 径向速度 $v_r^{ij}$ | $v_r^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = v_r^{ij}$ | ✅ 对称 |
| $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$ | 切向速度 $v_t^{ij}$ | $v_t^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_2^{ij}) = v_t^{ij}$ | ✅ 对称 |
| $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ | 法向速度 $v_b^{ij}$ | $v_b^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (+\mathbf{e}_3^{ij}) = -v_b^{ij}$ | ❌ **反对称！** |
| $\|\dot{\mathbf{x}}_{ij}\|$ | 相对速率 | 不变 | ✅ |

**问题发现**：第 4 个标量分量 $v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ 是**反对称**的，因为 $\dot{\mathbf{x}}_{ij}$ 反号而 $\mathbf{e}_3$ 不反号。

这意味着 $\boldsymbol{\sigma}_{ij} \neq \boldsymbol{\sigma}_{ji}$，所以 $\alpha_{1,2}^{ij} \neq \alpha_{1,2}^{ji}$，整个 proof 的第 4 步不成立！

**修复方案**（三选一）：

**(a)** 在标量化时对 $v_b$ 取绝对值：$|v_b|$ 是对称的。但丢失了符号信息。

**(b)** 从标量特征中移除 $v_b$，只用 4 个标量 $[\|\mathbf{r}\|, v_r, v_t, \|\dot{\mathbf{x}}\|]$。但丢失了法向速度信息。

**(c)** 将 $v_b$ 同时用于 $\alpha_3$ 的反对称构造（如同 $v_r$ 的角色），但不进入 $\alpha_{1,2}$ 的 MLP。即：
$$\alpha_{1,2} = \text{MLP}_{\alpha}([\|\mathbf{r}\|, v_r, v_t, \|\dot{\mathbf{x}}\|, \mathbf{h}_i, \mathbf{h}_j])$$
$$\alpha_3 = v_r \cdot g_\theta([\|\mathbf{r}\|, |v_r|, v_t, |v_b|, \|\dot{\mathbf{x}}\|, \mathbf{h}_i, \mathbf{h}_j])$$

但等等——$v_r$ 自身呢？让我重新检查：

$v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$，我上面刚验证了 $v_r^{ji} = v_r^{ij}$（对称）。所以 $v_r$ 进入 $\alpha_{1,2}$ 的 MLP 是安全的。

但如果 $v_r$ 是对称的，那用 $v_r$ 乘以 $g(\cdot)$ 得到的 $\alpha_3$ 也是对称的，不是反对称的！

等等，让我重新算。文档说 "$v_r^{ij} = -v_r^{ji}$（which satisfies antisymmetry）"。让我仔细重算：

$$v_r^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} = v_r^{ij}$$

**$v_r$ 是对称的！** 两个负号抵消了！

**这是算法文档中的一个关键错误。** §2.4 的 $\alpha_3$ 修复依赖于 "$v_r^{ij}$ 是反对称的"这一假设，但实际上 $v_r^{ij} = v_r^{ji}$（对称），所以 $\alpha_3^{ij} = v_r^{ij} \cdot g(\boldsymbol{\sigma}^{sym}) = v_r^{ji} \cdot g(\boldsymbol{\sigma}^{sym}) = \alpha_3^{ji}$（也对称），$\alpha_3 \mathbf{e}_3$ 不抵消。

**⚠️ 严重度：致命。Proof 不成立。**

让我帮你找到正确的反对称标量。需要的是一个在交换 $i \leftrightarrow j$ 时变号的标量：

- $v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ → $v_b^{ji} = -v_b^{ij}$ ← **这个是反对称的！**

所以正确的修复应该是：

$$\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{sym})$$

其中 $\boldsymbol{\sigma}^{sym}$ 必须只包含对称标量：$[\|\mathbf{r}\|, v_r, v_t, \|\dot{\mathbf{x}}\|]$（注意：$v_r$ 和 $v_t$ 都是对称的）。

验证：$\alpha_3^{ji} = v_b^{ji} \cdot g_\theta(\boldsymbol{\sigma}^{sym}) = -v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{sym}) = -\alpha_3^{ij}$ ✅

那么 $\alpha_3^{ij} \mathbf{e}_3^{ij} + \alpha_3^{ji} \mathbf{e}_3^{ji} = \alpha_3^{ij} \mathbf{e}_3^{ij} + (-\alpha_3^{ij}) \mathbf{e}_3^{ij} = 0$ ✅ （因为 $\mathbf{e}_3^{ji} = \mathbf{e}_3^{ij}$）

**结论：用 $v_b$（不是 $v_r$）做反对称 marker。这是一个微妙但关键的错误。**

同时，$\alpha_{1,2}$ 的 MLP 中也不能包含 $v_b$（反对称标量会破坏 $\alpha_{1,2}$ 的对称性）。所以 $\alpha_{1,2}$ 的输入必须严格限制为：

$$\boldsymbol{\sigma}^{sym} = [\|\mathbf{r}\|, v_r, v_t, \|\dot{\mathbf{x}}\|, \mathbf{h}_i, \mathbf{h}_j]$$

不含 $v_b$。

**Gap 2：$\mathbf{h}_i, \mathbf{h}_j$ 的对称性**

Proof 假设 $\alpha_k^{ij} = \alpha_k^{ji}$，这依赖于 MLP 输入中 $[\mathbf{h}_i, \mathbf{h}_j]$ 在交换 $i \leftrightarrow j$ 时变为 $[\mathbf{h}_j, \mathbf{h}_i]$。

但如果 MLP 不是对输入的前后两半对称的，$\text{MLP}([\mathbf{h}_i, \mathbf{h}_j]) \neq \text{MLP}([\mathbf{h}_j, \mathbf{h}_i])$！

**修复方案**：使用节点特征的**对称聚合**：

$$[\mathbf{h}_i + \mathbf{h}_j, \; |\mathbf{h}_i - \mathbf{h}_j|, \; \mathbf{h}_i \odot \mathbf{h}_j]$$

或者更简单地：$[\mathbf{h}_i + \mathbf{h}_j, \; \mathbf{h}_i \odot \mathbf{h}_j]$

这样交换 $i \leftrightarrow j$ 后输入不变，保证 MLP 输出不变。

**严重度：高。当前 proof 只对 $\mathbf{h}_i = \mathbf{h}_j$ 或 MLP 恰好对称时成立。**

### A2. $\alpha_3$ Antisymmetrization — 遗漏审查

如上分析，**存在关键遗漏**：

| 问题 | 当前状态 | 正确做法 |
|------|---------|---------|
| $v_r$ 的对称性 | 文档误认为反对称 | $v_r^{ij} = v_r^{ji}$（对称），不能用作 $\alpha_3$ 的 sign flip |
| $v_b$ 的反对称性 | 未使用 | $v_b^{ij} = -v_b^{ji}$（反对称），应作为 $\alpha_3$ 的 sign flip |
| 当 $v_b \approx 0$ 时 | 未讨论 | $\alpha_3 \to 0$（合理：无法向运动意味着无法向力）|
| $\mathbf{h}_i, \mathbf{h}_j$ 排列对称性 | 未处理 | 需对称聚合：$\mathbf{h}_i + \mathbf{h}_j$ 或 $\mathbf{h}_i \odot \mathbf{h}_j$ |

**修正后的 $\alpha_3$ 公式**：

$$\alpha_3^{ij} = v_b^{ij} \cdot g_\theta\bigl([\|\mathbf{r}\|, v_r, v_t, \|\dot{\mathbf{x}}\|, \; \mathbf{h}_i + \mathbf{h}_j]\bigr)$$

**修正后的伪代码（替换 §4.2 中的关键行）**：

```python
# SCALARIZATION
v_r = dot(v_rel, e1)      # 对称 (用于 α₁₂ 输入)
v_t = dot(v_rel, e2)      # 对称 (用于 α₁₂ 输入)
v_b = dot(v_rel, e3)      # 反对称!!! (只用于 α₃ 的 sign flip)

scalars_sym = [d_ij, v_r, v_t, norm(v_rel)]           # ALL symmetric
h_sym = h[src] + h[dst]                                # 对称聚合
scalars = cat(scalars_sym, h_sym)

# SCALAR MLP
alpha12 = self.alpha12_mlp(scalars)                    # [α₁, α₂], symmetric
alpha3_mag = self.alpha3_mag_mlp(scalars)              # |α₃|, symmetric

# ANTISYMMETRIZE α₃ using v_b (NOT v_r!)
alpha3 = v_b * alpha3_mag                              # v_b is antisymmetric → α₃ antisymmetric ✅
```

### A3. Angular Momentum — 放弃是否合理？

**结论：合理，但论文中需要更好的论述。**

算法文档的 Remark 说：

> "Conservation of angular momentum additionally requires $\mathbf{F}_{ij} \parallel \mathbf{r}_{ij}$ (central forces), i.e., $\alpha_2 = \alpha_3 = 0$. This is overly restrictive for contact-rich manipulation (friction is tangential)."

**审查意见**：

**(a)** 物理上正确。Angular momentum conservation 要求力是 central force（沿连线方向），这排除了摩擦力。对于机器人操作（大量接触+摩擦），放弃角动量守恒是合理的工程取舍。✅

**(b)** 但需要讨论**什么时候**角动量守恒重要。例如：
- 自由空间中的多体系统（天体力学、分子模拟）→ 角动量守恒很重要
- 有外力矩的系统（机器人臂施加 torque）→ 角动量本来就不守恒
- 有摩擦的接触系统 → 切向力破坏角动量守恒

在我们的 PushBox 场景中，机器人臂施加的是外力矩，所以**即使我们的 GNN 满足角动量守恒，系统总角动量也不守恒**（因为有外力矩）。所以放弃是完全合理的。

**(c)** **建议在论文中**加一段话明确说明：
> "We note that our system is open (the robot arm applies external forces and torques), so neither linear nor angular momentum of the full system is conserved. What we conserve is the **internal interaction forces** between objects: $\sum_{j} \mathbf{F}_{ij}^{\text{internal}} + \sum_j \mathbf{F}_{ji}^{\text{internal}} = 0$. This encodes Newton's Third Law as an inductive bias, even though the total system momentum changes due to external actuation."

这一点很重要——可能有审稿人会问 "系统有外力，动量怎么可能守恒？" 需要提前回答。

### A4. Stop-Gradient — 是否过于保守？

**结论：合理的初始设计，但需要消融实验来验证。**

文档的设计：
- RL loss 只 backprop 到 Policy Stream + Fusion
- Physics Stream 只通过 $\mathcal{L}_{phys}$ 训练
- `sg(â_box)` 阻止 RL 梯度进入 Physics Stream

**赞同理由**：
1. 防止 RL 的高方差梯度污染物理模型的学习 ✅
2. 物理模型的目标是准确预测动力学，不是最大化 reward ✅
3. 如果不 stop-gradient，RL 可能学会 "hack" 物理预测（输出有利于策略的虚假加速度）

**质疑理由**：
1. 完全切断梯度意味着 Physics Stream **不知道哪些物理特征对 policy 有用**
2. $\mathcal{L}_{phys}$ 训练物理模型预测所有物体的加速度，但 policy 可能只需要 box 相对于 goal 的加速度分量
3. 一些论文（如 CURL, DrQ）发现 RL 梯度 + auxiliary loss 的联合训练效果更好

**建议**：这应该是一个**消融实验**，不是一个设计决策。需要对比：

| 配置 | 描述 |
|------|------|
| (a) Full stop-gradient | 当前设计。Physics stream 只通过 $\mathcal{L}_{phys}$ 训练 |
| (b) No stop-gradient | RL + physics loss 同时训练整个网络 |
| (c) Scaled gradient | Physics stream 收到 RL 梯度但乘以小系数 $\beta = 0.01$ |
| (d) Alternating | 偶数 epoch 用 RL loss，奇数 epoch 用 physics loss |

我预期 **(a)** 在 in-distribution 上略差（因为特征不 task-specific），但在 OOD 上更好（因为物理模型更通用）。**(c)** 可能是最优折中。

**这个消融应该加入实验计划。** 优先级 P1。

### A5. 其他算法审查意见

**A5.1 Degeneracy Handling（§2.2）**

文档提出了 fallback 链：relative velocity → gravity-aligned → $\hat{y}$。

**问题**：fallback 之间的切换是不连续的。当 $\|\mathbf{v}_{ij}^{\perp}\|$ 从 $\epsilon_{deg} + \delta$ 变到 $\epsilon_{deg} - \delta$ 时，$\mathbf{e}_2$ 突然从 velocity-based 跳到 gravity-based。这个不连续性会导致 $\alpha$ 值的跳变，可能引起训练不稳定。

**修复**：使用**smooth blending**：
$$\mathbf{e}_2^{ij} = w \cdot \mathbf{e}_2^{\text{vel}} + (1-w) \cdot \mathbf{e}_2^{\text{grav}}, \quad w = \sigma\bigl((\|\mathbf{v}^{\perp}\| - \epsilon_{\text{deg}}) / \tau\bigr)$$
然后重新正交化。$\sigma$ 是 sigmoid，$\tau$ 是温度参数。

**严重度：中。** 不影响理论正确性，但影响实际训练稳定性。

**A5.2 Physics Auxiliary Loss（§2.8.2）**

有限差分加速度 $\mathbf{a}_i^{fd} = (\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ 是 noisy 的（特别是 MuJoCo 的 5 substep 积分）。

**建议**：
- 使用多步差分减少噪声：$\mathbf{a}_i^{fd} = (\dot{\mathbf{x}}_i^{t+k} - \dot{\mathbf{x}}_i^t) / (k \Delta t)$
- 或使用 Savitzky-Golay 滤波
- 在 loss 中加权：离散化误差大的 transition 给低权重

**严重度：低-中。** 影响物理模型的训练质量，但不影响理论框架。

**A5.3 §7 Expected Results 过于保守**

文档 claim "2–5× sample efficiency improvement"（对比第一轮的 12.5×），这更诚实。但 "30–50% better OOD generalization" 仍需实验验证。

**建议**：在论文中不 claim 具体倍数，只说 "statistically significant improvement"，让数据说话。

---

## Part B：实验设计评审

### B1. 5 Seeds 够不够？

**结论：PushBox 上够了，Multi-Object 上可能不够。**

**PushBox（低方差环境）**：
- 2-DOF 臂 + 单箱 → 相对确定性的动力学
- 预期 5 seeds 的 success rate std ≤ 10%
- 5 seeds + Welch's t-test 足以检测 15% 以上的差异（power ~0.8 at α=0.05）
- ✅ 够了

**Multi-Object（高方差环境）**：
- 3-5 个箱子 → 初始条件组合爆炸
- 预期 std 可能 > 15%
- 5 seeds 可能 power 不足
- **建议**：Multi-Object 实验跑 **8-10 seeds**

**如果 5 seeds 的结果中两个方法差异 < 10% 且 p > 0.1**：
- 不能 claim "PhysRobot > PPO"
- 需要增加到 10 seeds 或 claim "comparable performance"

**统计学补充建议**：
- 除了 Welch's t-test，建议增加 **bootstrap confidence interval**（对 5 个点的 mean 做 10000 次 bootstrap resample）
- 效果量报告 **Cohen's d** 或 **rank-biserial correlation**

### B2. Baselines 对 ICRA/CoRL 够吗？

**ICRA**：

| Baseline | 有 | 够？ |
|----------|-----|------|
| Pure PPO (on-policy) | ✅ | ✅ |
| SAC (off-policy) | ✅ | ✅ |
| TD3 (off-policy) | ✅ | ✅（但和 SAC 有些重复） |
| GNS (graph, no physics) | ✅ | ✅ |
| HNN (energy conservation) | ✅ | ✅ 很好的物理对比 |
| Dreamer v3 (model-based) | ⚠️ P2 | ✅ 如果能做 |
| **缺少** | | |
| EGNN + PPO | ❌ | **必须加。** 最直接的等变 GNN 对比 |
| Param-matched MLP | ❌ | **强烈建议。** 证明不是参数量的功劳 |

**CoRL**（更高标准）：

上述全部 + 以下至少 1 个：
- **TD-MPC / TD-MPC2** — Model-based RL 的最新 SOTA
- **Equivariant Actor-Critic**（Wang et al., 2022）— 等变策略方法
- **Oracle with true physics** — Performance upper bound

**修改建议**：
1. **必须加 EGNN**。代码已有 PyG 依赖，实现 EGNN layer 只需 ~30 行。Priority P1。
2. **强烈建议加 param-matched MLP**。把 PPO 的 MLP 扩大到和 PhysRobot 相同参数量（~6K → 用 [128, 64] 的 net_arch）。如果 large MLP 能达到 PhysRobot 的性能，PhysRobot 的贡献就不存在了。
3. **TD3 可以降为 P2**。和 SAC 的 insight 重叠太多——两个 off-policy 方法在同一个简单环境上不会给出不同的 insight。省出的 GPU 时间给 EGNN。

### B3. OOD Evaluation 够吗？

**当前 OOD 设计（§3.1 M3）**：
- Mass: 7 points [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
- Friction: 5 points [0.1, 0.3, 0.5, 0.7, 1.0]
- Size: 4 points [0.03, 0.05, 0.07, 0.1]
- Goal distance: 5 points [0.1, 0.2, 0.3, 0.5, 0.7]

**评审意见**：

**(a)** Mass sweep 设计良好 ✅。7 个点 + 包含训练值 0.5 作为参考。范围 [0.1, 5.0] 覆盖 50× 变化。

**(b)** Friction 和 Size 的范围可能不够极端。建议：
- Friction 扩展到 [0.01, 2.0]（冰面到粗糙橡胶）
- Size 扩展到 [0.02, 0.15]（更大的变化范围）

**(c)** ⚠️ **缺少组合 OOD（compositional OOD）**。所有当前测试只变一个参数。审稿人可能要求：
- Mass + Friction 同时变化（2D grid）
- 例如：mass ∈ {0.25, 1.0, 5.0} × friction ∈ {0.1, 0.5, 1.0} = 9 个组合
- 这测试的是 "物理先验是否帮助理解参数之间的交互"，比单维度 OOD 更有说服力

**(d)** ⚠️ **缺少 structural OOD**。改变质量/摩擦是 parametric OOD。更强的测试：
- 改变物体形状（cube → sphere → cylinder）
- 改变桌面倾斜角
- 改变机器人运动学参数

建议至少加一个 structural OOD（如 cube → sphere）作为 "stretch test"。

**(e)** OOD robustness score（AUC 归一化）是好指标 ✅。但建议额外报告 **worst-case degradation**：$\min_{p} \text{SR}(p) / \text{SR}(p_{\text{train}})$。

### B4. 实验方案的其他审查意见

**B4.1 Variant 4（No-Conservation）标记为 "deferred to V3" — 这不能推迟**

这是**最关键的消融实验**。它回答的问题是："守恒律约束到底有没有用？"

如果没有这个消融，审稿人会问："你怎么知道不是 GNN 结构/相对特征/双流架构带来的提升，而是守恒律？" 唯一能回答这个问题的实验就是 Variant 4。

**建议**：将 Variant 4 提升到 **P0**，和核心实验同时跑。如果 V3 code 还没 ready，就用 V2（PhysRobot-V2 without any physics loss）作为 V4 的 proxy——这就是当前的 V3 = V4 degeneracy，但它 still 能给出 "no explicit conservation" 的 baseline 数据。

**B4.2 Variant 7（Symplectic Integrator）的实现有问题**

```python
# Symplectic Euler: p_new = p - dt*dH/dq, q_new = q + dt*dH/dp
```

在 RL 的单步特征提取场景中，symplectic integration 的优势（长时间能量守恒）无法体现。这和我在第一轮 critique 中对辛积分器的意见一致。

**建议**：降低到 P3 或移除。如果保留，必须设计 multi-step rollout 实验来展示辛积分器的长时间优势。

**B4.3 Dreamer v3 的实现估计过于乐观**

实验文档说 "Option A: Use official DreamerV3 repo, adapt to PushBox env (preferred)"。

实际上：DreamerV3 的 codebase 是 JAX-based，和 SB3 (PyTorch) 不兼容。适配 PushBox 需要：
- 将 env 封装为 DreamerV3 的 env API
- 可能需要图像 observation（DreamerV3 默认用 vision）
- 调参困难（DreamerV3 的超参对环境敏感）

**更实际的替代**：使用 **MBPO**（Model-Based Policy Optimization, Janner et al. 2019），PyTorch 实现现成。或者使用 **TD-MPC2**（Haarnoja et al. 2024），有 PyTorch 实现且更 recent。

**B4.4 缺少计算开销分析**

实验设计中有 M4（Training Time）和 M5（Parameter Count），但缺少：
- **推理延迟**（inference latency per step）— 对 real-time robot 控制很重要
- **GPU 内存占用** — 影响 scalability
- **FLOPs 分析** — 更客观的计算量度量

算法文档的 §5.3 有估算（PhysRobot ~0.20ms/step, <15% overhead），建议实验中**实际测量**并报告。

**B4.5 Missing: Reward Curve Comparison**

Learning curve（Fig 3 in figure plan）是实验部分最重要的图。但实验方案中没有明确说明：
- X 轴是 timesteps 还是 episodes 还是 wall-clock time？
- **建议**：报告三种（timesteps 作为主图，wall-clock 作为补充），因为不同方法的 step cost 不同

### B5. Multi-Object 环境的设计审查

**总体评价**：设计合理 ✅

**细节建议**：

1. **Success criteria "All boxes within 0.15m of respective goals"** 太严格。3 个箱子同时到位的概率 ≈ (单箱成功率)³。如果单箱 SR = 60%，三箱 = 21.6%。建议设置分级指标：
   - **Partial SR**: 至少 1 个箱子到位的比例
   - **Full SR**: 所有箱子到位的比例
   - **Average boxes placed**: 平均有几个箱子到位

2. **Mass distribution "Uniform [0.3, 1.0] kg per box"** — 这不是 OOD 测试。如果训练时就随机化了质量，那测试不同质量不算 OOD。OOD 应该测 [0.1, 0.3] 和 [1.0, 5.0] 范围。

3. **Environment 3 (Sorting)** 是一个很好的高难度任务，但可能太难了。建议先确保 Multi-3Box 跑通，再决定是否做 Sorting。

---

## Part C：综合评估

### 第一轮 → 第二轮的改进

| 第一轮问题 | 状态 | 评价 |
|-----------|------|------|
| 反对称性破坏 | 🟡 提出了 SV-pipeline 修复 | 方向正确但有 $v_r$ vs $v_b$ 错误 |
| 2-node graph 无意义 | 🟢 设计了 Multi-Object 环境 | 很好 |
| 论文 overclaim | 🟢 Expected results 更保守 | 很好 |
| Baseline 不够 | 🟢 6 baselines + 7 ablation variants | 大幅改善 |
| 统计显著性 | 🟢 5 seeds + Welch's t-test | 基本够 |
| OOD 测试不够 | 🟡 Mass + friction + size + goal dist | 缺组合 OOD 和 structural OOD |
| 辛积分器未使用 | 🟡 作为 Variant 7 | 优先级可降 |
| ee_vel = zeros | ⚠️ 未在算法文档中明确提及修复 | 需要确认 |

### 当前推荐评级变化

| Venue | 第一轮 | 第二轮（如果修复所有问题） |
|-------|--------|------------------------|
| **ICRA** | Weak Reject | **Borderline → Weak Accept** |
| **CoRL** | Reject | **Weak Reject → Borderline** |

进步明显，但以下 **3 个 blocking issues** 必须在投稿前解决：

### ⭐ Blocking Issues（必须修复才能投稿）

#### Blocking Issue 1：$v_r$ vs $v_b$ 对称性错误

$v_r^{ij} = v_r^{ji}$（对称），不能用作 $\alpha_3$ 的反对称 marker。应改用 $v_b^{ij} = -v_b^{ji}$（反对称）。这影响 §2.4 的 proof 和 §4.2 的伪代码。

**修复工作量**：~2 小时（改公式 + 改代码 + 改 proof）

#### Blocking Issue 2：$[\mathbf{h}_i, \mathbf{h}_j]$ 排列不变性

当前 MLP 输入 $[\mathbf{h}_i \| \mathbf{h}_j]$ 不具有排列对称性，导致 $\alpha_k^{ij} \neq \alpha_k^{ji}$。必须使用对称聚合 $\mathbf{h}_i + \mathbf{h}_j$ 或 $[\mathbf{h}_i + \mathbf{h}_j, \mathbf{h}_i \odot \mathbf{h}_j]$。

**修复工作量**：~1 小时

#### Blocking Issue 3：缺少 EGNN baseline

EGNN 是最直接的竞争方法。没有 EGNN 对比，审稿人会直接 desk reject。

**修复工作量**：~4 小时（实现 EGNN layer + 集成到 PPO + 5 seeds 实验）

### 建议的优先级调整

| 原优先级 | 修改后 | 原因 |
|---------|--------|------|
| Variant 4 (No-Conservation): P1 | **P0** | 最关键的消融 |
| EGNN baseline: 未列入 | **P0** | 审稿人必问 |
| TD3 baseline: P1 | **P2** | 与 SAC 重复 |
| Variant 7 (Symplectic): P2 | **P3 / 移除** | 单步特征提取中无意义 |
| Param-matched MLP: 未列入 | **P1** | 证明 GNN 结构的价值 |
| Compositional OOD: 未列入 | **P1** | 更强的 OOD 证据 |
| Stop-gradient 消融: 未列入 | **P1** | 验证设计决策 |
| Multi-Object 增加到 8-10 seeds | **P1** | 高方差环境需要更多 seeds |

---

## 附录：修正后的 Proof（供算法组参考）

**Theorem (Linear Momentum Conservation, Corrected).** *Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph with symmetric edge set ($\forall (i,j) \in \mathcal{E}, (j,i) \in \mathcal{E}$). For each edge $(i,j)$, define the force:*

$$\mathbf{F}_{ij} = \alpha_1^{ij}\,\mathbf{e}_1^{ij} + \alpha_2^{ij}\,\mathbf{e}_2^{ij} + \alpha_3^{ij}\,\mathbf{e}_3^{ij}$$

*where:*
- *$\mathbf{e}_1^{ij}, \mathbf{e}_2^{ij}$ are antisymmetric: $\mathbf{e}_k^{ji} = -\mathbf{e}_k^{ij}$ for $k=1,2$*
- *$\mathbf{e}_3^{ij}$ is symmetric: $\mathbf{e}_3^{ji} = +\mathbf{e}_3^{ij}$*
- *$\alpha_1^{ij}, \alpha_2^{ij}$ are symmetric in $(i,j)$: $\alpha_k^{ij} = \alpha_k^{ji}$, ensured by using the same MLP on permutation-invariant inputs $\boldsymbol{\sigma}^{sym}_{ij} = [\|\mathbf{r}_{ij}\|, v_r, v_t, \|\dot{\mathbf{x}}_{ij}\|, \mathbf{h}_i + \mathbf{h}_j]$ where $v_r, v_t$ are symmetric projections*
- *$\alpha_3^{ij}$ is antisymmetric in $(i,j)$: $\alpha_3^{ij} = -\alpha_3^{ji}$, ensured by $\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{sym}_{ij})$ where $v_b^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ satisfies $v_b^{ji} = -v_b^{ij}$*

*Then: $\sum_{i \in \mathcal{V}} \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij} = \mathbf{0}$.*

*Proof.* 将求和改写为对无序边对 $\{i,j\}$ 的求和：

$$\sum_i \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij} = \sum_{\{i,j\} \in \mathcal{E}/\sim} (\mathbf{F}_{ij} + \mathbf{F}_{ji})$$

对每个边对：

$$\mathbf{F}_{ij} + \mathbf{F}_{ji} = \underbrace{\alpha_1^{ij} \mathbf{e}_1^{ij} + \alpha_1^{ji} \mathbf{e}_1^{ji}}_{\alpha_1(\mathbf{e}_1 - \mathbf{e}_1) = 0} + \underbrace{\alpha_2^{ij} \mathbf{e}_2^{ij} + \alpha_2^{ji} \mathbf{e}_2^{ji}}_{\alpha_2(\mathbf{e}_2 - \mathbf{e}_2) = 0} + \underbrace{\alpha_3^{ij} \mathbf{e}_3^{ij} + \alpha_3^{ji} \mathbf{e}_3^{ji}}_{(\alpha_3 - \alpha_3) \mathbf{e}_3 = 0} = \mathbf{0}$$

因为每个边对贡献为零，总和为零。$\square$

*Remark 1.* 这对任意网络参数 $\theta$ 成立（硬约束）。

*Remark 2.* Angular momentum conservation 额外需要 $\alpha_2 = \alpha_3 = 0$（中心力），这在有摩擦的接触中过于限制。我们只保证线动量守恒。

*Remark 3.* 当 $v_b \to 0$ 时，$\alpha_3 \to 0$，法向力分量消失。这在物理上合理：没有法向相对运动意味着没有法向力（在弹性碰撞以外的场景）。但如果希望保留静态法向力，可以用 $\alpha_3^{ij} = s_{ij} \cdot g_\theta(\boldsymbol{\sigma}^{sym})$，其中 $s_{ij}$ 是任意反对称标量（如 $m_i - m_j$，若质量已知且不同）。

---

*第二轮审稿完成。总结：方向明确、进步明显。修复 $v_r \to v_b$ 错误、$\mathbf{h}$ 对称聚合、加 EGNN baseline，即可达到 ICRA Weak Accept 水平。*
