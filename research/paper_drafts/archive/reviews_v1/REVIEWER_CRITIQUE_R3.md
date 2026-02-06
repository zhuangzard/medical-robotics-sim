# PhysRobot — 第三轮审稿意见 (Reviewer 2, Round 3)

**审稿人**: Reviewer 2 (Devil's Advocate)  
**日期**: 2026-02-06  
**审稿对象**:
1. `paper_drafts/PAPER_OUTLINE.md` (新版论文大纲, 212 行)
2. `paper_drafts/RELATED_WORK.md` (完整 related work, 38 ref)

**上下文**: R1 critique → 算法组+实验组修改 → R2 critique (发现 $v_r$/$v_b$ 错误 + $\mathbf{h}$ 排列问题) → 写作组提交新大纲 + Related Work → 本轮 R3

---

## 总评

**写作组做了关键性的修正**：标题从 "Foundation Models" 改为 "Physics-Informed GNNs for Sample-Efficient Manipulation"——这是正确的定位转变，直接解决了 R1 中最严重的 overclaim 问题。Related Work 覆盖全面（40 篇），positioning 表格清晰。

但仍有若干问题需要修正。以下分三部分评审。

---

## Part 1：参考文献覆盖审查

### 1.1 已覆盖的关键领域 ✅

| 领域 | 代表文献 | 覆盖？ |
|------|---------|--------|
| PINNs | Raissi 2019 [1] | ✅ |
| HNN | Greydanus 2019 [2] | ✅ |
| LNN | Cranmer 2020 [3] | ✅ |
| DeLaN | Lutter 2019 [4] | ✅ |
| Port-Hamiltonian NN | [6] | ✅ |
| EGNN | Satorras 2021 [7] | ✅ |
| SEGNN | Brandstetter 2022 [8] | ✅ |
| GNS | Sanchez-Gonzalez 2020 [11] | ✅ |
| DimeNet | Gasteiger 2020 [14] | ✅ |
| PaiNN | Schütt 2021 [15] | ✅ |
| NequIP | Batzner 2022 [16] | ✅ |
| MACE | Batatia 2022 [17] | ✅ |
| Allegro | Musaelian 2023 [18] | ✅ |
| PPO/SAC/TD3 | [24][25][26] | ✅ |
| Dreamer v3 | Hafner 2023 [28] | ✅ |
| TD-MPC2 | Hansen 2024 [31] | ✅ |
| RT-2 | Brohan 2023 [32] | ✅ |
| Octo | 2024 [33] | ✅ |
| π₀ | Physical Intelligence 2024 [34] | ✅ |
| NRI | Kipf 2018 [22] | ✅ |
| Isaac Gym | Makoviychuk 2021 [27] | ✅ |
| Differentiable Physics RL | Heiden 2021 [36] | ✅ |
| Symmetry RL | van der Pol 2020 [39] | ✅ |
| Structured World Models | Kipf 2020 [40] | ✅ |

Related Work 在广度上是**优秀的**。40 篇文献覆盖了几乎所有审稿人可能提到的方向。

### 1.2 ⚠️ 遗漏的关键文献（必须补充）

以下文献直接与 PhysRobot 竞争或高度相关，遗漏任何一篇都可能被审稿人认为"没做 homework"：

#### 遗漏 1：**Dynami-CAL (Sharma & Fink, 2025)**

**这是最致命的遗漏。** PhysRobot 的 SV-pipeline 直接基于 Dynami-CAL 的边帧方法。可行性报告和算法设计文档反复引用此论文，但 Related Work 中**完全没有提及**。

任何审稿人看到 PhysRobot 的边帧设计后，第一反应就是 Google "antisymmetric edge frame GNN" → 找到 Dynami-CAL → 发现你没引用 → 认为你在隐瞒来源。

**必须做**：在 §2.1 或 §2.2 末尾添加一段专门讨论 Dynami-CAL，明确说明：
> "Our physics stream draws inspiration from the Dynami-CAL framework [X], which introduced antisymmetric edge-local coordinate frames for particle simulation. We extend their approach in three ways: (1) integration into an RL policy rather than a standalone simulator, (2) a corrected $\alpha_3$ antisymmetrization using the binormal velocity component, and (3) dynamic contact graphs for manipulation."

#### 遗漏 2：**Equivariant Transporter Networks / SE(3)-equivariant policies for manipulation**

近年来有多篇关于等变策略的 manipulation 论文，它们是 PhysRobot 最直接的竞争者之一：

- **Simeonov et al. (2023)** — "SE(3)-Equivariant Relational Rearrangement with Neural Descriptors" (CoRL 2023)
- **Ryu et al. (2023/2024)** — "Equivariant Descriptor Fields" for manipulation
- **Huang et al. (2022)** — "Equivariant Transporter Network" (RSS 2022)
- **Zhu et al. (2022)** — "Sample Efficient Grasp Learning Using Equivariant Models" (RSS 2022)
- **Wang et al. (2022)** — "On-Robot Learning With Equivariant Models" (CoRL 2022)

这些论文 argue "等变性足以带来样本效率提升"，PhysRobot 的反驳应该是"等变性只是必要条件，守恒律是额外的、更强的约束"。不引用它们就无法做这个对比。

**必须做**：在 §2.4 中添加一段讨论 equivariant manipulation policies。

#### 遗漏 3：**Graph-based RL for manipulation**

- **Li et al. (2019)** — "Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids" (ICLR 2019)
- **Lin et al. (2022)** — GNN-based manipulation policy
- **Driess et al. (2022)** — "Learning Multi-Object Dynamics with Compositional Neural Radiance Fields" (CoRL 2022) — 虽然不完全是 GNN，但关于 compositional object reasoning

**Li et al. 2019** 特别关键——它是 GNN 用于 manipulation 的开创性工作。必须引用。

#### 遗漏 4：**Constraint-based RL / Safe RL**

PhysRobot 的守恒律约束在形式上类似于 constrained RL（Constrained MDP）：

- **Altman (1999)** — Constrained Markov Decision Processes（经典教材）
- **Achiam et al. (2017)** — "Constrained Policy Optimization (CPO)" (ICML 2017)
- **Stooke et al. (2020)** — "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"

虽然 PhysRobot 的约束是架构层面的（不是 Lagrangian multiplier），但讨论与 constrained RL 的关系会加强 positioning。

#### 遗漏 5：**Residual Physics / Hybrid Models**

- **Ajay et al. (2019)** — "Augmenting Physical Simulators with Stochastic Neural Networks"
- **Heiden et al. (2021)** — "NeuralSim" [已引用，但不够]
- **Kloss et al. (2020)** — "Combining Learned and Analytical Models for Predicting Action Effects"

这些工作结合了解析物理和学习模型。PhysRobot 可以被视为这个大类的一个实例（物理 GNN = analytical prior，policy MLP = learned component）。

### 1.3 建议的补充引用优先级

| 优先级 | 文献 | 原因 |
|--------|------|------|
| **必须** | Dynami-CAL (Sharma & Fink 2025) | 方法直接基于此 |
| **必须** | Li et al. 2019 (particle dynamics manipulation) | GNN + manipulation 开创性工作 |
| **必须** | 至少 1 篇 equivariant manipulation policy (e.g., Huang 2022 or Simeonov 2023) | 最直接的竞争者 |
| **强烈建议** | Achiam 2017 (CPO) | 约束 RL 的关联 |
| **建议** | Ajay 2019 或 Kloss 2020 | Residual physics positioning |

补充后总引用量约 42-45 篇，对 8 页论文完全合理。

### 1.4 现有引用的准确性问题

| Ref # | 问题 | 修正 |
|-------|------|------|
| [2] | Greydanus 的共同作者应是 Dzamba 和 **Yosinski**（不是 Sprague） | 核实作者列表 |
| [6] | "Port-Hamiltonian Neural Networks" — Greydanus and Dzamba 2021 — 需要验证这是否是正确引用。更知名的 Port-Hamiltonian NN 论文可能是 Desai et al. 2021 或 Eidnes et al. 2023 | 核实 |
| [9] | 标注为 "Equivariant Polynomials" 但引用的是 Batzner et al. 2023。Batzner 的 2023 工作可能不叫这个名字。需要确认 | 核实 |
| [13] | "Learning Physics Simulations with Constraints" — Li et al. NeurIPS Workshop 2022 — 需要确认这篇论文是否存在。可能与 Huang et al. (2022) "Constrained Graph Dynamics" 混淆 | 核实 |
| [21] | LoCS 标注为 Kofinas et al. NeurIPS 2024，但论文描述为 "Learning on Continuous Structures"。需要确认标题和年份 | 核实 |
| [23] | 引用了 Battaglia 2016 (Interaction Networks) 但标注为 "C-GNS" (Compositional GNS)。这是两篇不同的论文！Battaglia 2016 是 Interaction Networks，C-GNS 可能指 Kossen et al. 2020 或类似工作 | **必须修正** |
| [38] | 引用的是 Li et al. 2021 (FNO)，但描述为 "Physics-Informed Neural Operator RL"。FNO 本身不是 RL 论文。如果有 follow-up，需要给出正确引用 | 核实 |

**Ref [23] 是一个明显的错误**——把 Interaction Networks (Battaglia 2016) 标记为 C-GNS。审稿人如果查到这个错误，会对整篇论文的 scholarly rigor 产生怀疑。必须修正。

---

## Part 2：Positioning 评审

### 2.1 新标题和故事线评估

**旧标题**: "Physics-Informed Foundation Models for Robotic Manipulation: Integrating Conservation Laws with Vision-Language-Action Models"

**新标题**: "PhysRobot: Physics-Informed Graph Neural Networks for Sample-Efficient Robot Manipulation"

**评价**: 大幅改善 ✅。新标题准确反映了实际贡献——不再 overclaim "foundation model"，聚焦于核心贡献（physics-informed GNN + sample efficiency）。

**建议微调**：
> "PhysRobot: Conservation-Aware Graph Neural Networks for Sample-Efficient Multi-Object Manipulation"

加入 "Conservation-Aware" 突出核心技术贡献（区别于普通的 physics-informed）；加入 "Multi-Object" 突出 GNN 的 scalability 优势。去掉 "Robot" 因为 manipulation 已经暗示了 robot。

### 2.2 Positioning Summary Table 评估

Related Work 末尾的表格设计很好 ✅。清晰地用 5 个维度（Physics Structure, Where Applied, Active Control, Multi-Object, Conservation）区分了所有方法。

**改进建议**：

1. 表格中缺少 **Dynami-CAL** 行。它应该是：
   | Dynami-CAL [X] | Conservation laws | Simulator | ❌ | ✅ | ✅ (hard) |
   
   这样 PhysRobot 和 Dynami-CAL 的区别就一目了然：**相同的 physics structure，但 PhysRobot 用于 policy，Dynami-CAL 用于 simulator**。

2. 建议增加一列 **"Sample Efficiency"** 维度，区分：
   - ❌ 需要 10⁶+ steps (PPO/SAC)
   - ⚠️ 中等 (Dreamer, model-based)
   - ✅ 高 (PhysRobot)

3. **Equivariant manipulation policies** 也应该出现在表格中（如果补充了引用）：
   | Equiv. Manip. [Y,Z] | SE(3) equivariance | Policy | ✅ | Limited | ❌ |

### 2.3 "Physics in Policy vs Physics in World Model" 的 positioning

这是新大纲中**最关键的 positioning insight**，出现在多处：

> "Unlike model-based methods that encode physics into the *world model*, we encode it into the *policy*."

**评价**：清晰且正确 ✅。但需要更深入地讨论这个选择的 trade-off：

**优势（论文应强调的）**：
- Policy 中的物理约束在**每一步推理**中都生效，不依赖 world model 的准确性
- World model 的物理约束只在 planning 阶段生效；policy 执行时没有约束
- 对 model-free RL 天然兼容（不需要学 world model）

**劣势（论文应诚实讨论的）**：
- Policy 中的物理约束不能用于 planning / imagination（Dreamer 式的 rollout）
- Policy 只输出当前步的动作，不能做多步物理推理
- 与 model-based RL 结合时，两份物理先验可能冲突

**建议**：在 §1.2 或 §5 (Conclusion/Limitations) 中加一段讨论 "Policy vs World Model" 的 trade-off。这展示了作者的深度思考，审稿人会 appreciate。

### 2.4 与算法文档的一致性检查

**⚠️ 存在多处不一致**：

| 维度 | 论文大纲 (PAPER_OUTLINE.md) | 算法文档 (ALGORITHM_DESIGN.md) | 不一致 |
|------|---------------------------|-------------------------------|--------|
| 物理约束类型 | "antisymmetric exchange: $m_{ij} = -m_{ji}$" | SV-pipeline with $\alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3$ | 大纲过于简化 |
| 守恒性 | "momentum conservation" + "energy channel" | 只做线动量守恒，能量是 soft regularizer | 大纲 overclaim "energy conservation" |
| EdgeFrame | "local frame from displacement + up-vector" | 从 relative velocity 构建 e2（不是 up-vector） | 大纲用旧方法 |
| 融合机制 | "PPO integration: per-node action embeddings → MLP" | "stop-gradient fusion: $\text{sg}(\hat{a}_{box})$" | 不同的设计 |
| Conservation loss | "$L_{cons} = \lambda_1 \|\sum F_i\|^2 + \lambda_2 \max(0, -\Delta E)$" | Physics aux loss: MSE(predicted acc, FD acc) | 不同的 loss 设计 |
| 环境设计 | "18-dim state, 4-dim action, 200 steps/ep" | "16-dim state, 2-dim action, 500 steps/ep" | 不匹配 |

**这些不一致必须在写正式论文前统一。** 当前的情况是：算法组设计了一个方案，写作组写了一个不同的方案，实验组又在用第三个方案跑实验。

**建议**：
1. 以算法文档为 ground truth（它是最新、最详细的设计）
2. 论文大纲中的所有技术描述必须与算法文档对齐
3. 特别注意：**不要在论文中 claim "energy conservation"**。当前设计只保证线动量守恒，能量只有 soft regularizer。

### 2.5 Abstract 评估

新 Abstract 的 5 个 key messages 结构清晰。但有问题：

**Claim "3–5× fewer environment steps"**：
- 这是算法文档中 "2–5×" 的乐观版本
- **仍然未经实验验证**
- 建议：写成 "significantly fewer environment steps"，投稿时用实际数字替换

**Claim "zero-shot to unseen object counts/masses"**：
- "zero-shot to unseen object counts" 意味着 train on 3 objects, test on 5-10
- 这需要 GNN 的可变图大小能力——当前代码未实现
- 对 "masses" 的 zero-shot 更容易验证
- 建议：如果 multi-object 实验没有跑通，降级为 "robust to unseen masses"

**Claim "First demonstration that embedding conservation-law structure directly into a GNN policy..."**：
- "First" 是一个 strong claim。需要确保文献中确实没有人做过这件事
- 即使没有完全相同的工作，如果有类似的（如 equivariant policy），审稿人可能 argue "not the first"
- 建议：加限定词 "To the best of our knowledge, the first..."

---

## Part 3：论文结构适合 ICRA/CoRL 吗？

### 3.1 结构评估

| Section | 页数预算 | 评价 |
|---------|---------|------|
| §1 Introduction | 1.5p | ✅ 合理 |
| §2 Related Work | 1p | ✅ 对 CoRL 合理（ICRA 可压缩到 0.75p）|
| §3 Method | 2p | ⚠️ 偏少——SV-pipeline + proof 需要空间 |
| §4 Experiments | 2p | ⚠️ 偏少——3 environments + ablation + OOD |
| §5 Conclusion | 0.5p | ✅ 合理 |
| Appendix | 补充 | ✅ |
| **Total** | **7p + ref** | 偏紧 |

**问题**：Method 2p + Experiments 2p = 4p，但实际内容量（SV-pipeline 数学 + proof + 3 environments + ablation + OOD）需要约 5p。这意味着要么：

**(a)** 压缩 Method 到 1.5p（把 proof 移到 appendix）
**(b)** 压缩 Experiments 到 1.5p（减少 to 2 environments）
**(c)** 压缩 Introduction + Related Work 到 2p（从 2.5p）

**建议采用 (a) + (c) 的混合**：
- Introduction 压缩到 1p（去掉 §1.5 Paper Organization，这在 8p 论文中浪费空间）
- Related Work 压缩到 0.75p（只留最关键的 positioning，详细讨论移到 appendix）
- Method 保留 2p（核心技术需要详细解释）
- Experiments 扩展到 2.5p（实验是论文最重要的部分）
- Conclusion 0.75p（包含 limitations）

### 3.2 ICRA vs CoRL 的结构差异

**CoRL 偏好**：
- 更强的 learning contribution（新算法 > 新 application）
- 更多实验（包括 ablation 和 analysis）
- Video 补充材料很重要
- 真实机器人实验加分很大（但不是必须）

**ICRA 偏好**：
- 可以偏 systems/engineering contribution
- 实验不需要那么多（但 baselines 必须足够）
- 更接受仿真-only 论文
- 更重视可复现性

**当前论文结构**：更适合 **CoRL**（learning-centric, 强调 sample efficiency 和 generalization）。但需要：
- 至少 5 seeds + significance test ← 实验组已计划 ✅
- 完整的 ablation ← 实验组已计划 ✅
- 补充视频（showing learned behaviors）← **尚未计划，建议添加**

### 3.3 §4 实验设计与实验组方案的一致性

**大纲 §4.1 的环境**：

| 大纲 | 实验组方案 | 一致？ |
|------|-----------|--------|
| PushBox: "18-dim state, 4-dim action" | "16-dim state, 2-dim action" | ❌ |
| PushBox: "7 DoF robot" | "2 DoF robot" | ❌ |
| PushBox: "200 steps/ep" | "500 steps/ep" | ❌ |
| MultiPush: 3-5 objects | Multi-3Box, Multi-5Box | ✅ 大致一致 |
| Sort: 2 colors, 4-6 objects | Sorting-3: 3 colored boxes | ⚠️ 规模不同 |

**大纲 §4.2 的 baselines**：

| 大纲 | 实验组 | 一致？ |
|------|--------|--------|
| PPO | PPO (B1) | ✅ |
| SAC | SAC (B2) | ✅ |
| GNS-Policy | GNS (B4) | ✅ |
| HNN-Policy (MPC) | HNN (B5, but as feature extractor, not MPC) | ⚠️ |
| PPO + Data Aug | 未列入 | ❌ 大纲有但实验组没有 |
| TD3 | TD3 (B3) | ✅ |
| Dreamer v3 | Dreamer (B6, P2) | ✅ |
| **EGNN** | **未列入** | ❌ 两边都缺 |

**严重问题**：大纲中的环境参数（7-DoF, 18-dim state）和实际代码（2-DoF, 16-dim state）完全不同。如果投稿时论文写的是 7-DoF 但实验跑的是 2-DoF，这是 **fatal inconsistency**。

**建议**：
1. **统一到实际实现**：论文中描述 2-DOF 环境，不要虚构 7-DoF
2. 如果要升级到 7-DoF（如 Franka Panda），需要实际实现和实验
3. 或者明确标注 "proof-of-concept with simplified 2-DOF system; extension to 7-DoF planned"

### 3.4 缺失的论文元素

以下在正式论文中必须包含但当前大纲没有：

1. **Limitations section**（大纲 §5 有提到，但太简略）
   - 当前 limitation "assumes ground-truth state" 是正确的
   - 还需要讨论：不保证角动量、$v_b \to 0$ 时 $\alpha_3$ 退化、dynamic graph construction overhead

2. **Supplementary video** — 对 CoRL 非常重要
   - 展示 learned pushing behavior（PPO vs PhysRobot）
   - 展示 OOD generalization（mass 变化时的行为差异）
   - 展示 multi-object scaling（3 → 5 → 10 objects）

3. **Code release plan** — CoRL 强烈鼓励 open-source
   - Anonymous GitHub repo for review
   - 完整的 reproduction scripts

4. **Computation budget** — 审稿人会问 "这要多少 GPU 时间"
   - 算法文档有估算（<15% overhead），需要实验验证
   - 报告每个方法的 training wall-clock time

---

## Part 4：综合评估与评级更新

### 改进追踪

| 第一轮问题 | R2 状态 | R3 状态 |
|-----------|---------|---------|
| 反对称性破坏 | 🟡 SV-pipeline 提出，但 $v_r$ 错误 | 🟡 写作组未改代码，依赖算法组修复 |
| Overclaim "Foundation Model" | — | 🟢 标题已修正 |
| 2-node graph 无意义 | 🟢 Multi-Object 计划 | 🟢 大纲中包含 |
| Baseline 不够 | 🟢 6 baselines | 🟡 仍缺 EGNN |
| 缺 Dynami-CAL 引用 | — | 🔴 Related Work 未引用 |
| 论文结构 | — | 🟡 基本合理但有一致性问题 |
| 定量 claim 未验证 | 🟢 更保守 (3-5×) | 🟡 仍未验证 |
| 环境参数不一致 | — | 🔴 大纲写 7-DoF，实际是 2-DoF |

### 当前评级

| Venue | R1 | R2（预测） | R3 |
|-------|-----|-----------|-----|
| **ICRA** | Weak Reject | Borderline → Weak Accept | **Borderline** (修复一致性后可升 Weak Accept) |
| **CoRL** | Reject | Weak Reject → Borderline | **Weak Reject** (需要更多工作) |

### ⭐ 投稿前必须完成的 Checklist

#### Blocking Issues（不修则不能投）

- [ ] **引用 Dynami-CAL**。不引用 = 学术不端嫌疑。
- [ ] **修正 Ref [23]**（Battaglia 2016 ≠ C-GNS）。
- [ ] **统一环境参数**（大纲/算法/实验/代码之间）。确定到底是 2-DOF 还是 7-DOF，然后所有文档一致。
- [ ] **修正 $v_r \to v_b$ 错误**（R2 已指出，算法组需执行，写作组需更新大纲中的对应公式）。
- [ ] **修正 $[\mathbf{h}_i, \mathbf{h}_j]$ 排列问题**（R2 已指出）。
- [ ] **加 EGNN baseline**。

#### Should-Fix（不修影响评分但不致命）

- [ ] 加 equivariant manipulation policy 引用（Huang 2022 或 Simeonov 2023）
- [ ] 加 Li et al. 2019 (particle dynamics manipulation) 引用
- [ ] 统一 conservation loss 设计（大纲 vs 算法文档）
- [ ] 核实所有引用的准确性（作者名、年份、标题）
- [ ] 加 param-matched MLP baseline
- [ ] 加 compositional OOD 实验
- [ ] 加 stop-gradient 消融

#### Nice-to-Have（加分项）

- [ ] 补充视频计划
- [ ] 讨论 "policy vs world model" trade-off
- [ ] 微调标题加入 "Conservation-Aware" 和 "Multi-Object"
- [ ] 在大纲中加入 computation budget 分析
- [ ] 讨论与 Constrained RL (CPO) 的关系

---

## 附录：完整 Consistency Matrix

为所有团队提供一个统一参考，标记当前文档之间的不一致：

| 参数 | 可行性报告 | 算法文档 | 实验文档 | 论文大纲 | 实际代码 | **应统一为** |
|------|-----------|---------|---------|---------|---------|------------|
| Robot DOF | 2 | 未指定 | 2 | 7 | 2 | **2（实际）** |
| State dim | 16 | 灵活 | 16 | 18 | 16 | **16** |
| Action dim | 2 | 灵活 | 2 | 4 | 2 | **2** |
| Episode length | 500 | 未指定 | 500 | 200 | 500 | **500** |
| Physics stream output | 加速度 (3D) | 加速度 via SV | 加速度 (3D) | "force-like messages" | 加速度 (3D) | **加速度** |
| Conservation type | 线动量 | 线动量(hard) + 能量(soft) | 线动量 | 线动量 + 能量 | 无(broken) | **线动量(hard)** |
| $\alpha_3$ marker | — | $v_r$ (**错**) | — | 未指定 | 未实现 | **$v_b$** |
| Node aggregation | — | $[\mathbf{h}_i \| \mathbf{h}_j]$ (**不对称**) | — | 未指定 | 未实现 | **$\mathbf{h}_i + \mathbf{h}_j$** |
| EdgeFrame reference | up = [0,0,1] | rel. velocity | — | "displacement + up-vector" | up = [0,0,1] | **rel. velocity** |
| Fusion method | concat + ReLU | stop-gradient + concat | — | "per-node embeddings → MLP" | concat + ReLU | **stop-gradient + concat** |
| Hidden dim | 128 | 64 recommended | 32 (V2) | 未指定 | 128 (broken), 32 (V2) | **64** |
| MP layers | 3 | 2 recommended | 1 (V2) | 3 | 3 (broken), 1 (V2) | **2** |
| Baseline count | 3 | — | 6 | 5 | 3 | **≥6 + EGNN = 7** |
| Seeds | — | — | 5 | 5 | 1 | **5 (PushBox), 8-10 (MultiObj)** |

**所有团队（算法/实验/写作）应以此表最右列为准，统一所有文档。**

---

*第三轮审稿完成。写作组的修改方向正确（标题改好、related work 全面），但存在致命的引用遗漏（Dynami-CAL）和跨文档一致性问题。这些是投稿前必须解决的 blocking issues。修复后，论文具备 ICRA Weak Accept 的潜力。*
