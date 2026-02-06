# PhysRobot Paper SPEC — Single Source of Truth

**Version**: 1.0
**Created**: 2026-02-06
**Authority**: This document is the **sole** authoritative reference for all shared parameters, architectural choices, and notation. All other documents (ALGORITHM_DESIGN.md, EXPERIMENT_DESIGN.md, PAPER_OUTLINE.md, code) MUST align to this SPEC. Any change MUST first be reflected here.

**Conflict resolution principle**: Code that passes tests > SPEC > ALGORITHM_DESIGN > EXPERIMENT_DESIGN > PAPER_OUTLINE.

---

## 1. 环境规格

| Property | Value | Source of truth |
|----------|-------|-----------------|
| **Robot DOF** | **2** (shoulder + elbow, planar arm) | `environments/push_box_env.py` |
| **Observation dim** | **16** | `environments/push_box_env.py` |
| **Action dim** | **2** (shoulder torque, elbow torque) | `environments/push_box_env.py` |
| **Action range** | [-10, 10] Nm | `environments/push_box_env.py` |
| **Episode length** | 500 steps | EXPERIMENT_DESIGN §2.1 |
| **Simulator substeps** | 5 | `environments/push_box_env.py` |
| **Success threshold** | **0.15 m** | EXPERIMENT_DESIGN §2.1 (V2) |
| **dt** | 0.002 s (MuJoCo timestep) × 5 substeps → 0.01 s control dt | MuJoCo XML |

### 1.1 Observation layout (16-dim)

| Index | Name | Dim | Description |
|-------|------|-----|-------------|
| 0–1 | `joint_pos` | 2 | Joint angles [shoulder, elbow] (rad) |
| 2–3 | `joint_vel` | 2 | Joint angular velocities (rad/s) |
| 4–6 | `ee_pos` | 3 | End-effector Cartesian position (m) |
| 7–9 | `box_pos` | 3 | Box center-of-mass position (m) |
| 10–12 | `box_vel` | 3 | Box linear velocity (m/s) |
| 13–15 | `goal_pos` | 3 | Goal target position (m) |

### 1.2 Physical parameters (default / training)

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Box mass | $m_{\text{box}}$ | 0.5 kg | OOD range: [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0] kg |
| Box friction | $\mu_{\text{box}}$ | 0.5 | OOD range: [0.1, 0.3, 0.5, 0.7, 1.0] |
| Box size (half-extent) | — | 0.05 m | OOD range: [0.03, 0.05, 0.07, 0.1] m |
| Arm link 1 length | $l_1$ | ~0.2 m | From MuJoCo XML |
| Arm link 2 length | $l_2$ | ~0.2 m | From MuJoCo XML |
| Gravity | $g$ | 9.81 m/s² | Standard |

### 1.3 Randomization (training initial conditions)

| Parameter | Range |
|-----------|-------|
| Joint angles | ±0.5 rad from default |
| Box x-position | [0.25, 0.45] m |
| Box y-position | [-0.15, 0.15] m |

### ⚠️ Resolved contradictions

| Contradiction ID | Issue | Resolution |
|------------------|-------|------------|
| **X4** (obs dim: 10 vs 16 vs 18) | `push_box.py` uses 10-dim; `push_box_env.py` uses 16-dim; paper outline says 18-dim | **16-dim** is canonical. `push_box.py` (10-dim) is deprecated. Paper outline must be corrected from 18 to 16. |
| **X5** (DOF: unspecified vs 2 vs 7) | Algorithm doc flexible; code 2-DOF; paper outline 7-DOF | **2-DOF** is the implemented system. Paper must describe 2-DOF. |
| **BUG-1** (two PushBoxEnv) | `push_box.py` and `push_box_env.py` conflict | Use **`push_box_env.py`** (16-dim) as the single environment. Merge or deprecate `push_box.py`. |

---

## 2. 网络架构规格

### 2.1 PPO Baseline

| Component | Specification |
|-----------|--------------|
| Policy network (`pi`) | MLP: 16 → 64 → ReLU → 64 → ReLU → 2 |
| Value network (`vf`) | MLP: 16 → 64 → ReLU → 64 → ReLU → 1 |
| Shared layers | None (separate `pi` and `vf` heads) |
| SB3 `net_arch` | `dict(pi=[64, 64], vf=[64, 64])` |
| Activation | ReLU |
| **Total params** | **~10K** |

### 2.2 GNS (Graph Network Simulator, V2 lightweight)

| Component | Specification |
|-----------|--------------|
| Node encoder | Linear(6 → 32) + ReLU |
| Edge encoder | Linear(4 → 32) + ReLU |
| GNS layer | 1× `GNSGraphLayerV2` (MessagePassing, hidden=32) |
| Decoder | Linear(32 → 3) |
| Feature projection | Linear(19 → 64) + ReLU |
| Edges | Bidirectional: `[[0,1],[1,0]]` (ee ↔ box) |
| **Total params** | **~5K** |

### 2.3 PhysRobot (SV) — Canonical architecture

**This section is authoritative. Derived from `physics_core/sv_message_passing.py` (verified, 24 tests passed).**

| Component | Specification | Code reference |
|-----------|--------------|----------------|
| **Physics Stream** | `SVPhysicsCore` | `sv_message_passing.py` L242–314 |
| Node encoder | MLP: 6 → 32 → LayerNorm → ReLU → 32 | `sv_message_passing.py` L275 |
| SV layers ($L$) | **1** (for 2-node PushBox) | `sv_message_passing.py` L278 |
| Hidden dim ($d_h$) | **32** | `sv_message_passing.py` L263 |
| Force MLP input | 5 scalars + 2×32 node embeddings = **69-dim** | `sv_message_passing.py` L144–148 |
| Force MLP output | 3 ($\alpha_1, \alpha_2, \alpha_3$) | `sv_message_passing.py` L148 |
| Force MLP hidden | 32 | `sv_message_passing.py` L145 |
| Node update MLP | (32 + 3) → 32 → 32 | `sv_message_passing.py` L151–154 |
| Physics stream params | **6,019** | Self-test output |
| **Policy Stream** | MLP: 16 → 64 → ReLU → 64 → ReLU | `sv_message_passing.py` L354–358 |
| Policy stream params | ~5K | |
| **Fusion** | Linear(64+3 → 64) + ReLU | `sv_message_passing.py` L361–364 |
| Fusion input | `[z_policy ‖ sg(â_box)]` (stop-gradient on physics) | `sv_message_passing.py` L418–421 |
| Fusion params | ~4K | |
| PPO heads (actor+critic) | Standard SB3 heads on top of 64-dim features | |
| **Total params (extractor)** | **15,619** | Self-test output |
| **Total params (with PPO heads)** | **~25K** (estimated) | |

### 2.4 Fusion mechanism — **FINAL DECISION: Stop-gradient concatenation**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Type** | `z = ReLU(W_f [z_policy ‖ sg(â_box)] + b_f)` | Simple, proven, implemented and tested |
| Stop-gradient | **Yes** — `z_physics.detach()` before fusion | Prevents RL loss from distorting physics dynamics |
| NOT cross-attention | Cross-attention was mentioned in paper outline but never implemented; stop-grad concat is what the tested code uses |
| NOT plain concat | We DO apply stop-gradient (unlike BUG-3 in Colab) |

### ⚠️ Resolved contradictions

| Contradiction ID | Issue | Resolution |
|------------------|-------|------------|
| **X3** (Fusion: stop-grad concat vs concat+ReLU vs cross-attention) | Algorithm says stop-grad concat; code had plain concat; paper said cross-attention | **Stop-gradient concatenation** — matches `sv_message_passing.py` L418–421. Paper outline and Colab must be corrected. |
| **X9** (Hidden dim: 64 vs 128 vs 32) | Algorithm recommends 64; old code 128; V2 code 32 | **32** for PushBox (2-node, lightweight). **64** for multi-object (5+ nodes). Code is the authority. |
| **X10** (MP layers: 1 vs 2 vs 3) | Algorithm recommends 2; old code 3; V2 code 1 | **1** for PushBox. **2** for multi-object. Matches code. |
| **CONSIST-2** (Fusion three-way conflict) | See X3 | See X3 resolution above |

---

## 3. 训练超参数

**All methods use identical PPO training hyperparameters for fair comparison.**

| Hyperparameter | Symbol | Value | Source |
|---------------|--------|-------|--------|
| Total timesteps | $T$ | **500,000** | EXPERIMENT_DESIGN §A.1 (V2) |
| Parallel envs | `n_envs` | 4 | EXPERIMENT_DESIGN §A.1 |
| Steps per rollout | `n_steps` | 2,048 | EXPERIMENT_DESIGN §A.1 |
| Batch size | — | 64 | EXPERIMENT_DESIGN §A.1 |
| PPO epochs per update | `n_epochs` | 10 | SB3 default |
| Learning rate | `lr` | 3×10⁻⁴ | SB3 default |
| Discount factor | $\gamma$ | 0.99 | Standard |
| GAE lambda | $\lambda_{\text{GAE}}$ | 0.95 | Standard |
| Clip range | $\epsilon_{\text{clip}}$ | 0.2 | SB3 default |
| Entropy coefficient | `ent_coef` | **0.01** | V2: encourage exploration |
| Value function coefficient | `vf_coef` | 0.5 | SB3 default |
| Max gradient norm | — | 0.5 | SB3 default |

### 3.1 PhysRobot-specific training parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Physics auxiliary loss weight | $\lambda_{\text{phys}} = 0.1$ | Physics loss = MSE(predicted acc, FD acc) |
| Physics loss warmup | $T_{\text{warmup}} = 50{,}000$ steps | Linear ramp from 0 to $\lambda_{\text{phys}}$ |
| Conservation regularizer weight | $\lambda_{\text{reg}} = 0.01$ | Soft energy-aware regularizer |

### ⚠️ Resolved contradictions

| Contradiction ID | Issue | Resolution |
|------------------|-------|------------|
| **X7** (Training steps: unspecified vs 16K vs 500K) | Algorithm doc unspecified; code hardcoded 16K; experiment plan 500K | **500,000** timesteps. BUG-4 (16K) must be fixed. |
| **X6** (Conservation loss: FD acc MSE vs none vs λ₁‖ΣF‖²+λ₂) | Algorithm uses FD acc MSE; code has none; paper outline uses different formula | **FD acceleration MSE** as primary physics loss (ALGORITHM_DESIGN §2.8.2). The ‖ΣF‖² term is unnecessary because SV-pipeline provides architectural conservation. Code must implement FD acc loss. |
| **BUG-4** (PhysRobot steps hardcoded 16K) | Produces only 2 PPO updates | Fix to 500K. |

---

## 4. 评估协议

### 4.1 In-distribution evaluation

| Parameter | Value |
|-----------|-------|
| Evaluation episodes | **100** per method per seed |
| Evaluation policy | Deterministic (no exploration noise) |
| Evaluation seeds | `range(10000, 10100)` — 100 fixed seeds |
| Training seeds | `[42, 123, 256, 789, 1024]` — 5 seeds |
| Evaluation frequency | Every 50K timesteps during training |

### 4.2 OOD evaluation (zero-shot, no fine-tuning)

| OOD Parameter | Training value | Test values | # Points |
|---------------|---------------|-------------|----------|
| Box mass | 0.5 kg | [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0] | 7 |
| Box friction | 0.5 | [0.1, 0.3, 0.5, 0.7, 1.0] | 5 |
| Box size | 0.05 m | [0.03, 0.05, 0.07, 0.1] | 4 |
| Goal distance | ~0.3 m | [0.1, 0.2, 0.3, 0.5, 0.7] | 5 |

### 4.3 Statistical analysis

| Method | Details |
|--------|---------|
| Central tendency | **Mean ± standard deviation** across 5 seeds |
| Significance test | **Welch's t-test**, significance at **p < 0.05** |
| Confidence interval | **95% CI** |
| Sample efficiency | **Median** across seeds (robust to outliers) |
| OOD robustness score | AUC of SR-vs-parameter curve, normalized by in-distribution SR |

### 4.4 Primary metrics

| ID | Metric | Definition |
|----|--------|-----------|
| M1 | **Success Rate (SR)** | Fraction of 100 eval episodes where box is within 0.15m of goal |
| M2 | **Sample Efficiency (SE)** | Timesteps to first success; timesteps to sustained >50% SR |
| M3 | **OOD Generalization** | SR on modified physics parameters (zero-shot) |
| M4 | **Training Time** | Wall-clock time on Colab T4 GPU |
| M5 | **Parameter Count** | `sum(p.numel() for p in model.parameters())` |

---

## 5. 数学符号表

**All documents and the paper MUST use these symbols consistently.**

### 5.1 Environment & state

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $N$ | Number of interacting bodies (nodes) | $\mathbb{Z}^+$ |
| $\mathbf{x}_i$ | Position of body $i$ | $\mathbb{R}^3$ |
| $\dot{\mathbf{x}}_i$ | Velocity of body $i$ | $\mathbb{R}^3$ |
| $\boldsymbol{\phi}_i$ | Intrinsic properties of body $i$ (mass, friction, geometry) | $\mathbb{R}^k$ |
| $\mathbf{s}_i^t$ | Full state of body $i$ at time $t$: $(\mathbf{x}_i^t, \dot{\mathbf{x}}_i^t, \boldsymbol{\phi}_i)$ | |
| $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ | Interaction graph at time $t$ | |
| $\mathcal{N}(i)$ | Neighbors of node $i$ in graph | |
| $r_{\text{cut}}$ | Contact graph cutoff radius | $\mathbb{R}^+$ |

### 5.2 Edge-local frame

| Symbol | Meaning | Symmetry under $i \leftrightarrow j$ |
|--------|---------|--------------------------------------|
| $\mathbf{r}_{ij}$ | Displacement: $\mathbf{x}_j - \mathbf{x}_i$ | **Antisymmetric** |
| $d_{ij}$ | Distance: $\|\mathbf{r}_{ij}\|$ | Symmetric |
| $\dot{\mathbf{x}}_{ij}$ | Relative velocity: $\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$ | Antisymmetric |
| $\mathbf{e}_1^{ij}$ | Radial unit vector: $\mathbf{r}_{ij} / d_{ij}$ | **Antisymmetric** |
| $\mathbf{v}_{ij}^{\perp}$ | Perpendicular velocity component | Antisymmetric |
| $\mathbf{e}_2^{ij}$ | Tangential unit vector: $\mathbf{v}_{ij}^{\perp} / \|\mathbf{v}_{ij}^{\perp}\|$ | **Antisymmetric** |
| $\mathbf{e}_3^{ij}$ | Binormal unit vector: $\mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$ | **Symmetric** (⚠️) |

### 5.3 SV-pipeline

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\boldsymbol{\sigma}_{ij}$ | Scalar invariants: $(d_{ij}, v_r, v_t, v_b, \|\dot{\mathbf{x}}_{ij}\|) \in \mathbb{R}^5$ | Rotation-invariant |
| $v_r$ | Radial relative velocity: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ | **Antisymmetric** under swap |
| $v_t$ | Tangential relative velocity: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$ | Symmetric |
| $v_b$ | Binormal relative velocity: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ | **Antisymmetric** |
| $\mathbf{h}_i \in \mathbb{R}^{d_h}$ | Node embedding of body $i$ | Learned |
| $\alpha_1, \alpha_2, \alpha_3$ | Scalar force coefficients | Output of force MLP |
| $\mathbf{F}_{ij}$ | Force on node $j$ due to node $i$: $\alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ | |
| $\mathbf{F}_i$ | Net force on node $i$: $\sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}$ | |
| $d_h$ | Hidden dimension of node embeddings | 32 (PushBox), 64 (multi-obj) |
| $L$ | Number of SV message-passing layers | 1 (PushBox), 2 (multi-obj) |

### 5.4 Dual-stream architecture

| Symbol | Meaning |
|--------|---------|
| $\mathbf{z}_{\text{policy}}$ | Policy stream output | 
| $\mathbf{z}_{\text{physics}}$ | Physics stream output (predicted box acceleration) |
| $\text{sg}(\cdot)$ | Stop-gradient operator |
| $\hat{\mathbf{a}}_i$ | Predicted acceleration of body $i$ |
| $\mathbf{a}_i^{\text{fd}}$ | Finite-difference acceleration: $(\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ |

### 5.5 Loss functions

| Symbol | Meaning | Formula |
|--------|---------|---------|
| $\mathcal{L}$ | Total loss | $\mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$ |
| $\mathcal{L}_{\text{RL}}$ | PPO clipped surrogate objective | Standard PPO (Schulman et al., 2017) |
| $\mathcal{L}_{\text{phys}}$ | Physics auxiliary loss | $\frac{1}{|\mathcal{B}|} \sum \|\hat{\mathbf{a}}_i - \mathbf{a}_i^{\text{fd}}\|^2$ |
| $\mathcal{L}_{\text{reg}}$ | Energy regularizer (soft) | Work-energy consistency term |
| $\lambda_{\text{phys}}$ | Physics loss weight | 0.1 (after warmup) |
| $\lambda_{\text{reg}}$ | Regularizer weight | 0.01 |
| $T_{\text{warmup}}$ | Physics loss warmup period | 50,000 steps |

### 5.6 Symmetry analysis summary

**This table resolves the core mathematical issue (Reviewer 2 MATH-1, MATH-2).**

The code in `sv_message_passing.py` uses the **undirected-pair processing** approach (lines 155–162), which avoids the $v_r$ vs $v_b$ debate entirely:

| Aspect | ALGORITHM_DESIGN.md approach | **Code (authoritative)** approach |
|--------|------------------------------|-----------------------------------|
| Edge processing | Process all directed edges $(i,j)$; need $\alpha_3$ to be antisymmetric | Process each **undirected pair** once ($i < j$); assign $+\mathbf{F}$ to $j$ and $-\mathbf{F}$ to $i$ |
| Conservation mechanism | Relies on $\alpha_k^{ij} = \alpha_k^{ji}$ for $k=1,2$ and $\alpha_3^{ij} = -\alpha_3^{ji}$ | **Hard-coded Newton's 3rd law**: `scatter_add(+F, j)` and `scatter_add(-F, i)` |
| $\alpha_3$ antisymmetrization | Multiply by $v_r$ (ALGORITHM_DESIGN) or $v_b$ (Reviewer 2 suggestion) | **Not needed** — each pair computed once; $\pm$ assignment handles antisymmetry |
| Node embedding symmetry | $[\mathbf{h}_i \| \mathbf{h}_j]$ is order-dependent (MATH-2 issue) | Use $\mathbf{h}_i + \mathbf{h}_j$ (symmetric) + $|\mathbf{h}_i - \mathbf{h}_j|$ (symmetric) |
| Correctness | Correct in theory if $v_r$ fix is applied (but fragile) | **Correct by construction** — Σ F = 0 verified for 100 random trials |

**Decision**: The **code approach** (undirected pairs + hard ±F assignment) is simpler, more robust, and verified. The paper should describe this approach. ALGORITHM_DESIGN.md §2.4's directed-edge proof is mathematically interesting but the code uses a cleaner method.

### ⚠️ Resolved contradictions

| Contradiction ID | Issue | Resolution |
|------------------|-------|------------|
| **MATH-1** ($v_r$ not antisymmetric) | Reviewer 2 pointed out $v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ sign analysis | **Moot**: code uses undirected-pair approach, doesn't rely on directed $\alpha_3$ antisymmetrization |
| **MATH-2** ($[\mathbf{h}_i, \mathbf{h}_j]$ order-dependence) | Concatenation is not permutation-invariant | **Fixed in code**: uses $\mathbf{h}_i + \mathbf{h}_j$ and $|\mathbf{h}_i - \mathbf{h}_j|$ (both symmetric). See `sv_message_passing.py` L215–216 |
| **X1** (Antisymmetry: SV-pipeline vs MLP+ReLU) | Old `edge_frame.py` uses MLP that breaks antisymmetry | **Resolved**: `sv_message_passing.py` doesn't use the old EdgeFrame. Conservation is guaranteed by ±F assignment, not by function antisymmetry. |
| **X2** ($\alpha_3$ marker choice) | $v_r$ vs $v_b$ debate | **Resolved**: undirected-pair approach eliminates the need for an antisymmetric marker entirely. |
| **BUG-2** (`check_antisymmetry()` never passes) | Old EdgeFrame encoder breaks antisymmetry | **Resolved**: `sv_message_passing.py` replaces the old EdgeFrame entirely. The old `check_antisymmetry()` is not used. |

---

## 6. 代码-论文映射

**Reference file**: `physics_core/sv_message_passing.py` (verified, all tests pass)

### 6.1 Core SV-Pipeline

| Paper Formula / Concept | Code Location | Lines | Notes |
|------------------------|---------------|-------|-------|
| **Edge frame construction** (§2.2): $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ | `build_edge_frames()` | L57–112 | Includes degeneracy fallback (z→y) |
| $\mathbf{e}_1^{ij} = \mathbf{r}_{ij} / \|\mathbf{r}_{ij}\|$ | `build_edge_frames()` | L73–75 | Radial unit vector |
| $\mathbf{v}_{ij}^{\perp}$, $\mathbf{e}_2^{ij}$ from rel. velocity | `build_edge_frames()` | L78–86 | With degeneracy check |
| $\mathbf{e}_3^{ij} = \mathbf{e}_1 \times \mathbf{e}_2$ | `build_edge_frames()` | L102 | Binormal (symmetric!) |
| **Scalarization** (§2.3.1): $\boldsymbol{\sigma}_{ij} \in \mathbb{R}^5$ | `SVMessagePassing.forward_with_forces()` | L198–206 | $[d, v_r, v_t, v_b, \|\mathbf{v}\|]$ |
| Node embedding symmetrization: $\mathbf{h}_i + \mathbf{h}_j$, $|\mathbf{h}_i - \mathbf{h}_j|$ | `SVMessagePassing.forward_with_forces()` | L209–216 | Resolves MATH-2 |
| **Scalar MLP** (§2.3.2): $(\alpha_1, \alpha_2, \alpha_3)$ | `self.force_mlp` | L144–148, L219 | 2-layer MLP with LayerNorm |
| **Vectorization** (§2.3.3): $\mathbf{F}_{ij} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ | `SVMessagePassing.forward_with_forces()` | L223–224 | Force reconstruction |
| **Newton's 3rd law**: $+\mathbf{F}$ to $j$, $-\mathbf{F}$ to $i$ | `scatter_add_` | L227–230 | Hard-coded conservation |
| **Node update** (residual): $\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \text{MLP}([\mathbf{h}^{(\ell)}, \mathbf{F}_i])$ | `SVMessagePassing.forward_with_forces()` | L233–234 | Residual connection |

### 6.2 Full Physics Stream

| Paper Concept | Code Location | Lines | Notes |
|---------------|---------------|-------|-------|
| **Node encoder**: $\mathbf{h}_i^{(0)} = \text{MLP}([\mathbf{x}_i, \dot{\mathbf{x}}_i])$ | `SVPhysicsCore.__init__` → `self.encoder` | L275 | 6→32→32 with LayerNorm+ReLU |
| **$L$ rounds of SV message passing** | `SVPhysicsCore.forward()` loop | L302–306 | Iterates over `self.sv_layers` |
| **Output: per-node forces** (NOT decoded through MLP) | `SVPhysicsCore.forward()` return | L308 | Direct force output preserves conservation |

### 6.3 Dual-Stream Architecture

| Paper Concept | Code Location | Lines | Notes |
|---------------|---------------|-------|-------|
| **Policy Stream**: MLP(obs) → $\mathbf{z}_{\text{policy}}$ | `PhysRobotFeaturesExtractorV3.policy_stream` | L354–358 | 16→64→64 with ReLU |
| **Physics Stream**: SV-GNN(graph) → $\hat{\mathbf{a}}_{\text{box}}$ | `PhysRobotFeaturesExtractorV3.physics_core` | L348–352 | SVPhysicsCore instance |
| **Obs → graph conversion** (16-dim → positions + velocities) | `_obs_to_graph()` | L372–389 | Extracts ee/box pos/vel |
| **Stop-gradient**: $\text{sg}(\hat{\mathbf{a}}_{\text{box}})$ | `z_physics_sg = z_physics.detach()` | L418 | Prevents RL→physics gradient |
| **Fusion**: $\mathbf{z} = \text{ReLU}(\mathbf{W}_f [\mathbf{z}_{\text{policy}} \| \text{sg}(\hat{\mathbf{a}}_{\text{box}})] + \mathbf{b}_f)$ | `self.fusion` | L361–364 | Linear(67→64)+ReLU |

### 6.4 Conservation Verification

| Paper Concept | Code Location | Lines | Notes |
|---------------|---------------|-------|-------|
| **Theorem: $\sum_i \mathbf{F}_i = \mathbf{0}$** | `verify_momentum_conservation()` | L434–471 | 100 random trials, N=4, tol=1e-4 |
| **Variable graph sizes** | Self-test section 3 | L488–499 | N=2,3,5,8 all pass |
| **Gradient flow verification** | Self-test section 4 | L502–514 | All active params have gradients |

### 6.5 Mapping from paper equations to code approach

> **Important**: The paper's momentum conservation proof (ALGORITHM_DESIGN §2.4) uses the directed-edge formulation with $\alpha_3$ antisymmetrization. The code uses a simpler and more robust **undirected-pair** approach. The paper should present the code's approach as the primary method, and optionally mention the directed-edge proof in an appendix.

| Paper proof step | Code implementation |
|------------------|-------------------|
| "For each directed edge $(i,j)$, compute $\mathbf{F}_{ij}$" | "For each **undirected pair** $(i,j)$ with $i<j$, compute one force $\mathbf{F}$" |
| "Show $\mathbf{F}_{ij} + \mathbf{F}_{ji} = 0$ via scalar/basis symmetry" | "Assign $+\mathbf{F}$ to $j$ and $-\mathbf{F}$ to $i$ **by construction**" |
| "Requires $\alpha_3^{ij} = -\alpha_3^{ji}$ (needs $v_r$ marker)" | "Not needed — each pair processed once" |
| "Requires $\boldsymbol{\sigma}_{ij}^{\text{sym}} = \boldsymbol{\sigma}_{ji}^{\text{sym}}$" | "Automatic — only one direction processed" |

---

## Appendix A: Cross-document contradiction resolution summary

This appendix lists all 12 contradictions identified in REVIEWER_FINAL_VERDICT.md and their resolutions.

| ID | Description | Resolution | Action required |
|----|-------------|-----------|-----------------|
| **X1** | Antisymmetry: SV-pipeline (algo) vs MLP+ReLU (code) vs "antisymmetric exchange" (paper) | Code's undirected-pair approach replaces old EdgeFrame. Conservation verified. | Update paper to describe undirected-pair method. |
| **X2** | $\alpha_3$ marker: $v_r$ (algo) vs unimplemented (code) vs unspecified (paper) | Moot — undirected-pair approach doesn't need a marker. | Algorithm doc footnote; paper describes code approach. |
| **X3** | Fusion: stop-grad concat (algo) vs concat+ReLU (code) vs cross-attention (paper) | **Stop-gradient concatenation** per `sv_message_passing.py`. | Fix paper outline; fix old Colab code. |
| **X4** | Obs dim: 10 (push_box.py) vs 16 (push_box_env.py) vs 18 (paper) | **16-dim**. Deprecate push_box.py. Fix paper. | Merge envs; fix paper. |
| **X5** | Robot DOF: unspecified (algo) vs 2 (code) vs 7 (paper) | **2-DOF**. Paper must match code. | Fix paper. |
| **X6** | Conservation loss: FD acc MSE (algo) vs none (code) vs λ‖ΣF‖² (paper) | **FD acceleration MSE** (architectural conservation makes ‖ΣF‖² unnecessary). | Implement FD acc loss in training code. |
| **X7** | Training steps: unspecified (algo) vs 16K (code) vs 500K (experiment) | **500,000**. | Fix BUG-4. |
| **X8** | EdgeFrame: rel. velocity (algo) vs up=[0,0,1] (code) vs "displacement + up" (paper) | **Relative velocity** per `build_edge_frames()`. Old code's up=[0,0,1] is deprecated. | No action (sv_message_passing.py is correct). |
| **X9** | Hidden dim: 64 (algo) vs 128/32 (code) vs unspecified (paper) | **32** (PushBox), **64** (multi-object). | Paper specifies both configs. |
| **X10** | MP layers: 2 (algo) vs 3/1 (code) vs 3 (paper) | **1** (PushBox), **2** (multi-object). | Paper specifies both configs. |
| **X11** | Missing Dynami-CAL citation in Related Work | **Must add**. Sharma & Fink (2025). | Writing team: add citation. |
| **X12** | Multi-object env: MuJoCo (experiment) vs analytical physics (code team) | **MuJoCo** for paper experiments (authoritative, reproducible). Analytical version for rapid prototyping only. | Confirm MuJoCo multi-object env as the experimental standard. |

## Appendix B: Architecture configs for different environments

### B.1 PushBox (2-node, primary benchmark)

```yaml
physics_stream:
  node_input_dim: 6
  hidden_dim: 32        # d_h
  n_layers: 1           # L
  params: 6,019

policy_stream:
  input_dim: 16
  layers: [64, 64]
  activation: ReLU

fusion:
  type: stop_grad_concat
  input_dim: 67          # 64 + 3
  output_dim: 64

total_extractor_params: 15,619
```

### B.2 Multi-Object (5-node, scaling benchmark)

```yaml
physics_stream:
  node_input_dim: 6
  hidden_dim: 64        # d_h (larger for more interactions)
  n_layers: 2           # L (deeper for multi-hop reasoning)
  params: ~46K          # estimated per ALGORITHM_DESIGN §5.1

policy_stream:
  input_dim: 16 + 6*(N-1)  # grows with number of boxes
  layers: [128, 128]
  activation: ReLU

fusion:
  type: stop_grad_concat
  input_dim: 128 + 3*N     # policy + per-node acc
  output_dim: 128

total_params: ~80K          # estimated
```

---

*End of SPEC. This document supersedes all other parameter/architecture specifications.*
*Next update: after V2 experiment results are available (ETA 2026-02-07).*
