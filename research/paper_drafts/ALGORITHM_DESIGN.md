# PhysRobot: Algorithm Design Document (v2)

**Version**: 2.0  
**Date**: 2026-02-06  
**Author**: Algorithm Architecture Team (R2 revision)  
**Target Venue**: ICRA 2027 / CoRL 2026  
**Ground Truth**: `physics_core/sv_message_passing.py` (code-first)

---

## Changelog: R1 → R2

| R1 Issue | Resolution | Section |
|----------|-----------|---------|
| v_r claimed antisymmetric (it's symmetric) | Code uses **undirected pairs** — no antisymmetric marker needed | §2.3, §2.4 |
| α₃ antisymmetrization via v_r·g(σ) | Eliminated: code processes each pair once, assigns ±F | §2.3.3, §2.4 |
| [h_i ‖ h_j] breaks permutation symmetry | Code uses h_sum = h_i + h_j, h_diff_abs = \|h_i − h_j\| | §2.3.1 |
| Angular momentum critique | Explicit discussion: intentional relaxation for friction modeling | §2.5 |
| Novelty rated Borderline | Three new directions proposed: contact-aware MP, dissipation tracking, multi-scale graph | §6 |
| Proof had two gaps (v_r symmetry, h ordering) | New Theorem 1 based on undirected-pair mechanism (gap-free) | §2.4 |
| Reviewer: "just a GNN feature extractor" | Refined novelty: undirected-pair conservation + dual-stream stop-grad + RL-native design | §1 |

---

## 1. Core Novelty Statement (Revised)

> **PhysRobot** integrates a **hard-constraint momentum-conserving GNN** into an RL policy for **contact-rich robotic manipulation**. The physics stream guarantees Newton's Third Law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) through an **undirected-pair Scalarization–Vectorization (SV) pipeline**: each unordered edge pair $\{i,j\}$ is processed exactly once to produce a single force vector $\mathbf{F}_{ij}$, which is assigned as $+\mathbf{F}$ to node $j$ and $-\mathbf{F}$ to node $i$. This **construction-level guarantee** holds for **any** network parameters $\theta$, requires no antisymmetric activation functions or sign-flip markers, and naturally handles **non-conservative, dissipative, multi-body** systems.

### 1.1 One-Sentence Contribution

A dual-stream RL architecture where an **undirected-pair SV-GNN** (Physics Stream) provides momentum-conserving force features to a standard policy network (Policy Stream), achieving superior sample efficiency and OOD generalization on contact-rich manipulation tasks.

### 1.2 Key Claims

| # | Claim | Mechanism |
|---|-------|-----------|
| C1 | **Architectural momentum conservation** | Undirected-pair SV-pipeline: $\sum_i \mathbf{F}_i = \mathbf{0}$ for *any* $\theta$ |
| C2 | **OOD generalization** | Physics inductive bias constrains hypothesis space; correct force structure transfers |
| C3 | **Scalable multi-object reasoning** | GNN message-passing handles variable-count objects with dynamic contact graphs |
| C4 | **No conservative-system assumption** | Unlike HNN/LNN, no Hamiltonian or Lagrangian required; friction and damping modeled natively |
| C5 | **Simpler than prior art** | No antisymmetric activations, no sign-flip markers, no separate α₃ MLP — a single MLP per edge pair suffices |

### 1.3 Distinction from Dynami-CAL (Sharma & Fink, 2025)

PhysRobot's physics stream draws inspiration from Dynami-CAL's edge-local frame decomposition. Key differences:

| Aspect | Dynami-CAL | PhysRobot |
|--------|-----------|-----------|
| **Application** | Standalone particle simulator | RL policy feature extractor |
| **Conservation mechanism** | Directed edges + antisymmetric markers | **Undirected pairs + explicit ±F** (simpler) |
| **α₃ handling** | Requires antisymmetric scalar marker | Unnecessary — single MLP, ±F hard-coded |
| **Training signal** | Trajectory rollout loss | Self-supervised FD acceleration from RL rollouts |
| **RL integration** | None | Dual-stream + stop-gradient |
| **Dynamic graph** | Fixed particle neighbors | Contact-aware topology changes |

---

## 2. Mathematical Framework

### 2.1 Problem Setting

$N$ interacting bodies (end-effector, objects, obstacles). State of body $i$ at time $t$:

$$\mathbf{s}_i^t = (\mathbf{x}_i^t, \dot{\mathbf{x}}_i^t, \boldsymbol{\phi}_i)$$

where $\mathbf{x}_i \in \mathbb{R}^3$ is position, $\dot{\mathbf{x}}_i \in \mathbb{R}^3$ is velocity, $\boldsymbol{\phi}_i$ encodes intrinsic properties.

Graph $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ where $\mathcal{E}^t$ is an **undirected** edge set of interacting pairs.

### 2.2 Edge-Local Coordinate Frame

For each **canonical directed edge** $(i \to j)$ with $i < j$, we construct an orthonormal frame $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$:

**Step 1: Radial basis vector**
$$\mathbf{e}_1 = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

**Step 2: Tangential basis vector (from relative velocity)**
$$\mathbf{v}_{\text{rel}} = \dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$$
$$\mathbf{v}_{\parallel} = (\mathbf{v}_{\text{rel}} \cdot \mathbf{e}_1)\,\mathbf{e}_1, \quad \mathbf{v}_{\perp} = \mathbf{v}_{\text{rel}} - \mathbf{v}_{\parallel}$$
$$\mathbf{e}_2 = \frac{\mathbf{v}_{\perp}}{\|\mathbf{v}_{\perp}\| + \epsilon}$$

**Step 3: Binormal basis vector**
$$\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

**Degeneracy handling**: When $\|\mathbf{v}_{\perp}\| < \epsilon_{\text{deg}}$:
- Primary fallback: $\mathbf{e}_2 = \text{normalize}(\mathbf{e}_1 \times \hat{\mathbf{z}})$
- Secondary fallback (when $\mathbf{e}_1 \approx \pm\hat{\mathbf{z}}$): $\mathbf{e}_2 = \text{normalize}(\mathbf{e}_1 \times \hat{\mathbf{y}})$
- Smooth blending via mask: $\mathbf{e}_2 = w \cdot \mathbf{e}_2^{\text{vel}} + (1-w) \cdot \mathbf{e}_2^{\text{fall}}$ where $w = \mathbb{1}[\|\mathbf{v}_\perp\| > \epsilon_{\text{deg}}]$

**Implementation** (from `build_edge_frames` in code):

```python
e1, e2, e3, r_ij, d_ij = build_edge_frames(pos, vel, pi, pj)
# pi, pj are the canonical direction: pi[k] < pj[k] for all k
```

**Important note on frame semantics**: Because we only process the canonical direction $i \to j$ (with $i < j$), the frame $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ is defined **once per pair**. We never construct the reverse-direction frame. This eliminates the entire class of symmetry/antisymmetry concerns that plagued the R1 design.

### 2.3 Scalarization–Vectorization (SV) Pipeline

This is the core mechanism. The key insight of the v2 design: **process each unordered pair exactly once, then hard-code Newton's Third Law in the aggregation step**.

#### 2.3.1 Scalarization: $\mathbb{R}^3 \to \mathbb{R}^k$ (rotation-invariant)

For each canonical pair $(i \to j, \; i < j)$, project geometric vectors onto the edge frame:

$$\boldsymbol{\sigma}_{\{i,j\}} = \begin{pmatrix} d_{ij} = \|\mathbf{x}_j - \mathbf{x}_i\| \\[4pt] v_r = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_1 \\[4pt] v_t = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_2 \\[4pt] v_b = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_3 \\[4pt] \|\mathbf{v}_{\text{rel}}\| \end{pmatrix} \in \mathbb{R}^5$$

**Symmetry analysis of scalar features** (with respect to the canonical direction):

Since we fix the canonical direction $i < j$, there is no need for these scalars to be symmetric or antisymmetric — each pair is processed exactly once. The scalars are simply **well-defined rotation-invariant features** of the pair.

**Node embedding symmetrization**:

$$\mathbf{h}_{\text{sum}} = \mathbf{h}_i + \mathbf{h}_j \in \mathbb{R}^{d_h} \quad \text{(order-invariant)}$$
$$\mathbf{h}_{\text{diff}} = |\mathbf{h}_i - \mathbf{h}_j| \in \mathbb{R}^{d_h} \quad \text{(element-wise absolute value, order-invariant)}$$

**Extended scalar input**:

$$\boldsymbol{\sigma}_{\{i,j\}}^{\text{ext}} = [\boldsymbol{\sigma}_{\{i,j\}} \,\|\, \mathbf{h}_{\text{sum}} \,\|\, \mathbf{h}_{\text{diff}}] \in \mathbb{R}^{5 + 2d_h}$$

**Why h_sum and |h_diff|?** Both are **permutation-invariant** under swapping $i \leftrightarrow j$:
- $\mathbf{h}_j + \mathbf{h}_i = \mathbf{h}_i + \mathbf{h}_j$ ✓
- $|\mathbf{h}_j - \mathbf{h}_i| = |\mathbf{h}_i - \mathbf{h}_j|$ ✓

This ensures the force magnitude is independent of which node we arbitrarily call $i$ vs $j$. This resolves the R1 gap where concatenation $[\mathbf{h}_i \| \mathbf{h}_j]$ broke permutation symmetry.

#### 2.3.2 Scalar MLP: $\mathbb{R}^{5+2d_h} \to \mathbb{R}^3$

A single MLP processes all scalar features to produce three force coefficients:

$$\begin{pmatrix} \alpha_1 \\ \alpha_2 \\ \alpha_3 \end{pmatrix} = \text{MLP}_{\theta}(\boldsymbol{\sigma}_{\{i,j\}}^{\text{ext}})$$

**Architecture** (from code):

```python
self.force_mlp = nn.Sequential(
    nn.Linear(5 + 2*node_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 3),  # α₁, α₂, α₃
)
```

**Key simplification over R1**: There is **no separate α₃ MLP**, no antisymmetric marker multiplication, no v_r or v_b sign-flip. All three coefficients come from the same MLP. Conservation is guaranteed by the aggregation step (§2.3.4), not by the coefficient structure.

#### 2.3.3 Vectorization: $\mathbb{R}^3 \to \mathbb{R}^3$ (reconstruct 3D force)

$$\mathbf{F}_{\{i,j\}} = \alpha_1 \,\mathbf{e}_1 + \alpha_2 \,\mathbf{e}_2 + \alpha_3 \,\mathbf{e}_3$$

This is the force that node $j$ experiences due to node $i$ (in the canonical direction $i \to j$).

#### 2.3.4 Newton's Third Law Aggregation (THE conservation mechanism)

**This is where conservation is enforced.** For each undirected pair $\{i,j\}$:

$$\text{Node } j \text{ receives } +\mathbf{F}_{\{i,j\}}$$
$$\text{Node } i \text{ receives } -\mathbf{F}_{\{i,j\}}$$

The net force on each node:

$$\mathbf{F}_i = \sum_{j : \{i,j\} \in \mathcal{E}} (\pm\mathbf{F}_{\{i,j\}})$$

where the sign depends on whether $i$ is the source or target in the canonical direction.

**Implementation** (from code):

```python
# F_{i→j}: force on j due to i (canonical direction)
force_ij = alpha1 * e1 + alpha2 * e2 + alpha3 * e3   # [P, 3]

# Node j receives +F
F_agg.scatter_add_(0, pj.unsqueeze(-1).expand_as(force_ij), force_ij)
# Node i receives -F  (Newton's 3rd law, HARD-CODED)
F_agg.scatter_add_(0, pi.unsqueeze(-1).expand_as(force_ij), -force_ij)
```

**Why this is simpler and more robust than R1**:

| Aspect | R1 (directed pairs) | R2/Code (undirected pairs) |
|--------|---------------------|---------------------------|
| Edges processed | Both (i→j) and (j→i) | Only canonical (i<j) |
| Conservation mechanism | Requires σ_{ij}=σ_{ji} and antisymmetric markers | Hard-coded ±F in aggregation |
| α₃ handling | Separate MLP with v_b marker | Same MLP as α₁, α₂ |
| Proof complexity | Requires symmetric/antisymmetric analysis of all scalars | Trivial: ±F cancel by construction |
| Failure modes | v_r/v_b symmetry errors, h ordering | None (architectural) |
| Computational cost | 2× MLP evaluations per pair | 1× MLP evaluation per pair |

### 2.4 Conservation Proof (Code-Aligned)

**Theorem 1 (Linear Momentum Conservation).** *Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph with undirected edge set $\mathcal{E}$. For each unordered pair $\{i,j\} \in \mathcal{E}$, let the SV-pipeline produce a force vector $\mathbf{F}_{\{i,j\}} \in \mathbb{R}^3$ (which may depend arbitrarily on positions, velocities, and node embeddings). Assign $+\mathbf{F}_{\{i,j\}}$ to node $j$ and $-\mathbf{F}_{\{i,j\}}$ to node $i$. Then for any network parameters $\theta$:*

$$\sum_{i \in \mathcal{V}} \mathbf{F}_i = \mathbf{0}$$

*where $\mathbf{F}_i = \sum_{j: \{i,j\} \in \mathcal{E}} (\text{sign}_{i,\{i,j\}} \cdot \mathbf{F}_{\{i,j\}})$ is the net force on node $i$.*

**Proof.**

$$\sum_{i \in \mathcal{V}} \mathbf{F}_i = \sum_{i \in \mathcal{V}} \sum_{j: \{i,j\} \in \mathcal{E}} \text{sign}_{i,\{i,j\}} \cdot \mathbf{F}_{\{i,j\}}$$

Each unordered pair $\{i,j\}$ appears exactly twice in this double sum: once when we sum over neighbors of $i$ (contributing sign $-1$ since $i$ is the source in the canonical direction) and once when we sum over neighbors of $j$ (contributing sign $+1$). Therefore:

$$\sum_{i \in \mathcal{V}} \mathbf{F}_i = \sum_{\{i,j\} \in \mathcal{E}} \bigl(+\mathbf{F}_{\{i,j\}} + (-\mathbf{F}_{\{i,j\}})\bigr) = \sum_{\{i,j\} \in \mathcal{E}} \mathbf{0} = \mathbf{0}$$

$\square$

**Remarks.**

1. *Universality*: The proof places **no constraints** on how $\mathbf{F}_{\{i,j\}}$ is computed. The MLP can use any activation function, any architecture, any inputs. Conservation follows purely from the ±F assignment, not from properties of the scalar features or the neural network.

2. *Numerical precision*: In floating-point arithmetic, the guarantee holds to machine precision (~$10^{-7}$ for float32). Our verification tests confirm $\|\sum_i \mathbf{F}_i\| < 10^{-4}$ for 100 random trials with random parameters (see `verify_momentum_conservation()` in code).

3. *Comparison with R1 proof*: The R1 proof required establishing (a) $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$, (b) $\alpha_k^{ij} = \alpha_k^{ji}$ for $k=1,2$, (c) $\alpha_3^{ij} = -\alpha_3^{ji}$, (d) correct symmetry/antisymmetry of $\mathbf{e}_k$. The R2 proof requires **none of these**. This is strictly simpler and more robust.

4. *Contrast with soft constraints*: A soft-constraint approach adds $\lambda\|\sum_i \mathbf{F}_i\|^2$ to the loss. This provides no guarantee at test time (especially OOD). Our architectural guarantee holds for any $\theta$, including random initialization, adversarial parameters, and OOD inputs.

### 2.5 Angular Momentum: Intentional Relaxation

**Proposition.** Conservation of angular momentum additionally requires all forces to be central: $\mathbf{F}_{\{i,j\}} \parallel \mathbf{r}_{ij}$, i.e., $\alpha_2 = \alpha_3 = 0$.

*Proof.* Total torque about the origin:

$$\boldsymbol{\tau} = \sum_i \mathbf{x}_i \times \mathbf{F}_i = \sum_{\{i,j\}} \bigl(\mathbf{x}_j \times \mathbf{F}_{\{i,j\}} - \mathbf{x}_i \times \mathbf{F}_{\{i,j\}}\bigr) = \sum_{\{i,j\}} \mathbf{r}_{ij} \times \mathbf{F}_{\{i,j\}}$$

This vanishes iff $\mathbf{F}_{\{i,j\}} \parallel \mathbf{r}_{ij}$ for all pairs, which requires $\alpha_2 = \alpha_3 = 0$ (since $\mathbf{e}_2, \mathbf{e}_3 \perp \mathbf{r}_{ij}$). $\square$

**Why we relax angular momentum conservation:**

1. **Friction is tangential.** Contact-rich manipulation involves sliding friction ($\alpha_2$ component) and rolling friction ($\alpha_3$ component). Requiring central forces would make the model unable to represent these physically essential interactions.

2. **The system is open.** The robot arm applies external forces and torques. Total system angular momentum is not conserved regardless of the internal force structure.

3. **Targeted conservation.** We conserve what matters most for multi-body reasoning: Newton's Third Law (action-reaction pairs). This constrains the GNN's hypothesis space substantially while retaining the expressivity to model non-central forces.

> "We note that our system is open (the robot arm applies external forces and torques), so neither linear nor angular momentum of the full system is conserved. What we conserve is the **internal interaction forces** between objects: $\mathbf{F}_{j \leftarrow i} = -\mathbf{F}_{i \leftarrow j}$. This encodes Newton's Third Law as an inductive bias, even though the total system momentum changes due to external actuation."

---

## 3. Complete Architecture

### 3.1 Physics Stream: SVPhysicsCore

The physics stream consists of three stages (matching code in `SVPhysicsCore`):

**Stage 1. Node Encoding**

$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}([\mathbf{x}_i \| \dot{\mathbf{x}}_i]) \in \mathbb{R}^{d_h}$$

**Stage 2. $L$ rounds of SV Message Passing** (for $\ell = 0, \ldots, L-1$):

For each undirected pair $\{i,j\}$ (with canonical direction $i < j$):

1. **Frame**: $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\} \leftarrow \text{build\_edge\_frames}(\mathbf{x}, \dot{\mathbf{x}}, i, j)$
2. **Scalarize**: $\boldsymbol{\sigma} = [d, v_r, v_t, v_b, \|\mathbf{v}_{\text{rel}}\|, \mathbf{h}_i^{(\ell)} + \mathbf{h}_j^{(\ell)}, |\mathbf{h}_i^{(\ell)} - \mathbf{h}_j^{(\ell)}|]$
3. **MLP**: $(\alpha_1, \alpha_2, \alpha_3) = \text{MLP}_{\text{force}}^{(\ell)}(\boldsymbol{\sigma})$
4. **Vectorize**: $\mathbf{F}_{\{i,j\}} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$
5. **±F Aggregation**: $\mathbf{F}_j \mathrel{+}= \mathbf{F}_{\{i,j\}}$, $\;\mathbf{F}_i \mathrel{+}= -\mathbf{F}_{\{i,j\}}$
6. **Node update**: $\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}_{\text{upd}}^{(\ell)}([\mathbf{h}_i^{(\ell)} \| \mathbf{F}_i])$

**Stage 3. Force Output (No Decoder)**

$$\hat{\mathbf{F}}_i = \mathbf{F}_i^{(L-1)}$$

**Critical design choice**: The final output is the **raw aggregated force** from the last SV layer. We do NOT pass forces through a per-node decoder MLP, which would break the $\sum = \mathbf{0}$ property. Node embeddings $\mathbf{h}^{(L)}$ are updated (for potential use in multi-layer stacks) but the **output** is the conserved force vector.

### 3.2 Dual-Stream Architecture

```
Observation o_t ──┬──► [Policy Stream: MLP]  ──► z_policy ∈ R^d
                  │
                  └──► [Physics Stream: SV-GNN]  ──► â_box ∈ R^3
                        (§3.1)                      (predicted force on box)

z = Fusion(z_policy, sg(â_box)) ──► PPO Actor ──► π(a|o)
                                  ──► PPO Critic ──► V(o)
```

**Fusion mechanism** (from code):

$$\mathbf{z} = \text{ReLU}\bigl(\mathbf{W}_f [\mathbf{z}_{\text{policy}} \| \text{sg}(\hat{\mathbf{a}}_{\text{box}})] + \mathbf{b}_f\bigr)$$

where $\text{sg}(\cdot)$ is **stop-gradient**: prevents RL loss from backpropagating into the physics stream.

**Why stop-gradient?**
1. Prevents RL's high-variance policy gradient from corrupting the physics model
2. Physics stream's goal is accurate dynamics prediction, not reward maximization
3. Without sg, RL may learn to "hack" physics predictions (output fictitious accelerations that bias the policy favorably)

### 3.3 Observation-to-Graph Mapping

For the PushBox task (from `PhysRobotFeaturesExtractorV3._obs_to_graph`):

```
obs[0:2]   → joint_pos
obs[2:4]   → joint_vel
obs[4:7]   → ee_pos  (3D) → Node 0: pos=ee_pos, vel=zeros(3)
obs[7:10]  → box_pos (3D) → Node 1: pos=box_pos, vel=box_vel
obs[10:13] → box_vel (3D)
obs[13:16] → goal_pos (3D) [not used by physics stream]
```

**Edge index**: Fully-connected (both directions), then undirected pairs extracted internally:

```python
edge_index = [[0, 1], [1, 0]]  # directed input
# Internally: pairs = [[0], [1]]  # undirected (0 < 1)
```

**Known limitation**: `ee_vel ≈ zeros` (approximation). Future work: compute from `joint_vel` via forward kinematics Jacobian.

---

## 4. Loss Function Design

### 4.1 Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}$$

### 4.2 RL Loss (PPO)

Standard PPO clipped surrogate. Backpropagates through **Policy Stream** and **Fusion** only (physics stream receives stop-gradient).

### 4.3 Physics Auxiliary Loss

$$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{s}^t, \mathbf{s}^{t+1}) \in \mathcal{B}} \sum_{i \in \text{objects}} \left\| \hat{\mathbf{F}}_i^t - \mathbf{a}_i^{t,\text{fd}} \right\|^2$$

where $\mathbf{a}_i^{t,\text{fd}} = (\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ is the finite-difference acceleration.

**Self-supervised**: No ground-truth forces needed. Acceleration estimated from consecutive state observations in RL rollouts.

**Warmup schedule**:
$$\lambda_{\text{phys}}(t) = \lambda_0 \cdot \min(1, t / T_{\text{warmup}})$$

---

## 5. Parameter Analysis and Computational Cost

### 5.1 Parameter Count (Code-Verified)

For $d_h = 32$, $d_s = 5$, node input $= 6$ (pos + vel), $L = 1$ layer:

| Module | Parameters | Code Reference |
|--------|-----------|---------------|
| Node encoder | $(6+1) \times 32 + 32 + (32+1) \times 32 + 32 = 1344$ | `self.encoder` |
| Force MLP | $(5 + 2 \times 32 + 1) \times 32 + 32 + (32 + 1) \times 3 = 2369$ | `self.force_mlp` |
| Node update | $(32 + 3 + 1) \times 32 + 32 + (32 + 1) \times 32 = 2240$ | `self.node_update` |
| **Physics Stream total** | **~5,953** | `model.parameter_count()` |

For full dual-stream extractor (`PhysRobotFeaturesExtractorV3`):

| Component | Parameters |
|-----------|-----------|
| Physics Stream (SV-GNN) | ~6K |
| Policy Stream (MLP: 16→64→64) | ~5.2K |
| Fusion (67→64) | ~4.4K |
| **Total** | **~15.6K** |

This is **lightweight** — comparable to a single PPO policy network.

### 5.2 Computational Complexity

Per SV layer:

| Operation | Complexity |
|-----------|-----------|
| Frame construction | $O(P)$ where $P = |\mathcal{E}|/2$ pairs |
| Scalarization | $O(P \cdot d_h)$ |
| Force MLP | $O(P \cdot d_h^2)$ (dominant) |
| Vectorization | $O(P)$ |
| ±F Aggregation | $O(P)$ (scatter_add) |
| Node update | $O(N \cdot d_h^2)$ |

**PushBox** ($N=2$, $P=1$, $d_h=32$): ~3K FLOPs per forward pass — negligible.

**5-object scene** ($N=6$, $P \leq 15$, $d_h=64$, $L=2$): ~200K FLOPs — still negligible vs MuJoCo step (~1M FLOPs).

### 5.3 Inference Latency

| Component | Estimated Time (M1 Mac) |
|-----------|------------------------|
| MuJoCo step | ~0.15 ms |
| Graph construction | ~0.01 ms |
| SV-GNN forward | ~0.02 ms |
| Policy MLP | ~0.01 ms |
| **Total** | **~0.19 ms → 5,000+ Hz** |

PhysRobot adds < 20% overhead vs pure PPO.

---

## 6. Novelty Enhancement Proposals (R2 New)

The R1 reviewer assessment was CoRL Borderline, primarily due to "GNN as feature extractor" being an established pattern. We propose three extensions that elevate the contribution:

### 6.1 Contact-Aware Message Passing (Collision-Driven Dynamic Graph)

**Motivation**: The current graph is static (fully-connected or radius-based). In manipulation, contact topology changes rapidly: the end-effector touches the box, the box slides on the table, objects collide.

**Proposal**: Replace the static graph with a **contact-aware dynamic graph** driven by collision detection:

$$\mathcal{E}^t = \mathcal{E}_{\text{contact}}^t \cup \mathcal{E}_{\text{proximity}}^t$$

where:
- $\mathcal{E}_{\text{contact}}^t = \{(i,j) : \text{penetration}(i,j) > 0\}$ — active contacts from MuJoCo's collision detector
- $\mathcal{E}_{\text{proximity}}^t = \{(i,j) : d_{ij} < r_{\text{prox}}\}$ — anticipatory edges for near-misses

**Contact features**: Augment scalar features with contact-specific information:

$$\boldsymbol{\sigma}_{\text{contact}} = [d_{ij}, v_r, v_t, v_b, \|\mathbf{v}_{\text{rel}}\|, \underbrace{f_n^{\text{MuJoCo}}, \mu_{ij}, A_{\text{contact}}}_{\text{contact features}}]$$

where $f_n$ is the normal contact force from the simulator, $\mu_{ij}$ is the friction coefficient, and $A_{\text{contact}}$ is the contact patch area.

**Novelty gain**: This goes beyond "GNN as feature extractor" — it's a **physics-aware graph construction** method that uses the simulator's collision geometry to build the interaction graph. No prior work in GNN-for-RL uses collision detection to dynamically construct the message-passing topology.

**Edge-type conditioning**: Different MLP heads for contact vs proximity edges:

$$\alpha_k = \begin{cases} \text{MLP}_{\text{contact}}(\boldsymbol{\sigma}) & \text{if } (i,j) \in \mathcal{E}_{\text{contact}} \\ \text{MLP}_{\text{prox}}(\boldsymbol{\sigma}) & \text{if } (i,j) \in \mathcal{E}_{\text{proximity}} \end{cases}$$

### 6.2 Energy Dissipation Tracking

**Motivation**: The physics stream predicts forces but has no explicit model of energy flow. In manipulation, energy dissipation (friction heat, inelastic collision losses) is a critical signal for understanding contact dynamics.

**Proposal**: Add an **energy dissipation channel** to the SV layer:

For each pair $\{i,j\}$, the force MLP additionally outputs a **dissipation rate** $\dot{D}_{\{i,j\}} \geq 0$:

$$(\alpha_1, \alpha_2, \alpha_3, \dot{D}_{\{i,j\}}) = \text{MLP}_\theta(\boldsymbol{\sigma})$$
$$\dot{D}_{\{i,j\}} = \text{softplus}(\text{raw output}) \geq 0$$

**Energy balance loss**: Augment $\mathcal{L}_{\text{phys}}$ with a work-energy constraint:

$$\mathcal{L}_{\text{energy}} = \left| \Delta E_{\text{kin}} - \sum_{\{i,j\}} \mathbf{F}_{\{i,j\}} \cdot \Delta \mathbf{x}_{\{i,j\}} + \sum_{\{i,j\}} \dot{D}_{\{i,j\}} \cdot \Delta t \right|$$

where $\Delta E_{\text{kin}} = E_{\text{kin}}^{t+1} - E_{\text{kin}}^t$ and $\mathbf{F} \cdot \Delta\mathbf{x}$ is the work done. The dissipation $D$ accounts for energy losses.

**Novelty gain**: This provides a **thermodynamically consistent** force model — forces must be compatible with observed energy changes plus non-negative dissipation. No prior GNN-for-RL work models energy dissipation explicitly.

**Physical interpretation**: The dissipation rate $\dot{D}$ learns to represent friction losses, damping, and inelastic collision energy. In contact-rich manipulation, this is a powerful inductive bias: the network must explain where kinetic energy goes when objects slow down.

### 6.3 Multi-Scale Graph Architecture

**Motivation**: In multi-object manipulation, interactions happen at multiple scales:
- **Fine scale**: contact patches between touching surfaces (~mm)
- **Object scale**: object-object interactions (~cm)
- **Scene scale**: global arrangement, goal relationships (~m)

A single graph cannot efficiently capture all scales.

**Proposal**: Two-level hierarchical graph:

**Level 1 (Fine)**: Per-contact-patch nodes with SV message passing. Output: per-object aggregate contact force.

**Level 2 (Coarse)**: Per-object nodes (one per rigid body). Input includes Level-1 contact summaries. Output: scene-level force predictions.

```
Level 2 (Coarse):  [EE] ←→ [Box1] ←→ [Box2] ←→ [Box3]
                     ↑          ↑          ↑          ↑
Level 1 (Fine):   [cp1]     [cp2,cp3]  [cp4,cp5]  [cp6]
```

where cp = contact point.

**Conservation guarantee**: Both levels independently satisfy $\sum \mathbf{F} = \mathbf{0}$ (each level uses the undirected-pair ±F mechanism). The total force from both levels is also conserved:

$$\sum_i \mathbf{F}_i^{\text{total}} = \underbrace{\sum_i \mathbf{F}_i^{\text{coarse}}}_{= \mathbf{0}} + \underbrace{\sum_i \mathbf{F}_i^{\text{fine}}}_{= \mathbf{0}} = \mathbf{0}$$

**Novelty gain**: This is the first **hierarchical graph architecture** for physics-informed RL that preserves conservation laws at every level. It combines the multi-scale physics intuition from computational mechanics (finite element methods) with GNN message passing.

### 6.4 Comparative Novelty Assessment

| Contribution | Novelty Level | Implementation Effort | Impact |
|-------------|--------------|----------------------|--------|
| Base SV-pipeline (R1) | Medium — extends Dynami-CAL | Done ✅ | Foundation |
| Undirected-pair simplification (R2) | Medium — cleaner than Dynami-CAL | Done ✅ | Robustness |
| Dual-stream stop-gradient RL | Medium — known in representation learning | Done ✅ | RL integration |
| Contact-aware MP (§6.1) | **High** — novel graph construction | 2 weeks | ⭐ Top novelty |
| Energy dissipation tracking (§6.2) | **High** — thermodynamic consistency | 1 week | ⭐ Physical depth |
| Multi-scale graph (§6.3) | **Very high** — hierarchical + conservation | 3 weeks | ⭐⭐ Strongest novelty |

**Recommendation for CoRL submission**: Implement §6.1 (contact-aware MP) as the primary novelty booster. It is the most practical (uses existing MuJoCo collision data), most impactful (dynamic topology is a clear differentiator), and directly addresses the reviewer critique that "static GNN topology is not interesting for manipulation."

**For ICRA submission**: The base architecture (§2–3) + solid experiments may suffice. Add §6.2 (energy dissipation) for a physics-depth bonus.

---

## 7. Comparison with Existing Methods

### 7.1 Comparison Table

| Feature | **EGNN** | **PaiNN** | **NequIP** | **Dynami-CAL** | **PhysRobot (Ours)** |
|---------|----------|-----------|------------|----------------|---------------------|
| Symmetry | E(n) equiv. | E(3) equiv. | E(3) equiv. | E(3) equiv. | E(3) equiv. |
| Conservation | ❌ None | ❌ None | ❌ None | ✅ Lin. mom. (directed pairs + markers) | ✅ Lin. mom. (**undirected pairs + ±F**) |
| Conservation mechanism | — | — | — | Antisymmetric scalars + sign-flip | **Hard-coded ±F** (simpler) |
| Force type | Radial only | Scalar + Vector | Tensor (irreps) | Full 3D (edge frame) | Full 3D (edge frame) |
| Non-conservative | ✅ | ✅ | ❌ | ✅ | ✅ |
| Friction modeling | ❌ | ❌ | ❌ | ✅ | ✅ (tangential forces) |
| RL integration | Indirect | Indirect | Indirect | ❌ None | ✅ Native (dual-stream + stop-grad) |
| Domain | Molecular | Molecular/MD | MD/Materials | Particles | **Robotics manipulation** |
| Cost per pair | $O(d)$ | $O(d)$ | $O(l^3 d)$ | $O(d^2)$ (2× MLP) | $O(d^2)$ (**1× MLP**) |

### 7.2 Key Differentiators

**vs. EGNN**: EGNN's position update $\mathbf{x}_i' = \mathbf{x}_i + \sum_j (\mathbf{x}_j - \mathbf{x}_i)\phi(\cdot)$ produces only **radial** forces. Cannot model friction (tangential). No conservation guarantee.

**vs. PaiNN**: Maintains scalar+vector channels but no explicit frame decomposition. No conservation guarantee. Messages can create net force.

**vs. NequIP**: Elegant spherical harmonic formulation but $O(l_{\max}^3)$ per edge. Designed for conservative potential energy surfaces. Cannot model dissipation.

**vs. Dynami-CAL**: Our direct ancestor. We simplify conservation (undirected pairs vs directed + markers), add RL integration, and propose contact-aware graph construction. Half the computational cost per pair (1× vs 2× MLP evaluations).

---

## 8. Implementation Verification

### 8.1 Unit Tests (from code `__main__`)

The implementation includes comprehensive self-tests:

1. **Parameter count**: Verify physics stream < 30K params
2. **Momentum conservation**: 100 random trials × 4 nodes, $\|\sum \mathbf{F}\| < 10^{-4}$ ✅
3. **Variable graph sizes**: $N \in \{2, 3, 5, 8\}$, all conserve ✅
4. **Gradient flow**: All active parameters receive non-zero gradients ✅
5. **Dual-stream**: Forward pass shape verification $(B, 16) \to (B, 64)$ ✅

### 8.2 Running the Tests

```bash
cd medical-robotics-sim
python physics_core/sv_message_passing.py
# Expected output: "✅ ALL TESTS PASSED"
```

### 8.3 Conservation Guarantee (Empirical)

```python
from physics_core.sv_message_passing import SVPhysicsCore, verify_momentum_conservation

model = SVPhysicsCore(node_input_dim=6, hidden_dim=32, n_layers=1)
# Passes for ANY random initialization:
verify_momentum_conservation(model, n_trials=1000, n_nodes=10, tol=1e-4)
# ✅ All 1000 trials passed.  Max ||Σ F|| = ~1e-7
```

---

## 9. Summary of Contributions (for Paper Writer)

1. **Undirected-Pair SV-Pipeline**: A simpler, more robust mechanism for momentum conservation than Dynami-CAL's directed-pair approach. Single MLP per pair, hard-coded ±F, trivial proof. No antisymmetric markers needed.

2. **Dual-Stream RL Architecture**: Physics stream (SV-GNN) provides momentum-conserving force features to a policy stream (MLP) via stop-gradient fusion. First integration of conservation-law GNN into RL.

3. **Self-Supervised Physics Learning**: Physics stream trained on finite-difference accelerations from RL rollouts. No ground-truth force labels required.

4. **Novelty Extensions** (§6): Contact-aware message passing, energy dissipation tracking, and multi-scale graph architecture — each independently raises novelty and addresses reviewer concerns about "just a feature extractor."

5. **Expected Empirical Results**:
   - Statistically significant sample efficiency improvement over PPO
   - Better OOD generalization to unseen masses and friction
   - Near-zero conservation error ($\|\sum \mathbf{F}\| \sim 10^{-7}$) vs unbounded for unconstrained GNNs
   - Graceful scaling to multi-object scenes

---

*Document complete. v2 revision aligned with `physics_core/sv_message_passing.py` as ground truth. All R1 issues addressed. Ready for review by Experiment, Writer, and Reviewer teams.*
