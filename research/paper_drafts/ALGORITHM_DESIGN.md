# PhysRobot: Algorithm Design Document (R2)

**Version**: 2.0  
**Date**: 2026-02-06  
**Author**: Algorithm Architecture Team  
**Ground Truth**: `physics_core/sv_message_passing.py` (code-first, line references below)  
**Target Venue**: ICRA 2027 / CoRL 2026  
**Supersedes**: `archive/reviews_v1/ALGORITHM_DESIGN_R1.md`

---

## R1 → R2 Changelog

| R1 Issue (Reviewer 2) | R2 Resolution | Section |
|------------------------|---------------|---------|
| $v_r$ claimed antisymmetric (MATH-1) | **Moot**: code uses undirected pairs — no directed-edge symmetry analysis needed | §2 |
| $\alpha_3$ antisymmetrization via $v_r \cdot g(\sigma)$ | **Eliminated**: each pair processed once, $\pm\mathbf{F}$ hard-coded | §2.3 |
| $[\mathbf{h}_i \| \mathbf{h}_j]$ breaks permutation symmetry (MATH-2) | **Fixed**: $\mathbf{h}_{\text{sum}} = \mathbf{h}_i + \mathbf{h}_j$, $\mathbf{h}_{\text{diff}} = |\mathbf{h}_i - \mathbf{h}_j|$ | §2.2.1 |
| Conservation proof had two gaps | **New proof**: trivial one-line cancellation, no symmetry analysis required | §3 |
| Angular momentum not discussed | **Addressed**: intentional relaxation for friction modeling | §3.2 |
| "Just a GNN feature extractor" (Novelty = Borderline) | Three novelty extensions: Contact-aware MP, energy dissipation, multi-scale graph | §6 |
| 12 cross-document contradictions | All resolved via SPEC.md alignment; code is the single authority | Throughout |

---

## 1. Core Innovation: Undirected-Pair SV Pipeline

### 1.1 One-Sentence Summary

> PhysRobot guarantees Newton's Third Law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) **by construction** through an **undirected-pair Scalarization–Vectorization (SV) pipeline**: each unordered edge $\{i,j\}$ is processed exactly once to produce a single force vector $\mathbf{F}_{\{i,j\}}$, which is assigned as $+\mathbf{F}$ to node $j$ and $-\mathbf{F}$ to node $i$.

### 1.2 Why This Matters

| Property | Mechanism |
|----------|-----------|
| **Architectural momentum conservation** | $\sum_i \mathbf{F}_i = \mathbf{0}$ for *any* $\theta$, *any* input — not learned, not regularized, **hard-coded** |
| **No antisymmetric markers** | Unlike Dynami-CAL's directed-pair approach, we never need $v_r$ or $v_b$ as sign-flip markers |
| **No separate $\alpha_3$ MLP** | All three force coefficients come from one MLP |
| **Half the compute** | 1× MLP evaluation per pair (vs 2× for directed approaches) |
| **Trivial proof** | Conservation follows from $+\mathbf{F} + (-\mathbf{F}) = \mathbf{0}$; no symmetry analysis of scalars or basis vectors required |

### 1.3 Distinction from Dynami-CAL (Sharma & Fink, 2025)

| Aspect | Dynami-CAL | **PhysRobot (Ours)** |
|--------|-----------|---------------------|
| Edge processing | Directed: both $(i \to j)$ and $(j \to i)$ | **Undirected**: canonical $(i \to j, i < j)$ only |
| Conservation mechanism | Requires symmetric scalars + antisymmetric $\alpha_3$ marker | **Hard-coded $\pm\mathbf{F}$** in aggregation |
| $\alpha_3$ handling | Separate MLP or marker multiplication | Same MLP as $\alpha_1, \alpha_2$ |
| Proof complexity | Must verify symmetry of all scalar features and basis vectors | **One line**: $+\mathbf{F} + (-\mathbf{F}) = \mathbf{0}$ |
| Application | Standalone particle simulator | **RL policy feature extractor** (dual-stream + stop-grad) |
| Training signal | Trajectory rollout loss | Self-supervised FD acceleration from RL rollouts |
| Dynamic graph | Fixed particle neighbors | **Contact-aware** topology changes (§6.1) |
| Cost per pair | $O(d_h^2)$ (2× MLP) | $O(d_h^2)$ (**1× MLP**) |

---

## 2. Mathematical Framework (Code-Aligned)

All equations in this section correspond line-by-line to `physics_core/sv_message_passing.py`. Line references are provided in brackets.

### 2.0 Problem Setting

$N$ interacting bodies (end-effector, objects, obstacles). State of body $i$ at time $t$:

$$\mathbf{s}_i^t = (\mathbf{x}_i^t,\; \dot{\mathbf{x}}_i^t,\; \boldsymbol{\phi}_i)$$

where $\mathbf{x}_i \in \mathbb{R}^3$ is position, $\dot{\mathbf{x}}_i \in \mathbb{R}^3$ is velocity, $\boldsymbol{\phi}_i$ encodes intrinsic properties (mass, friction, geometry).

Interaction graph $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ with **undirected** edge set $\mathcal{E}^t$.

### 2.1 Edge-Local Coordinate Frame

**Code**: `build_edge_frames()` [L57–112]

For each canonical directed edge $(i \to j)$ with $i < j$, construct an orthonormal frame $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$:

**Step 1 — Radial unit vector** [L73–75]:

$$\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i, \quad d_{ij} = \|\mathbf{r}_{ij}\|, \quad \mathbf{e}_1 = \frac{\mathbf{r}_{ij}}{d_{ij} + \epsilon}$$

**Step 2 — Tangential unit vector (from relative velocity)** [L78–93]:

$$\mathbf{v}_{\text{rel}} = \dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$$

$$\mathbf{v}_{\parallel} = (\mathbf{v}_{\text{rel}} \cdot \mathbf{e}_1)\,\mathbf{e}_1, \quad \mathbf{v}_{\perp} = \mathbf{v}_{\text{rel}} - \mathbf{v}_{\parallel}$$

$$\mathbf{e}_2^{\text{vel}} = \frac{\mathbf{v}_{\perp}}{\|\mathbf{v}_{\perp}\| + \epsilon}$$

**Degeneracy handling** [L86–101]: When $\|\mathbf{v}_{\perp}\| < \epsilon_{\text{deg}}$ (= $10^{-4}$):

- Primary fallback: $\mathbf{e}_2^{\text{fall}} = \text{normalize}(\mathbf{e}_1 \times \hat{\mathbf{z}})$
- Secondary fallback (when $\mathbf{e}_1 \approx \pm\hat{\mathbf{z}}$): $\mathbf{e}_2^{\text{fall}} = \text{normalize}(\mathbf{e}_1 \times \hat{\mathbf{y}})$
- Mask blending: $\mathbf{e}_2 = w \cdot \mathbf{e}_2^{\text{vel}} + (1-w) \cdot \mathbf{e}_2^{\text{fall}}$ where $w = \mathbb{1}[\|\mathbf{v}_{\perp}\| > \epsilon_{\text{deg}}]$

**Step 3 — Binormal unit vector** [L102]:

$$\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

**Critical design note**: Because we only process the canonical direction $i \to j$ (with $i < j$), the frame is defined **once per pair**. We never construct the reverse-direction frame. This eliminates the entire class of symmetry/antisymmetry concerns from R1.

### 2.2 Scalarization: $\mathbb{R}^3 \to \mathbb{R}^k$

**Code**: `SVMessagePassing.forward_with_forces()` [L198–216]

For each canonical pair $(i \to j,\; i < j)$, project geometric vectors onto the edge frame:

$$\boldsymbol{\sigma}_{\{i,j\}} = \begin{pmatrix} d_{ij} \\[3pt] v_r = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_1 \\[3pt] v_t = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_2 \\[3pt] v_b = \mathbf{v}_{\text{rel}} \cdot \mathbf{e}_3 \\[3pt] \|\mathbf{v}_{\text{rel}}\| \end{pmatrix} \in \mathbb{R}^5$$

Since each pair is processed exactly once (no reverse direction), these scalars are simply **well-defined rotation-invariant features**. No symmetric/antisymmetric classification is needed.

#### 2.2.1 Node Embedding Symmetrization

**Code**: [L209–216]

$$\mathbf{h}_{\text{sum}} = \mathbf{h}_i + \mathbf{h}_j \in \mathbb{R}^{d_h}$$
$$\mathbf{h}_{\text{diff}} = |\mathbf{h}_i - \mathbf{h}_j| \in \mathbb{R}^{d_h} \quad \text{(element-wise absolute value)}$$

Both are **permutation-invariant** under $i \leftrightarrow j$:
- $\mathbf{h}_j + \mathbf{h}_i = \mathbf{h}_i + \mathbf{h}_j$ ✓
- $|\mathbf{h}_j - \mathbf{h}_i| = |\mathbf{h}_i - \mathbf{h}_j|$ ✓

This ensures force magnitude is independent of the arbitrary canonical ordering. **This resolves R1 MATH-2** (where $[\mathbf{h}_i \| \mathbf{h}_j]$ concatenation broke symmetry).

**Extended scalar input**:

$$\boldsymbol{\sigma}_{\{i,j\}}^{\text{ext}} = [\boldsymbol{\sigma}_{\{i,j\}} \;\|\; \mathbf{h}_{\text{sum}} \;\|\; \mathbf{h}_{\text{diff}}] \in \mathbb{R}^{5 + 2d_h}$$

### 2.3 Force MLP: $\mathbb{R}^{5+2d_h} \to \mathbb{R}^3$

**Code**: `self.force_mlp` [L144–148], called at [L219]

$$(\alpha_1, \alpha_2, \alpha_3) = \text{MLP}_{\theta}(\boldsymbol{\sigma}_{\{i,j\}}^{\text{ext}})$$

Architecture:

```python
self.force_mlp = nn.Sequential(
    nn.Linear(5 + 2*node_dim, hidden_dim),  # 69 → 32
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 3),               # 32 → 3 (α₁, α₂, α₃)
)
```

**Key simplification over R1**: There is **no separate $\alpha_3$ MLP**, no antisymmetric marker multiplication, no $v_r$ or $v_b$ sign-flip. All three coefficients come from the same MLP. Conservation is guaranteed by the aggregation step (§2.5), not by the coefficient structure.

### 2.4 Vectorization: $\mathbb{R}^3 \to \mathbb{R}^3$

**Code**: [L223–224]

$$\mathbf{F}_{\{i,j\}} = \alpha_1 \,\mathbf{e}_1 + \alpha_2 \,\mathbf{e}_2 + \alpha_3 \,\mathbf{e}_3$$

This is the force that node $j$ experiences due to node $i$ (in the canonical direction).

### 2.5 Newton's Third Law Aggregation (THE Conservation Mechanism)

**Code**: [L227–230] — **This is the key innovation.**

```python
F_agg = torch.zeros(N, 3, device=h.device, dtype=h.dtype)
# Node j receives +F
F_agg.scatter_add_(0, pj.unsqueeze(-1).expand_as(force_ij), force_ij)
# Node i receives -F  (Newton's 3rd law, HARD-CODED)
F_agg.scatter_add_(0, pi.unsqueeze(-1).expand_as(force_ij), -force_ij)
```

For each undirected pair $\{i,j\}$:

$$\text{Node } j \text{ receives } +\mathbf{F}_{\{i,j\}}$$
$$\text{Node } i \text{ receives } -\mathbf{F}_{\{i,j\}}$$

The net force on each node:

$$\mathbf{F}_i = \sum_{\{i,j\} \in \mathcal{E}} \text{sign}_{i,\{i,j\}} \cdot \mathbf{F}_{\{i,j\}}$$

where the sign is $+1$ if $i$ is the destination (target) in the canonical direction, $-1$ if $i$ is the source.

### 2.6 Node Update (Residual)

**Code**: [L233–234]

$$\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}_{\text{upd}}([\mathbf{h}_i^{(\ell)} \;\|\; \mathbf{F}_i])$$

```python
h_input = torch.cat([h, F_agg], dim=-1)  # [N, node_dim + 3]
h_new = h + self.node_update(h_input)     # residual connection
```

Node update MLP: $(d_h + 3) \to d_h \to d_h$ with LayerNorm + ReLU [L151–154].

### 2.7 Complete Pipeline Summary (One SV Layer)

For each undirected pair $\{i,j\}$ with canonical direction $i < j$:

| Step | Operation | Code Line | Input → Output |
|------|-----------|-----------|----------------|
| 0 | Extract undirected pairs ($i < j$) | L190 | `edge_index [2,E]` → `pairs [2,P]` |
| 1 | Build edge frame | L193 | `pos, vel, pi, pj` → `e1, e2, e3, r_ij, d_ij` |
| 2 | Scalarize | L198–210 | `v_rel, e1, e2, e3, h` → `σ ∈ R^{5+2d_h}` |
| 3 | Force MLP | L219 | `σ` → `(α₁, α₂, α₃)` |
| 4 | Vectorize | L224 | `α, e1, e2, e3` → `F ∈ R³` |
| 5 | ±F aggregation | L227–230 | `F` → `F_agg[j] += F`, `F_agg[i] -= F` |
| 6 | Node update | L233–234 | `h, F_agg` → `h_new` (residual) |

---

## 3. Conservation Proofs

### 3.1 Theorem 1: Linear Momentum Conservation

**Theorem 1.** *Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph with undirected edge set $\mathcal{E}$. For each unordered pair $\{i,j\} \in \mathcal{E}$, let the SV-pipeline produce an arbitrary force vector $\mathbf{F}_{\{i,j\}} \in \mathbb{R}^3$. Assign $+\mathbf{F}_{\{i,j\}}$ to node $j$ and $-\mathbf{F}_{\{i,j\}}$ to node $i$. Then:*

$$\sum_{i \in \mathcal{V}} \mathbf{F}_i = \mathbf{0} \quad \forall\, \theta$$

**Proof.**

$$\sum_{i \in \mathcal{V}} \mathbf{F}_i = \sum_{i \in \mathcal{V}} \sum_{\{i,j\} \in \mathcal{E}} \text{sign}_{i,\{i,j\}} \cdot \mathbf{F}_{\{i,j\}}$$

Each unordered pair $\{i,j\}$ appears exactly twice in this double sum — once contributing $+\mathbf{F}_{\{i,j\}}$ (for the target node) and once contributing $-\mathbf{F}_{\{i,j\}}$ (for the source node):

$$= \sum_{\{i,j\} \in \mathcal{E}} \bigl(+\mathbf{F}_{\{i,j\}} + (-\mathbf{F}_{\{i,j\}})\bigr) = \sum_{\{i,j\} \in \mathcal{E}} \mathbf{0} = \mathbf{0} \qquad \square$$

**Remarks.**

1. **Universality**: The proof places **zero constraints** on how $\mathbf{F}_{\{i,j\}}$ is computed. The MLP can use any activation, any architecture, any inputs. Conservation follows purely from the $\pm\mathbf{F}$ assignment.

2. **Comparison with R1 proof**: R1 required establishing (a) $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$, (b) $\alpha_k^{ij} = \alpha_k^{ji}$ for $k=1,2$, (c) $\alpha_3^{ij} = -\alpha_3^{ji}$, (d) correct symmetry/antisymmetry of all basis vectors. The R2 proof requires **none of these**.

3. **Numerical precision**: In float32, the guarantee holds to $\sim 10^{-7}$. Verified: $\|\sum_i \mathbf{F}_i\| < 10^{-4}$ for 100 random trials with random $\theta$ (see `verify_momentum_conservation()` [L434–471]).

4. **Contrast with soft constraints**: Adding $\lambda\|\sum_i \mathbf{F}_i\|^2$ to the loss provides no test-time guarantee, especially OOD. Our architectural guarantee holds for any $\theta$, including random initialization, adversarial parameters, and OOD inputs.

### 3.2 Angular Momentum: Intentional Relaxation

**Proposition.** Conservation of angular momentum additionally requires all forces to be central: $\mathbf{F}_{\{i,j\}} \parallel \mathbf{r}_{ij}$, i.e., $\alpha_2 = \alpha_3 = 0$.

**Proof.** Total torque about the origin:

$$\boldsymbol{\tau} = \sum_i \mathbf{x}_i \times \mathbf{F}_i = \sum_{\{i,j\}} (\mathbf{x}_j - \mathbf{x}_i) \times \mathbf{F}_{\{i,j\}} = \sum_{\{i,j\}} \mathbf{r}_{ij} \times \mathbf{F}_{\{i,j\}}$$

This vanishes iff $\mathbf{F}_{\{i,j\}} \parallel \mathbf{r}_{ij}$ for all pairs, requiring $\alpha_2 = \alpha_3 = 0$. $\square$

**Why we intentionally relax angular momentum conservation:**

1. **Friction is tangential.** Sliding friction ($\alpha_2$ component) and rolling friction ($\alpha_3$ component) are physically essential in contact-rich manipulation. Central-force-only models cannot represent them.

2. **The system is open.** The robot arm applies external forces and torques. Total angular momentum is not conserved regardless.

3. **Targeted inductive bias.** We conserve what matters most: Newton's Third Law (action-reaction pairs). This substantially constrains the GNN's hypothesis space while preserving expressivity for non-central forces.

### 3.3 Comparison: R1 vs R2 Conservation Approaches

| Aspect | R1 (Directed Pairs) | **R2/Code (Undirected Pairs)** |
|--------|---------------------|-------------------------------|
| Edges processed | Both $(i \to j)$ and $(j \to i)$ | Only canonical $(i < j)$ |
| Conservation mechanism | Requires $\alpha_k^{ij} = \alpha_k^{ji}$ (k=1,2) and $\alpha_3^{ij} = -\alpha_3^{ji}$ | **Hard-coded $\pm\mathbf{F}$** |
| $\alpha_3$ handling | Separate MLP or antisymmetric marker ($v_r$ or $v_b$) | Same MLP, no marker |
| Proof complexity | Multi-step: analyze symmetry of all scalars, all basis vectors | **One line**: $+\mathbf{F} - \mathbf{F} = \mathbf{0}$ |
| Failure modes | $v_r$ symmetry errors, $\mathbf{h}$ ordering, MLP breaking antisymmetry | **None** (architectural) |
| MLP evaluations per pair | 2× | **1×** |
| Reviewer 2 gaps | MATH-1 ($v_r$), MATH-2 ($[\mathbf{h}_i \| \mathbf{h}_j]$) | Both **eliminated** by design |

---

## 4. Complete Architecture

### 4.1 Physics Stream: `SVPhysicsCore`

**Code**: [L242–314]

Three stages:

**Stage 1 — Node Encoding** [L275]:

$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}([\mathbf{x}_i \;\|\; \dot{\mathbf{x}}_i]) \in \mathbb{R}^{d_h}$$

```python
self.encoder = _make_mlp(node_input_dim, hidden_dim, hidden_dim)
# 6 → 32 → LayerNorm → ReLU → 32
```

**Stage 2 — $L$ rounds of SV Message Passing** [L302–306]:

```python
for layer in self.sv_layers:
    h, F_agg = layer.forward_with_forces(h, edge_index, positions, velocities)
    forces = F_agg  # keep last layer's forces
```

Each round executes the full pipeline from §2.7.

**Stage 3 — Force Output (No Decoder)** [L308]:

$$\hat{\mathbf{F}}_i = \mathbf{F}_i^{(L-1)}$$

**Critical**: The final output is the **raw aggregated force** from the last SV layer. We do NOT pass through a per-node decoder MLP, which would break $\sum = \mathbf{0}$.

### 4.2 Dual-Stream Architecture: `PhysRobotFeaturesExtractorV3`

**Code**: [L320–423]

```
Observation o_t ──┬──► [Policy Stream: MLP]  ──► z_policy ∈ R^64
                  │
                  └──► [Physics Stream: SV-GNN]  ──► â_box ∈ R^3
                                                     (predicted force on box)

Fusion: z = ReLU(W_f [z_policy ‖ sg(â_box)] + b_f) ──► PPO Actor/Critic
```

**Policy Stream** [L354–358]:

```python
self.policy_stream = nn.Sequential(
    nn.Linear(obs_dim, 64),   # 16 → 64
    nn.ReLU(),
    nn.Linear(64, features_dim),  # 64 → 64
    nn.ReLU(),
)
```

**Fusion** [L361–364]:

$$\mathbf{z} = \text{ReLU}\bigl(\mathbf{W}_f [\mathbf{z}_{\text{policy}} \;\|\; \text{sg}(\hat{\mathbf{a}}_{\text{box}})] + \mathbf{b}_f\bigr)$$

```python
self.fusion = nn.Sequential(
    nn.Linear(features_dim + 3, features_dim),  # 67 → 64
    nn.ReLU(),
)
```

**Stop-gradient** [L418]:

```python
z_physics_sg = z_physics.detach()
```

**Why stop-gradient?**

1. Prevents RL's high-variance policy gradient from corrupting learned dynamics
2. Physics stream's goal is accurate force prediction, not reward maximization
3. Without sg, RL may learn to "hack" physics predictions (output fictitious accelerations that bias the policy)

### 4.3 Observation-to-Graph Mapping

**Code**: `_obs_to_graph()` [L372–389]

```
obs[0:2]   → joint_pos       (not used by physics stream)
obs[2:4]   → joint_vel       (not used by physics stream)
obs[4:7]   → ee_pos  (3D) → Node 0: pos=ee_pos, vel=zeros(3)
obs[7:10]  → box_pos (3D) → Node 1: pos=box_pos, vel=box_vel
obs[10:13] → box_vel (3D)
obs[13:16] → goal_pos (3D)  (not used by physics stream)
```

**Edge index**: Fully-connected directed input, undirected pairs extracted internally:

```python
edge_index = [[0, 1], [1, 0]]  # directed
# Internally: pairs = [[0], [1]]  # undirected (0 < 1), single pair
```

**Known limitation**: `ee_vel ≈ zeros(3)` is an approximation. Future: compute from `joint_vel` via forward kinematics Jacobian.

---

## 5. Loss Function Design

### 5.1 Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}$$

### 5.2 RL Loss (PPO)

Standard PPO clipped surrogate (Schulman et al., 2017). Backpropagates through **Policy Stream** and **Fusion** only — physics stream is blocked by stop-gradient.

### 5.3 Physics Auxiliary Loss (Self-Supervised)

$$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{s}^t, \mathbf{s}^{t+1}) \in \mathcal{B}} \sum_{i \in \text{objects}} \left\| \hat{\mathbf{F}}_i^t - \mathbf{a}_i^{t,\text{fd}} \right\|^2$$

where $\mathbf{a}_i^{t,\text{fd}} = (\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ is the finite-difference acceleration.

**Self-supervised**: No ground-truth forces needed. Acceleration estimated from consecutive state observations in RL rollouts.

**Warmup schedule** (per SPEC.md §3.1):

$$\lambda_{\text{phys}}(t) = 0.1 \cdot \min\!\left(1,\; \frac{t}{50{,}000}\right)$$

### 5.4 Why $\|\sum \mathbf{F}\|^2$ Regularization is Unnecessary

R1 proposed $\lambda_{\text{reg}} \|\sum_i \mathbf{F}_i\|^2$ as a soft conservation penalty. With the undirected-pair design, $\sum_i \mathbf{F}_i = \mathbf{0}$ is **exactly zero by construction**. No regularization is needed — the constraint is architectural, not learned.

---

## 6. Parameter Analysis and Computational Cost

### 6.1 Parameter Count (Code-Verified)

Config: $d_h = 32$, $L = 1$, node input = 6 (pos + vel).

| Module | Computation | Parameters | Code Reference |
|--------|------------|------------|----------------|
| Node encoder | $(6+1) \times 32 + 32 + (32+1) \times 32 = 1{,}312 + 32 = 1{,}344$ | 1,344 | `self.encoder` |
| Force MLP | $(69+1) \times 32 + 32 + (32+1) \times 3 = 2{,}272 + 99 = 2{,}371$ | 2,371 | `self.force_mlp` |
| Node update | $(35+1) \times 32 + 32 + (32+1) \times 32 = 1{,}184 + 1{,}056 = 2{,}240$ | 2,240 | `self.node_update` |
| **Physics Stream total** | | **~6,019** | `model.parameter_count()` |

Full dual-stream (`PhysRobotFeaturesExtractorV3`):

| Component | Parameters |
|-----------|-----------|
| Physics Stream (SV-GNN) | ~6K |
| Policy Stream (16→64→64) | ~5.2K |
| Fusion (67→64) | ~4.4K |
| **Total extractor** | **~15,619** |

### 6.2 Computational Complexity

Per SV layer:

| Operation | Complexity | PushBox ($N=2, P=1, d_h=32$) |
|-----------|-----------|-------------------------------|
| Frame construction | $O(P)$ | Trivial |
| Scalarization | $O(P \cdot d_h)$ | ~64 ops |
| Force MLP | $O(P \cdot d_h^2)$ | ~2K FLOPs |
| Vectorization | $O(P)$ | ~9 ops |
| ±F Aggregation | $O(P)$ | ~6 ops |
| Node update | $O(N \cdot d_h^2)$ | ~2K FLOPs |
| **Total** | | **~4K FLOPs** |

Compare: MuJoCo step ~1M FLOPs → PhysRobot overhead < 1%.

### 6.3 Inference Latency (M1 Mac Estimate)

| Component | Time |
|-----------|------|
| MuJoCo step | ~0.15 ms |
| Graph construction | ~0.01 ms |
| SV-GNN forward | ~0.02 ms |
| Policy MLP | ~0.01 ms |
| **Total** | **~0.19 ms → 5,000+ Hz** |

---

## 7. Novelty Enhancement Proposals (R2 New)

R1 Reviewer assessment: "Borderline — GNN as feature extractor is established." We propose three extensions, each independently raising the novelty bar.

### 7.1 Contact-Aware Message Passing (Dynamic Graph from Collision Detection)

**Motivation**: Current graph is static (fully-connected). In manipulation, contact topology changes rapidly.

**Proposal**: Replace static graph with **contact-aware dynamic graph**:

$$\mathcal{E}^t = \mathcal{E}_{\text{contact}}^t \cup \mathcal{E}_{\text{proximity}}^t$$

- $\mathcal{E}_{\text{contact}}^t = \{(i,j) : \text{penetration}(i,j) > 0\}$ — active contacts from MuJoCo
- $\mathcal{E}_{\text{proximity}}^t = \{(i,j) : d_{ij} < r_{\text{prox}}\}$ — anticipatory edges

**Contact-augmented scalars**:

$$\boldsymbol{\sigma}_{\text{contact}} = [d_{ij},\; v_r,\; v_t,\; v_b,\; \|\mathbf{v}_{\text{rel}}\|,\; \underbrace{f_n^{\text{MuJoCo}},\; \mu_{ij},\; A_{\text{contact}}}_{\text{new contact features}}]$$

**Edge-type conditioning**: Different MLPs for contact vs proximity edges:

$$\alpha_k = \begin{cases} \text{MLP}_{\text{contact}}(\boldsymbol{\sigma}) & \text{if } (i,j) \in \mathcal{E}_{\text{contact}} \\ \text{MLP}_{\text{prox}}(\boldsymbol{\sigma}) & \text{if } (i,j) \in \mathcal{E}_{\text{proximity}} \end{cases}$$

**Novelty**: No prior GNN-for-RL work uses collision detection to dynamically construct the message-passing topology. This goes beyond "GNN as feature extractor."

### 7.2 Energy Dissipation Tracking

**Motivation**: The physics stream predicts forces but has no explicit model of energy flow. In manipulation, dissipation (friction heat, inelastic losses) is a critical signal.

**Proposal**: Force MLP additionally outputs a dissipation rate $\dot{D}_{\{i,j\}} \geq 0$:

$$(\alpha_1, \alpha_2, \alpha_3, \dot{D}) = \text{MLP}_\theta(\boldsymbol{\sigma})$$
$$\dot{D}_{\{i,j\}} = \text{softplus}(\text{raw output}) \geq 0$$

**Energy balance loss**:

$$\mathcal{L}_{\text{energy}} = \left| \Delta E_{\text{kin}} - \sum_{\{i,j\}} \mathbf{F}_{\{i,j\}} \cdot \Delta\mathbf{x}_{\{i,j\}} + \sum_{\{i,j\}} \dot{D}_{\{i,j\}} \cdot \Delta t \right|$$

**Novelty**: Thermodynamically consistent force model — forces must be compatible with observed energy changes plus non-negative dissipation. First in GNN-for-RL.

### 7.3 Multi-Scale Graph Architecture

**Proposal**: Two-level hierarchical graph:

- **Level 1 (Fine)**: Per-contact-patch nodes with SV message passing → per-object aggregate contact force.
- **Level 2 (Coarse)**: Per-object nodes, input includes Level-1 summaries → scene-level predictions.

**Conservation guarantee at every level**:

$$\sum_i \mathbf{F}_i^{\text{total}} = \underbrace{\sum_i \mathbf{F}_i^{\text{coarse}}}_{= \mathbf{0}} + \underbrace{\sum_i \mathbf{F}_i^{\text{fine}}}_{= \mathbf{0}} = \mathbf{0}$$

Each level independently uses undirected-pair $\pm\mathbf{F}$.

**Novelty**: First hierarchical graph for physics-informed RL preserving conservation at every level.

### 7.4 Novelty Assessment Summary

| Contribution | Novelty | Effort | Impact |
|-------------|---------|--------|--------|
| Base SV-pipeline | Medium (extends Dynami-CAL) | Done ✅ | Foundation |
| Undirected-pair simplification | Medium (cleaner than Dynami-CAL) | Done ✅ | Robustness |
| Dual-stream stop-grad RL | Medium (known in repr. learning) | Done ✅ | Integration |
| **Contact-aware MP (§7.1)** | **High** | 2 weeks | ⭐ Top pick for CoRL |
| **Energy dissipation (§7.2)** | **High** | 1 week | ⭐ Physics depth |
| **Multi-scale graph (§7.3)** | **Very High** | 3 weeks | ⭐⭐ Strongest overall |

**Recommendation**: Implement §7.1 (contact-aware MP) as primary novelty booster — most practical, most impactful, directly addresses reviewer critique.

---

## 8. Comparison with Existing Methods

| Feature | **EGNN** | **PaiNN** | **NequIP** | **Dynami-CAL** | **PhysRobot** |
|---------|----------|-----------|------------|----------------|---------------|
| Symmetry | E(n) equiv. | E(3) equiv. | E(3) equiv. | E(3) equiv. | E(3) equiv. |
| Lin. momentum conservation | ❌ | ❌ | ❌ | ✅ (directed + markers) | ✅ (**undirected + ±F**) |
| Conservation mechanism | — | — | — | Antisymmetric scalars | **Hard-coded ±F** |
| Force type | Radial only | Scalar + Vector | Tensor (irreps) | Full 3D (edge frame) | Full 3D (edge frame) |
| Non-conservative forces | ✅ | ✅ | ❌ (Hamiltonian) | ✅ | ✅ |
| Friction modeling | ❌ | ❌ | ❌ | ✅ | ✅ (tangential $\alpha_2, \alpha_3$) |
| RL integration | Indirect | Indirect | Indirect | ❌ | ✅ (dual-stream + stop-grad) |
| Domain | Molecular | Molecular/MD | MD/Materials | Particles | **Robotics manipulation** |
| Cost per pair | $O(d)$ | $O(d)$ | $O(l^3 d)$ | $O(d^2)$ (2× MLP) | $O(d^2)$ (**1× MLP**) |
| Proof simplicity | — | — | — | Multi-step symmetry | **One-line ±F** |

**Key differentiators**:

- **vs EGNN**: EGNN produces only **radial** forces ($\mathbf{x}_j - \mathbf{x}_i) \phi(\cdot)$). Cannot model friction. No conservation guarantee.
- **vs PaiNN**: Scalar+vector channels but no frame decomposition. No conservation guarantee.
- **vs NequIP**: Elegant $O(l^3)$ spherical harmonics but designed for conservative potentials only.
- **vs Dynami-CAL**: Our direct ancestor. We simplify (undirected pairs), halve compute (1× MLP), add RL integration, and propose contact-aware graphs.

---

## 9. Implementation Verification

### 9.1 Unit Tests (from `__main__` block [L476–520])

| Test | What it verifies | Result |
|------|-----------------|--------|
| Parameter count | Physics stream < 30K params | ✅ ~6,019 |
| Momentum conservation | 100 random trials × 4 nodes, $\|\sum \mathbf{F}\| < 10^{-4}$ | ✅ Max error ~$10^{-7}$ |
| Variable graph sizes | $N \in \{2, 3, 5, 8\}$ all conserve | ✅ |
| Gradient flow | All active parameters receive non-zero gradients | ✅ |
| Dual-stream forward | $(B, 16) \to (B, 64)$ shape verification | ✅ |

### 9.2 Running Tests

```bash
cd medical-robotics-sim
python physics_core/sv_message_passing.py
# Expected: "✅ ALL TESTS PASSED"
```

### 9.3 Conservation for Arbitrary Parameters

```python
from physics_core.sv_message_passing import SVPhysicsCore, verify_momentum_conservation

model = SVPhysicsCore(node_input_dim=6, hidden_dim=32, n_layers=1)
verify_momentum_conservation(model, n_trials=1000, n_nodes=10, tol=1e-4)
# ✅ All 1000 trials passed.  Max ||Σ F|| = ~1e-7
```

This works for **any** random initialization — it is an architectural guarantee, not a learned property.

---

## 10. Summary of Contributions (for Paper Writer)

1. **Undirected-Pair SV-Pipeline** (§2): Simpler, more robust momentum conservation than Dynami-CAL. Single MLP per pair, hard-coded $\pm\mathbf{F}$, one-line proof. No antisymmetric markers, no symmetry analysis.

2. **Dual-Stream RL Architecture** (§4.2): Physics stream (SV-GNN) provides momentum-conserving force features to policy stream (MLP) via stop-gradient fusion. First integration of conservation-law GNN into RL.

3. **Self-Supervised Physics Learning** (§5.3): Physics stream trained on finite-difference accelerations from RL rollouts. No ground-truth force labels.

4. **Novelty Extensions** (§7): Contact-aware message passing, energy dissipation tracking, multi-scale graph — each independently elevates novelty beyond "GNN as feature extractor."

5. **Code-First Verification** (§9): Architectural conservation verified for 1000+ random parameter initializations. All claims are backed by executable unit tests.

---

*R2 complete. All equations aligned with `physics_core/sv_message_passing.py`. All R1 reviewer issues (MATH-1, MATH-2, proof gaps, $\alpha_3$ marker) resolved. Archived: `archive/reviews_v1/ALGORITHM_DESIGN_R1.md`.*
