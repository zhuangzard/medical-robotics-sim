# PhysRobot: Algorithm Design Document

**Version**: 1.0  
**Date**: 2026-02-06  
**Author**: Algorithm Architecture Team  
**Target Venue**: ICRA 2027 / CoRL 2026  

---

## 1. Core Novelty Statement

> **PhysRobot** is the first framework to integrate a **hard-constraint momentum-conserving Graph Neural Network** into a reinforcement learning policy for **contact-rich robotic manipulation**. Unlike soft-constraint approaches (physics-informed losses) or conservative-system methods (HNN/LNN), PhysRobot architecturally guarantees Newton's Third Law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) through a **Scalarization–Vectorization (SV) pipeline** operating in edge-local coordinate frames, while remaining applicable to **non-conservative, dissipative, multi-body** systems with time-varying contact topology.

### 1.1 One-Sentence Contribution

We propose a dual-stream RL architecture where a **geometrically-constrained GNN** (the *Physics Stream*) provides momentum-conserving dynamical features to a standard policy network (the *Policy Stream*), yielding superior **sample efficiency** and **out-of-distribution generalization** on contact-rich manipulation tasks.

### 1.2 Key Claims

| # | Claim | Mechanism |
|---|-------|-----------|
| C1 | **Architectural momentum conservation** | SV-pipeline on edge frames: $\sum_i \mathbf{F}_i = \mathbf{0}$ for *any* network parameters |
| C2 | **OOD generalization to unseen physics** | Physics inductive bias constrains the hypothesis space; correct force structure transfers across mass/friction changes |
| C3 | **Scalable multi-object reasoning** | GNN message-passing naturally handles variable-count objects with dynamic contact graphs |
| C4 | **No conservative-system assumption** | Unlike HNN/LNN, we do not require a Hamiltonian or Lagrangian; friction, damping, and external forces are modeled natively |

---

## 2. Mathematical Framework

### 2.1 Problem Setting

We consider a robotic manipulation system with $N$ interacting bodies (end-effector, objects, obstacles). The state of body $i$ at time $t$ is:

$$\mathbf{s}_i^t = (\mathbf{x}_i^t, \dot{\mathbf{x}}_i^t, \boldsymbol{\phi}_i)$$

where $\mathbf{x}_i \in \mathbb{R}^3$ is position, $\dot{\mathbf{x}}_i \in \mathbb{R}^3$ is velocity, and $\boldsymbol{\phi}_i$ encodes intrinsic properties (mass, friction coefficient, geometry descriptor).

The system is represented as a directed graph $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ where:
- $\mathcal{V} = \{1, \ldots, N\}$ is the set of bodies (nodes)
- $\mathcal{E}^t = \{(i,j) : \|\mathbf{x}_i^t - \mathbf{x}_j^t\| < r_{\text{cutoff}} \}$ is the set of interaction edges (time-varying contact graph)

### 2.2 Edge-Local Coordinate Frame

For each directed edge $(i \to j)$, we construct an orthonormal frame $\{\mathbf{e}_1^{ij}, \mathbf{e}_2^{ij}, \mathbf{e}_3^{ij}\}$:

**Step 1: Radial basis vector**
$$\mathbf{e}_1^{ij} = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

**Step 2: Tangential basis vector (from relative velocity)**
$$\mathbf{v}_{ij}^{\perp} = \dot{\mathbf{x}}_{ij} - (\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij})\,\mathbf{e}_1^{ij}, \quad \mathbf{e}_2^{ij} = \frac{\mathbf{v}_{ij}^{\perp}}{\|\mathbf{v}_{ij}^{\perp}\| + \epsilon}$$

where $\dot{\mathbf{x}}_{ij} = \dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$ is the relative velocity.

**Step 3: Binormal basis vector**
$$\mathbf{e}_3^{ij} = \mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$$

**Degeneracy handling**: When $\|\mathbf{v}_{ij}^{\perp}\| < \epsilon_{\text{deg}}$, we fall back to a gravity-aligned frame:
$$\mathbf{e}_2^{ij} = \frac{\mathbf{e}_1^{ij} \times \hat{\mathbf{z}}}{\|\mathbf{e}_1^{ij} \times \hat{\mathbf{z}}\| + \epsilon}$$
with a further fallback to $\hat{\mathbf{y}}$ when $\mathbf{e}_1^{ij} \approx \pm\hat{\mathbf{z}}$.

**Critical antisymmetry property**:
$$\mathbf{e}_1^{ij} = -\mathbf{e}_1^{ji}, \quad \mathbf{e}_2^{ij} = -\mathbf{e}_2^{ji}, \quad \mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$$

*Proof*: $\mathbf{e}_1^{ji} = (\mathbf{x}_i - \mathbf{x}_j)/\|\cdot\| = -\mathbf{e}_1^{ij}$. Since $\dot{\mathbf{x}}_{ji} = -\dot{\mathbf{x}}_{ij}$, projection removal gives $\mathbf{v}_{ji}^{\perp} = -\mathbf{v}_{ij}^{\perp}$, hence $\mathbf{e}_2^{ji} = -\mathbf{e}_2^{ij}$. For the cross product: $\mathbf{e}_3^{ji} = \mathbf{e}_1^{ji} \times \mathbf{e}_2^{ji} = (-\mathbf{e}_1^{ij}) \times (-\mathbf{e}_2^{ij}) = +\mathbf{e}_3^{ij}$. $\square$

### 2.3 Scalarization–Vectorization (SV) Pipeline

This is the **core architectural innovation** that guarantees momentum conservation.

#### 2.3.1 Scalarization: $\mathbb{R}^3 \to \mathbb{R}^k$ (rotation-invariant)

Raw geometric quantities are projected onto the edge frame to produce **scalar invariants**:

$$\boldsymbol{\sigma}_{ij} = \begin{pmatrix} \|\mathbf{x}_j - \mathbf{x}_i\| \\[4pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} \\[4pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij} \\[4pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij} \\[4pt] \|\dot{\mathbf{x}}_{ij}\| \end{pmatrix} \in \mathbb{R}^5$$

These scalars are **invariant under global rotation** and contain the full geometric information of the edge.

**Extended scalar features** (including node attributes):

$$\boldsymbol{\sigma}_{ij}^{\text{ext}} = [\boldsymbol{\sigma}_{ij} \,\|\, \mathbf{h}_i \,\|\, \mathbf{h}_j] \in \mathbb{R}^{5 + 2d_h}$$

where $\mathbf{h}_i, \mathbf{h}_j \in \mathbb{R}^{d_h}$ are learned node embeddings.

#### 2.3.2 Scalar MLP: $\mathbb{R}^{5+2d_h} \to \mathbb{R}^3$

A standard MLP processes the scalar features:

$$\begin{pmatrix} \alpha_1^{ij} \\ \alpha_2^{ij} \\ \alpha_3^{ij} \end{pmatrix} = \text{MLP}_{\theta}(\boldsymbol{\sigma}_{ij}^{\text{ext}})$$

These are **scalar force coefficients**. Because the MLP operates on rotation-invariant inputs, the coefficients are rotation-invariant.

#### 2.3.3 Vectorization: $\mathbb{R}^3 \to \mathbb{R}^3$ (reconstruct 3D force)

The 3D force vector is reconstructed by combining scalar coefficients with frame basis vectors:

$$\mathbf{F}_{ij} = \alpha_1^{ij}\,\mathbf{e}_1^{ij} + \alpha_2^{ij}\,\mathbf{e}_2^{ij} + \alpha_3^{ij}\,\mathbf{e}_3^{ij}$$

### 2.4 Momentum Conservation Proof

**Theorem (Linear Momentum Conservation).** *For any network parameters $\theta$, the total predicted force on the system vanishes:*

$$\sum_{i=1}^{N} \mathbf{F}_i = \mathbf{0}$$

*where $\mathbf{F}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}$ is the net force on node $i$.*

**Proof.** The total force is:

$$\sum_i \mathbf{F}_i = \sum_i \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij} = \sum_{(i,j) \in \mathcal{E}} (\mathbf{F}_{ij} + \mathbf{F}_{ji})$$

Now consider each component:

$$\mathbf{F}_{ij} + \mathbf{F}_{ji} = \alpha_1^{ij}\mathbf{e}_1^{ij} + \alpha_2^{ij}\mathbf{e}_2^{ij} + \alpha_3^{ij}\mathbf{e}_3^{ij} + \alpha_1^{ji}\mathbf{e}_1^{ji} + \alpha_2^{ji}\mathbf{e}_2^{ji} + \alpha_3^{ji}\mathbf{e}_3^{ji}$$

**Key insight**: The scalar features satisfy $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$ (the distance, speed norms, and squared projections are symmetric). Since the same MLP processes both, we have $\alpha_k^{ij} = \alpha_k^{ji}$ for $k = 1, 2, 3$.

Using the antisymmetry of the frame ($\mathbf{e}_1^{ji} = -\mathbf{e}_1^{ij}$, $\mathbf{e}_2^{ji} = -\mathbf{e}_2^{ij}$, $\mathbf{e}_3^{ji} = +\mathbf{e}_3^{ij}$):

$$\mathbf{F}_{ij} + \mathbf{F}_{ji} = \alpha_1^{ij}(\mathbf{e}_1^{ij} - \mathbf{e}_1^{ij}) + \alpha_2^{ij}(\mathbf{e}_2^{ij} - \mathbf{e}_2^{ij}) + \alpha_3^{ij}(\mathbf{e}_3^{ij} + \mathbf{e}_3^{ij})$$

$$= 2\alpha_3^{ij}\mathbf{e}_3^{ij}$$

This does NOT cancel! The $\mathbf{e}_3$ component is **symmetric**, not antisymmetric.

**Fix: Antisymmetrize $\alpha_3$.**

We must enforce $\alpha_3^{ij} = -\alpha_3^{ji}$ while keeping $\alpha_1, \alpha_2$ free (they are automatically canceled by the antisymmetric basis vectors). Since $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$ in the base scalar features, we need an **antisymmetric marker**.

**Solution**: Augment the scalar input with a **signed radial velocity** $v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ (which satisfies $v_r^{ij} = -v_r^{ji}$), and decompose the $\alpha_3$ MLP as:

$$\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\boldsymbol{\sigma}_{ij}^{\text{sym}})$$

where $g_\theta$ is a scalar-valued MLP acting on *symmetric* scalar features only, and $v_r^{ij}$ provides the antisymmetric sign flip. Then:

$$\alpha_3^{ij} + \alpha_3^{ji} = v_r^{ij}\, g_\theta(\boldsymbol{\sigma}^{\text{sym}}) + v_r^{ji}\, g_\theta(\boldsymbol{\sigma}^{\text{sym}}) = (v_r^{ij} - v_r^{ij})\, g_\theta(\cdot) = 0$$

Now $\mathbf{F}_{ij} + \mathbf{F}_{ji} = \mathbf{0}$ and hence $\sum_i \mathbf{F}_i = \mathbf{0}$. $\square$

**Remark on Angular Momentum**: Conservation of angular momentum additionally requires $\mathbf{F}_{ij} \parallel \mathbf{r}_{ij}$ (central forces), i.e., $\alpha_2 = \alpha_3 = 0$. This is overly restrictive for contact-rich manipulation (friction is tangential). We deliberately relax this to model friction and only enforce **linear** momentum conservation.

### 2.5 Complete Forward Pass: Physics Stream

Given a graph $\mathcal{G}^t$ with node states $\{\mathbf{s}_i^t\}$:

**Step 1. Node Encoding**
$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}([\mathbf{x}_i, \dot{\mathbf{x}}_i, \boldsymbol{\phi}_i])$$

**Step 2. $L$ rounds of SV Message Passing** (for $\ell = 0, \ldots, L-1$):

For each edge $(i,j)$:
1. Compute edge frame $\{\mathbf{e}_k^{ij}\}$ (Section 2.2)
2. Scalarize: $\boldsymbol{\sigma}_{ij}^{(\ell)} = \text{Scalar}(\mathbf{x}_i, \mathbf{x}_j, \dot{\mathbf{x}}_i, \dot{\mathbf{x}}_j, \mathbf{h}_i^{(\ell)}, \mathbf{h}_j^{(\ell)})$
3. Compute scalar coefficients: $(\alpha_1, \alpha_2) = \text{MLP}_{\alpha}^{(\ell)}(\boldsymbol{\sigma}_{ij}^{(\ell)})$, $\alpha_3 = v_r^{ij} \cdot g^{(\ell)}(\boldsymbol{\sigma}_{ij}^{\text{sym},(\ell)})$
4. Vectorize: $\mathbf{m}_{ij}^{(\ell)} = \alpha_1 \mathbf{e}_1^{ij} + \alpha_2 \mathbf{e}_2^{ij} + \alpha_3 \mathbf{e}_3^{ij}$
5. Aggregate: $\mathbf{M}_i^{(\ell)} = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(\ell)}$
6. Update node: $\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}_{\text{upd}}^{(\ell)}([\mathbf{h}_i^{(\ell)}, \mathbf{M}_i^{(\ell)}])$

**Step 3. Dynamics Decoding**
$$\hat{\mathbf{a}}_i = \text{MLP}_{\text{dec}}(\mathbf{h}_i^{(L)}) \quad \text{(predicted acceleration of body } i\text{)}$$

### 2.6 Dual-Stream RL Architecture

The complete PhysRobot architecture consists of two parallel streams feeding into a PPO policy:

```
Observation o_t ──┬──► [Policy Stream]  ──► z_policy ∈ R^d
                  │     (Standard MLP)
                  │
                  └──► [Physics Stream] ──► z_physics ∈ R^{3N}
                        (SV-GNN, §2.5)      (predicted accelerations)
                        
z = Fusion(z_policy, z_physics) ──► PPO Actor ──► π(a|o)
                                 ──► PPO Critic ──► V(o)
```

**Fusion mechanism**:
$$\mathbf{z} = \text{ReLU}(\mathbf{W}_f [\mathbf{z}_{\text{policy}} \| \text{sg}(\hat{\mathbf{a}}_{\text{box}})] + \mathbf{b}_f)$$

where $\text{sg}(\cdot)$ denotes **stop-gradient** on the physics predictions during RL training. This prevents the RL loss from distorting the physics network's learned dynamics. The physics stream is trained with an auxiliary physics loss (Section 2.8).

### 2.7 Multi-Object Graph Construction

For $N$ objects in the scene (including the end-effector):

**Nodes**: Each rigid body $i$ contributes one node with features:
$$\mathbf{x}_i^{\text{node}} = [\mathbf{x}_i, \dot{\mathbf{x}}_i, m_i, \mu_i, \mathbf{g}_i]$$
where $m_i$ is mass, $\mu_i$ is friction coefficient, $\mathbf{g}_i$ is a geometry descriptor (e.g., bounding-box dimensions).

**Edges** (dynamic contact graph):
$$\mathcal{E}^t = \{(i,j) : \|\mathbf{x}_i^t - \mathbf{x}_j^t\| < r_{\text{cut}} \} \cup \{(i, j_{\text{ee}}) : i \in \text{objects}\}$$

The end-effector always maintains edges to all objects (it can potentially interact with any of them). Object-object edges are activated within a cutoff radius $r_{\text{cut}}$.

**Edge types**: We use a 1-hot edge type indicator $\tau_{ij} \in \{$ee-obj, obj-obj, obj-ground$\}$ appended to scalar features.

**Scaling**: For $N$ objects, the graph has $|\mathcal{V}| = N+1$ nodes and $|\mathcal{E}| = O(N + N_{\text{contact}})$ edges, where $N_{\text{contact}}$ is the number of active contacts. This is **linear** in the number of objects for sparse contact configurations.

### 2.8 Loss Function Design

The total training loss has three components:

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

#### 2.8.1 RL Loss (PPO)

Standard PPO clipped surrogate objective:

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_t \left[\min\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] + c_v \mathcal{L}_{\text{VF}} - c_e H[\pi_\theta]$$

The RL loss only backpropagates through the **Policy Stream** and **Fusion module** (the Physics Stream receives stop-gradient).

#### 2.8.2 Physics Auxiliary Loss

The Physics Stream is trained with a **dynamics prediction loss** using transitions collected during RL rollouts:

$$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{s}^t, \mathbf{s}^{t+1}) \in \mathcal{B}} \sum_{i \in \text{objects}} \left\| \hat{\mathbf{a}}_i^t - \mathbf{a}_i^{t,\text{fd}} \right\|^2$$

where $\mathbf{a}_i^{t,\text{fd}} = (\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ is the finite-difference acceleration from consecutive observations.

**Key advantage**: This loss is **self-supervised** — no labeled force data or simulator access is needed; acceleration is estimated from observed state transitions.

#### 2.8.3 Conservation Regularizer

Although the SV-pipeline provides *architectural* momentum conservation for the force-like messages, we add a soft regularizer to encourage energy-aware predictions:

$$\mathcal{L}_{\text{reg}} = \frac{1}{|\mathcal{B}|} \sum_{t} \left| \hat{E}_{\text{kin}}^{t+1} - \hat{E}_{\text{kin}}^t + W_{\text{ext}}^t + D^t \right|$$

where:
- $\hat{E}_{\text{kin}}^t = \sum_i \frac{1}{2} m_i \|\dot{\mathbf{x}}_i^t\|^2$ (kinetic energy from observed velocities)
- $W_{\text{ext}}^t$ is estimated external work (from applied torques)
- $D^t \geq 0$ is a learned dissipation estimate: $D^t = \text{MLP}_D(\mathbf{s}^t)$

This loss encourages the physics stream to produce predictions consistent with the work-energy theorem, without requiring strict energy conservation (which would preclude friction modeling).

**Loss weight schedule**:
$$\lambda_{\text{phys}}(t) = \lambda_0 \cdot \min(1, t / T_{\text{warmup}})$$

We linearly warm up the physics loss over $T_{\text{warmup}}$ steps to let the RL policy stabilize before coupling with physics predictions.

---

## 3. Comparison with Existing Methods

### 3.1 Comparison Table

| Feature | **EGNN** | **DimeNet** | **PaiNN** | **NequIP** | **PhysRobot (Ours)** |
|---------|----------|-------------|-----------|------------|---------------------|
| **Symmetry group** | E(n) equiv. | SE(3) inv. (distances + angles) | E(3) equiv. (scalar + vector) | E(3) equiv. (irreps) | E(3) equiv. (edge frames) |
| **Conservation guarantee** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Linear momentum (hard) |
| **Message type** | Scalar | Scalar | Scalar + Vector | Tensor (irreps) | Scalar → Vector (SV pipeline) |
| **Handles non-conservative** | ✅ | ✅ | ✅ | ❌ (energy conserv. bias) | ✅ |
| **Contact/friction** | ❌ Not designed | ❌ Not designed | ❌ Not designed | ❌ Not designed | ✅ Tangential forces via $\alpha_2, \alpha_3$ |
| **RL integration** | Indirect | Indirect | Indirect | Indirect | ✅ Native (dual-stream + stop-grad) |
| **Domain** | Molecular / general | Molecular | Molecular / MD | MD / materials | Robotics manipulation |
| **Computational cost** | $O(N^2 d)$ | $O(N^2 d)$ | $O(N^2 d)$ | $O(N^2 l_{\max}^3 d)$ | $O(|\mathcal{E}| \cdot d)$ |

### 3.2 Detailed Differentiators

**vs. EGNN (Satorras et al., 2021)**:
- EGNN updates positions equivariantly: $\mathbf{x}_i' = \mathbf{x}_i + \sum_j (\mathbf{x}_j - \mathbf{x}_i) \phi(\mathbf{h}_i, \mathbf{h}_j, \|\mathbf{x}_j - \mathbf{x}_i\|)$
- This is always a **radial** update — no tangential force component
- Cannot model friction (tangential to contact surface)
- No momentum conservation guarantee
- PhysRobot: full 3D force with tangential components + conservation guarantee

**vs. DimeNet (Gasteiger et al., 2020)**:
- Uses distances and angles between triplets — higher expressivity
- But all operations are **scalar** (invariant, not equivariant)
- Cannot directly output 3D vectors; needs extra projection step
- PhysRobot: native 3D force output via Vectorization step

**vs. PaiNN (Schütt et al., 2021)**:
- Maintains parallel scalar + vector channels: $(\mathbf{s}_i, \mathbf{V}_i)$
- Vector updates: $\mathbf{V}_i' = \mathbf{V}_i + \sum_j \mathbf{r}_{ij} \odot \phi_s(\mathbf{s}_i, \mathbf{s}_j, \|\mathbf{r}_{ij}\|)$
- **Closest to our approach**, but:
  - No explicit frame construction (uses position-weighted messages)
  - No conservation guarantee (messages can create net force)
  - No RL integration mechanism
- PhysRobot: explicit frame decomposition enables provable conservation

**vs. NequIP (Batzner et al., 2022)**:
- Uses spherical harmonics and irreducible representations — most theoretically elegant
- But: $O(l_{\max}^3)$ per edge (expensive for $l_{\max} > 2$)
- Designed for conservative systems (potential energy surfaces)
- PhysRobot: simpler $O(d)$ per edge, handles non-conservative forces

**vs. Dynami-CAL (Sharma & Fink, 2025)**:
- PhysRobot's physics stream is **inspired by** Dynami-CAL
- Key differences:
  1. Dynami-CAL is for **pure physics simulation**; we integrate into **RL**
  2. We add the **dual-stream architecture** with stop-gradient
  3. We add the **$\alpha_3$ antisymmetrization fix** (Dynami-CAL's $e_3$ component breaks conservation in the general case — see Section 2.4)
  4. We introduce **dynamic contact graphs** for multi-object manipulation
  5. We design the **physics auxiliary loss** from RL rollout data (self-supervised)

---

## 4. Antisymmetry Fix: The SV Pipeline in Detail

### 4.1 The Problem with Naïve Implementations

The current codebase has **two broken implementations**:

**Bug 1** (`physics_core/edge_frame.py`): Passes raw vectors $[\mathbf{r}_{ij}, \|\mathbf{r}_{ij}\|, \dot{\mathbf{x}}_{ij}, \|\dot{\mathbf{x}}_{ij}\|]$ through an MLP. The MLP **destroys antisymmetry** because:
- Input contains both antisymmetric ($\mathbf{r}_{ij}$) and symmetric ($\|\mathbf{r}_{ij}\|$) components
- A general MLP $f$ satisfies $f([-\mathbf{r}, d, -\mathbf{v}, s]) \neq -f([\mathbf{r}, d, \mathbf{v}, s])$
- After MLP, $\mathbf{e}_{ij} \neq -\mathbf{e}_{ji}$ → momentum conservation fails

**Bug 2** (`baselines/physics_informed.py`): Uses a fixed `up = [0,0,1]` vector for frame construction instead of relative velocity. This causes:
- Gram-Schmidt degeneracy when $\mathbf{e}_1 \approx \hat{\mathbf{z}}$
- Loss of velocity information in the frame
- $\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$ is symmetric (not antisymmetric), but the code doesn't antisymmetrize $\alpha_3$

### 4.2 The Correct SV Pipeline (Pseudocode)

```python
class SVMessagePassing(MessagePassing):
    """Scalarization-Vectorization Message Passing Layer."""
    
    def __init__(self, node_dim, hidden_dim):
        super().__init__(aggr='add')
        n_scalar = 5  # ||r||, v_r, v_t, v_b, ||v||
        
        # MLPs for α₁, α₂ (symmetric scalar features → auto-canceled by antisymmetric basis)
        self.alpha12_mlp = MLP(n_scalar + 2*node_dim, hidden_dim, out=2)
        
        # MLP for α₃ magnitude (symmetric features only)
        self.alpha3_mag_mlp = MLP(n_scalar + 2*node_dim, hidden_dim, out=1)
        
        # Node update
        self.node_update = MLP(node_dim + 3, hidden_dim, out=node_dim)
    
    def forward(self, h, edge_index, pos, vel):
        src, dst = edge_index
        
        # === FRAME CONSTRUCTION ===
        r_ij = pos[dst] - pos[src]                    # antisymmetric
        d_ij = norm(r_ij, keepdim=True) + eps
        e1 = r_ij / d_ij                              # antisymmetric
        
        v_rel = vel[dst] - vel[src]                    # antisymmetric
        v_perp = v_rel - (v_rel · e1) * e1             # antisymmetric
        v_perp_norm = norm(v_perp, keepdim=True) + eps
        e2 = v_perp / v_perp_norm                      # antisymmetric
        # (with degeneracy fallback)
        
        e3 = cross(e1, e2)                             # SYMMETRIC!
        
        # === SCALARIZATION ===
        v_r = dot(v_rel, e1)      # radial relative velocity (antisymmetric!)
        v_t = dot(v_rel, e2)      # tangential component (symmetric)
        v_b = dot(v_rel, e3)      # binormal component (antisymmetric)
        
        scalars_sym = [d_ij, |v_r|, v_t, |v_b|, norm(v_rel)]  # ALL symmetric
        scalars = cat(scalars_sym, h[src], h[dst])
        
        # === SCALAR MLP ===
        alpha12 = self.alpha12_mlp(scalars)            # [α₁, α₂] ∈ R²
        alpha3_mag = self.alpha3_mag_mlp(scalars)      # |α₃| ∈ R¹
        
        # === ANTISYMMETRIZE α₃ ===
        alpha3 = v_r * alpha3_mag                      # v_r is antisymmetric → α₃ is antisymmetric
        
        # === VECTORIZATION ===
        force = alpha12[:,0:1]*e1 + alpha12[:,1:2]*e2 + alpha3*e3
        
        # === AGGREGATE + UPDATE ===
        F_i = scatter_add(force, dst, dim=0)           # ΣF_ij for each node i
        h_new = h + self.node_update(cat(h, F_i))
        
        return h_new
```

### 4.3 Verification Test

```python
def test_momentum_conservation(model, pos, vel, edge_index):
    """Verify Σ F_i = 0 for arbitrary network parameters."""
    acc = model(pos, vel, edge_index)   # [N, 3]
    masses = ones(N)
    total_force = (masses.unsqueeze(-1) * acc).sum(dim=0)
    assert norm(total_force) < 1e-5, f"Conservation violated: ||ΣF|| = {norm(total_force)}"
```

This test must pass for **any** random initialization of network parameters, not just after training.

---

## 5. Parameter Analysis and Computational Complexity

### 5.1 Parameter Count

For hidden dimension $d_h$, scalar feature dimension $d_s = 5$, node feature dimension $d_n$, and $L$ message-passing layers:

| Module | Parameters | Formula |
|--------|-----------|---------|
| Node encoder | $(d_n + 1) d_h + (d_h + 1) d_h$ | $\approx 2 d_h^2$ |
| Per-layer $\alpha_{1,2}$ MLP | $(d_s + 2d_h + 1) d_h + (d_h+1) \cdot 2$ | $\approx d_h(d_s + 2d_h)$ |
| Per-layer $\alpha_3$ MLP | $(d_s + 2d_h + 1) d_h + (d_h+1)$ | $\approx d_h(d_s + 2d_h)$ |
| Per-layer node update | $(d_h + 3 + 1) d_h + (d_h + 1) d_h$ | $\approx 2 d_h^2$ |
| Decoder | $(d_h + 1) d_h + (d_h + 1) \cdot 3$ | $\approx d_h^2$ |
| **Total (Physics Stream)** | | $\approx (3 + 4L) d_h^2 + 2L \cdot d_h \cdot d_s$ |

For our recommended settings ($d_h = 64$, $d_s = 5$, $L = 2$):
- Physics Stream: $(3 + 8) \cdot 64^2 + 4 \cdot 64 \cdot 5 = 45,056 + 1,280 \approx$ **46K params**
- Policy Stream ($d_h = 64$, 2-layer MLP): $\approx$ **12K params**
- Fusion: $\approx$ **4K params**
- PPO Actor/Critic: $\approx$ **17K params**
- **Total: ~79K params**

For lightweight settings ($d_h = 32$, $L = 1$):
- Physics Stream: $(3 + 4) \cdot 32^2 + 2 \cdot 32 \cdot 5 = 7,168 + 320 \approx$ **7.5K params**
- Policy Stream: $\approx$ **6K params**
- Fusion + Actor/Critic: $\approx$ **10K params**
- **Total: ~24K params**

### 5.2 Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Frame construction | $O(|\mathcal{E}|)$ | Per edge: 2 cross products + norms |
| Scalarization | $O(|\mathcal{E}| \cdot d_h)$ | Per edge: projections + concat |
| Scalar MLP | $O(|\mathcal{E}| \cdot d_h^2)$ | Dominant cost |
| Vectorization | $O(|\mathcal{E}|)$ | 3 scalar-vector multiplies |
| Aggregation | $O(|\mathcal{E}|)$ | Scatter-add |
| Node update | $O(|\mathcal{V}| \cdot d_h^2)$ | Per node MLP |
| **Per layer total** | $O(|\mathcal{E}| \cdot d_h^2 + |\mathcal{V}| \cdot d_h^2)$ | |
| **$L$ layers total** | $O(L (|\mathcal{E}| + |\mathcal{V}|) d_h^2)$ | |

For the PushBox task ($|\mathcal{V}| = 2$, $|\mathcal{E}| = 2$, $L = 1$, $d_h = 32$):
- ~8K FLOPs per forward pass (negligible)

For 5-object scene ($|\mathcal{V}| = 6$, $|\mathcal{E}| \approx 20$, $L = 2$, $d_h = 64$):
- ~328K FLOPs per forward pass (still negligible vs. MuJoCo step ~1M FLOPs)

### 5.3 Inference Latency Budget

| Component | Time (estimated, RTX 3080) |
|-----------|---------------------------|
| MuJoCo step (5 substeps) | ~0.15 ms |
| Graph construction | ~0.01 ms |
| Physics Stream forward | ~0.02 ms |
| Policy Stream forward | ~0.01 ms |
| PPO policy sample | ~0.01 ms |
| **Total per step** | **~0.20 ms** → **5,000 Hz** |

PhysRobot adds < 15% overhead compared to pure PPO.

---

## 6. Implementation Roadmap

### Phase 1: Core SV Pipeline (Week 1)

1. Implement `SVMessagePassing` layer with correct antisymmetrization
2. Implement degeneracy-safe frame construction
3. Write `test_momentum_conservation()` unit test
4. Verify: test passes for 1000 random parameter initializations

### Phase 2: Dual-Stream Integration (Week 2)

5. Implement `PhysRobotFeaturesExtractorV3` with SV-GNN physics stream
6. Implement stop-gradient between physics and RL streams
7. Implement physics auxiliary loss (finite-difference acceleration)
8. Run PushBox (2-object) experiments: match or beat PPO

### Phase 3: Multi-Object (Week 3–4)

9. Implement dynamic contact graph builder
10. Create Multi-Object PushBox environment (3–5 objects)
11. Run multi-object experiments: demonstrate GNN scaling advantage
12. Run OOD experiments: mass 0.1x–10x, friction 0.1–1.0

### Phase 4: Paper-Ready (Week 5–6)

13. Full ablation study (5 seeds × 5 variants × 3 tasks)
14. Comparison with EGNN, HNN baselines
15. Compute conservation error metrics
16. Generate learning curves, OOD generalization plots

---

## 7. Summary of Contributions (for Paper Writer)

1. **PhysRobot**: A dual-stream RL architecture that integrates a momentum-conserving GNN (Physics Stream) with a standard policy network (Policy Stream) via stop-gradient fusion.

2. **Corrected SV Pipeline**: We identify and fix a subtle bug in the Dynami-CAL framework where the $\mathbf{e}_3$ (binormal) component breaks antisymmetry. Our fix—multiplying $\alpha_3$ by the signed radial velocity $v_r$—restores the conservation guarantee with minimal computational overhead.

3. **Self-Supervised Physics Learning**: The Physics Stream is trained on finite-difference accelerations from RL rollouts, requiring no ground-truth force labels or differentiable simulator.

4. **Empirical Results** (expected):
   - 2–5× sample efficiency improvement over PPO on contact-rich tasks
   - 30–50% better OOD generalization to unseen physical parameters
   - Near-zero momentum conservation error (architectural guarantee) vs. >10% for unconstrained GNNs
   - Graceful scaling to 5+ objects where PPO degrades

---

*Document complete. Ready for review by Experiment, Writer, and Reviewer teams.*
