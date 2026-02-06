# 3. Method

We present PhysRobot, a physics-informed GNN architecture for reinforcement learning in contact-rich manipulation. We first formulate the problem (§3.1), then describe scene graph construction (§3.2), the Scalarization–Vectorization (SV) message-passing mechanism (§3.3), its conservation guarantees (§3.4), and integration with PPO via a dual-stream architecture (§3.5).

## 3.1 Problem Formulation

We formulate contact-rich manipulation as a Markov Decision Process (MDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ augmented with a dynamic scene graph that captures the relational structure among interacting bodies.

**State space.** The scene contains $N$ rigid bodies (one end-effector and $N-1$ objects). The state of body $i$ at time $t$ is:

$$\mathbf{s}_i^t = \left(\mathbf{x}_i^t,\; \dot{\mathbf{x}}_i^t,\; \boldsymbol{\phi}_i\right)$$

where $\mathbf{x}_i^t \in \mathbb{R}^3$ is position, $\dot{\mathbf{x}}_i^t \in \mathbb{R}^3$ is velocity, and $\boldsymbol{\phi}_i$ encodes intrinsic physical properties (mass $m_i$, friction coefficient $\mu_i$, geometry descriptor $\mathbf{g}_i$). The full state is the union of all body states along with robot joint configuration: $\mathbf{s}^t = (\mathbf{q}^t, \dot{\mathbf{q}}^t, \{\mathbf{s}_i^t\}_{i=1}^N)$.

**Action space.** Actions are continuous joint torques $\mathbf{a}^t \in \mathbb{R}^{d_a}$ applied to the robot's actuated joints, bounded by torque limits $[\mathbf{a}_{\min}, \mathbf{a}_{\max}]$.

**Scene graph.** At each time step, we represent the multi-body system as a directed graph $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ where $\mathcal{V} = \{1, \ldots, N\}$ is the fixed set of body nodes and $\mathcal{E}^t$ is a time-varying edge set encoding potential pairwise interactions (defined in §3.2). The graph provides a relational inductive bias: rather than reasoning over a flat state vector, the policy operates on a structured representation where each node maintains local state and communicates with neighbors through physically-constrained messages.

**Objective.** We seek a policy $\pi_\theta(\mathbf{a}^t | \mathbf{s}^t)$ that maximizes the expected discounted return $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(\mathbf{s}^t, \mathbf{a}^t)\right]$ with minimum environment interaction, where $R$ combines task-specific rewards (e.g., object-goal proximity) with optional regularization (energy penalty).

## 3.2 Scene Graph Construction

The scene graph $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ is constructed at each time step from the current state.

**Nodes.** Each rigid body $i \in \mathcal{V}$ is endowed with a feature vector:

$$\mathbf{x}_i^{\text{node}} = \left[\mathbf{x}_i,\; \dot{\mathbf{x}}_i,\; m_i,\; \mu_i,\; \mathbf{g}_i,\; \boldsymbol{\tau}_i\right]$$

where $\boldsymbol{\tau}_i \in \{0, 1\}^{|\mathcal{T}|}$ is a one-hot node-type indicator over the type set $\mathcal{T} = \{\texttt{end-effector},\; \texttt{object},\; \texttt{goal}\}$. The node encoder maps raw features to a $d_h$-dimensional embedding:

$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}\!\left(\mathbf{x}_i^{\text{node}}\right) \in \mathbb{R}^{d_h}$$

**Edges.** The edge set $\mathcal{E}^t$ is the union of two subsets:

1. **Proximity edges:** For every pair of bodies $(i, j)$ with $i \neq j$, an edge is created if $\|\mathbf{x}_i^t - \mathbf{x}_j^t\| < r_{\text{cut}}$, where $r_{\text{cut}}$ is a distance cutoff.

2. **End-effector edges:** The end-effector node maintains edges to *all* object nodes regardless of distance, reflecting the robot's ability to potentially interact with any object:

$$\mathcal{E}^t = \left\{(i,j) : \|\mathbf{x}_i^t - \mathbf{x}_j^t\| < r_{\text{cut}},\; i \neq j \right\} \;\cup\; \left\{(i, j_{\text{ee}}) : i \in \mathcal{V}_{\text{obj}}\right\}$$

Each edge carries a type indicator $\boldsymbol{\tau}_{ij} \in \{\texttt{ee-obj},\; \texttt{obj-obj},\; \texttt{obj-ground}\}$ appended to its scalar features.

**Scaling.** For $N$ bodies, the graph has $|\mathcal{V}| = N$ nodes and $|\mathcal{E}| = O(N + N_{\text{contact}})$ edges, where $N_{\text{contact}}$ is the number of active pairwise contacts. This is *linear* in $N$ for sparse contact configurations typical of tabletop manipulation. The graph is recomputed at each time step to reflect the changing contact topology — edges appear and disappear as objects come into and out of proximity.

## 3.3 Scalarization–Vectorization (SV) Message Passing

The core innovation of PhysRobot is a message-passing mechanism that produces inter-body force-like messages satisfying Newton's third law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) by construction. The mechanism proceeds in five stages: (1) edge-local frame construction, (2) scalarization (vector $\to$ scalar), (3) scalar force-coefficient computation, (4) vectorization (scalar $\to$ vector), and (5) aggregation with node update. We describe each stage below and prove the conservation guarantee in §3.4.

### 3.3.1 Edge-Local Coordinate Frames

For each *undirected* body pair $\{i, j\}$ with canonical ordering $i < j$, we construct an orthonormal frame $\{\mathbf{e}_1^{ij}, \mathbf{e}_2^{ij}, \mathbf{e}_3^{ij}\}$ anchored to the pair's geometric configuration.

**Radial basis vector.** The first axis is aligned with the inter-body displacement:

$$\mathbf{e}_1^{ij} = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

**Tangential basis vector.** The second axis captures the direction of tangential relative motion. Let $\dot{\mathbf{x}}_{ij} = \dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$ be the relative velocity. We project out the radial component to obtain the tangential residual:

$$\mathbf{v}_{ij}^{\perp} = \dot{\mathbf{x}}_{ij} - \left(\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}\right) \mathbf{e}_1^{ij}, \qquad \mathbf{e}_2^{ij} = \frac{\mathbf{v}_{ij}^{\perp}}{\|\mathbf{v}_{ij}^{\perp}\| + \epsilon}$$

**Degeneracy handling.** When $\|\mathbf{v}_{ij}^{\perp}\| < \epsilon_{\text{deg}}$ (bodies have negligible tangential relative motion), the frame degenerates. We fall back to a gravity-aligned construction: $\mathbf{e}_2^{ij} = \text{normalize}(\mathbf{e}_1^{ij} \times \hat{\mathbf{z}})$. If additionally $\mathbf{e}_1^{ij} \approx \pm\hat{\mathbf{z}}$, we substitute $\hat{\mathbf{y}}$ for $\hat{\mathbf{z}}$. A smooth blending factor $\beta = \sigma((\|\mathbf{v}_{ij}^\perp\| - \epsilon_{\text{deg}}) / \tau)$ interpolates between the velocity-based and fallback frames to avoid discontinuities.

**Binormal basis vector.** The third axis completes the right-handed frame:

$$\mathbf{e}_3^{ij} = \mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$$

### 3.3.2 Scalarization: Geometric Vectors $\to$ Rotation-Invariant Scalars

Raw 3D vectors are projected onto the edge frame to produce rotation-invariant scalar features. For the canonical direction $i \to j$ ($i < j$):

$$\boldsymbol{\sigma}_{ij} = \begin{pmatrix} \|\mathbf{x}_j - \mathbf{x}_i\| \\[3pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} \\[3pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij} \\[3pt] \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij} \\[3pt] \|\dot{\mathbf{x}}_{ij}\| \end{pmatrix} \in \mathbb{R}^5$$

These five scalars — distance, radial velocity $v_r$, tangential velocity $v_t$, binormal velocity $v_b$, and speed — capture the full geometric configuration of the pair in a rotation-invariant representation.

**Node embedding symmetrization.** To ensure the scalar features are invariant to the ordering of the pair, we symmetrize the node embeddings before concatenation:

$$\boldsymbol{\sigma}_{ij}^{\text{ext}} = \left[\boldsymbol{\sigma}_{ij} \;\|\; (\mathbf{h}_i + \mathbf{h}_j) \;\|\; |\mathbf{h}_i - \mathbf{h}_j|\right] \in \mathbb{R}^{5 + 2d_h}$$

where $\mathbf{h}_i + \mathbf{h}_j$ is the element-wise sum (symmetric) and $|\mathbf{h}_i - \mathbf{h}_j|$ is the element-wise absolute difference (also symmetric). This resolves the ordering ambiguity noted by prior work on undirected graph networks [22].

### 3.3.3 Scalar Force-Coefficient MLP

A standard MLP processes the symmetric scalar features to produce three force coefficients:

$$\left(\alpha_1^{ij},\; \alpha_2^{ij},\; \alpha_3^{ij}\right) = \text{MLP}_\theta\!\left(\boldsymbol{\sigma}_{ij}^{\text{ext}}\right) \in \mathbb{R}^3$$

Because the MLP operates on rotation-invariant, pair-order-invariant inputs, the resulting coefficients are deterministic functions of the *geometric* configuration — they do not depend on the global coordinate frame or the arbitrary labeling of $i$ vs. $j$.

### 3.3.4 Vectorization: Scalar Coefficients $\to$ 3D Force

The three scalar coefficients are combined with the frame basis vectors to reconstruct a 3D force vector for the canonical direction $i \to j$:

$$\mathbf{F}_{ij} = \alpha_1^{ij}\,\mathbf{e}_1^{ij} + \alpha_2^{ij}\,\mathbf{e}_2^{ij} + \alpha_3^{ij}\,\mathbf{e}_3^{ij}$$

The three components have direct physical interpretations:
- $\alpha_1 \mathbf{e}_1$: **normal force** along the line connecting the bodies (repulsion/attraction),
- $\alpha_2 \mathbf{e}_2$: **tangential friction** along the relative sliding direction,
- $\alpha_3 \mathbf{e}_3$: **lateral force** perpendicular to both the normal and sliding directions.

### 3.3.5 Antisymmetric Assignment and Aggregation

This is the step that enforces Newton's third law. Rather than independently computing $\mathbf{F}_{ij}$ and $\mathbf{F}_{ji}$ (which would require careful symmetry management across separate MLP evaluations), we process each *undirected* pair $\{i, j\}$ exactly once and assign equal-and-opposite forces:

$$\text{Node } j \text{ receives } +\mathbf{F}_{ij}, \qquad \text{Node } i \text{ receives } -\mathbf{F}_{ij}$$

The net force on each node is obtained by aggregation over all incident pairs:

$$\mathbf{F}_i = \sum_{j:\{i,j\} \in \mathcal{E}} \mathbf{F}_{\{i,j\} \to i}$$

where $\mathbf{F}_{\{i,j\} \to i} = -\mathbf{F}_{ij}$ if $i < j$ and $\mathbf{F}_{\{i,j\} \to i} = +\mathbf{F}_{ji}$ if $j < i$.

### 3.3.6 Node Update

Node embeddings are updated using the aggregated forces via a residual MLP:

$$\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}_{\text{upd}}^{(\ell)}\!\left(\left[\mathbf{h}_i^{(\ell)} \;\|\; \mathbf{F}_i^{(\ell)}\right]\right)$$

The full SV message-passing layer is applied for $L$ rounds ($\ell = 0, \ldots, L-1$), with separate parameters per layer. The frame and scalar features are recomputed at each layer using the original geometric state (positions and velocities are *not* updated by message passing; only the learned embeddings $\mathbf{h}_i$ evolve).

## 3.4 Conservation Guarantees

We now state and prove the central theoretical property of SV message passing.

**Theorem 1 (Linear Momentum Conservation).** *For any network parameters $\theta$ and any input state $\{\mathbf{s}_i\}_{i=1}^N$, the total predicted force over all bodies vanishes:*

$$\sum_{i=1}^{N} \mathbf{F}_i = \mathbf{0}$$

*Proof.* By the aggregation rule in §3.3.5, each undirected pair $\{i, j\} \in \mathcal{E}$ contributes $+\mathbf{F}_{ij}$ to node $j$ and $-\mathbf{F}_{ij}$ to node $i$. Summing over all nodes:

$$\sum_{i=1}^{N} \mathbf{F}_i = \sum_{\{i,j\} \in \mathcal{E}} \left(+\mathbf{F}_{ij} + (-\mathbf{F}_{ij})\right) = \sum_{\{i,j\} \in \mathcal{E}} \mathbf{0} = \mathbf{0}$$

The cancellation is *exact* (up to floating-point precision) and holds for any force vector $\mathbf{F}_{ij} \in \mathbb{R}^3$ — it does not depend on the specific values produced by the MLP, the structure of the graph, or the training stage. $\square$

**Remark 1 (Comparison with directed-edge approaches).** An alternative design computes $\mathbf{F}_{ij}$ and $\mathbf{F}_{ji}$ independently via directed edges and relies on input symmetry ($\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$) plus frame antisymmetry to guarantee cancellation. While mathematically equivalent for exact arithmetic, this approach introduces subtle failure modes. For the binormal component, $\mathbf{e}_3^{ij} = \mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$ is *symmetric* ($\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$, since both $\mathbf{e}_1$ and $\mathbf{e}_2$ flip sign), so $\alpha_3$ must be explicitly antisymmetrized — e.g., by multiplying by the signed binormal velocity $v_b^{ij}$ which satisfies $v_b^{ij} = -v_b^{ji}$. Our undirected-pair approach avoids this complexity entirely: by computing each force once and hard-coding the sign flip, conservation is guaranteed *trivially*, regardless of frame symmetry properties.

**Remark 2 (Angular momentum).** Conservation of angular momentum additionally requires all forces to be central ($\mathbf{F}_{ij} \parallel \mathbf{r}_{ij}$), i.e., $\alpha_2 = \alpha_3 = 0$. This constraint is deliberately *not* enforced, as tangential forces (friction) are essential for manipulation. PhysRobot conserves linear momentum but not angular momentum — a principled trade-off between physical fidelity and modeling capacity.

**Remark 3 (Energy).** Energy conservation is not architecturally enforced because manipulation involves non-conservative processes (friction dissipation, external work by actuators). Instead, we include an optional soft energy regularizer (§3.5) that encourages predictions consistent with the work-energy theorem.

## 3.5 Dual-Stream PPO Integration

PhysRobot integrates the SV-GNN physics module into a PPO-based RL agent through a dual-stream architecture with stop-gradient fusion.

### 3.5.1 Architecture Overview

The complete architecture consists of two parallel feature-extraction streams and a fusion module:

```
Observation s_t ──┬──► [Policy Stream]  ──────────► z_policy ∈ R^d
                  │     MLP(s_t)
                  │
                  └──► [Physics Stream] ──► sg(·) ──► z_physics ∈ R^{3N_obj}
                        SV-GNN(G^t)          ↑
                                        stop-gradient
                        
z = Fusion(z_policy, z_physics) ──► PPO Actor ──► π(a | s)
                                 ──► PPO Critic ──► V(s)
```

**Policy stream.** A standard 2-layer MLP processes the full observation vector:

$$\mathbf{z}_{\text{policy}} = \text{MLP}_{\text{policy}}(\mathbf{s}^t) \in \mathbb{R}^d$$

This stream handles non-physical aspects of the task (goal specification, joint limits, etc.) and provides a fallback representation when the physics stream is uninformative.

**Physics stream.** The SV-GNN (§3.3) processes the scene graph and produces per-node force predictions. The output is the concatenation of predicted accelerations (forces under unit-mass assumption) for the object nodes:

$$\mathbf{z}_{\text{physics}} = \left[\hat{\mathbf{a}}_1,\; \hat{\mathbf{a}}_2,\; \ldots,\; \hat{\mathbf{a}}_{N_{\text{obj}}}\right] \in \mathbb{R}^{3N_{\text{obj}}}$$

where $\hat{\mathbf{a}}_i$ is the predicted acceleration of object $i$, obtained as the net force on node $i$ from the last SV message-passing layer.

**Stop-gradient.** The physics features are detached from the computational graph before fusion:

$$\mathbf{z}_{\text{physics}}^{\text{sg}} = \text{sg}\!\left(\mathbf{z}_{\text{physics}}\right)$$

This is a critical design choice. Without stop-gradient, the PPO policy gradient would backpropagate through the fusion layer into the SV-GNN, distorting its learned dynamics to maximize reward rather than predict accurate physics. The stop-gradient decouples the two learning objectives: the physics stream learns dynamics, and the policy stream learns to *use* physics features for action selection.

**Fusion.** The two streams are concatenated and projected to a common feature space:

$$\mathbf{z} = \text{ReLU}\!\left(\mathbf{W}_f \left[\mathbf{z}_{\text{policy}} \;\|\; \mathbf{z}_{\text{physics}}^{\text{sg}}\right] + \mathbf{b}_f\right) \in \mathbb{R}^d$$

The fused features feed into the PPO actor (outputting action mean $\boldsymbol{\mu}$ and log-standard-deviation $\log\boldsymbol{\sigma}$) and critic (outputting scalar value estimate $\hat{V}$).

### 3.5.2 Loss Function

The total training loss combines three components:

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{phys}}\,\mathcal{L}_{\text{phys}} + \lambda_{\text{reg}}\,\mathcal{L}_{\text{reg}}$$

**RL loss (PPO).** The standard clipped surrogate objective with value function loss and entropy bonus:

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] + c_v \mathcal{L}_{\text{VF}} - c_e H[\pi_\theta]$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{\text{old}}}(a_t | s_t)$ is the importance ratio, $\hat{A}_t$ is the GAE advantage estimate, and $\epsilon = 0.2$ is the clipping threshold. The RL loss backpropagates only through the policy stream and fusion module (not the physics stream, due to stop-gradient).

**Physics auxiliary loss.** The physics stream is trained with a self-supervised dynamics prediction loss using state transitions collected during RL rollouts:

$$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{s}^t, \mathbf{s}^{t+1}) \in \mathcal{B}} \sum_{i \in \mathcal{V}_{\text{obj}}} \left\|\hat{\mathbf{a}}_i^t - \mathbf{a}_i^{t, \text{fd}}\right\|^2$$

where $\mathbf{a}_i^{t, \text{fd}} = (\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ is the finite-difference acceleration computed from consecutive observations. This loss is *self-supervised*: it requires no ground-truth force labels, no access to the simulator's internal state, and no differentiable physics engine. The data comes for free from the RL rollout buffer.

**Energy regularizer (optional).** A soft regularizer encourages energy-consistent predictions:

$$\mathcal{L}_{\text{reg}} = \frac{1}{|\mathcal{B}|} \sum_t \left|\hat{E}_{\text{kin}}^{t+1} - \hat{E}_{\text{kin}}^t + W_{\text{ext}}^t + D^t\right|$$

where $\hat{E}_{\text{kin}}^t = \sum_i \frac{1}{2}m_i \|\dot{\mathbf{x}}_i^t\|^2$ is the kinetic energy, $W_{\text{ext}}^t$ is estimated external work from applied torques, and $D^t = \text{MLP}_D(\mathbf{s}^t) \geq 0$ is a learned dissipation estimate. This loss encourages consistency with the work-energy theorem without requiring strict energy conservation.

**Loss weight schedule.** The physics loss weight is linearly warmed up over the first $T_{\text{warmup}}$ steps:

$$\lambda_{\text{phys}}(t) = \lambda_0 \cdot \min\!\left(1,\; t / T_{\text{warmup}}\right)$$

This allows the RL policy to stabilize before the physics stream's gradient signals become significant, preventing early-training instabilities from conflicting gradients.

### 3.5.3 Computational Complexity

The SV-GNN adds minimal overhead compared to a standard MLP policy. Per SV message-passing layer, the dominant cost is the scalar MLP evaluated once per undirected edge pair: $O(|\mathcal{E}| \cdot d_h^2)$, where $d_h$ is the hidden dimension. Node updates cost $O(|\mathcal{V}| \cdot d_h^2)$. For $L$ layers, the total is $O(L(|\mathcal{E}| + |\mathcal{V}|) d_h^2)$.

For the PushBox task ($|\mathcal{V}| = 2$, $|\mathcal{E}| = 2$, $L = 1$, $d_h = 32$), this amounts to $\sim$8K FLOPs per forward pass — negligible compared to a single MuJoCo simulation step ($\sim$1M FLOPs). For multi-object scenes ($|\mathcal{V}| = 6$, $|\mathcal{E}| \approx 20$, $L = 2$, $d_h = 64$), the cost grows to $\sim$330K FLOPs, still well within real-time budgets. Total parameter count for the recommended configuration ($d_h = 64$, $L = 2$) is approximately 46K for the physics stream and 79K for the full architecture, keeping the model lightweight and suitable for high-frequency control.
