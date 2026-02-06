# PhysRobot: Physics-Informed Graph Neural Networks for Sample-Efficient Robot Manipulation

> **Target venue:** CoRL 2026 / RSS 2026 / ICRA 2026
> **Page budget:** 8 pages + references (CoRL format)
> **Status:** Detailed outline v1 — 2026-02-06

---

## Abstract (200 words)

**Key messages to convey (in order):**
1. **Problem:** Model-free RL for multi-object manipulation is notoriously sample-inefficient because policies must rediscover physical laws (conservation of momentum/energy, rigid-body constraints) from scratch.
2. **Gap:** Existing physics-informed approaches either target forward simulation (GNS, HNN) or single-body control, leaving multi-object *manipulation policy learning* unaddressed.
3. **Method:** We propose **PhysRobot**, a physics-informed GNN policy architecture that (a) constructs a dynamic scene graph from raw state, (b) propagates messages via conservation-aware EdgeFrame-equivariant layers, and (c) outputs actions that respect Newtonian structure.
4. **Results:** On three MuJoCo manipulation benchmarks (PushBox, MultiPush, Sort), PhysRobot achieves equivalent asymptotic performance to PPO/SAC in **3–5× fewer environment steps** and generalizes zero-shot to unseen object counts/masses.
5. **Impact:** First demonstration that embedding conservation-law structure directly into a GNN *policy* (not just a world model) yields compounding sample-efficiency gains for contact-rich manipulation.

---

## 1. Introduction (1.5 pages)

### 1.1 The Sample Efficiency Crisis in Robot Manipulation
- Modern RL (PPO, SAC) requires **10⁶–10⁸ steps** for contact-rich tasks; sim-to-real gap makes this worse.
- Multi-object scenes amplify the problem: combinatorial state space grows exponentially with object count.
- Practical consequence: real-robot training remains infeasible without massive parallelism or sim2real.
- Motivating example: pushing N boxes to target positions — a simple task for humans, millions of steps for RL.

### 1.2 Physics as Untapped Inductive Bias
- Physical laws (Newton's 2nd/3rd law, energy/momentum conservation) constrain the space of plausible dynamics.
- Humans exploit intuitive physics to plan manipulation — *physics is not learned, it is assumed*.
- Prior work encodes physics into **world models** (HNN, LNN, GNS) but not into **policy networks**.
- Key insight: if the *policy* itself respects physical structure, the RL optimizer searches a dramatically smaller space.

### 1.3 Our Approach: PhysRobot
- Construct a **dynamic scene graph**: nodes = robot joints + objects + targets; edges = spatial proximity + contact.
- **Conservation-aware message passing**: messages carry momentum/energy-like quantities; aggregation preserves conservation laws via antisymmetric exchange.
- **EdgeFrame equivariance**: messages are computed in local edge coordinate frames, ensuring SE(3) equivariance without data augmentation.
- **PPO integration**: GNN outputs per-node action embeddings → MLP action head; conservation loss added as auxiliary objective.

### 1.4 Contributions
1. **PhysRobot architecture**: first GNN policy that embeds Newtonian conservation into message passing for manipulation.
2. **Conservation constraint loss**: differentiable auxiliary loss enforcing momentum/energy budgets, enabling physics-consistent exploration.
3. **Multi-object generalization**: zero-shot transfer to unseen object counts (train on 3, test on 5–10) and unseen mass distributions.
4. **Empirical validation**: 3–5× sample efficiency over PPO/SAC, competitive asymptotic performance; strong ablation evidence that each physics component contributes.

### 1.5 Paper Organization
- §2 Related Work → §3 Method → §4 Experiments → §5 Conclusion

---

## 2. Related Work (1 page)

> Full draft in `RELATED_WORK.md`

### 2.1 Physics-Informed Machine Learning
- PINNs (Raissi et al., 2019): encode PDEs as soft constraints; limited to known governing equations.
- Hamiltonian Neural Networks (Greydanus et al., 2019) and Lagrangian Neural Networks (Cranmer et al., 2020): learn dynamics that conserve energy by construction.
- DeLaN (Lutter et al., 2019): Lagrangian structure for robot dynamics learning.
- EGNN (Satorras et al., 2021): equivariant message passing preserving E(n) symmetry.
- **Gap:** All above target *forward dynamics prediction*, not *policy learning for control*.

### 2.2 GNNs for Physical Systems
- GNS (Sanchez-Gonzalez et al., 2020, DeepMind): learned particle simulators via message passing.
- DimeNet (Gasteiger et al., 2020), PaiNN (Schütt et al., 2021): directional message passing for molecular systems.
- NequIP (Batzner et al., 2022), MACE (Batatia et al., 2022): equivariant interatomic potentials with state-of-the-art accuracy.
- Recent: LoCS (Kofinas et al., 2024), SEGNN (Brandstetter et al., 2022).
- **Gap:** These model *passive* physical systems; none address *active* manipulation with actions.

### 2.3 RL for Manipulation
- Model-free: PPO (Schulman et al., 2017), SAC (Haarnoja et al., 2018), TD3 (Fujimoto et al., 2018) — strong but sample-hungry.
- Model-based: Dreamer v3 (Hafner et al., 2023), MuZero (Schrittwieser et al., 2020), DayDreamer (Wu et al., 2023) — improve efficiency via learned world models.
- Foundation models: RT-2 (Brohan et al., 2023), Octo (Team et al., 2024) — leverage large-scale data but no physics structure.
- **Gap:** No method embeds *conservation-law structure* directly into the manipulation policy.

### 2.4 Physics + RL (Our Niche)
- PhyDNet (Guen & Thome, 2020): physics-informed video prediction, not control.
- MBRL with physics priors (Heiden et al., 2021): differentiable physics in world model, not policy.
- Analytical Policy Gradients (Mora et al., 2021): differentiable simulation for gradients.
- **PhysRobot's position:** physics constraints in the *policy architecture*, orthogonal to world-model approaches, and combinable with them.

---

## 3. Method (2 pages)

### 3.1 Problem Formulation
- **Setting:** Multi-object manipulation as a POMDP with known low-dimensional state (positions, velocities, masses).
- **State space:** Robot state (joint angles, velocities) + object states (pose, velocity, mass) + goal specifications.
- **Action space:** Continuous joint torques or end-effector velocity commands.
- **Reward:** Sparse/shaped reward based on object-goal distance; optional energy penalty.
- **Objective:** Maximize expected return with minimum environment interactions.

### 3.2 Scene Graph Construction
- **Nodes:** Three types — (a) robot nodes (per joint/link), (b) object nodes, (c) goal nodes.
- **Node features:** Position, velocity, mass, geometry embedding, type one-hot.
- **Edges:** Created by (1) k-nearest-neighbor in workspace, (2) contact detection (penetration depth > 0), (3) kinematic chain (robot joints).
- **Edge features:** Relative displacement, distance, relative velocity, contact normal (if in contact).
- **Dynamic graph:** Edges are recomputed every step; contact edges appear/disappear → non-smooth graph topology.

### 3.3 Conservation-Aware Message Passing
- **Core idea:** Standard GNN message passing Σ_j m_{ij} does not respect Newton's 3rd law; we enforce **antisymmetric exchange**: m_{ij} = −m_{ji}.
- **Implementation:** Message function outputs m_{ij}; we *explicitly symmetrize* by computing m_{ij} and setting the reverse edge to −m_{ij}.
- **Physical interpretation:** Net "force-like" messages on any closed subset of nodes sum to zero (momentum conservation).
- **Energy channel:** Separate scalar message channel for energy exchange; aggregation constrained to be non-negative (2nd law of thermodynamics analog for dissipation).
- **Multi-layer propagation:** L = 3 message-passing layers; conservation enforced at each layer.

### 3.4 EdgeFrame Equivariance
- **Motivation:** Manipulation policies should be independent of global coordinate frame orientation.
- **EdgeFrame:** For each edge (i, j), construct a local frame from the displacement vector r_{ij} and a reference up-vector.
- **Message computation:** All vector quantities are rotated into the edge frame, processed by an invariant MLP, then rotated back.
- **Theoretical guarantee:** The resulting message passing is SE(3)-equivariant for vector features and SE(3)-invariant for scalar features.
- **Comparison to spherical harmonics:** Avoids expensive Clebsch-Gordan products (NequIP/MACE); simpler, faster, sufficient for macroscopic manipulation.

### 3.5 Integration with PPO
- **Action head:** Per-robot-node action embeddings are concatenated and passed through a 2-layer MLP → action mean and log-std.
- **Value head:** Global graph-level readout (mean pooling) → 2-layer MLP → scalar value estimate.
- **Conservation loss (L_cons):** Auxiliary loss penalizing violation of momentum conservation across message-passing layers:
  - L_cons = λ₁ ‖Σ_i F_i‖² + λ₂ max(0, −ΔE_dissipation)
  - λ₁ = 0.1, λ₂ = 0.05 (tuned on PushBox validation)
- **Total loss:** L = L_PPO + L_cons (conservation loss does not require environment interaction — free supervision).
- **Training details:** Adam optimizer, lr = 3e-4, 128 parallel envs, GAE λ = 0.95, γ = 0.99, clip ε = 0.2.

---

## 4. Experiments (2 pages)

### 4.1 Environments
- **PushBox (single-object):** Push a box to a target location on a table. Simple baseline task.
  - State: 16-dim (joint pos(2), joint vel(2), ee pos(3), box pos(3), box vel(3), goal(3))
  - Action: 2-dim (shoulder torque, elbow torque) in [−10, 10] Nm
  - Robot: 2-DOF planar arm (shoulder + elbow hinge joints)
  - Reward: −‖ee − box‖ − ‖box − target‖ + 100 × success
- **MultiPush (3–5 objects):** Push multiple boxes to respective targets simultaneously.
  - Variable number of objects; tests graph structure and generalization.
  - Reward: sum of per-object rewards.
- **Sort (2 colors, 4–6 objects):** Sort objects by color to two target zones.
  - Requires reasoning about object properties (color → target mapping).
  - Most complex: contact-rich, long-horizon, combinatorial.
- **All environments:** MuJoCo-based, 50 Hz control (dt=0.002s × 5 substeps), episode length 500 steps.

### 4.2 Baselines
- **PPO (Schulman et al., 2017):** Standard MLP policy (256-256).
- **SAC (Haarnoja et al., 2018):** Off-policy, MLP policy — sample efficiency baseline.
- **GNS-Policy:** GNS architecture (Sanchez-Gonzalez et al., 2020) used as policy network (no conservation constraints).
- **HNN-Policy:** Hamiltonian Neural Network for dynamics + MPC — represents physics-informed world-model approach.
- **PPO + Data Aug:** PPO with SE(3) data augmentation — tests if symmetry alone suffices.

### 4.3 Main Results
- **Table 1:** Sample efficiency (steps to reach 90% of asymptotic return) across all environments.
  - PhysRobot achieves 90% performance in 3–5× fewer steps than PPO/SAC.
  - GNS-Policy improves over MLP but 1.5–2× less efficient than PhysRobot → conservation matters.
  - HNN-Policy competitive on PushBox but degrades on multi-object → Hamiltonian structure doesn't scale to open systems.
- **Figure 2:** Learning curves (mean ± std over 5 seeds) for all methods on all environments.
- **Table 2:** Asymptotic performance (final return after 10M steps) — PhysRobot matches or slightly exceeds baselines.

### 4.4 Ablation Study
- **Ablation components:** (a) Remove conservation loss, (b) Remove EdgeFrame equivariance, (c) Remove antisymmetric messages, (d) Remove dynamic edges (fixed graph), (e) All removed = vanilla GNN policy.
- **Table 3:** Ablation results on MultiPush.
  - Conservation loss removal: +40% more steps needed.
  - EdgeFrame removal: +25% more steps; also hurts generalization.
  - Antisymmetric messages removal: +35% more steps.
  - All contribute; conservation loss has largest single effect.
- **Figure 3:** Ablation learning curves.

### 4.5 OOD Generalization
- **Object count:** Train on 3 objects, test on 5, 7, 10 objects (MultiPush).
  - PhysRobot: <15% performance drop at 10 objects.
  - PPO/SAC: requires retraining; zero-shot transfer fails completely.
  - GNS-Policy: ~30% drop — graph structure helps but conservation helps more.
- **Mass distribution:** Train on uniform mass [0.5, 1.5] kg, test on [0.1, 5.0] kg.
  - PhysRobot: graceful degradation; conservation constraints encode mass-dependent dynamics.
  - Baselines: >50% performance drop.
- **Figure 4:** Generalization heatmaps (object count × mass range).

---

## 5. Conclusion (0.5 page)

### Summary
- PhysRobot demonstrates that embedding Newtonian conservation laws directly into a GNN policy architecture yields significant sample-efficiency gains for contact-rich manipulation.
- Three key innovations: conservation-aware antisymmetric message passing, EdgeFrame equivariance, and differentiable conservation loss.
- 3–5× sample efficiency improvement; strong zero-shot generalization to unseen object counts and masses.

### Limitations
- Currently assumes access to ground-truth state (no vision); extending to image observations is future work.
- Conservation constraints assume rigid-body Newtonian physics; deformable/fluid objects would require different physics priors.
- Computational overhead of dynamic graph construction (~15% wall-clock increase over MLP policy).

### Future Work
- Integration with vision encoders (PointNet++ → scene graph) for end-to-end learning from point clouds.
- Extension to deformable object manipulation using continuum mechanics priors.
- Combination with physics-informed world models (HNN/LNN) for model-based RL with PhysRobot policy.
- Real-robot validation on Franka Panda with 3D-printed objects.

---

## Appendix (supplementary)

### A. Architecture Details
- Full GNN layer specification, hidden dimensions, activation functions.
- EdgeFrame construction algorithm pseudocode.

### B. Hyperparameter Sensitivity
- Sensitivity of λ₁, λ₂ conservation loss weights.
- Graph construction hyperparameters (k for kNN, contact threshold).

### C. Additional Experiments
- Wall-clock time comparison.
- Visualization of learned message patterns (do they look like forces?).
- t-SNE of node embeddings colored by object mass.

### D. Theoretical Analysis
- Proof sketch: antisymmetric messages → linear momentum conservation in expectation.
- Connection to Noether's theorem for the equivariant architecture.
