# PhysRobot — Complete Experiment Design

**Author**: Experiment Lead (paper-experiment)  
**Date**: 2026-02-06  
**Status**: V1 — Awaiting V2 Colab results  
**Target**: ICRA 2027 / CoRL 2026 submission  

---

## 0. Executive Summary

This document defines the full experimental campaign for the PhysRobot paper. It covers:
- **7 ablation variants × 5 seeds** on the core PushBox task
- **3 environments** of increasing difficulty
- **6 baselines** (RL + physics-informed)
- **5 evaluation metrics** with statistical rigor
- **Colab training plan** with GPU time estimates and parallelization strategy

### Current Status (V1 → V2)

| Method | V1 Result | V2 Target | V2 Status |
|--------|-----------|-----------|-----------|
| Pure PPO | 6% | >50% | ⏳ Running on Colab |
| GNS | 0% | >30% | ⏳ Running on Colab |
| PhysRobot | 0% | >30% | ⏳ Running on Colab |

V2 fixes (from `EXPERT_DEBATE_AND_SOLUTION.md`):
1. Reward redesign: progress-based + 500 success bonus + action penalty
2. Parameter reduction: GNS 500K→5K, PhysRobot 391K→6K
3. Timesteps: 200K→500K
4. Exploration: ent_coef=0.01
5. Success threshold: 0.1→0.15m

---

## 1. Ablation Study

### 1.1 Overview

**Design**: 5 seeds × 7 variants × 3 environments = **105 training runs**  
**Seeds**: `[42, 123, 256, 789, 1024]`  
**Reporting**: Mean ± std across seeds; Welch's t-test for significance (p < 0.05)

### 1.2 Variant Definitions

#### Variant 1: Pure PPO (Baseline)

The minimal RL baseline. No graph structure, no physics priors.

```python
# Config
agent = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, ent_coef=0.01)
# net_arch: default [64, 64] shared → pi + vf heads
# Params: ~10K
```

**Purpose**: Establishes the "pure RL" performance ceiling on each environment. All other methods must beat this to justify their added complexity.

#### Variant 2: GNS (Graph without Physics)

Graph neural network over the object scene, but no physics constraints (no conservation laws, no antisymmetric forces).

```python
# Config (V2 — lightweight)
features_extractor = GNSFeaturesExtractorV2(obs_space, features_dim=64)
# - node_encoder: Linear(6→32) + ReLU
# - edge_encoder: Linear(4→32) + ReLU
# - 1× GNSGraphLayerV2 (MessagePassing, hidden=32)
# - decoder: Linear(32→3)
# - feature_proj: Linear(19→64) + ReLU
# Params: ~5K
```

**Purpose**: Isolates the value of graph structure alone. Difference between Variant 2 and Variant 1 = "what does a graph representation buy you?"

#### Variant 3: PhysRobot-Full (Our Method)

Full proposed method: physics-informed feature extractor with relative-geometry physics stream + policy stream + fusion.

```python
# Config (V2 — lightweight MLP physics, no GNN on 2-node)
features_extractor = PhysRobotFeaturesExtractorV2(obs_space, features_dim=64)
# - physics_net: Linear(9→32→3) — predicts box acceleration from relative geometry
# - policy_stream: Linear(16→64→64) — processes raw obs
# - fusion: Linear(67→64) + ReLU — merges physics + policy
# Params: ~6K
```

**Purpose**: The full system we claim in the paper. Must demonstrate improvement over Variant 1 (PPO) in at least one axis: sample efficiency, OOD generalization, or multi-object scaling.

#### Variant 4: PhysRobot-no-Conservation (Ablation)

PhysRobot with physics_net but **no conservation loss**. The physics stream predicts box acceleration but is not constrained to conserve momentum.

```python
# Same as Variant 3 but:
# - No auxiliary conservation loss (Lcons = 0)
# - Physics_net trained purely by RL gradient
# In V2 this is actually identical to Variant 3 (no conservation loss yet)
# In V3+ we add conservation loss, and this ablation removes it
```

**Purpose**: Measures the contribution of the conservation law constraint. If Variant 3 ≈ Variant 4, the conservation loss isn't helping (on in-distribution). If Variant 3 >> Variant 4 on OOD, conservation is key for generalization.

**Implementation note**: This ablation becomes meaningful only after V3 adds the explicit conservation loss. For V2, Variant 4 = Variant 3 (degenerate). Mark as "deferred to V3."

#### Variant 5: PhysRobot-no-EdgeFrame (Ablation)

PhysRobot with physics but **no edge-local coordinate frame**. Forces are computed in the global frame rather than relative to the edge direction.

```python
# Variant 5: replace physics_net input
# Instead of [rel_pos, rel_vel, dist, goal_dir] (edge-local-ish)
# Use [ee_pos, box_pos, box_vel, goal_pos] (global frame, 12-dim)
self.physics_net_global = nn.Sequential(
    nn.Linear(12, 32), nn.ReLU(),
    nn.Linear(32, 3)
)
# Params: ~1.5K (slightly different from V3)
```

**Purpose**: Measures the contribution of relative/edge-local features vs global features. This tests whether the relative-geometry inductive bias matters.

#### Variant 6: PhysRobot Graph Scaling (2-node vs 5-node vs 10-node)

Tests GNN scalability by varying graph resolution.

```
6a: 2-node graph = [end-effector, box] (current)
6b: 5-node graph = [base, shoulder, elbow, ee, box]
6c: 10-node graph = [base, shoulder, elbow, ee, box, goal, 4×waypoints]
```

**Implementation**:
```python
# 6b: 5-node arm graph
# Nodes: [base(fixed), shoulder(joint), elbow(joint), ee(actuated), box(free)]
# Edges: base→shoulder, shoulder→elbow, elbow→ee, ee↔box (kinematic chain + contact)
# Node features: [pos(3), vel(3)] = 6-dim per node
# Edge features: [rel_pos(3), dist(1)] = 4-dim per edge

# 6c: 10-node enriched graph
# Additional nodes: goal(target), 4×workspace_grid_points (spatial anchors)
# Edges: k-nearest-neighbor (k=3) + kinematic chain
```

**Purpose**: The key paper claim is that PhysRobot scales to complex scenes via GNN. This variant proves it. On 2-node, GNN ≈ MLP. On 5+ nodes, GNN should outperform. **This is essential for the multi-object environment.**

**Note**: Variant 6 only makes sense in the multi-object environment (Section 2.2). On PushBox (1 box), a 5-node graph is overhead.

#### Variant 7: PhysRobot + Symplectic Integrator

Uses a symplectic (energy-conserving) numerical integrator inside the physics stream instead of a vanilla MLP predictor.

```python
# Replace physics_net with a symplectic step
# H(q, p) learned by HNN-style network
# q_next, p_next = symplectic_euler(H, q, p, dt)
class SymplecticPhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(6, 32), nn.Softplus(),  # smooth activation for H
            nn.Linear(32, 1)
        )
    def forward(self, q, p):
        # q = rel_pos [B,3], p = rel_vel [B,3] (momentum proxy)
        qp = torch.cat([q, p], dim=-1)
        qp.requires_grad_(True)
        H = self.hamiltonian_net(qp)
        dH = torch.autograd.grad(H.sum(), qp, create_graph=True)[0]
        dHdq, dHdp = dH[:, :3], dH[:, 3:]
        # Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq
        # Symplectic Euler: p_new = p - dt*dH/dq, q_new = q + dt*dH/dp
        dt = 0.01
        p_new = p - dt * dHdq
        q_new = q + dt * dHdp
        acc = (p_new - p) / dt  # effective acceleration
        return acc  # [B, 3]
# Params: ~1.3K
```

**Purpose**: Tests whether energy-conserving structure in the physics predictor improves long-horizon prediction and policy stability. Links our work to the HNN/SympNet literature.

### 1.3 Ablation Summary Table

| ID | Variant | Physics | Graph | Conservation | EdgeFrame | Symplectic | Params | Priority |
|----|---------|---------|-------|-------------|-----------|------------|--------|----------|
| V1 | Pure PPO | ✗ | ✗ | ✗ | ✗ | ✗ | ~10K | P0 |
| V2 | GNS | ✗ | ✓ | ✗ | ✗ | ✗ | ~5K | P0 |
| V3 | PhysRobot-full | ✓ | ✓* | ✓** | ✓ | ✗ | ~6K | P0 |
| V4 | No-conservation | ✓ | ✓* | ✗ | ✓ | ✗ | ~6K | P1 (deferred) |
| V5 | No-edgeframe | ✓ | ✗ | ✗ | ✗ | ✗ | ~5K | P1 |
| V6a-c | Graph scaling | ✓ | ✓ (2/5/10) | ✓** | ✓ | ✗ | 6K–20K | P2 |
| V7 | + Symplectic | ✓ | ✓* | ✓ | ✓ | ✓ | ~5K | P2 |

\* V2 uses MLP (no GNN) on 2-node; switches to GNN for multi-object  
\** Conservation loss added in V3+ implementation

---

## 2. Environments

### 2.1 Environment 1: PushBox (Current)

**Difficulty**: Easy  
**Description**: 2-DOF planar arm pushes a single box to a goal position.

| Property | Value |
|----------|-------|
| State dim | 16 (joint_pos(2), joint_vel(2), ee_pos(3), box_pos(3), box_vel(3), goal(3)) |
| Action dim | 2 (shoulder torque, elbow torque) |
| Action range | [-10, 10] Nm |
| Episode length | 500 steps |
| Success threshold | 0.15m (V2) |
| Box mass (train) | 0.5 kg |
| Randomization | Joint angles ±0.5 rad, box_x [0.25, 0.45], box_y [-0.15, 0.15] |

**Role in paper**: Primary benchmark. All ablations run here first.

### 2.2 Environment 2: Multi-Object PushBox

**Difficulty**: Medium  
**Description**: Same arm pushes 3–5 boxes to respective goal positions.

| Property | Value |
|----------|-------|
| State dim | 16 + 6×(N-1) per extra box (pos(3), vel(3)) |
| Action dim | 2 |
| Boxes | 3 (easy), 4 (medium), 5 (hard) |
| Success criteria | All boxes within 0.15m of respective goals |
| Mass distribution | Uniform [0.3, 1.0] kg per box |

**Implementation plan**:
```xml
<!-- Additional boxes in MuJoCo XML -->
<body name="box2" pos="0.35 0.15 0.05">
  <freejoint name="box2_freejoint"/>
  <geom name="box2_geom" type="box" size="0.05 0.05 0.05"
        mass="0.7" rgba="0.2 0.2 0.8 1"/>
</body>
<!-- + goal sites -->
<site name="goal2" pos="0.5 -0.2 0.02" size="0.06" rgba="0 0 1 0.4"/>
```

**Why this matters**: 
- **GNN advantage**: With 5 boxes, the interaction graph has 6 nodes (ee + 5 boxes) and up to 30 edges. GNN naturally handles variable-sized graphs; MLP cannot.
- **Physics advantage**: Box-box collisions require understanding of contact physics. Conservation laws become more impactful with more interacting objects.
- **Scalability story**: Same model works for 3, 4, 5 boxes without retraining (zero-shot transfer via GNN).

### 2.3 Environment 3: Sorting Task

**Difficulty**: Hard  
**Description**: Arm must push colored boxes to matching colored goal zones. Requires planning — pushing box A to goal A may require first moving box B out of the way.

| Property | Value |
|----------|-------|
| Boxes | 3 (colored: red, green, blue) |
| Goals | 3 (colored zones, randomly placed) |
| Success criteria | Each box in its matching goal zone |
| Additional obs | Box color encoding (one-hot per box) |
| Episode length | 1000 steps (longer for planning) |

**Why this matters**:
- Tests whether physics priors help with **sequential planning** (not just reactive pushing)
- GNN message passing can propagate "obstacle awareness" (box B blocks path to goal A)
- Strongest test of generalization: agent must compose primitive skills

**Implementation priority**: P2 (after PushBox and Multi-Object are solid)

### 2.4 Environment Summary

| Env | Nodes | Edges | Episode | Difficulty | Priority |
|-----|-------|-------|---------|------------|----------|
| PushBox | 2 | 2 | 500 | Easy | P0 |
| Multi-3Box | 4 | 12 | 500 | Medium | P1 |
| Multi-5Box | 6 | 30 | 500 | Medium-Hard | P1 |
| Sorting-3 | 4 | 12 | 1000 | Hard | P2 |

---

## 3. Evaluation Metrics

### 3.1 Primary Metrics

#### M1: Success Rate (SR)

**Definition**: Fraction of evaluation episodes where the task is completed.  
**Protocol**: 
- 100 episodes per evaluation (V2; 50 in V1 was too noisy)
- Deterministic policy (no exploration noise)
- Random initial conditions (same seed set across all methods for fair comparison)
- **Report**: Mean ± std across 5 seeds

```python
# Fixed evaluation seeds for reproducibility
EVAL_SEEDS = list(range(10000, 10100))  # 100 fixed seeds
```

**Statistical test**: Welch's t-test between methods; significance at p < 0.05. Report 95% CI.

#### M2: Sample Efficiency (SE)

**Definition**: Number of training episodes until first success (and until sustained >50% success).  
**Protocol**:
- Log success at every episode during training via `SuccessTrackingCallback`
- "First success" = first episode with `info['success'] == True`
- "Sustained success" = rolling window of 20 episodes with >50% success rate
- **Report**: Median across 5 seeds (more robust than mean for this metric)

```python
# Two measures:
SE_first = episodes_to_first_success  # lower is better
SE_50 = episodes_to_sustained_50pct   # lower is better
```

#### M3: OOD Generalization

**Definition**: Performance on environments with modified physical parameters, tested on a trained policy (zero-shot, no fine-tuning).

**OOD conditions**:

| Parameter | Train Value | OOD Test Range | N Points |
|-----------|------------|---------------|----------|
| Box mass | 0.5 kg | [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0] | 7 |
| Box friction | 0.5 | [0.1, 0.3, 0.5, 0.7, 1.0] | 5 |
| Box size | 0.05m | [0.03, 0.05, 0.07, 0.1] | 4 |
| Goal distance | ~0.3m | [0.1, 0.2, 0.3, 0.5, 0.7] | 5 |

**Reporting**:
- **OOD robustness score**: Area under the SR-vs-parameter curve, normalized by in-distribution SR
- **Degradation ratio**: SR(OOD) / SR(in-distribution), averaged across conditions
- Plots: SR vs parameter value for each method (with 95% CI bands)

**This is PhysRobot's key differentiator.** If PhysRobot-full degrades less than PPO on mass changes, the physics prior is validated.

#### M4: Training Time

**Definition**: Wall-clock time for the full training run.  
**Protocol**: Measured on Colab T4 GPU (consistent hardware across all runs).

**Report**: Mean ± std across 5 seeds. Also report as a ratio to PPO time (overhead factor).

#### M5: Parameter Count

**Definition**: Total trainable parameters (features extractor + policy + value networks).  
**Protocol**: `sum(p.numel() for p in model.parameters())`

**Report**: Single number per variant.

### 3.2 Secondary Metrics (for analysis section)

| Metric | Description | Purpose |
|--------|-------------|---------|
| M6: Reward curve | Episode return vs training step | Visualize learning dynamics |
| M7: Physics prediction error | MSE of predicted vs actual box acceleration | Validate physics stream |
| M8: Gradient norm | Per-layer gradient norm during training | Diagnose training stability |
| M9: Value function accuracy | MSE of V(s) vs actual returns | PPO-specific diagnostic |
| M10: Contact frequency | % of steps where ee touches box | Behavioral analysis |

### 3.3 Metric Collection Infrastructure

```python
class ExperimentLogger:
    """Unified logging for all metrics."""
    def __init__(self, method_name, seed, save_dir):
        self.method = method_name
        self.seed = seed
        self.save_dir = save_dir
        self.data = {
            'episodes': [],
            'rewards': [],
            'successes': [],
            'physics_errors': [],
            'gradient_norms': [],
            'timestamps': []
        }
    
    def log_episode(self, reward, success, info):
        self.data['episodes'].append(len(self.data['episodes']))
        self.data['rewards'].append(reward)
        self.data['successes'].append(success)
        self.data['timestamps'].append(time.time())
    
    def save(self):
        path = f"{self.save_dir}/{self.method}_seed{self.seed}.json"
        with open(path, 'w') as f:
            json.dump(self.data, f)
```

---

## 4. Baselines

### 4.1 RL Baselines

#### B1: PPO (Proximal Policy Optimization)

- **Implementation**: Stable-Baselines3 `PPO`
- **Config**: lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.01
- **net_arch**: pi=[64,64], vf=[64,64]
- **Params**: ~10K
- **Reference**: Schulman et al. (2017)

#### B2: SAC (Soft Actor-Critic)

- **Implementation**: Stable-Baselines3 `SAC`
- **Config**: lr=3e-4, buffer_size=100K, batch_size=256, tau=0.005
- **net_arch**: pi=[64,64], qf=[64,64]
- **Params**: ~15K
- **Reference**: Haarnoja et al. (2018)
- **Why**: SAC is the standard off-policy baseline. Expected to be more sample-efficient than PPO on continuous control.

```python
from stable_baselines3 import SAC
sac_agent = SAC('MlpPolicy', env, learning_rate=3e-4,
    buffer_size=100_000, batch_size=256, tau=0.005,
    ent_coef='auto', verbose=0)
```

#### B3: TD3 (Twin Delayed DDPG)

- **Implementation**: Stable-Baselines3 `TD3`
- **Config**: lr=1e-3, buffer_size=100K, batch_size=256
- **Params**: ~15K
- **Reference**: Fujimoto et al. (2018)
- **Why**: Another strong off-policy baseline, often best on continuous robotic tasks.

```python
from stable_baselines3 import TD3
td3_agent = TD3('MlpPolicy', env, learning_rate=1e-3,
    buffer_size=100_000, batch_size=256, verbose=0)
```

### 4.2 Physics-Informed Baselines

#### B4: GNS (Graph Network Simulator — DeepMind Style)

- **Implementation**: Our `GNSFeaturesExtractorV2` (V2: lightweight)
- **Original reference**: Sanchez-Gonzalez et al. (2020)
- **Adaptation**: GNS predicts next-state acceleration from graph; used as PPO features extractor
- **Params**: ~5K (V2)
- **Why**: Closest existing method to PhysRobot but without conservation constraints.

#### B5: HNN (Hamiltonian Neural Network)

- **Implementation**: Custom `HNNFeaturesExtractor`
- **Reference**: Greydanus et al. (2019)
- **Architecture**: Learn H(q,p), derive forces via autograd
- **Adaptation**: HNN predicts dynamics; predictions fed as features to PPO

```python
class HNNFeaturesExtractor(BaseFeaturesExtractor):
    """Hamiltonian Neural Network as PPO feature extractor."""
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        # Hamiltonian network: H(q, p) → scalar
        self.H_net = nn.Sequential(
            nn.Linear(6, 64), nn.Softplus(),
            nn.Linear(64, 64), nn.Softplus(),
            nn.Linear(64, 1)
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(3 + 16, features_dim), nn.ReLU()
        )
    
    def forward(self, observations):
        # q = relative position, p = relative velocity (momentum proxy)
        ee_pos = observations[:, 4:7]
        box_pos = observations[:, 7:10]
        box_vel = observations[:, 10:13]
        q = box_pos - ee_pos
        p = box_vel
        
        qp = torch.cat([q, p], dim=-1).requires_grad_(True)
        H = self.H_net(qp)
        dH = torch.autograd.grad(H.sum(), qp, create_graph=True)[0]
        # Hamilton's equations: dp/dt = -dH/dq → acceleration
        acc = -dH[:, :3]
        
        combined = torch.cat([acc, observations], dim=-1)
        return self.feature_proj(combined)
```

- **Params**: ~10K
- **Why**: HNN is the standard energy-conserving baseline. Tests whether Hamiltonian structure alone is sufficient (vs our graph + conservation approach).

#### B6: Dreamer v3

- **Implementation**: DreamerV3 codebase (Hafner et al., 2023)
- **Reference**: Hafner et al. (2023) "Mastering Diverse Domains through World Models"
- **Adaptation**: World-model based RL; learns environment dynamics internally
- **Why**: State-of-the-art model-based RL. Tests whether an implicit world model matches our explicit physics model.

**Implementation note**: Dreamer v3 requires significant engineering effort. Options:
1. **Option A**: Use official DreamerV3 repo, adapt to PushBox env (preferred)
2. **Option B**: Use MBPO (Model-Based Policy Optimization) as a simpler stand-in
3. **Option C**: Cite Dreamer results from literature if our env is standard enough

**Priority**: P2 (implement after core results are solid)

### 4.3 Baseline Summary

| ID | Method | Type | On-Policy | Physics | Params | Priority |
|----|--------|------|-----------|---------|--------|----------|
| B1 | PPO | RL | ✓ | ✗ | ~10K | P0 |
| B2 | SAC | RL | ✗ | ✗ | ~15K | P1 |
| B3 | TD3 | RL | ✗ | ✗ | ~15K | P1 |
| B4 | GNS | Physics+RL | ✓ | Partial | ~5K | P0 |
| B5 | HNN | Physics+RL | ✓ | ✓ (energy) | ~10K | P1 |
| B6 | Dreamer v3 | Model-based | ✗ | Implicit | ~500K | P2 |

---

## 5. Colab Training Plan

### 5.1 GPU Resource Estimation

**Hardware**: Google Colab T4 GPU (free tier: ~12h/day; Pro: ~24h, A100 available)

**Per-run time estimates** (based on V1 timings, scaled to V2):

| Method | V1 Time (200K) | V2 Estimate (500K) | Per Seed |
|--------|-----------------|---------------------|----------|
| Pure PPO | 4.5 min | ~12 min | 12 min |
| SAC | — | ~15 min | 15 min |
| TD3 | — | ~15 min | 15 min |
| GNS V2 | ~10 min* | ~25 min | 25 min |
| PhysRobot V2 | ~10 min* | ~25 min | 25 min |
| HNN | — | ~20 min | 20 min |
| Eval (OOD) | — | ~10 min | 10 min |

\* V1 GNS took 28 min with 500K params; V2 with 5K params should be much faster.

### 5.2 Total GPU Time Budget

#### Phase 1: Core Results (P0) — MUST COMPLETE

| Experiment | Runs | Time/Run | Total |
|------------|------|----------|-------|
| PPO × 5 seeds | 5 | 12 min | 1h |
| GNS × 5 seeds | 5 | 25 min | 2h |
| PhysRobot × 5 seeds | 5 | 25 min | 2h |
| OOD eval (3 methods × 7 masses × 5 seeds) | 15 | 10 min | 2.5h |
| **Phase 1 Total** | | | **7.5h** |

#### Phase 2: Extended Baselines (P1)

| Experiment | Runs | Time/Run | Total |
|------------|------|----------|-------|
| SAC × 5 seeds | 5 | 15 min | 1.25h |
| TD3 × 5 seeds | 5 | 15 min | 1.25h |
| HNN × 5 seeds | 5 | 20 min | 1.7h |
| Variant 5 (no-edgeframe) × 5 seeds | 5 | 25 min | 2h |
| OOD eval (4 new methods) | 20 | 10 min | 3.3h |
| **Phase 2 Total** | | | **9.5h** |

#### Phase 3: Multi-Object + Advanced (P2)

| Experiment | Runs | Time/Run | Total |
|------------|------|----------|-------|
| Multi-3Box (PPO, GNS, PhysRobot × 5 seeds) | 15 | 40 min | 10h |
| Multi-5Box (same) | 15 | 60 min | 15h |
| Variant 6b (5-node) × 5 seeds | 5 | 30 min | 2.5h |
| Variant 6c (10-node) × 5 seeds | 5 | 40 min | 3.3h |
| Variant 7 (symplectic) × 5 seeds | 5 | 25 min | 2h |
| Sorting task (3 methods × 5 seeds) | 15 | 90 min | 22.5h |
| Dreamer v3 (if implemented) | 5 | 120 min | 10h |
| **Phase 3 Total** | | | **65h** |

#### Grand Total

| Phase | GPU Hours | Calendar Days (Colab free) | Calendar Days (Colab Pro) |
|-------|-----------|---------------------------|--------------------------|
| P0 | 7.5h | 1 day | 1 day |
| P1 | 9.5h | 1 day | 1 day |
| P2 | 65h | 6 days | 3 days |
| **Total** | **82h** | **8 days** | **5 days** |

### 5.3 Parallelization Strategy

#### What CAN be parallelized (independent runs):

```
Colab Session 1                    Colab Session 2
├── PPO seed 42                    ├── PPO seed 123
├── PPO seed 256                   ├── PPO seed 789
├── PPO seed 1024                  │
├── GNS seed 42                    ├── GNS seed 123
├── GNS seed 256                   ├── GNS seed 789
└── GNS seed 1024                  └── ...
```

With 2 Colab sessions: **Phase 1 time halved to ~4h**

#### What CANNOT be parallelized:
- Runs within the same session (GPU memory constraint)
- OOD evaluation depends on trained models from the same method+seed

#### Recommended execution order:

```
Day 1 (P0):
  Session A: PPO (5 seeds) → GNS (5 seeds)     [~3h]
  Session B: PhysRobot (5 seeds) → OOD eval     [~4.5h]

Day 2 (P1):
  Session A: SAC (5 seeds) → TD3 (5 seeds)      [~2.5h]
  Session B: HNN (5 seeds) → Variant 5 (5 seeds) [~3.7h]
  Session C: OOD eval for new methods            [~3.3h]

Day 3-7 (P2):
  Multi-object and sorting (sequential, larger runs)
```

### 5.4 Data Collection Strategy

#### Saved artifacts per run:

```
{SAVE_DIR}/
├── models/
│   ├── {method}_seed{seed}.zip          # SB3 model checkpoint
│   └── {method}_seed{seed}_best.zip     # Best model (by eval SR)
├── results/
│   ├── {method}_seed{seed}_train.json   # Training log (per-episode)
│   ├── {method}_seed{seed}_eval.json    # Evaluation results (100 ep)
│   ├── {method}_seed{seed}_ood.json     # OOD results (7 masses × 100 ep)
│   └── summary.json                     # Aggregated results
├── logs/
│   └── {method}_seed{seed}/             # TensorBoard logs
└── figures/
    ├── success_rates.png
    ├── ood_generalization.png
    ├── learning_curves.png
    └── ablation_table.png
```

#### Checkpointing strategy:

```python
class CheckpointCallback(BaseCallback):
    """Save model every N steps and on best eval performance."""
    def __init__(self, save_freq=50000, save_dir='', method='', seed=0):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.method = method
        self.seed = seed
        self.best_sr = 0
    
    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_dir}/models/{self.method}_seed{self.seed}_step{self.num_timesteps}"
            self.model.save(path)
        return True
```

#### Google Drive sync:
- All results auto-saved to Drive (survives Colab disconnects)
- Periodic backup: `!cp -r {SAVE_DIR} /content/drive/MyDrive/backup_$(date +%Y%m%d)/`

### 5.5 Notebook Structure

One master notebook per phase:

```
colab/
├── week1_full_training_v2.ipynb     # Phase 1: PPO, GNS, PhysRobot (current)
├── week1_baselines_v2.ipynb         # Phase 2: SAC, TD3, HNN, ablations
├── week2_multiobject.ipynb          # Phase 3a: Multi-object environments
├── week2_sorting.ipynb              # Phase 3b: Sorting task
├── week2_dreamer.ipynb              # Phase 3c: Dreamer v3 (if implemented)
└── analysis/
    ├── aggregate_results.ipynb      # Combine all results, generate tables/figures
    └── statistical_tests.ipynb      # Significance tests, confidence intervals
```

---

## 6. Expected Results & Hypotheses

### 6.1 Hypothesis Matrix

| # | Hypothesis | How to Test | Expected Outcome | Evidence For/Against |
|---|-----------|-------------|------------------|---------------------|
| H1 | PhysRobot-full ≥ PPO on SR | V3 vs V1, PushBox | PhysRobot ≥ PPO (small margin) | V2 results pending |
| H2 | PhysRobot > PPO on OOD | V3 vs V1, mass sweep | PhysRobot degrades 30% less | Key paper claim |
| H3 | GNN helps on multi-object | V6b vs V1, Multi-3Box | GNS/PhysRobot >> PPO | GNN's inductive bias |
| H4 | Conservation loss helps OOD | V3 vs V4, mass sweep | V3 > V4 on OOD, ≈ on in-dist | Needs V3 conservation impl |
| H5 | Edge-local frames > global | V3 vs V5, PushBox | V3 ≥ V5 (SE or OOD) | Tests relative-feature value |
| H6 | PhysRobot is sample-efficient | V3 vs V1, SE metric | PhysRobot SE_first < PPO | Physics priors guide exploration |
| H7 | Symplectic integration helps | V7 vs V3, long episodes | V7 ≥ V3 on stability | Energy conservation |

### 6.2 Expected Results Table (Post-V2)

#### PushBox (500K steps, success threshold 0.15m)

| Method | SR (in-dist) | SR (mass=2.0) | SE_first | Time | Params |
|--------|-------------|---------------|----------|------|--------|
| PPO | 55±8% | 15±5% | 150±50 | 12m | 10K |
| SAC | 65±6% | 20±5% | 100±30 | 15m | 15K |
| TD3 | 60±7% | 18±5% | 120±40 | 15m | 15K |
| GNS V2 | 45±10% | 25±8% | 200±80 | 25m | 5K |
| HNN | 50±8% | 30±7% | 180±60 | 20m | 10K |
| **PhysRobot V2** | **55±8%** | **35±6%** | **120±40** | 25m | 6K |
| Dreamer v3 | 70±5% | 25±8% | 80±30 | 120m | 500K |

**Key takeaway**: PhysRobot should match PPO in-distribution but significantly outperform on OOD (mass=2.0: 35% vs 15%).

#### Multi-3Box (1M steps)

| Method | SR | Time |
|--------|-----|------|
| PPO | 20±10% | 30m |
| GNS V2 | 35±8% | 40m |
| **PhysRobot V2** | **40±8%** | 40m |

**Key takeaway**: GNN structure becomes essential with multiple objects. PPO's flat MLP cannot handle variable-size scenes.

### 6.3 Failure Modes to Watch For

1. **PhysRobot ≈ PPO on everything**: Physics stream adds no value. Mitigation: check if physics_pred has non-zero gradient; if not, the fusion layer is ignoring it.

2. **GNS > PhysRobot**: Graph structure helps, but physics constraints hurt. Mitigation: ablate conservation loss (V4); if V4 > V3, conservation is too restrictive.

3. **All methods <30% on PushBox**: Environment is harder than expected. Mitigation: increase to 1M steps; try position control; further relax success threshold.

4. **OOD generalization flat across methods**: All methods equally bad at transfer. Mitigation: try larger mass range (0.01 to 100 kg); add friction/size OOD.

5. **High variance across seeds**: 5 seeds insufficient for statistical claims. Mitigation: increase to 10 seeds for key comparisons.

---

## 7. Paper Figure Plan

Based on the experiment results, these figures will go into the paper:

| Fig # | Content | Section | Data Source |
|-------|---------|---------|-------------|
| 1 | Architecture diagram (PhysRobot) | Method (§3) | Diagram (paper-writer) |
| 2 | Bar chart: Success rate comparison (6 methods) | Experiments (§4) | Phase 1+2 |
| 3 | Learning curves (reward vs timesteps, 5 seeds CI) | Experiments (§4) | Phase 1 |
| 4 | OOD generalization plot (SR vs mass, all methods) | Experiments (§4) | Phase 1 OOD |
| 5 | Ablation table (formatted as small figure) | Analysis (§5) | Phase 1+2 |
| 6 | Multi-object scaling (SR vs #objects) | Analysis (§5) | Phase 3 |
| 7 | Sample efficiency (episodes to first success, boxplot) | Analysis (§5) | Phase 1+2 |

---

## 8. Timeline & Milestones

| Date | Milestone | Depends On |
|------|-----------|-----------|
| 2026-02-06 | V2 Colab running | ✅ Done |
| 2026-02-07 | V2 results available | Colab completion |
| 2026-02-07 | Phase 1 (P0) complete: PPO, GNS, PhysRobot × 5 seeds | V2 results ≥ targets |
| 2026-02-08 | Phase 2 (P1) complete: SAC, TD3, HNN, ablations | Phase 1 OK |
| 2026-02-09 | Conservation loss implemented (V3) | paper-algorithm |
| 2026-02-10 | Variant 4 ablation (no-conservation) runnable | V3 code |
| 2026-02-12 | Multi-object environment ready | paper-code |
| 2026-02-14 | Phase 3 (P2) complete: multi-object, sorting | Phase 2 OK + env |
| 2026-02-16 | All results aggregated, figures generated | All phases |
| 2026-02-18 | Paper draft with full experimental section | paper-writer |

---

## 9. Communication to Other Agents

### To `paper-algorithm`:
- **Request**: Implement conservation loss (Lcons) for Variant 4 ablation; implement proper Scalarization-Vectorization for the full PhysRobot architecture (V3). Current V2 uses a simple MLP physics stream — we need the theoretically grounded version for the paper.
- **Priority**: High — we need V3 code by Feb 9 to run Phase 2 ablations.

### To `paper-writer`:
- **Delivery**: This EXPERIMENT_DESIGN.md contains the full experimental protocol. Use Section 6.2 (Expected Results) as a template for the paper's experiment section. Update with actual numbers once V2 results arrive.
- **Figures**: See Section 7 for the figure plan. Begin drafting figure captions.

### To `paper-reviewer`:
- **Review request**: Evaluate whether these experiments are sufficient for ICRA/CoRL. Specifically:
  - Is 5 seeds enough? (Standard is 3–10; we use 5)
  - Are the baselines complete? (Missing: MPC, ILC, other control baselines)
  - Is the OOD evaluation convincing? (Mass sweep + friction + size)
  - Are there obvious experiments we're missing?

### To `paper-code`:
- **Request**: Implement the multi-object PushBox environment (Section 2.2) and the additional baselines (SAC, TD3, HNN — Section 4). The Colab notebooks need to be extended per Section 5.5.
- **Priority**: Multi-object env is P1 (needed by Feb 12); SAC/TD3 are P1; HNN is P1; Dreamer is P2.

---

## Appendix A: Detailed Hyperparameter Grid

### A.1 PPO Hyperparameters (Fixed)

| Param | Value | Justification |
|-------|-------|--------------|
| learning_rate | 3e-4 | SB3 default, proven on MuJoCo |
| n_steps | 2048 | Balance between update frequency and data |
| batch_size | 64 | Fits T4 GPU memory |
| n_epochs | 10 | SB3 default, sufficient for 2048 batch |
| gamma | 0.99 | Standard for continuous control |
| gae_lambda | 0.95 | Standard |
| clip_range | 0.2 | SB3 default |
| ent_coef | 0.01 | V2 change: encourage exploration |
| vf_coef | 0.5 | SB3 default |
| max_grad_norm | 0.5 | SB3 default |
| n_envs | 4 | Parallel environments for sample efficiency |

### A.2 SAC Hyperparameters

| Param | Value |
|-------|-------|
| learning_rate | 3e-4 |
| buffer_size | 100,000 |
| learning_starts | 1,000 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |
| ent_coef | auto |

### A.3 TD3 Hyperparameters

| Param | Value |
|-------|-------|
| learning_rate | 1e-3 |
| buffer_size | 100,000 |
| learning_starts | 1,000 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |
| policy_delay | 2 |
| target_noise_std | 0.2 |

---

## Appendix B: Reproducibility Checklist

- [x] Random seeds specified (42, 123, 256, 789, 1024)
- [x] Hardware specified (Colab T4 GPU)
- [x] Library versions pinned (SB3, MuJoCo, PyTorch, PyG)
- [x] Evaluation seeds fixed (10000–10099)
- [x] Hyperparameters fully listed
- [x] Environment XML included in notebook
- [ ] Code released on GitHub (after submission)
- [ ] Docker/conda environment file
- [ ] Pre-trained model checkpoints
- [ ] Raw training logs
