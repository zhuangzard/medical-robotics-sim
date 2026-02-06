# Environment Specification — Canonical Reference

> **Status**: AUTHORITATIVE — all code, paper text, and experiments MUST match this document.  
> **Last updated**: 2026-02-06  
> **Resolves**: BUG-1 (two conflicting PushBoxEnv), paper outline 7-DOF/18-dim mismatch

---

## 1. PushBoxEnv (Single-Object Push)

### 1.1 Overview

| Property | Value |
|----------|-------|
| **Source file** | `environments/push_box.py` |
| **Class** | `PushBoxEnv` |
| **Factory** | `make_push_box_env(box_mass=0.5)` |
| **Physics engine** | MuJoCo ≥ 3.0 |
| **MuJoCo XML** | `environments/assets/push_box.xml` |

### 1.2 Robot

| Property | Value |
|----------|-------|
| Type | 2-DOF planar arm |
| Joint 1 (shoulder) | Hinge, axis=[0,0,1], range=[-180°, 180°], damping=0.5 |
| Joint 2 (elbow) | Hinge, axis=[0,0,1], range=[-180°, 180°], damping=0.5 |
| Upper arm length | 0.4 m (capsule, radius=0.025 m) |
| Forearm length | 0.3 m (capsule, radius=0.025 m) |
| Max reach | 0.7 m from origin |
| End-effector | Site at tip of forearm |

### 1.3 Object (Box)

| Property | Value |
|----------|-------|
| Shape | Box, half-extents = [0.05, 0.05, 0.05] m |
| Default mass | 0.5 kg |
| OOD mass range | [0.25, 3.0] kg |
| Joint | Free joint (6-DOF: 3 translation + 3 rotation) |
| Friction | μ_slide=0.5, μ_torsion=0.005, μ_roll=0.0001 |
| Initial position | x ∈ [0.25, 0.45], y ∈ [-0.15, 0.15], z = 0.05 |

### 1.4 Goal

| Property | Value |
|----------|-------|
| Position | Fixed at [0.5, 0.3, 0.02] |
| Success threshold | 0.1 m (box center to goal, 2D Euclidean) |
| Termination | Immediate on success |

### 1.5 Observation Space (16-dim)

| Index | Name | Dim | Description |
|-------|------|-----|-------------|
| 0-1 | `joint_pos` | 2 | Shoulder, elbow angles [rad] |
| 2-3 | `joint_vel` | 2 | Shoulder, elbow angular velocities [rad/s] |
| 4-6 | `ee_pos` | 3 | End-effector Cartesian position [m] (from MuJoCo site) |
| 7-9 | `box_pos` | 3 | Box center position [m] (from freejoint qpos) |
| 10-12 | `box_vel` | 3 | Box translational velocity [m/s] (from freejoint qvel) |
| 13-15 | `goal_pos` | 3 | Goal position [m] |
| **Total** | | **16** | `np.float32` |

### 1.6 Action Space (2-dim)

| Index | Name | Range | Unit |
|-------|------|-------|------|
| 0 | `shoulder_torque` | [-10, 10] | Nm |
| 1 | `elbow_torque` | [-10, 10] | Nm |

### 1.7 Reward Function

```
r1 = -‖ee_pos[:2] − box_pos[:2]‖₂     (reach: encourage EE near box)
r2 = -‖box_pos[:2] − goal_pos[:2]‖₂    (push: encourage box near goal)
reward = 0.5 × r1 + r2 + 100 × 𝟙[success]
```

### 1.8 Physics / Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Timestep (dt) | 0.002 s |
| Substeps per env step | 5 |
| Effective control frequency | 50 Hz (0.002 × 5 = 0.01 s per step) |
| Integrator | Euler (MuJoCo default) |
| Gravity | [0, 0, -9.81] m/s² |
| Max episode steps | 500 |
| Max episode time | 5.0 s |

### 1.9 Observation Index Map (for downstream code)

```python
# Canonical observation slicing — use these constants
OBS_JOINT_POS  = slice(0, 2)     # joint_pos
OBS_JOINT_VEL  = slice(2, 4)     # joint_vel
OBS_EE_POS     = slice(4, 7)     # ee_pos (3D)
OBS_BOX_POS    = slice(7, 10)    # box_pos (3D)
OBS_BOX_VEL    = slice(10, 13)   # box_vel (3D)
OBS_GOAL_POS   = slice(13, 16)   # goal_pos (3D)
```

---

## 2. MultiObjectPushEnv (Multi-Object Push)

### 2.1 Overview

| Property | Value |
|----------|-------|
| **Source file** | `environments/multi_object_push.py` |
| **Class** | `MultiObjectPushEnv` |
| **Factory** | `make_multi_push_env(n_objects=3, task_mode='push_to_goals')` |
| **Physics engine** | Analytical (no MuJoCo dependency) |
| **Portability** | Colab-compatible, no native library needed |

### 2.2 Robot

| Property | Value |
|----------|-------|
| Type | Point-mass (2D) |
| Mass | 1.0 kg (fixed) |
| Radius | 0.04 m |

### 2.3 Objects

| Property | Value |
|----------|-------|
| Count | 2–8 (configurable, default 3) |
| Shape | Circle, radius = 0.05 m |
| Mass range | [0.5, 2.0] kg (randomized per episode) |
| Collision | Elastic, restitution = 0.6 |

### 2.4 Observation Space (variable dim)

```
obs_dim = 4 + n_objects × 5 + n_objects × 2
        = 4 + 7 × n_objects
```

| Component | Dim | Description |
|-----------|-----|-------------|
| Robot pos + vel | 4 | [x, y, vx, vy] |
| Per object (×N) | 5 | [x, y, vx, vy, mass] |
| Per goal (×N) | 2 | [gx, gy] |

Examples: N=3 → 25-dim, N=5 → 39-dim

### 2.5 Action Space (2-dim)

| Index | Name | Range | Unit |
|-------|------|-------|------|
| 0 | `force_x` | [-1, 1] | normalized (scaled to ±10 N internally) |
| 1 | `force_y` | [-1, 1] | normalized |

### 2.6 Task Modes

| Mode | Description | Goal |
|------|-------------|------|
| `push_to_goals` | Each object → assigned random goal | Per-object distance < 0.1 m |
| `sort_by_mass` | Sort left→right by mass | Objects at evenly-spaced x positions |
| `group_by_color` | Group by mass category | Light/medium/heavy in 3 zones |
| `stack` | Push all to single zone | All objects within 0.1 m of [0.5, 0.5] |

### 2.7 Physics Parameters

| Parameter | Value |
|-----------|-------|
| Simulation dt | 0.02 s |
| Substeps | 5 |
| Ground friction | 0.5 |
| Restitution | 0.6 |
| Workspace | [-1, 1] × [-1, 1] m |
| Max episode steps | 500 |

### 2.8 Graph Interface (for GNN models)

```python
graph = env.build_graph()
# Returns: {'x': [N+1, 6], 'pos': [N+1, 2], 'edge_index': [2, E], 'edge_attr': [E, 5]}
# Nodes: [robot, obj_0, ..., obj_{N-1}], fully connected (no self-loops)
# E = (N+1) × N

graph_3d = env.build_graph_3d()
# Returns: {'positions': [N+1, 3], 'velocities': [N+1, 3], 'edge_index': [2, E], 'masses': [N+1]}
# 2D→3D padding (z=0) for compatibility with EdgeFrame/DynamicalGNN
```

---

## 3. Deprecated Files

| File | Reason | Replacement |
|------|--------|-------------|
| `environments/push_box_deprecated.py` | Old 10-dim obs version; missing ee_pos, inconsistent reward | `environments/push_box.py` |
| `environments/push_box_env.py` | Duplicate of canonical (was the source before unification) | `environments/push_box.py` |

---

## 4. Version History

| Date | Version | Change |
|------|---------|--------|
| 2026-02-05 | v0.1 | Initial `push_box.py` (10-dim obs) |
| 2026-02-06 | v0.2 | Added `push_box_env.py` (16-dim obs, fixed reward) |
| 2026-02-06 | v0.3 | Added `multi_object_push.py` (analytical physics, graph interface) |
| 2026-02-06 | **v1.0** | **Unified**: 16-dim version becomes canonical `push_box.py`; old version deprecated; all imports updated; paper outline corrected from 7-DOF/18-dim to 2-DOF/16-dim |

---

## 5. Verification Checklist

- [x] `environments/push_box.py` obs_dim = 16
- [x] `environments/push_box.py` action_dim = 2
- [x] `environments/__init__.py` exports `PushBoxEnv`, `MultiObjectPushEnv`
- [x] `baselines/*.py` all import from `environments.push_box`
- [x] `training/*.py` all import from `environments.push_box`
- [x] `PAPER_OUTLINE.md` says 16-dim state, 2-dim action, 2-DOF
- [x] `EXPERIMENT_DESIGN.md` says 16-dim state, 2-dim action
- [x] Colab notebook uses inline 16-dim PushBoxEnv
- [x] No remaining imports of `push_box_env` (except deprecated file)
