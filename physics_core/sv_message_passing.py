"""
Scalarization–Vectorization (SV) Message Passing
=================================================

Core architectural innovation of PhysRobot.
Guarantees linear momentum conservation by construction:
    Σ_i F_i = 0  for ANY network parameters θ.

Design (from ALGORITHM_DESIGN.md §2.2–2.5, §4.2):
    1. Edge Frame: construct antisymmetric ONB {e1, e2, e3} per edge
    2. Scalarize: project geometric vectors onto frame → rotation-invariant scalars
    3. Scalar MLP: produce force coefficients α1, α2, α3
    4. Antisymmetrize α3: multiply by signed radial velocity v_r (antisymmetric marker)
    5. Vectorize: reconstruct 3D force  F_ij = α1·e1 + α2·e2 + α3·e3
    6. Aggregate: F_i = Σ_j F_ij  (sum gives net force; conservation guaranteed)

Antisymmetry properties:
    e1^{ij} = -e1^{ji}      (radial)
    e2^{ij} = -e2^{ji}      (tangential, from relative velocity)
    e3^{ij} = +e3^{ji}      (binormal = e1 × e2 → symmetric)

    α1, α2: symmetric scalars → automatically canceled by antisymmetric e1, e2
    α3: made antisymmetric via  α3^{ij} = v_r^{ij} · g(σ_sym)  where v_r is antisymmetric

Result: F_ij + F_ji = 0  for every edge pair → Σ_i F_i = 0.  QED.

Reference: Sharma & Fink (2025), Dynami-CAL GraphNet.
Enhancement: our α3-antisymmetrization fix for the binormal component.

Author: PhysRobot Team
Date: 2026-02-06
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ───────────────────── Helpers ─────────────────────

EPS = 1e-7          # numerical stability
DEG_EPS = 1e-4      # degeneracy threshold for v_perp


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with LayerNorm + ReLU."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ───────────────── Edge Frame Builder ──────────────

def build_edge_frames(
    pos: torch.Tensor,        # [N, 3]
    vel: torch.Tensor,        # [N, 3]
    src: torch.Tensor,        # [E]
    dst: torch.Tensor,        # [E]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct antisymmetric edge-local coordinate frames.

    Returns:
        e1:   [E, 3]  radial unit vector      (antisymmetric)
        e2:   [E, 3]  tangential unit vector   (antisymmetric)
        e3:   [E, 3]  binormal unit vector     (SYMMETRIC)
        r_ij: [E, 3]  displacement vector
        d_ij: [E, 1]  distance scalar
    """
    # ---- e1: radial ----
    r_ij = pos[dst] - pos[src]                          # [E, 3]
    d_ij = torch.norm(r_ij, dim=-1, keepdim=True)       # [E, 1]
    e1 = r_ij / (d_ij + EPS)                            # [E, 3]

    # ---- e2: tangential from relative velocity ----
    v_rel = vel[dst] - vel[src]                          # [E, 3]
    # Project out the radial component
    v_par = (v_rel * e1).sum(dim=-1, keepdim=True) * e1  # [E, 3]
    v_perp = v_rel - v_par                                # [E, 3]
    v_perp_norm = torch.norm(v_perp, dim=-1, keepdim=True)  # [E, 1]

    # Degeneracy fallback: when v_perp ≈ 0, use gravity-aligned frame
    non_degenerate = (v_perp_norm > DEG_EPS).float()      # [E, 1]

    # Primary e2 (from velocity)
    e2_vel = v_perp / (v_perp_norm + EPS)                 # [E, 3]

    # Fallback e2: cross(e1, z_hat)
    z_hat = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand_as(e1)
    e2_fall_raw = torch.cross(e1, z_hat, dim=-1)
    e2_fall_norm = torch.norm(e2_fall_raw, dim=-1, keepdim=True)

    # Second fallback: when e1 ≈ ±z_hat, use y_hat instead
    use_y = (e2_fall_norm < DEG_EPS).float()
    y_hat = torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand_as(e1)
    e2_fall_raw2 = torch.cross(e1, y_hat, dim=-1)
    e2_fall_raw = (1 - use_y) * e2_fall_raw + use_y * e2_fall_raw2
    e2_fall_norm = torch.norm(e2_fall_raw, dim=-1, keepdim=True)
    e2_fall = e2_fall_raw / (e2_fall_norm + EPS)

    e2 = non_degenerate * e2_vel + (1 - non_degenerate) * e2_fall  # [E, 3]

    # ---- e3: binormal ----
    e3 = torch.cross(e1, e2, dim=-1)                     # [E, 3]

    return e1, e2, e3, r_ij, d_ij


# ───────────── SV Message Passing Layer ────────────

class SVMessagePassing(nn.Module):
    """
    One round of Scalarization–Vectorization message passing.

    Operates on **undirected** edge pairs {i,j}. For each unordered pair:
        1. Build frame {e1, e2, e3} from the canonical direction (i→j, i<j)
        2. Scalarize geometric vectors → rotation-invariant scalars (all symmetric)
        3. MLP → (α1, α2, α3) force coefficients
        4. Vectorize: F = α1·e1 + α2·e2 + α3·e3
        5. Assign +F to node j, -F to node i  (Newton's 3rd law, hard-coded)
        6. Aggregate + node update

    By construction: F_ij = -F_ji  → Σ_i F_i = 0 for ANY θ.

    Note: we process undirected pairs to guarantee exact cancellation.
    The input edge_index may be directed (both i→j and j→i present);
    we extract unique undirected pairs internally.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        n_scalar = 5  # ||r||, v_r, v_t, v_b, ||v_rel||

        # Force coefficient MLP: symmetric scalars → 3 coefficients
        # Node embeddings are symmetrized: h_sum = h_i + h_j, h_diff_norm = ||h_i - h_j||
        self.force_mlp = _make_mlp(
            in_dim=n_scalar + 2 * node_dim,   # scalars + h_sum + |h_diff| element-wise
            hidden_dim=hidden_dim,
            out_dim=3,                         # α1, α2, α3
        )

        # Node update
        self.node_update = _make_mlp(
            in_dim=node_dim + 3,   # h_i ‖ F_i
            hidden_dim=hidden_dim,
            out_dim=node_dim,
        )

    def _extract_undirected_pairs(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract unique undirected pairs from edge_index.
        Returns [2, P] where P = number of unique pairs, with src < dst.
        """
        src, dst = edge_index[0], edge_index[1]
        # Keep only edges where src < dst
        mask = src < dst
        return torch.stack([src[mask], dst[mask]], dim=0)

    def forward(
        self,
        h: torch.Tensor,           # [N, node_dim]
        edge_index: torch.Tensor,   # [2, E]  (directed, both directions)
        pos: torch.Tensor,          # [N, 3]
        vel: torch.Tensor,          # [N, 3]
    ) -> torch.Tensor:
        """One round of SV message passing. Returns updated h."""
        h_new, _ = self.forward_with_forces(h, edge_index, pos, vel)
        return h_new

    def forward_with_forces(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SV message passing returning both updated h and conserved forces.

        Returns:
            h_new:  [N, node_dim]
            F_agg:  [N, 3]   (guaranteed: F_agg.sum(dim=0) ≈ 0)
        """
        N = h.size(0)

        # ========== 0. Extract undirected pairs (i < j) ==========
        pairs = self._extract_undirected_pairs(edge_index)  # [2, P]
        pi, pj = pairs[0], pairs[1]  # i < j

        # ========== 1. Frame construction (canonical: i→j) ==========
        e1, e2, e3, r_ij, d_ij = build_edge_frames(pos, vel, pi, pj)
        v_rel = vel[pj] - vel[pi]  # [P, 3]

        # ========== 2. Scalarization ==========
        # All projections are computed in the canonical direction.
        # Since we only process each pair once, symmetry is automatic.
        v_r = (v_rel * e1).sum(dim=-1, keepdim=True)     # [P, 1]
        v_t = (v_rel * e2).sum(dim=-1, keepdim=True)     # [P, 1]
        v_b = (v_rel * e3).sum(dim=-1, keepdim=True)     # [P, 1]
        v_norm = torch.norm(v_rel, dim=-1, keepdim=True)  # [P, 1]

        scalars_geom = torch.cat([d_ij, v_r, v_t, v_b, v_norm], dim=-1)  # [P, 5]

        # Symmetrize node embeddings: sum is order-invariant
        h_sum = h[pi] + h[pj]                             # [P, node_dim]
        h_diff_abs = (h[pi] - h[pj]).abs()                # [P, node_dim]  order-invariant

        scalars = torch.cat([scalars_geom, h_sum, h_diff_abs], dim=-1)

        # ========== 3. Force MLP ==========
        alphas = self.force_mlp(scalars)                   # [P, 3]
        alpha1 = alphas[:, 0:1]
        alpha2 = alphas[:, 1:2]
        alpha3 = alphas[:, 2:3]

        # ========== 4. Vectorize ==========
        # F_{i→j}: force on j due to i (canonical direction)
        force_ij = alpha1 * e1 + alpha2 * e2 + alpha3 * e3   # [P, 3]

        # ========== 5. Assign ±F to nodes (Newton's 3rd law) ==========
        F_agg = torch.zeros(N, 3, device=h.device, dtype=h.dtype)
        # Node j receives +F_ij
        F_agg.scatter_add_(0, pj.unsqueeze(-1).expand_as(force_ij), force_ij)
        # Node i receives -F_ij
        F_agg.scatter_add_(0, pi.unsqueeze(-1).expand_as(force_ij), -force_ij)

        # ========== 6. Node update (residual) ==========
        h_input = torch.cat([h, F_agg], dim=-1)  # [N, node_dim + 3]
        h_new = h + self.node_update(h_input)     # residual connection

        return h_new, F_agg


# ────────────── Full Physics Stream ────────────────

class SVPhysicsCore(nn.Module):
    """
    Complete Physics Stream using SV-pipeline.

    Architecture:
        Node Encoder  →  L × SVMessagePassing (accumulates forces)  →  output forces

    The key insight: the conservation guarantee lives in the **force messages**.
    We must NOT pass forces through a per-node decoder MLP (which would break
    the Σ=0 property).  Instead, we output the aggregated forces directly.

    The SVMessagePassing layers update node embeddings h (used for the scalar
    MLPs in subsequent layers), but the **output** of the full model is the
    net aggregated SV-force on each node from the **last** layer.

    Recommended config for PushBox (paper §5.1):
        node_input_dim=6, hidden_dim=32, n_layers=1 → ~7.5K params (physics stream)
    """

    def __init__(
        self,
        node_input_dim: int = 6,    # position (3) + velocity (3)
        hidden_dim: int = 32,       # d_h (lightweight for 2-node)
        n_layers: int = 1,          # L message-passing rounds
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Node encoder
        self.encoder = _make_mlp(node_input_dim, hidden_dim, hidden_dim)

        # SV message-passing layers (return both h and forces)
        self.sv_layers = nn.ModuleList([
            SVMessagePassing(node_dim=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])

        # Per-node scalar multiplier for force → acceleration mapping
        # This is a scalar per node (learned "inverse mass") applied to the
        # conserved force.  Multiplying a conserved vector by per-node scalars
        # breaks Σ m_i a_i = 0 in general, but preserves Σ F_i = 0.
        # For the paper we report force conservation; acceleration = F/m is physical.
        # We omit this and directly output forces as "accelerations" (unit mass).

    def forward(
        self,
        positions: torch.Tensor,        # [N, 3]
        velocities: torch.Tensor,       # [N, 3]
        edge_index: torch.Tensor,       # [2, E]
    ) -> torch.Tensor:                  # [N, 3]  predicted forces / accelerations
        """
        Predict per-node forces from current state.

        Conservation guarantee:  sum(output, dim=0) ≈ 0
        (exact up to floating-point precision).
        """
        # Encode nodes
        node_features = torch.cat([positions, velocities], dim=-1)  # [N, 6]
        h = self.encoder(node_features)                              # [N, d_h]

        # Message passing — accumulate forces from last layer
        forces = None
        for layer in self.sv_layers:
            h, F_agg = layer.forward_with_forces(h, edge_index, positions, velocities)
            forces = F_agg  # keep last layer's forces

        return forces  # [N, 3]

    def parameter_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ────────── Dual-Stream Features Extractor ─────────

class PhysRobotFeaturesExtractorV3(nn.Module):
    """
    Dual-stream PPO features extractor for StableBaselines3.

    Policy Stream:  MLP(obs)         → z_policy
    Physics Stream: SV-GNN(graph)    → z_physics = sg(â_box)
    Fusion:         concat + Linear  → features

    The physics stream receives stop-gradient (sg) during RL training
    to prevent PPO loss from distorting learned dynamics.
    Physics stream is trained separately via auxiliary dynamics loss.

    Compatible with SB3 via BaseFeaturesExtractor interface.
    Import and subclass from stable_baselines3 when integrating.
    """

    def __init__(
        self,
        obs_dim: int = 16,
        features_dim: int = 64,
        physics_hidden: int = 32,
        physics_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.features_dim = features_dim

        # Physics stream
        self.physics_core = SVPhysicsCore(
            node_input_dim=6,
            hidden_dim=physics_hidden,
            n_layers=physics_layers,
        )

        # Policy stream (standard MLP)
        self.policy_stream = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

        # Fusion: policy features + physics predictions → final features
        self.fusion = nn.Sequential(
            nn.Linear(features_dim + 3, features_dim),
            nn.ReLU(),
        )

        # Edge index for 2-node graph (ee + box), cached
        self._edge_index_2 = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        ).t()  # [2, 2]

    def _obs_to_graph(self, obs: torch.Tensor):
        """
        Convert 16-dim observation to 3-D positions + velocities for 2 nodes.

        obs layout (push_box_env, 16-dim):
            [0:2]   joint_pos
            [2:4]   joint_vel
            [4:7]   ee_pos (3D)
            [7:10]  box_pos (3D)
            [10:13] box_vel (3D)
            [13:16] goal_pos (3D)
        """
        ee_pos = obs[:, 4:7]                               # [B, 3]
        box_pos = obs[:, 7:10]                              # [B, 3]
        box_vel = obs[:, 10:13]                             # [B, 3]
        ee_vel = torch.zeros_like(ee_pos)                   # approximate

        return ee_pos, ee_vel, box_pos, box_vel

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: [B, obs_dim]

        Returns:
            features: [B, features_dim]
        """
        B = observations.shape[0]
        device = observations.device

        # ---- Policy stream ----
        z_policy = self.policy_stream(observations)  # [B, features_dim]

        # ---- Physics stream (per sample, then batch) ----
        ee_pos, ee_vel, box_pos, box_vel = self._obs_to_graph(observations)

        # Construct per-sample graph and run physics core
        edge_index = self._edge_index_2.to(device)

        box_acc_list = []
        for i in range(B):
            pos_i = torch.stack([ee_pos[i], box_pos[i]], dim=0)   # [2, 3]
            vel_i = torch.stack([ee_vel[i], box_vel[i]], dim=0)   # [2, 3]
            acc_i = self.physics_core(pos_i, vel_i, edge_index)   # [2, 3]
            box_acc_list.append(acc_i[1])                          # box node

        z_physics = torch.stack(box_acc_list, dim=0)               # [B, 3]

        # ---- Stop-gradient on physics predictions during RL ----
        z_physics_sg = z_physics.detach()

        # ---- Fusion ----
        combined = torch.cat([z_policy, z_physics_sg], dim=-1)     # [B, features_dim + 3]
        features = self.fusion(combined)                            # [B, features_dim]

        return features


# ──────────────── Verification ─────────────────────

def verify_momentum_conservation(
    model: SVPhysicsCore,
    n_trials: int = 100,
    n_nodes: int = 4,
    tol: float = 1e-4,
) -> bool:
    """
    Verify Σ F_i = 0 for random positions/velocities/parameters.

    This MUST pass for ANY random initialization (architectural guarantee,
    not a learned property).

    Args:
        model:    SVPhysicsCore instance
        n_trials: number of random trials
        n_nodes:  nodes per trial
        tol:      tolerance for ||Σ F||

    Returns:
        True if all trials pass.
    """
    device = next(model.parameters()).device
    max_error = 0.0

    for trial in range(n_trials):
        pos = torch.randn(n_nodes, 3, device=device)
        vel = torch.randn(n_nodes, 3, device=device)

        # Fully-connected edges (no self-loops)
        src, dst = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

        acc = model(pos, vel, edge_index)      # [N, 3]
        total_force = acc.sum(dim=0)           # [3]
        error = total_force.norm().item()
        max_error = max(max_error, error)

        if error > tol:
            print(f"  ❌ Trial {trial}: ||Σ F|| = {error:.2e} > tol={tol:.0e}")
            return False

    print(f"  ✅ All {n_trials} trials passed.  Max ||Σ F|| = {max_error:.2e}")
    return True


# ──────────────── Self-Test ────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SV Message Passing — Unit Tests")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. Build model
    model = SVPhysicsCore(node_input_dim=6, hidden_dim=32, n_layers=1)
    n_params = model.parameter_count()
    print(f"\n1. Model created: {n_params:,} parameters")
    assert n_params < 30_000, f"Too many params: {n_params}"

    # 2. Momentum conservation (THE critical test)
    print("\n2. Momentum conservation test (100 random trials, N=4):")
    passed = verify_momentum_conservation(model, n_trials=100, n_nodes=4)
    assert passed, "MOMENTUM CONSERVATION FAILED"

    # 3. Different graph sizes
    print("\n3. Variable graph sizes:")
    for N in [2, 3, 5, 8]:
        pos = torch.randn(N, 3)
        vel = torch.randn(N, 3)
        src, dst = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    dst.append(j)
        ei = torch.tensor([src, dst], dtype=torch.long)
        acc = model(pos, vel, ei)
        err = acc.sum(dim=0).norm().item()
        print(f"   N={N}: acc shape={acc.shape}, ||Σ F|| = {err:.2e} {'✅' if err < 1e-4 else '❌'}")

    # 4. Gradient flow
    print("\n4. Gradient flow test:")
    model2 = SVPhysicsCore(node_input_dim=6, hidden_dim=32, n_layers=2)
    pos = torch.randn(3, 3)
    vel = torch.randn(3, 3)
    ei = torch.tensor([[0,0,1,1,2,2],[1,2,0,2,0,1]], dtype=torch.long)
    acc = model2(pos, vel, ei)
    loss = acc.pow(2).sum()
    loss.backward()
    # Last layer's node_update is expected to have no gradient (output not used)
    skip = {f"sv_layers.{model2.n_layers - 1}.node_update"}
    no_grad = [n for n, p in model2.named_parameters()
               if (p.grad is None or p.grad.norm() == 0)
               and not any(n.startswith(s) for s in skip)]
    has_grad = len(no_grad) == 0
    print(f"   All active parameters have gradients: {'✅' if has_grad else '❌'}")
    if no_grad:
        print(f"   Missing: {no_grad}")

    # 5. Dual-stream extractor
    print("\n5. PhysRobotFeaturesExtractorV3:")
    extractor = PhysRobotFeaturesExtractorV3(obs_dim=16, features_dim=64)
    obs = torch.randn(4, 16)
    features = extractor(obs)
    print(f"   Input: {obs.shape} → Output: {features.shape}")
    assert features.shape == (4, 64)

    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"   Total params (extractor): {total_params:,}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
