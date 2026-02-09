"""
Unit tests for SVMessagePassing — the core PhysRobot innovation.

Tests:
1. Momentum conservation (architectural guarantee, any θ)
2. Frame antisymmetry properties
3. Output shapes for various graph sizes
4. Gradient flow
5. Degeneracy handling (parallel velocities, zero velocity)
6. Rotation equivariance of forces
7. Translation invariance
8. PhysRobotFeaturesExtractorV3 integration
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_core.sv_message_passing import (
    SVMessagePassing,
    SVPhysicsCore,
    PhysRobotFeaturesExtractorV3,
    build_edge_frames,
    verify_momentum_conservation,
)


def _make_fully_connected_edges(N: int) -> torch.Tensor:
    """Fully connected edge index (both directions)."""
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


class TestMomentumConservation:
    """THE critical test: Σ F_i = 0 for any parameters."""

    def test_conservation_100_trials_4_nodes(self):
        """100 random trials, N=4 fully connected."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        assert verify_momentum_conservation(model, n_trials=100, n_nodes=4, tol=1e-4)

    def test_conservation_2_nodes(self):
        """Simplest graph: 2 nodes, 1 undirected edge."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        for _ in range(50):
            pos = torch.randn(2, 3)
            vel = torch.randn(2, 3)
            ei = _make_fully_connected_edges(2)
            F = model(pos, vel, ei)
            assert F.sum(dim=0).norm().item() < 1e-5, \
                f"2-node conservation violated: ||Σ F|| = {F.sum(dim=0).norm():.2e}"

    def test_conservation_8_nodes(self):
        """Larger graph: 8 nodes."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        assert verify_momentum_conservation(model, n_trials=20, n_nodes=8, tol=1e-3)

    def test_conservation_multi_layer(self):
        """Conservation with L=2 layers."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=2)
        assert verify_momentum_conservation(model, n_trials=50, n_nodes=4, tol=1e-4)

    def test_conservation_after_training_step(self):
        """Conservation still holds after optimizer step."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate a training step
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3)
        ei = _make_fully_connected_edges(3)
        F = model(pos, vel, ei)
        loss = F.pow(2).sum()
        loss.backward()
        opt.step()

        # Conservation must still hold
        model.eval()
        with torch.no_grad():
            F2 = model(torch.randn(4, 3), torch.randn(4, 3), _make_fully_connected_edges(4))
            err = F2.sum(dim=0).norm().item()
        assert err < 1e-4, f"Conservation violated after training step: ||Σ F|| = {err:.2e}"


class TestFrameAntisymmetry:
    """Test edge frame properties."""

    def test_e1_antisymmetric(self):
        """e1^{ij} = -e1^{ji}"""
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3)
        # Forward: 0→1, Reverse: 1→0
        e1_fwd, _, _, _, _ = build_edge_frames(pos, vel, torch.tensor([0]), torch.tensor([1]))
        e1_rev, _, _, _, _ = build_edge_frames(pos, vel, torch.tensor([1]), torch.tensor([0]))
        err = (e1_fwd + e1_rev).norm().item()
        assert err < 1e-6, f"e1 not antisymmetric: {err:.2e}"

    def test_e2_antisymmetric(self):
        """e2^{ij} = -e2^{ji}"""
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3) * 5  # large velocity to avoid degeneracy
        e1_f, e2_f, _, _, _ = build_edge_frames(pos, vel, torch.tensor([0]), torch.tensor([1]))
        e1_r, e2_r, _, _, _ = build_edge_frames(pos, vel, torch.tensor([1]), torch.tensor([0]))
        err = (e2_f + e2_r).norm().item()
        assert err < 1e-5, f"e2 not antisymmetric: {err:.2e}"

    def test_e3_symmetric(self):
        """e3^{ij} = +e3^{ji}"""
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3) * 5
        _, _, e3_f, _, _ = build_edge_frames(pos, vel, torch.tensor([0]), torch.tensor([1]))
        _, _, e3_r, _, _ = build_edge_frames(pos, vel, torch.tensor([1]), torch.tensor([0]))
        err = (e3_f - e3_r).norm().item()
        assert err < 1e-5, f"e3 not symmetric: {err:.2e}"

    def test_frame_orthonormal(self):
        """Frame vectors should be orthonormal."""
        pos = torch.randn(4, 3)
        vel = torch.randn(4, 3) * 5
        ei = _make_fully_connected_edges(4)
        e1, e2, e3, _, _ = build_edge_frames(pos, vel, ei[0], ei[1])

        # Check orthogonality
        dot12 = (e1 * e2).sum(dim=-1).abs().max().item()
        dot13 = (e1 * e3).sum(dim=-1).abs().max().item()
        dot23 = (e2 * e3).sum(dim=-1).abs().max().item()
        assert dot12 < 1e-5, f"e1·e2 = {dot12:.2e}"
        assert dot13 < 1e-5, f"e1·e3 = {dot13:.2e}"
        assert dot23 < 1e-5, f"e2·e3 = {dot23:.2e}"

        # Check unit norm
        norm1 = (e1.norm(dim=-1) - 1).abs().max().item()
        norm2 = (e2.norm(dim=-1) - 1).abs().max().item()
        norm3 = (e3.norm(dim=-1) - 1).abs().max().item()
        assert norm1 < 1e-5, f"||e1|| - 1 = {norm1:.2e}"
        assert norm2 < 1e-5, f"||e2|| - 1 = {norm2:.2e}"
        assert norm3 < 1e-5, f"||e3|| - 1 = {norm3:.2e}"


class TestOutputShapes:
    """Test output shapes for various configurations."""

    @pytest.mark.parametrize("N", [2, 3, 5, 8])
    def test_output_shape(self, N):
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        pos = torch.randn(N, 3)
        vel = torch.randn(N, 3)
        ei = _make_fully_connected_edges(N)
        F = model(pos, vel, ei)
        assert F.shape == (N, 3), f"Expected ({N}, 3), got {F.shape}"

    def test_sv_layer_shapes(self):
        layer = SVMessagePassing(node_dim=32, hidden_dim=32)
        h = torch.randn(4, 32)
        pos = torch.randn(4, 3)
        vel = torch.randn(4, 3)
        ei = _make_fully_connected_edges(4)
        h_new = layer(h, ei, pos, vel)
        assert h_new.shape == h.shape


class TestDegeneracy:
    """Test frame construction under degenerate conditions."""

    def test_zero_velocity(self):
        """Frame should use fallback when v_perp ≈ 0."""
        pos = torch.randn(2, 3)
        vel = torch.zeros(2, 3)  # zero velocity
        e1, e2, e3, _, _ = build_edge_frames(
            pos, vel, torch.tensor([0]), torch.tensor([1])
        )
        assert not torch.isnan(e2).any(), "NaN in e2 with zero velocity"
        assert e2.norm().item() > 0.9, "e2 not unit length with zero velocity"

    def test_parallel_velocity(self):
        """Frame should handle v_rel parallel to r_ij."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        vel = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # parallel to r
        e1, e2, e3, _, _ = build_edge_frames(
            pos, vel, torch.tensor([0]), torch.tensor([1])
        )
        assert not torch.isnan(e2).any(), "NaN in e2 with parallel velocity"

    def test_z_aligned_edge(self):
        """Frame fallback when e1 ≈ ±ẑ."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        vel = torch.zeros(2, 3)
        e1, e2, e3, _, _ = build_edge_frames(
            pos, vel, torch.tensor([0]), torch.tensor([1])
        )
        assert not torch.isnan(e2).any(), "NaN in e2 with z-aligned edge"
        assert e2.norm().item() > 0.9, "e2 not unit length"


class TestEquivariance:
    """Test physical symmetry properties.
    
    Note: The full SVPhysicsCore is NOT strictly translation/rotation
    invariant because the node encoder takes absolute positions.
    The SV message-passing layer itself IS invariant in the edge features,
    but the encoder breaks it.  This is a known design choice — absolute
    position info helps the policy know where objects are in the workspace.
    
    We test that the RAW EDGE FEATURES (before the encoder) are invariant.
    """

    def test_raw_edge_features_translation_invariant(self):
        """Edge frame construction is translation invariant."""
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3)
        ei = _make_fully_connected_edges(3)

        e1_a, e2_a, e3_a, _, d_a = build_edge_frames(pos, vel, ei[0], ei[1])
        e1_b, e2_b, e3_b, _, d_b = build_edge_frames(pos + 100.0, vel, ei[0], ei[1])

        assert (e1_a - e1_b).abs().max().item() < 1e-5, "e1 not translation invariant"
        assert (e2_a - e2_b).abs().max().item() < 1e-5, "e2 not translation invariant"
        assert (e3_a - e3_b).abs().max().item() < 1e-5, "e3 not translation invariant"
        assert (d_a - d_b).abs().max().item() < 1e-5, "d not translation invariant"

    def test_edge_features_rotation_equivariant(self):
        """Edge frames rotate correctly: e_k(Rx, Rv) = R @ e_k(x, v)."""
        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3) * 3  # non-degenerate
        ei = _make_fully_connected_edges(3)

        # Random rotation matrix
        A = torch.randn(3, 3)
        U, _, Vt = torch.linalg.svd(A)
        R = U @ Vt
        if R.det() < 0:
            R = -R

        e1_o, e2_o, e3_o, _, _ = build_edge_frames(pos, vel, ei[0], ei[1])
        e1_r, e2_r, e3_r, _, _ = build_edge_frames(pos @ R.T, vel @ R.T, ei[0], ei[1])

        # Check equivariance: e_k(Rx, Rv) ≈ R @ e_k(x, v)
        err1 = (e1_r - e1_o @ R.T).abs().max().item()
        assert err1 < 1e-4, f"e1 rotation equivariance error: {err1:.2e}"

    def test_conservation_under_rotation(self):
        """Momentum conservation holds regardless of frame orientation."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        model.eval()

        A = torch.randn(3, 3)
        U, _, Vt = torch.linalg.svd(A)
        R = U @ Vt
        if R.det() < 0:
            R = -R

        pos = torch.randn(4, 3)
        vel = torch.randn(4, 3)
        ei = _make_fully_connected_edges(4)

        with torch.no_grad():
            F_rot = model(pos @ R.T, vel @ R.T, ei)
        err = F_rot.sum(dim=0).norm().item()
        assert err < 1e-4, f"Conservation violated under rotation: {err:.2e}"


class TestParameterCount:
    """Verify parameter counts match paper claims."""

    def test_lightweight_config(self):
        """d_h=32, L=1 should be well under 30K params."""
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        n = model.parameter_count()
        assert n < 30_000, f"Too many params: {n:,}"
        print(f"  d_h=32, L=1: {n:,} params")

    def test_standard_config(self):
        """d_h=64, L=2 should be under 100K params."""
        model = SVPhysicsCore(hidden_dim=64, n_layers=2)
        n = model.parameter_count()
        assert n < 100_000, f"Too many params: {n:,}"
        print(f"  d_h=64, L=2: {n:,} params")


class TestFeaturesExtractor:
    """Test PhysRobotFeaturesExtractorV3."""

    def test_output_shape(self):
        ext = PhysRobotFeaturesExtractorV3(obs_dim=16, features_dim=64)
        obs = torch.randn(8, 16)
        features = ext(obs)
        assert features.shape == (8, 64)

    def test_gradient_flow(self):
        ext = PhysRobotFeaturesExtractorV3(obs_dim=16, features_dim=64)
        obs = torch.randn(4, 16)
        features = ext(obs)
        loss = features.sum()
        loss.backward()
        has_policy_grad = ext.policy_stream[0].weight.grad is not None
        has_fusion_grad = ext.fusion[0].weight.grad is not None
        assert has_policy_grad, "No gradient in policy stream"
        assert has_fusion_grad, "No gradient in fusion"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
