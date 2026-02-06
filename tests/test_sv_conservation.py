"""
Test suite: Momentum Conservation (architectural guarantee)
============================================================

These tests verify that SVPhysicsCore satisfies  sum_i F_i = 0
for ANY random initialization, graph size, and parameter state.

This is the single most important property of the SV-pipeline.
If any test here fails, the theoretical claim is broken.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_core.sv_message_passing import (
    SVPhysicsCore,
    SVMessagePassing,
    build_edge_frames,
    verify_momentum_conservation,
)


def _fc_edges(N: int) -> torch.Tensor:
    """Fully connected bidirectional edge index (no self-loops)."""
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


class TestConservationArchitectural:
    """Conservation must hold for ANY theta (untrained, trained, random)."""

    @pytest.mark.parametrize("seed", range(5))
    def test_random_init(self, seed):
        """Different random seeds all conserve momentum."""
        torch.manual_seed(seed)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        assert verify_momentum_conservation(model, n_trials=50, n_nodes=4, tol=1e-4)

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 8, 16])
    def test_various_graph_sizes(self, N):
        """Conservation for graphs of size 2..16."""
        torch.manual_seed(0)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        pos = torch.randn(N, 3)
        vel = torch.randn(N, 3)
        ei = _fc_edges(N)
        F = model(pos, vel, ei)
        err = F.sum(dim=0).norm().item()
        assert err < 1e-3, f"N={N}: ||sum F|| = {err:.2e}"

    def test_after_multiple_optimizer_steps(self):
        """Conservation persists after 50 optimizer steps."""
        torch.manual_seed(42)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        for step in range(50):
            pos = torch.randn(3, 3)
            vel = torch.randn(3, 3)
            ei = _fc_edges(3)
            F = model(pos, vel, ei)
            loss = F.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # After training, conservation must still hold
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                pos = torch.randn(5, 3)
                vel = torch.randn(5, 3)
                ei = _fc_edges(5)
                F = model(pos, vel, ei)
                err = F.sum(dim=0).norm().item()
                assert err < 1e-3, f"Post-training: ||sum F|| = {err:.2e}"

    def test_multi_layer_conservation(self):
        """L=1, L=2, L=3 all conserve."""
        for L in [1, 2, 3]:
            torch.manual_seed(0)
            model = SVPhysicsCore(hidden_dim=32, n_layers=L)
            pos = torch.randn(4, 3)
            vel = torch.randn(4, 3)
            ei = _fc_edges(4)
            F = model(pos, vel, ei)
            err = F.sum(dim=0).norm().item()
            assert err < 1e-3, f"L={L}: ||sum F|| = {err:.2e}"

    def test_large_coordinates(self):
        """Conservation with large coordinate values."""
        torch.manual_seed(0)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        pos = torch.randn(4, 3) * 1000
        vel = torch.randn(4, 3) * 100
        ei = _fc_edges(4)
        F = model(pos, vel, ei)
        err = F.sum(dim=0).norm().item()
        # Tolerance is relative to force magnitude
        F_mag = F.norm().item() + 1e-8
        rel_err = err / F_mag
        assert rel_err < 1e-3, f"Large coords: rel error = {rel_err:.2e}"

    def test_zero_velocity(self):
        """Conservation with zero velocities (degenerate frame)."""
        torch.manual_seed(0)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        pos = torch.randn(3, 3)
        vel = torch.zeros(3, 3)
        ei = _fc_edges(3)
        F = model(pos, vel, ei)
        err = F.sum(dim=0).norm().item()
        assert err < 1e-4, f"Zero vel: ||sum F|| = {err:.2e}"

    def test_coincident_nodes_no_crash(self):
        """Two nodes at same position should not crash."""
        torch.manual_seed(0)
        model = SVPhysicsCore(hidden_dim=32, n_layers=1)
        pos = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        vel = torch.randn(2, 3)
        ei = _fc_edges(2)
        F = model(pos, vel, ei)
        assert not torch.isnan(F).any(), "NaN with coincident nodes"
        err = F.sum(dim=0).norm().item()
        assert err < 1e-4


class TestNewtonThirdLaw:
    """F_ij + F_ji = 0 for every edge pair."""

    def test_pairwise_cancellation(self):
        """Check F_ij = -F_ji explicitly for a 2-node graph."""
        torch.manual_seed(0)
        layer = SVMessagePassing(node_dim=32, hidden_dim=32)
        h = torch.randn(2, 32)
        pos = torch.randn(2, 3)
        vel = torch.randn(2, 3)
        ei = _fc_edges(2)

        _, F_agg = layer.forward_with_forces(h, ei, pos, vel)
        # For 2 nodes: F_agg[0] = -F_agg[1]
        err = (F_agg[0] + F_agg[1]).norm().item()
        assert err < 1e-6, f"F_01 + F_10 = {err:.2e}"


class TestFrameProperties:
    """Additional frame tests beyond the sv_message_passing test suite."""

    def test_frame_consistent_across_batch(self):
        """Same edge in different positions within batch gives correct frame."""
        pos = torch.randn(10, 3)
        vel = torch.randn(10, 3)
        src = torch.arange(10)
        dst = (src + 1) % 10

        e1, e2, e3, r_ij, d_ij = build_edge_frames(pos, vel, src, dst)

        # Verify orthonormality for all edges
        for k in range(10):
            frame = torch.stack([e1[k], e2[k], e3[k]])  # [3, 3]
            gram = frame @ frame.T  # should be identity
            err = (gram - torch.eye(3)).abs().max().item()
            assert err < 1e-4, f"Edge {k}: frame not orthonormal (err={err:.2e})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
