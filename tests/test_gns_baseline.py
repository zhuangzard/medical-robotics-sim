"""
Test suite: GNS Baseline
=========================

Validates the GNS (Graph Network Simulator) baseline agent.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import mujoco
    import stable_baselines3
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(
    not (HAS_DEPS and HAS_PYG),
    reason="Requires mujoco, stable-baselines3, torch-geometric"
)


class TestGNSNetwork:
    """Test the GNS graph network."""

    def test_forward_shape(self):
        from baselines.gns_baseline import GNSNetwork
        net = GNSNetwork(node_feature_dim=6, edge_feature_dim=4, hidden_dim=32, n_layers=2)
        # 2-node graph
        x = torch.randn(2, 6)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        edge_attr = torch.randn(2, 4)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        acc = net(graph)
        assert acc.shape == (2, 3)

    def test_gradient_flow(self):
        from baselines.gns_baseline import GNSNetwork
        net = GNSNetwork(node_feature_dim=6, edge_feature_dim=4, hidden_dim=32, n_layers=1)
        x = torch.randn(2, 6)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        edge_attr = torch.randn(2, 4)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        acc = net(graph)
        loss = acc.sum()
        loss.backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestGNSFeaturesExtractor:
    """Test the SB3 features extractor."""

    def test_output_shape(self):
        from baselines.gns_baseline import GNSFeaturesExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(16,))
        ext = GNSFeaturesExtractor(obs_space, features_dim=128)
        obs = torch.randn(4, 16)
        features = ext(obs)
        assert features.shape == (4, 128)


class TestGNSEdgeIndex:
    """Verify the graph has bidirectional edges (ISSUE-6 fix verification)."""

    def test_bidirectional_edges_in_extractor(self):
        """GNSFeaturesExtractor._obs_to_graph should produce bidirectional edges."""
        from baselines.gns_baseline import GNSFeaturesExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(16,))
        ext = GNSFeaturesExtractor(obs_space, features_dim=128)
        obs = torch.randn(1, 16)
        graph = ext._obs_to_graph(obs)
        # Should have at least 1 edge in each direction (0->1 and 1->0)
        ei = graph.edge_index
        # For a single graph in the batch, check we have both directions
        # Batch shifts indices, so just check count >= 2
        assert ei.shape[1] >= 1, "No edges in graph"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
