"""
Test EdgeFrame antisymmetry and other properties
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_core.edge_frame import (
    EdgeFrame,
    construct_edge_features,
    fully_connected_edges,
)


class TestEdgeFrame:
    """Test suite for EdgeFrame"""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple 4-particle system"""
        N = 4
        positions = torch.randn(N, 3)
        velocities = torch.randn(N, 3)
        edge_index = fully_connected_edges(N, self_loops=False)
        return positions, velocities, edge_index
    
    def test_antisymmetry(self, simple_system):
        """
        Test antisymmetry property: e_ij = -e_ji
        
        Acceptance: error < 1e-5
        """
        positions, velocities, edge_index = simple_system
        
        edge_frame = EdgeFrame(hidden_dim=64)
        error = edge_frame.check_antisymmetry(positions, velocities, edge_index)
        
        print(f"Antisymmetry error: {error:.2e}")
        assert error < 1e-5, f"Antisymmetry violated: error={error}"
    
    def test_translation_invariance(self):
        """
        Test translation invariance: shifting all positions shouldn't change edge features
        """
        N = 3
        positions = torch.randn(N, 3)
        velocities = torch.randn(N, 3)
        edge_index = fully_connected_edges(N, self_loops=False)
        
        # Translate system
        translation = torch.tensor([10.0, -5.0, 3.0])
        positions_translated = positions + translation
        
        # Compute edge features
        edge_frame = EdgeFrame(hidden_dim=64)
        edge_features_1 = edge_frame(positions, velocities, edge_index)
        edge_features_2 = edge_frame(positions_translated, velocities, edge_index)
        
        # Should be identical
        diff = torch.max(torch.abs(edge_features_1 - edge_features_2))
        print(f"Translation invariance error: {diff:.2e}")
        assert diff < 1e-5, f"Translation invariance violated: diff={diff}"
    
    def test_output_shape(self):
        """Test output shapes are correct"""
        N = 5
        hidden_dim = 128
        
        positions = torch.randn(N, 3)
        velocities = torch.randn(N, 3)
        edge_index = fully_connected_edges(N, self_loops=False)
        
        edge_frame = EdgeFrame(hidden_dim=hidden_dim)
        edge_features = edge_frame(positions, velocities, edge_index)
        
        expected_num_edges = N * (N - 1)
        assert edge_features.shape == (expected_num_edges, hidden_dim), \
            f"Expected shape ({expected_num_edges}, {hidden_dim}), got {edge_features.shape}"
    
    def test_raw_features(self):
        """Test raw edge feature construction"""
        N = 3
        positions = torch.randn(N, 3)
        velocities = torch.randn(N, 3)
        edge_index = fully_connected_edges(N, self_loops=False)
        
        raw_features = construct_edge_features(positions, velocities, edge_index)
        
        # Should have 8 features: [dx, dy, dz, ||r||, dvx, dvy, dvz, ||v||]
        assert raw_features.shape[1] == 8, \
            f"Expected 8 features, got {raw_features.shape[1]}"
    
    def test_fully_connected_graph(self):
        """Test edge index construction"""
        N = 4
        edge_index = fully_connected_edges(N, self_loops=False)
        
        # Should have N*(N-1) edges
        expected_num_edges = N * (N - 1)
        assert edge_index.shape == (2, expected_num_edges), \
            f"Expected {expected_num_edges} edges, got {edge_index.shape[1]}"
        
        # No self-loops
        src, tgt = edge_index
        assert (src == tgt).sum() == 0, "Self-loops detected!"
    
    def test_batch_processing(self):
        """Test processing multiple systems (batched)"""
        # Note: Current implementation doesn't support batching
        # This is a placeholder for future enhancement
        pass


if __name__ == "__main__":
    print("Running EdgeFrame tests...\n")
    pytest.main([__file__, "-v", "-s"])
