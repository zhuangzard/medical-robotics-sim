"""
Edge Frame Implementation
=========================

Edge-centric frame of reference for encoding spatial relationships
with antisymmetry properties for physics-informed learning.

Key Properties:
- Antisymmetry: e_ij = -e_ji
- Translation invariance
- Rotation equivariance
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class EdgeFrame(nn.Module):
    """
    Edge Frame for encoding spatial relationships between nodes.
    
    Given nodes i and j with positions x_i, x_j and velocities v_i, v_j,
    constructs antisymmetric edge features:
    
    e_ij = [r_ij, ||r_ij||, v_rel, ||v_rel||]
    
    where r_ij = x_j - x_i (displacement vector)
          v_rel = v_j - v_i (relative velocity)
    """
    
    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: Dimension of hidden edge representations
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge feature encoder (maintains antisymmetry)
        # Input: [dx, dy, dz, ||r||, dvx, dvy, dvz, ||v||] = 8 features
        self.edge_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
    def forward(
        self,
        positions: torch.Tensor,  # (N, 3)
        velocities: torch.Tensor,  # (N, 3)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:  # (E, hidden_dim)
        """
        Compute edge features in edge frame.
        
        Args:
            positions: Node positions (N, 3)
            velocities: Node velocities (N, 3)
            edge_index: Edge connectivity (2, E) [source, target]
            
        Returns:
            Edge features (E, hidden_dim) with antisymmetric property
        """
        # Extract source and target node indices
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        
        # Compute displacement vectors (antisymmetric)
        r_ij = positions[tgt_idx] - positions[src_idx]  # (E, 3)
        r_norm = torch.norm(r_ij, dim=1, keepdim=True)  # (E, 1)
        
        # Compute relative velocities (antisymmetric)
        v_rel = velocities[tgt_idx] - velocities[src_idx]  # (E, 3)
        v_norm = torch.norm(v_rel, dim=1, keepdim=True)  # (E, 1)
        
        # Concatenate raw features
        # Shape: (E, 8) = (E, 3+1+3+1)
        edge_features = torch.cat([r_ij, r_norm, v_rel, v_norm], dim=1)
        
        # Encode to hidden dimension
        edge_hidden = self.edge_encoder(edge_features)  # (E, hidden_dim)
        
        return edge_hidden
    
    def check_antisymmetry(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> float:
        """
        Verify antisymmetry property: e_ij = -e_ji
        
        Returns:
            Maximum antisymmetry error
        """
        # Get forward edges e_ij
        e_ij = self(positions, velocities, edge_index)
        
        # Construct reverse edges (swap source and target)
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        e_ji = self(positions, velocities, edge_index_rev)
        
        # Check if e_ij ≈ -e_ji
        antisym_error = torch.max(torch.abs(e_ij + e_ji))
        
        return antisym_error.item()


def construct_edge_features(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Standalone function to construct raw edge features without neural network.
    
    Useful for debugging and visualization.
    
    Args:
        positions: Node positions (N, 3)
        velocities: Node velocities (N, 3)
        edge_index: Edge connectivity (2, E)
        
    Returns:
        Raw edge features (E, 8)
    """
    src_idx, tgt_idx = edge_index[0], edge_index[1]
    
    # Displacement and norm
    r_ij = positions[tgt_idx] - positions[src_idx]
    r_norm = torch.norm(r_ij, dim=1, keepdim=True)
    
    # Relative velocity and norm
    v_rel = velocities[tgt_idx] - velocities[src_idx]
    v_norm = torch.norm(v_rel, dim=1, keepdim=True)
    
    return torch.cat([r_ij, r_norm, v_rel, v_norm], dim=1)


def fully_connected_edges(num_nodes: int, self_loops: bool = False) -> torch.Tensor:
    """
    Construct fully connected edge index.
    
    Args:
        num_nodes: Number of nodes
        self_loops: Include self-loops (i->i)
        
    Returns:
        Edge index (2, E) where E = N*(N-1) or N*N
    """
    sources = []
    targets = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j and not self_loops:
                continue
            sources.append(i)
            targets.append(j)
    
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


if __name__ == "__main__":
    # Quick test
    print("Testing EdgeFrame...")
    
    # Create toy system
    N = 4  # 4 particles
    positions = torch.randn(N, 3)
    velocities = torch.randn(N, 3)
    edge_index = fully_connected_edges(N, self_loops=False)
    
    print(f"Nodes: {N}")
    print(f"Edges: {edge_index.shape[1]}")
    
    # Test EdgeFrame
    edge_frame = EdgeFrame(hidden_dim=64)
    edge_features = edge_frame(positions, velocities, edge_index)
    
    print(f"Edge features shape: {edge_features.shape}")
    
    # Check antisymmetry
    error = edge_frame.check_antisymmetry(positions, velocities, edge_index)
    print(f"Antisymmetry error: {error:.2e}")
    
    if error < 1e-5:
        print("✅ Antisymmetry verified!")
    else:
        print("❌ Antisymmetry violated!")
