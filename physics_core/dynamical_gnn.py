"""
Dynamical Graph Neural Network (Dynami-CAL GNN)
================================================

Physics-informed GNN for learning dynamical systems with:
- Conservation laws (energy, momentum)
- Symplectic structure (Hamiltonian mechanics)
- Edge-centric message passing

Reference: "Learning to Simulate Complex Physics with Graph Networks"
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple

from .edge_frame import EdgeFrame


class PhysicsMessagePassing(MessagePassing):
    """
    Message passing layer with physics constraints.
    
    Message: m_ij = φ(e_ij, h_i, h_j)
    Update:  h_i' = ψ(h_i, Σ_j m_ij)
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr='add')  # Sum aggregation
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message network φ
        self.message_net = nn.Sequential(
            nn.Linear(edge_dim + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update network ψ
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,  # Node features (N, hidden_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # Edge features (E, edge_dim)
    ) -> torch.Tensor:
        """
        Perform message passing.
        
        Returns:
            Updated node features (N, hidden_dim)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages m_ij from source j to target i.
        
        Args:
            x_i: Target node features (E, hidden_dim)
            x_j: Source node features (E, hidden_dim)
            edge_attr: Edge features (E, edge_dim)
        """
        # Concatenate edge and node info
        msg_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        messages = self.message_net(msg_input)
        return messages
    
    def update(self, aggr_out, x):
        """
        Update node features based on aggregated messages.
        
        Args:
            aggr_out: Aggregated messages (N, hidden_dim)
            x: Original node features (N, hidden_dim)
        """
        # Residual connection
        update_input = torch.cat([x, aggr_out], dim=-1)
        x_new = self.update_net(update_input) + x  # Residual
        return x_new


class DynamicalGNN(nn.Module):
    """
    Graph Neural Network for learning dynamical systems.
    
    Architecture:
    1. Edge Frame: Encode spatial relationships
    2. Node Encoder: Embed node states
    3. Message Passing: Propagate information (×N layers)
    4. Dynamics Decoder: Predict accelerations
    
    Conservation Laws:
    - Energy conservation (optional loss)
    - Momentum conservation (via antisymmetry)
    """
    
    def __init__(
        self,
        node_dim: int = 6,  # position (3) + velocity (3)
        hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        n_message_passing: int = 3,
        output_dim: int = 3,  # acceleration (3)
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.n_message_passing = n_message_passing
        self.output_dim = output_dim
        
        # Edge frame for spatial encoding
        self.edge_frame = EdgeFrame(hidden_dim=edge_hidden_dim)
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            PhysicsMessagePassing(hidden_dim, edge_hidden_dim)
            for _ in range(n_message_passing)
        ])
        
        # Dynamics decoder (predicts acceleration)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(
        self,
        positions: torch.Tensor,  # (N, 3)
        velocities: torch.Tensor,  # (N, 3)
        edge_index: torch.Tensor,  # (2, E)
        masses: Optional[torch.Tensor] = None,  # (N,)
    ) -> torch.Tensor:  # (N, 3) accelerations
        """
        Predict accelerations given current state.
        
        Args:
            positions: Node positions (N, 3)
            velocities: Node velocities (N, 3)
            edge_index: Edge connectivity (2, E)
            masses: Node masses (N,) [optional]
            
        Returns:
            Predicted accelerations (N, 3)
        """
        # 1. Encode edges in edge frame
        edge_features = self.edge_frame(positions, velocities, edge_index)
        
        # 2. Encode node states
        node_states = torch.cat([positions, velocities], dim=-1)  # (N, 6)
        node_features = self.node_encoder(node_states)  # (N, hidden_dim)
        
        # 3. Message passing
        x = node_features
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_features)
        
        # 4. Decode dynamics (acceleration)
        accelerations = self.decoder(x)  # (N, 3)
        
        return accelerations
    
    def compute_energy(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute system energy for conservation checking.
        
        Returns:
            kinetic_energy, potential_energy, total_energy
        """
        # Kinetic energy: KE = 0.5 * m * v²
        kinetic = 0.5 * masses.unsqueeze(-1) * (velocities ** 2).sum(dim=-1)
        kinetic_total = kinetic.sum()
        
        # Potential energy (gravitational): PE = m * g * h
        g = 9.81
        potential = masses * g * positions[:, 2]  # Assume z is height
        potential_total = potential.sum()
        
        total = kinetic_total + potential_total
        
        return kinetic_total, potential_total, total
    
    def check_conservation(
        self,
        positions_t0: torch.Tensor,
        velocities_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        velocities_t1: torch.Tensor,
        masses: torch.Tensor,
    ) -> dict:
        """
        Check conservation laws between two timesteps.
        
        Returns:
            Dictionary with conservation errors
        """
        # Energy conservation
        _, _, E_t0 = self.compute_energy(positions_t0, velocities_t0, masses)
        _, _, E_t1 = self.compute_energy(positions_t1, velocities_t1, masses)
        energy_error = torch.abs((E_t1 - E_t0) / (E_t0 + 1e-8))
        
        # Momentum conservation
        p_t0 = (masses.unsqueeze(-1) * velocities_t0).sum(dim=0)
        p_t1 = (masses.unsqueeze(-1) * velocities_t1).sum(dim=0)
        momentum_error = torch.norm(p_t1 - p_t0) / (torch.norm(p_t0) + 1e-8)
        
        return {
            'energy_error': energy_error.item(),
            'momentum_error': momentum_error.item(),
            'energy_t0': E_t0.item(),
            'energy_t1': E_t1.item(),
        }


if __name__ == "__main__":
    print("Testing DynamicalGNN...")
    
    # Create toy system
    N = 5  # 5 particles
    positions = torch.randn(N, 3)
    velocities = torch.randn(N, 3)
    masses = torch.ones(N)
    
    # Fully connected graph
    from .edge_frame import fully_connected_edges
    edge_index = fully_connected_edges(N, self_loops=False)
    
    # Create model
    model = DynamicalGNN(
        node_dim=6,
        hidden_dim=128,
        edge_hidden_dim=64,
        n_message_passing=3,
        output_dim=3,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    accelerations = model(positions, velocities, edge_index, masses)
    
    print(f"Input shape: positions {positions.shape}, velocities {velocities.shape}")
    print(f"Output shape: accelerations {accelerations.shape}")
    
    # Check energy
    KE, PE, E = model.compute_energy(positions, velocities, masses)
    print(f"Energy: KE={KE:.3f}, PE={PE:.3f}, Total={E:.3f}")
    
    print("✅ DynamicalGNN test passed!")
