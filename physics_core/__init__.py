"""
Physics-Informed Neural Networks for Robotics
==============================================

Core physics modules for learning robot dynamics with conservation laws.

Main components:
- SVPhysicsCore: momentum-conserving SV-pipeline (paper core innovation)
- EdgeFrame: basic edge feature encoding (legacy, replaced by SV-pipeline)
- DynamicalGNN: full GNN (requires torch_geometric)
- Integrators: Symplectic / RK4
"""

# Always-available modules (no torch_geometric dependency)
from .edge_frame import EdgeFrame, construct_edge_features
from .integrators import SymplecticIntegrator, RK4Integrator
from .sv_message_passing import (
    SVMessagePassing,
    SVPhysicsCore,
    PhysRobotFeaturesExtractorV3,
    build_edge_frames,
    verify_momentum_conservation,
)

# Optional: requires torch_geometric
try:
    from .dynamical_gnn import DynamicalGNN
except ImportError:
    DynamicalGNN = None  # torch_geometric not installed

__version__ = "0.2.0"
__all__ = [
    "EdgeFrame",
    "construct_edge_features",
    "DynamicalGNN",
    "SymplecticIntegrator",
    "RK4Integrator",
    "SVMessagePassing",
    "SVPhysicsCore",
    "PhysRobotFeaturesExtractorV3",
    "build_edge_frames",
    "verify_momentum_conservation",
]
