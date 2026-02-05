"""
Physics-Informed Neural Networks for Robotics
==============================================

Core physics modules for learning robot dynamics with conservation laws.
"""

from .edge_frame import EdgeFrame, construct_edge_features
from .dynamical_gnn import DynamicalGNN
from .integrators import SymplecticIntegrator, RK4Integrator

__version__ = "0.1.0"
__all__ = [
    "EdgeFrame",
    "construct_edge_features",
    "DynamicalGNN",
    "SymplecticIntegrator",
    "RK4Integrator",
]
