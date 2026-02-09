"""
Test conservation laws (energy, momentum, angular momentum)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_core.dynamical_gnn import DynamicalGNN
from physics_core.edge_frame import fully_connected_edges
from physics_core.integrators import SymplecticIntegrator


class TestConservation:
    """Test suite for conservation laws"""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple particle system"""
        N = 3
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        velocities = torch.tensor([
            [0.1, 0.0, 0.0],
            [-0.05, 0.05, 0.0],
            [-0.05, -0.05, 0.0],
        ])
        masses = torch.ones(N)
        edge_index = fully_connected_edges(N, self_loops=False)
        
        return positions, velocities, masses, edge_index
    
    def test_energy_computation(self, simple_system):
        """Test energy computation is correct"""
        positions, velocities, masses, edge_index = simple_system
        
        model = DynamicalGNN(hidden_dim=64, edge_hidden_dim=32, n_message_passing=2)
        KE, PE, E_total = model.compute_energy(positions, velocities, masses)
        
        # Check types
        assert isinstance(KE.item(), float), "KE should be scalar"
        assert isinstance(PE.item(), float), "PE should be scalar"
        assert isinstance(E_total.item(), float), "Total energy should be scalar"
        
        # Energy should be positive (or zero)
        assert E_total >= 0, f"Energy should be non-negative, got {E_total}"
        
        print(f"Energy: KE={KE:.4f}, PE={PE:.4f}, Total={E_total:.4f}")
    
    def test_momentum_conservation(self, simple_system):
        """
        Test momentum conservation in a closed system.
        
        Acceptance: momentum error < 0.1%
        """
        positions, velocities, masses, edge_index = simple_system
        
        # Initial momentum
        p_initial = (masses.unsqueeze(-1) * velocities).sum(dim=0)
        
        # Create model
        model = DynamicalGNN(hidden_dim=64, edge_hidden_dim=32, n_message_passing=2)
        
        # Simulate for a few steps with symplectic integrator
        integrator = SymplecticIntegrator(dt=0.01)
        
        def accel_fn(pos, vel):
            return model(pos, vel, edge_index, masses)
        
        # Rollout
        pos_traj, vel_traj = integrator.rollout(
            positions, velocities, accel_fn, n_steps=50
        )
        
        # Final momentum
        p_final = (masses.unsqueeze(-1) * vel_traj[-1]).sum(dim=0)
        
        # Check conservation
        momentum_error = torch.norm(p_final - p_initial) / (torch.norm(p_initial) + 1e-8)
        
        print(f"Momentum error: {momentum_error:.2%}")
        assert momentum_error < 0.001, f"Momentum not conserved: error={momentum_error:.2%}"
    
    def test_energy_conservation_symplectic(self, simple_system):
        """
        Test energy conservation with symplectic integrator.
        
        Acceptance: energy error < 0.1%
        """
        positions, velocities, masses, edge_index = simple_system
        
        model = DynamicalGNN(hidden_dim=64, edge_hidden_dim=32, n_message_passing=2)
        
        # Initial energy
        _, _, E_initial = model.compute_energy(positions, velocities, masses)
        
        # Simulate
        integrator = SymplecticIntegrator(dt=0.01)
        
        def accel_fn(pos, vel):
            return model(pos, vel, edge_index, masses)
        
        pos_traj, vel_traj = integrator.rollout(
            positions, velocities, accel_fn, n_steps=50
        )
        
        # Check energy at each timestep
        energies = []
        for i in range(len(pos_traj)):
            _, _, E = model.compute_energy(pos_traj[i], vel_traj[i], masses)
            energies.append(E.item())
        
        energies = torch.tensor(energies)
        energy_std = torch.std(energies) / (torch.mean(energies) + 1e-8)
        
        print(f"Energy drift (relative std): {energy_std:.2%}")
        
        # Symplectic integrator should have low energy drift
        assert energy_std < 0.01, f"Energy drift too large: {energy_std:.2%}"
    
    def test_conservation_check_function(self, simple_system):
        """Test the built-in conservation check function"""
        positions, velocities, masses, edge_index = simple_system
        
        model = DynamicalGNN(hidden_dim=64, edge_hidden_dim=32, n_message_passing=2)
        
        # Simulate one step
        integrator = SymplecticIntegrator(dt=0.01)
        
        def accel_fn(pos, vel):
            return model(pos, vel, edge_index, masses)
        
        pos_new, vel_new = integrator.step(positions, velocities, accel_fn)
        
        # Check conservation
        conservation_dict = model.check_conservation(
            positions, velocities, pos_new, vel_new, masses
        )
        
        print(f"Conservation check: {conservation_dict}")
        
        assert 'energy_error' in conservation_dict
        assert 'momentum_error' in conservation_dict
        
        # Errors should be small
        assert conservation_dict['energy_error'] < 0.001, \
            f"Energy error too large: {conservation_dict['energy_error']:.2%}"


class TestSymplecticProperties:
    """Test symplectic integrator properties"""
    
    def test_symplectic_vs_rk4_energy_drift(self):
        """
        Compare energy drift between symplectic and RK4 integrators.
        Symplectic should have less drift.
        """
        from physics_core.integrators import RK4Integrator
        
        # Simple harmonic oscillator
        def sho_accel(x, v):
            k = 1.0
            return -k * x
        
        x0 = torch.tensor([[1.0, 0.0, 0.0]])
        v0 = torch.tensor([[0.0, 0.0, 0.0]])
        
        # Symplectic
        symplectic = SymplecticIntegrator(dt=0.01)
        x_symp, v_symp = symplectic.rollout(x0, v0, sho_accel, n_steps=100)
        E_symp = 0.5 * (x_symp**2 + v_symp**2).sum(dim=-1)
        drift_symp = torch.std(E_symp) / torch.mean(E_symp)
        
        # RK4
        rk4 = RK4Integrator(dt=0.01)
        x_rk4, v_rk4 = rk4.rollout(x0, v0, sho_accel, n_steps=100)
        E_rk4 = 0.5 * (x_rk4**2 + v_rk4**2).sum(dim=-1)
        drift_rk4 = torch.std(E_rk4) / torch.mean(E_rk4)
        
        print(f"Symplectic energy drift: {drift_symp:.2e}")
        print(f"RK4 energy drift: {drift_rk4:.2e}")
        
        # Symplectic should be better
        assert drift_symp < drift_rk4, \
            "Symplectic integrator should have less energy drift than RK4"


if __name__ == "__main__":
    print("Running conservation law tests...\n")
    pytest.main([__file__, "-v", "-s"])
