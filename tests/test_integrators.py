"""
Test suite: Physics Integrators
=================================

Tests SymplecticIntegrator and RK4Integrator from physics_core.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires torch")


class TestSymplecticIntegrator:
    """Test the symplectic (Stoermer-Verlet) integrator."""

    def test_harmonic_oscillator_energy(self):
        """
        Harmonic oscillator: V = 0.5 k x^2, F = -kx
        Symplectic integrator should conserve energy much better than Euler.
        """
        from physics_core.integrators import SymplecticIntegrator

        integrator = SymplecticIntegrator(dt=0.01)

        # Initial conditions: x=1, v=0, k=1, m=1
        pos = torch.tensor([[1.0, 0.0, 0.0]])
        vel = torch.tensor([[0.0, 0.0, 0.0]])
        k = 1.0

        def force_fn(p, v):
            return -k * p

        initial_energy = 0.5 * k * (pos ** 2).sum() + 0.5 * (vel ** 2).sum()

        # Integrate for 1000 steps (10 oscillation periods)
        for _ in range(1000):
            pos, vel = integrator.step(pos, vel, force_fn)

        final_energy = 0.5 * k * (pos ** 2).sum() + 0.5 * (vel ** 2).sum()
        energy_drift = abs(final_energy.item() - initial_energy.item())

        # Symplectic integrator should have very low energy drift
        assert energy_drift < 0.01, f"Energy drift = {energy_drift:.4f}"

    def test_free_particle_momentum(self):
        """Free particle (F=0): momentum should be exactly conserved."""
        from physics_core.integrators import SymplecticIntegrator

        integrator = SymplecticIntegrator(dt=0.01)
        pos = torch.tensor([[0.0, 0.0, 0.0]])
        vel = torch.tensor([[1.0, 2.0, 3.0]])

        def zero_force(p, v):
            return torch.zeros_like(p)

        initial_momentum = vel.clone()

        for _ in range(100):
            pos, vel = integrator.step(pos, vel, zero_force)

        momentum_error = (vel - initial_momentum).norm().item()
        assert momentum_error < 1e-10, f"Momentum error = {momentum_error:.2e}"


class TestRK4Integrator:
    """Test the RK4 integrator."""

    def test_harmonic_oscillator_accuracy(self):
        """RK4 should be 4th-order accurate for smooth systems."""
        from physics_core.integrators import RK4Integrator

        integrator = RK4Integrator(dt=0.01)
        pos = torch.tensor([[1.0, 0.0, 0.0]])
        vel = torch.tensor([[0.0, 0.0, 0.0]])
        k = 1.0

        def force_fn(p, v):
            return -k * p

        # After half period (pi seconds), x should be -1, v should be ~0
        n_steps = int(3.14159 / 0.01)
        for _ in range(n_steps):
            pos, vel = integrator.step(pos, vel, force_fn)

        # Should be close to (-1, 0, 0)
        assert abs(pos[0, 0].item() + 1.0) < 0.01, \
            f"RK4 position error: x = {pos[0,0].item():.4f}, expected -1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
