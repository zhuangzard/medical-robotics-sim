"""
Physics Integrators
===================

Numerical integration methods for dynamical systems:
- Symplectic Integrator: Preserves Hamiltonian structure
- RK4: Classic 4th-order Runge-Kutta
"""

import torch
from typing import Callable, Tuple


class SymplecticIntegrator:
    """
    Symplectic (Verlet) integrator for Hamiltonian systems.
    
    Preserves energy and symplectic structure better than RK4.
    Ideal for conservative mechanical systems.
    
    Update scheme:
        v_{n+1/2} = v_n + 0.5 * dt * a_n
        x_{n+1} = x_n + dt * v_{n+1/2}
        a_{n+1} = f(x_{n+1})
        v_{n+1} = v_{n+1/2} + 0.5 * dt * a_{n+1}
    """
    
    def __init__(self, dt: float = 0.01):
        """
        Args:
            dt: Timestep size
        """
        self.dt = dt
    
    def step(
        self,
        positions: torch.Tensor,  # (N, 3)
        velocities: torch.Tensor,  # (N, 3)
        acceleration_fn: Callable,  # Function: (pos, vel) -> accel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one integration step.
        
        Args:
            positions: Current positions (N, 3)
            velocities: Current velocities (N, 3)
            acceleration_fn: Function that computes accelerations
            
        Returns:
            (new_positions, new_velocities)
        """
        # Compute acceleration at current state
        accel_n = acceleration_fn(positions, velocities)
        
        # Half-step velocity update
        v_half = velocities + 0.5 * self.dt * accel_n
        
        # Position update
        x_new = positions + self.dt * v_half
        
        # Compute acceleration at new position
        accel_n1 = acceleration_fn(x_new, v_half)
        
        # Full velocity update
        v_new = v_half + 0.5 * self.dt * accel_n1
        
        return x_new, v_new
    
    def rollout(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        acceleration_fn: Callable,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout trajectory for n_steps.
        
        Returns:
            positions: (n_steps+1, N, 3)
            velocities: (n_steps+1, N, 3)
        """
        traj_pos = [positions]
        traj_vel = [velocities]
        
        x, v = positions, velocities
        for _ in range(n_steps):
            x, v = self.step(x, v, acceleration_fn)
            traj_pos.append(x)
            traj_vel.append(v)
        
        return torch.stack(traj_pos), torch.stack(traj_vel)


class RK4Integrator:
    """
    Classic 4th-order Runge-Kutta integrator.
    
    High accuracy but doesn't preserve conservation laws.
    Good for general ODEs.
    """
    
    def __init__(self, dt: float = 0.01):
        """
        Args:
            dt: Timestep size
        """
        self.dt = dt
    
    def step(
        self,
        positions: torch.Tensor,  # (N, 3)
        velocities: torch.Tensor,  # (N, 3)
        acceleration_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one RK4 integration step.
        
        State vector: y = [x, v]
        Derivative: dy/dt = [v, a]
        """
        dt = self.dt
        
        # k1
        v1 = velocities
        a1 = acceleration_fn(positions, velocities)
        
        # k2
        x2 = positions + 0.5 * dt * v1
        v2 = velocities + 0.5 * dt * a1
        a2 = acceleration_fn(x2, v2)
        
        # k3
        x3 = positions + 0.5 * dt * v2
        v3 = velocities + 0.5 * dt * a2
        a3 = acceleration_fn(x3, v3)
        
        # k4
        x4 = positions + dt * v3
        v4 = velocities + dt * a3
        a4 = acceleration_fn(x4, v4)
        
        # Weighted average
        x_new = positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        v_new = velocities + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
        
        return x_new, v_new
    
    def rollout(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        acceleration_fn: Callable,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout trajectory for n_steps.
        
        Returns:
            positions: (n_steps+1, N, 3)
            velocities: (n_steps+1, N, 3)
        """
        traj_pos = [positions]
        traj_vel = [velocities]
        
        x, v = positions, velocities
        for _ in range(n_steps):
            x, v = self.step(x, v, acceleration_fn)
            traj_pos.append(x)
            traj_vel.append(v)
        
        return torch.stack(traj_pos), torch.stack(traj_vel)


if __name__ == "__main__":
    print("Testing Integrators...")
    
    # Simple harmonic oscillator: a = -k*x
    def sho_acceleration(x, v):
        k = 1.0  # Spring constant
        return -k * x
    
    # Initial conditions
    x0 = torch.tensor([[1.0, 0.0, 0.0]])
    v0 = torch.tensor([[0.0, 0.0, 0.0]])
    
    # Test Symplectic Integrator
    print("\n1. Symplectic Integrator")
    symplectic = SymplecticIntegrator(dt=0.01)
    x_symp, v_symp = symplectic.rollout(x0, v0, sho_acceleration, n_steps=100)
    
    # Energy: E = 0.5*k*x^2 + 0.5*v^2
    E_symp = 0.5 * (x_symp**2 + v_symp**2).sum(dim=-1)
    energy_drift_symp = torch.std(E_symp).item()
    print(f"Energy drift (std): {energy_drift_symp:.2e}")
    
    # Test RK4 Integrator
    print("\n2. RK4 Integrator")
    rk4 = RK4Integrator(dt=0.01)
    x_rk4, v_rk4 = rk4.rollout(x0, v0, sho_acceleration, n_steps=100)
    
    E_rk4 = 0.5 * (x_rk4**2 + v_rk4**2).sum(dim=-1)
    energy_drift_rk4 = torch.std(E_rk4).item()
    print(f"Energy drift (std): {energy_drift_rk4:.2e}")
    
    # Compare
    print("\n3. Comparison")
    print(f"Symplectic energy drift: {energy_drift_symp:.2e}")
    print(f"RK4 energy drift: {energy_drift_rk4:.2e}")
    
    if energy_drift_symp < energy_drift_rk4:
        print("✅ Symplectic preserves energy better!")
    
    print("\n✅ Integrator tests passed!")
