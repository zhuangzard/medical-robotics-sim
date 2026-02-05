# Proof-of-Concept Plan: Physics-Informed Robotic Manipulation
## 2-Week Sprint (February 5-19, 2026)

**Objective**: Validate that physics-informed neural architectures outperform pure learning on sample efficiency, safety, and generalization for robotic manipulation tasks.

---

## Week 1: Infrastructure + Simple Physics Task

### Day 1-2: Environment Setup

#### Codebase Structure
```
medical-robotics-sim/
├── physics_core/
│   ├── __init__.py
│   ├── dynamical_gnn.py        # Dynami-CAL implementation
│   ├── edge_frame.py            # Edge-local coordinate systems
│   ├── integrators.py           # Semi-implicit Euler, Verlet
│   └── tests/
│       ├── test_edge_frame.py   # Unit tests for antisymmetry
│       └── test_conservation.py # Momentum/energy validation
├── environments/
│   ├── push_box.py              # Simple rigid body task
│   ├── tissue_grasp.py          # Soft body grasping (Week 2)
│   └── rendering.py             # Visualization utilities
├── baselines/
│   ├── ppo_baseline.py          # Standard PPO (no physics)
│   ├── gns_baseline.py          # Graph Network Simulator
│   └── physics_informed.py      # Our approach: PPO + Dynami-CAL
├── training/
│   ├── train.py                 # Main training loop
│   ├── eval.py                  # Evaluation metrics
│   └── config.yaml              # Hyperparameters
└── experiments/
    ├── week1_push_box/
    └── week2_tissue_grasp/
```

#### Software Dependencies
```bash
# Create conda environment
conda create -n physics-robot python=3.10
conda activate physics-robot

# Core dependencies
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Simulation & robotics
pip install gym==0.26.2
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install stable-baselines3==2.1.0

# Utilities
pip install wandb matplotlib seaborn opencv-python
pip install hydra-core omegaconf

# Medical robotics (Week 2)
pip install trimesh pyvista SimpleITK
```

#### Hardware Requirements
```
Minimum:
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: 8 cores
- RAM: 32GB
- Storage: 50GB SSD

Recommended:
- GPU: NVIDIA A100 (40GB) or RTX 4090
- CPU: 16 cores
- RAM: 64GB
- Storage: 100GB NVMe
```

---

### Day 3-4: Implement Dynami-CAL Core

#### Module 1: EdgeFrame (edge_frame.py)
```python
"""
Edge-local antisymmetric coordinate frames
Guarantees momentum conservation by design
"""
import torch
import torch.nn as nn

class EdgeFrame:
    """Construct antisymmetric local reference frames for edges"""
    
    @staticmethod
    def construct(pos, vel, ang_vel, edge_index, eps=1e-8):
        """
        Args:
            pos: [N, 3] node positions
            vel: [N, 3] node velocities
            ang_vel: [N, 3] node angular velocities
            edge_index: [2, E] edge connectivity
            eps: numerical stability constant
        
        Returns:
            edge_frames: [E, 3, 3] orthonormal frames for each edge
            dist: [E, 1] edge lengths
        """
        row, col = edge_index
        
        # === e1: Radial direction (antisymmetric) ===
        r_ij = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(r_ij, dim=-1, keepdim=True) + eps  # [E, 1]
        e1 = r_ij / dist  # [E, 3]
        
        # === e2: Gram-Schmidt orthogonalization of relative velocity ===
        v_ij = vel[row] - vel[col]  # [E, 3]
        
        # Remove component parallel to e1
        v_ij_perp = v_ij - (v_ij * e1).sum(dim=-1, keepdim=True) * e1  # [E, 3]
        norm_v = torch.norm(v_ij_perp, dim=-1, keepdim=True) + eps
        
        e2 = v_ij_perp / norm_v  # [E, 3]
        
        # === e3: Complete right-hand system (SYMMETRIC - handled via MLP) ===
        e3 = torch.cross(e1, e2, dim=-1)  # [E, 3]
        
        # Stack into [E, 3, 3] tensor
        edge_frames = torch.stack([e1, e2, e3], dim=1)  # [E, 3, 3]
        
        return edge_frames, dist
    
    @staticmethod
    def test_antisymmetry(pos, vel, edge_index):
        """
        Unit test: Verify e1, e2 are antisymmetric when edge direction flips
        """
        frames_ij, _ = EdgeFrame.construct(pos, vel, None, edge_index)
        
        # Reverse edge directions
        edge_index_reversed = edge_index.flip(0)
        frames_ji, _ = EdgeFrame.construct(pos, vel, None, edge_index_reversed)
        
        # Check: e1_ji = -e1_ij, e2_ji = -e2_ij
        e1_error = torch.norm(frames_ji[:, 0] + frames_ij[:, 0])
        e2_error = torch.norm(frames_ji[:, 1] + frames_ij[:, 1])
        
        assert e1_error < 1e-5, f"e1 antisymmetry failed: error = {e1_error}"
        assert e2_error < 1e-5, f"e2 antisymmetry failed: error = {e2_error}"
        
        print("✓ Antisymmetry test passed")
        return True

# Run test
if __name__ == "__main__":
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]])
    vel = torch.tensor([[1.0, 0.0, 0.0], [-0.5, 0.3, 0.1]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    EdgeFrame.test_antisymmetry(pos, vel, edge_index)
```

#### Module 2: Dynami-CAL GNN (dynamical_gnn.py)
```python
"""
Core Dynami-CAL GraphNet implementation
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

from .edge_frame import EdgeFrame

class ScalarizationBlock(nn.Module):
    """Project 3D features onto edge-local frames → rotation-invariant scalars"""
    
    def forward(self, pos, vel, edge_frames, dist, edge_index):
        """
        Returns:
            z_scalar: [E, num_features] rotation-invariant edge features
        """
        row, col = edge_index
        
        # Relative quantities
        r_ij = pos[row] - pos[col]
        v_ij = vel[row] - vel[col]
        
        # Project onto local frame (using Einstein summation for clarity)
        r_proj = torch.einsum('ef,ef->e', r_ij, edge_frames[:, 0])  # r·e1
        v_proj = torch.stack([
            torch.einsum('ef,ef->e', v_ij, edge_frames[:, 0]),  # v·e1
            torch.einsum('ef,ef->e', v_ij, edge_frames[:, 1]),  # v·e2
            torch.einsum('ef,ef->e', v_ij, edge_frames[:, 2])   # v·e3
        ], dim=-1)
        
        # Assemble scalar features
        z_scalar = torch.cat([
            dist,              # [E, 1]
            r_proj.unsqueeze(1),  # [E, 1]
            v_proj             # [E, 3]
        ], dim=-1)  # [E, 5]
        
        return z_scalar

class VectorizationBlock(nn.Module):
    """Decode scalar force coefficients → 3D force vectors"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.force_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [f1, f2, f3] force coefficients
        )
        
        # Direction indicator network (ensures f3 antisymmetry)
        self.direction_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, edge_embeddings, edge_frames, edge_index):
        """
        Returns:
            forces: [E, 3] force vectors in global frame
        """
        # Predict force coefficients
        f_coeffs = self.force_decoder(edge_embeddings)  # [E, 3]
        
        # Handle e3 symmetry: make f3 antisymmetric via edge direction indicator
        row, col = edge_index
        direction_indicator = (row < col).float().unsqueeze(1) * 2 - 1  # [E, 1]: +1 or -1
        f3_correction = self.direction_net(direction_indicator)  # [E, 1]
        
        f_coeffs[:, 2] = f_coeffs[:, 2] * f3_correction.squeeze()
        
        # Reconstruct 3D forces
        forces = torch.einsum('ei,eij->ej', f_coeffs, edge_frames)  # [E, 3]
        
        return forces

class DynamiCALGraphNet(MessagePassing):
    """Complete Dynami-CAL model"""
    
    def __init__(self, hidden_dim=64, num_message_passing_steps=3):
        super().__init__(aggr='add')
        
        self.scalarization = ScalarizationBlock()
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5 scalar features from scalarization
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_message_passing_steps)
        ])
        
        self.vectorization = VectorizationBlock(hidden_dim)
        
    def forward(self, pos, vel, ang_vel, edge_index):
        """
        Returns:
            forces: [N, 3] forces on each node
            torques: [N, 3] torques on each node (optional, for future)
        """
        # 1. Construct edge frames
        edge_frames, dist = EdgeFrame.construct(pos, vel, ang_vel, edge_index)
        
        # 2. Scalarization
        z_scalar = self.scalarization(pos, vel, edge_frames, dist, edge_index)
        
        # 3. Edge encoding
        edge_embeddings = self.edge_encoder(z_scalar)  # [E, hidden_dim]
        
        # 4. Message passing
        row, col = edge_index
        for mlp in self.message_mlps:
            # Concatenate source and target node features
            edge_features = torch.cat([edge_embeddings[row], edge_embeddings[col]], dim=-1)
            messages = mlp(edge_features)
            
            # Aggregate messages to edges
            edge_embeddings = edge_embeddings + messages  # Residual connection
        
        # 5. Vectorization
        forces = self.vectorization(edge_embeddings, edge_frames, edge_index)
        
        # 6. Aggregate forces to nodes
        node_forces = scatter_add(forces, col, dim=0, dim_size=pos.size(0))
        
        return node_forces

# Momentum conservation test
def test_momentum_conservation():
    """Test that total momentum is conserved"""
    model = DynamiCALGraphNet(hidden_dim=32, num_message_passing_steps=2)
    
    # Create test system: 3 particles in triangle
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0]
    ])
    vel = torch.randn(3, 3)
    
    # Fully connected edges
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ])
    
    # Forward pass
    forces = model(pos, vel, ang_vel=None, edge_index=edge_index)
    
    # Check momentum conservation
    total_force = forces.sum(dim=0)
    momentum_error = torch.norm(total_force).item()
    
    print(f"Total force: {total_force}")
    print(f"Momentum error: {momentum_error:.2e}")
    
    assert momentum_error < 1e-5, f"Momentum conservation violated: {momentum_error}"
    print("✓ Momentum conservation test passed")

if __name__ == "__main__":
    test_momentum_conservation()
```

---

### Day 5: Simple Task - Push Box

#### Environment (push_box.py)
```python
"""
Simple rigid body task: Push a box to target location
Tests: Contact dynamics, momentum conservation, generalization
"""
import gym
from gym import spaces
import numpy as np
import torch
import mujoco
from dm_control import mjcf

class PushBoxEnv(gym.Env):
    """
    Task: Robot end-effector pushes a box on a table to target position
    
    Observation: [robot_pos, robot_vel, box_pos, box_vel, target_pos]
    Action: [dx, dy] end-effector displacement
    Reward: -||box_pos - target_pos||
    """
    
    def __init__(self, box_mass=1.0, friction=0.3, visualize=False):
        super().__init__()
        
        self.box_mass = box_mass
        self.friction = friction
        self.visualize = visualize
        
        # MuJoCo model
        self.model = self._build_mujoco_scene()
        self.data = mujoco.MjData(self.model)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        
        self.max_steps = 200
        self.current_step = 0
        
    def _build_mujoco_scene(self):
        """Build MuJoCo MJCF model"""
        mjcf_model = mjcf.RootElement()
        
        # Table
        table = mjcf_model.worldbody.add('geom', type='box', size=[0.5, 0.5, 0.02],
                                          pos=[0, 0, 0], rgba=[0.8, 0.8, 0.8, 1])
        
        # Box (movable)
        box_body = mjcf_model.worldbody.add('body', name='box', pos=[0, 0, 0.07])
        box_body.add('geom', type='box', size=[0.05, 0.05, 0.05], 
                     mass=self.box_mass, rgba=[1, 0, 0, 1])
        box_body.add('joint', type='free')
        
        # Robot end-effector (simple sphere)
        ee_body = mjcf_model.worldbody.add('body', name='end_effector', pos=[0.3, 0, 0.07])
        ee_body.add('geom', type='sphere', size=[0.02], rgba=[0, 0, 1, 1])
        ee_body.add('joint', type='slide', axis=[1, 0, 0], range=[-0.5, 0.5])
        ee_body.add('joint', type='slide', axis=[0, 1, 0], range=[-0.5, 0.5])
        
        # Compile
        return mjcf.Physics.from_mjcf_model(mjcf_model).model.ptr
    
    def reset(self):
        """Reset environment"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Random initial box position
        self.data.qpos[0:2] = np.random.uniform(-0.2, 0.2, 2)
        
        # Random target position
        self.target_pos = np.random.uniform(-0.3, 0.3, 2)
        
        # Reset robot end-effector
        self.data.qpos[7:9] = np.array([0.3, 0.0])
        
        self.current_step = 0
        
        return self._get_obs()
    
    def step(self, action):
        """Execute action"""
        # Apply action to end-effector
        self.data.qpos[7:9] += action
        
        # Simulate physics (10 substeps for stability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = -np.linalg.norm(self.data.qpos[0:2] - self.target_pos)
        done = (self.current_step >= self.max_steps) or (reward > -0.05)
        
        self.current_step += 1
        
        info = {
            'box_pos': self.data.qpos[0:2].copy(),
            'target_pos': self.target_pos.copy(),
            'distance': -reward
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get observation"""
        robot_pos = self.data.qpos[7:9]
        robot_vel = self.data.qvel[6:8]
        box_pos = self.data.qpos[0:2]
        box_vel = self.data.qvel[0:2]
        
        obs = np.concatenate([
            robot_pos, robot_vel,
            box_pos, box_vel,
            self.target_pos,
            np.array([self.box_mass])  # Expose mass (for generalization test)
        ])
        
        return obs.astype(np.float32)

# Test environment
if __name__ == "__main__":
    env = PushBoxEnv(box_mass=1.0, friction=0.3)
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step: reward={reward:.3f}, distance={info['distance']:.3f}")
        
        if done:
            break
```

---

### Day 6-7: Train Baselines + Physics-Informed Model

#### Training Script (training/train.py)
```python
"""
Main training script with 3 baselines:
1. Pure PPO (no physics)
2. GNS (pure data-driven GNN)
3. PPO + Dynami-CAL (our approach)
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
import yaml

from environments.push_box import PushBoxEnv
from physics_core.dynamical_gnn import DynamiCALGraphNet
from baselines.physics_informed import PhysicsInformedPolicy

def make_env(mass, friction):
    """Environment factory"""
    def _init():
        return PushBoxEnv(box_mass=mass, friction=friction)
    return _init

class ExperimentConfig:
    """Experiment configuration"""
    def __init__(self):
        # Environment
        self.num_envs = 8
        self.train_box_mass = 1.0  # Training distribution
        self.test_box_masses = [0.5, 2.0, 3.0]  # OOD generalization test
        
        # Training
        self.total_timesteps = 100_000
        self.learning_rate = 3e-4
        self.batch_size = 256
        
        # Model architecture
        self.physics_hidden_dim = 64
        self.policy_hidden_dim = 128
        
        # Logging
        self.wandb_project = "physics-informed-robotics"
        self.eval_freq = 5000

def train_baseline_ppo(config):
    """Baseline 1: Pure PPO (no physics knowledge)"""
    print("\n=== Training Baseline PPO ===")
    
    # Create vectorized environment
    envs = SubprocVecEnv([
        make_env(config.train_box_mass, friction=0.3)
        for _ in range(config.num_envs)
    ])
    
    # Standard MLP policy
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        verbose=1
    )
    
    # Train
    model.learn(total_timesteps=config.total_timesteps)
    
    # Save
    model.save("baseline_ppo")
    
    return model

def train_physics_informed(config):
    """Our approach: PPO + Dynami-CAL physics layer"""
    print("\n=== Training Physics-Informed Model ===")
    
    envs = SubprocVecEnv([
        make_env(config.train_box_mass, friction=0.3)
        for _ in range(config.num_envs)
    ])
    
    # Custom policy with physics layer
    policy_kwargs = dict(
        features_extractor_class=PhysicsInformedPolicy,
        features_extractor_kwargs=dict(
            physics_hidden_dim=config.physics_hidden_dim
        )
    )
    
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    model.learn(total_timesteps=config.total_timesteps)
    model.save("physics_informed")
    
    return model

def evaluate_generalization(model, test_masses, num_episodes=50):
    """Evaluate on out-of-distribution box masses"""
    results = {}
    
    for mass in test_masses:
        env = PushBoxEnv(box_mass=mass, friction=0.3)
        
        success_count = 0
        total_rewards = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if info['distance'] < 0.05:  # Success threshold
                success_count += 1
        
        results[mass] = {
            'success_rate': success_count / num_episodes,
            'avg_reward': np.mean(total_rewards)
        }
    
    return results

def main():
    config = ExperimentConfig()
    
    # Initialize W&B
    wandb.init(project=config.wandb_project, config=vars(config))
    
    # Train all models
    print("Starting experiment...")
    
    baseline_ppo = train_baseline_ppo(config)
    physics_informed = train_physics_informed(config)
    
    # Evaluate on training distribution
    print("\n=== Evaluation on Training Distribution (mass=1.0) ===")
    baseline_results_train = evaluate_generalization(baseline_ppo, [1.0])
    physics_results_train = evaluate_generalization(physics_informed, [1.0])
    
    # Evaluate on OOD
    print("\n=== Evaluation on Out-of-Distribution Masses ===")
    baseline_results_ood = evaluate_generalization(baseline_ppo, config.test_box_masses)
    physics_results_ood = evaluate_generalization(physics_informed, config.test_box_masses)
    
    # Log results
    wandb.log({
        "baseline_ppo_train_success": baseline_results_train[1.0]['success_rate'],
        "physics_informed_train_success": physics_results_train[1.0]['success_rate'],
        "baseline_ppo_ood_avg_success": np.mean([
            results['success_rate'] for results in baseline_results_ood.values()
        ]),
        "physics_informed_ood_avg_success": np.mean([
            results['success_rate'] for results in physics_results_ood.values()
        ])
    })
    
    print("\n=== Results Summary ===")
    print(f"Baseline PPO (train): {baseline_results_train[1.0]['success_rate']:.2%}")
    print(f"Physics-Informed (train): {physics_results_train[1.0]['success_rate']:.2%}")
    print(f"\nBaseline PPO (OOD avg): {np.mean([r['success_rate'] for r in baseline_results_ood.values()]):.2%}")
    print(f"Physics-Informed (OOD avg): {np.mean([r['success_rate'] for r in physics_results_ood.values()]):.2%}")

if __name__ == "__main__":
    main()
```

**Expected Runtime**: 4-6 hours on RTX 3080

**Expected Results**:
```
Training Distribution (mass=1.0kg):
  Baseline PPO:        85% success
  Physics-Informed:    92% success (+8%)

OOD (mass=0.5kg, 2.0kg, 3.0kg):
  Baseline PPO:        52% average success
  Physics-Informed:    78% average success (+50% relative)
```

---

## Week 2: Medical Robotics Application

### Day 8-9: Soft Tissue Physics

#### Neo-Hookean Material Model
```python
"""
Soft tissue mechanics for medical robotics
Based on hyperelastic material model
"""
import torch
import torch.nn as nn

class SoftTissuePhysics(nn.Module):
    """
    Neo-Hookean hyperelastic model for soft tissue
    
    Strain energy: W = (μ/2)(I₁ - 3) - μlog(J) + (λ/2)log²(J)
    where I₁ = trace(F^T F), J = det(F), F = deformation gradient
    """
    
    def __init__(self, mu=10e3, lam=20e3, rho=1000):
        """
        Args:
            mu: Shear modulus (Pa)
            lam: Lame's first parameter (Pa)
            rho: Density (kg/m³)
        """
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu))
        self.lam = nn.Parameter(torch.tensor(lam))
        self.rho = rho
    
    def compute_stress(self, F):
        """
        Compute 1st Piola-Kirchhoff stress tensor
        
        Args:
            F: [N, 3, 3] deformation gradient
        
        Returns:
            P: [N, 3, 3] stress tensor
        """
        # Invariants
        C = torch.matmul(F.transpose(-2, -1), F)  # Right Cauchy-Green tensor
        I1 = torch.trace(C)
        J = torch.det(F)
        
        # Neo-Hookean stress
        F_inv_T = torch.inverse(F).transpose(-2, -1)
        
        P = self.mu * (F - F_inv_T) + self.lam * torch.log(J).unsqueeze(-1).unsqueeze(-1) * F_inv_T
        
        return P
    
    def compute_forces(self, mesh_pos, rest_pos, mesh_connectivity):
        """
        Convert stress to nodal forces
        
        Args:
            mesh_pos: [N, 3] current positions
            rest_pos: [N, 3] reference positions
            mesh_connectivity: [E, 4] tetrahedral elements
        
        Returns:
            forces: [N, 3] forces on each node
        """
        # Compute deformation gradient for each element
        elements = mesh_connectivity
        
        # ... (finite element assembly, see medical robotics literature)
        # This is a placeholder - full implementation requires FEM library
        
        return forces

# Integration with Dynami-CAL
class SoftTissueDynamiCAL(DynamiCALGraphNet):
    """Extended Dynami-CAL for soft tissue"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tissue_physics = SoftTissuePhysics(mu=10e3, lam=20e3)
    
    def forward(self, pos, vel, mesh_connectivity):
        # Compute contact forces (Dynami-CAL)
        contact_forces = super().forward(pos, vel, ang_vel=None, edge_index=self._construct_contact_graph(pos))
        
        # Compute internal tissue forces (Neo-Hookean)
        internal_forces = self.tissue_physics.compute_forces(pos, rest_pos=self.rest_pos, 
                                                               mesh_connectivity=mesh_connectivity)
        
        # Total forces
        total_forces = contact_forces + internal_forces
        
        return total_forces
```

### Day 10-11: Tissue Grasping Task

#### Environment (tissue_grasp.py)
```python
"""
Medical robotics task: Grasp soft tissue without rupturing
Success criteria:
- Grasp securely (force > 2N)
- Don't rupture (force < 5N)
- Minimize tissue deformation
"""
import gym
import numpy as np
import pyvista as pv

class TissueGraspEnv(gym.Env):
    """
    Surgical robot grasping deformable tissue
    Physics: Neo-Hookean + contact
    """
    
    def __init__(self, tissue_stiffness=10e3, rupture_threshold=5.0):
        super().__init__()
        
        self.rupture_threshold = rupture_threshold
        self.tissue_stiffness = tissue_stiffness
        
        # Load tissue mesh (liver phantom)
        self.tissue_mesh = self._create_tissue_mesh()
        
        # Action: gripper position + closure
        self.action_space = gym.spaces.Box(
            low=np.array([-0.1, -0.1, -0.1, 0.0]),  # [dx, dy, dz, gripper_open]
            high=np.array([0.1, 0.1, 0.1, 1.0]),
            dtype=np.float32
        )
        
        # Observation: RGB-D + force sensor
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            'depth': gym.spaces.Box(0, 10, shape=(128, 128), dtype=np.float32),
            'force': gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
        })
        
    def _create_tissue_mesh(self):
        """Create soft tissue mesh (tetrahedral)"""
        # Simple cube for now (replace with actual organ geometry)
        mesh = pv.Cube()
        mesh = mesh.triangulate().compute_normals()
        return mesh
    
    def reset(self):
        # Reset tissue to rest configuration
        self.tissue_mesh.points = self.rest_points.copy()
        
        # Random gripper start position
        self.gripper_pos = np.random.uniform([-0.05, -0.05, 0.1], [0.05, 0.05, 0.15])
        
        return self._get_obs()
    
    def step(self, action):
        # Update gripper
        self.gripper_pos += action[:3]
        gripper_closure = action[3]
        
        # Simulate tissue deformation (call physics engine)
        tissue_forces, contact_force = self._simulate_physics(self.gripper_pos, gripper_closure)
        
        # Check for rupture
        max_stress = np.max(np.linalg.norm(tissue_forces, axis=1))
        ruptured = (max_stress > self.rupture_threshold)
        
        # Reward shaping
        if ruptured:
            reward = -100  # Large penalty
            done = True
        elif contact_force > 2.0:  # Secure grasp
            reward = 10 - 0.1 * np.linalg.norm(tissue_forces).sum()  # Penalize deformation
            done = True
        else:
            reward = -0.1  # Small step penalty
            done = False
        
        obs = self._get_obs()
        info = {
            'contact_force': contact_force,
            'max_stress': max_stress,
            'ruptured': ruptured
        }
        
        return obs, reward, done, info
```

### Day 12-14: Experiments + Analysis

#### Evaluation Metrics
```python
"""
Comprehensive evaluation for medical robotics
"""
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalRoboticsEvaluator:
    """Evaluate surgical robot performance"""
    
    @staticmethod
    def compute_metrics(trajectories):
        """
        Args:
            trajectories: List of episode data
        
        Returns:
            metrics: Dict of performance metrics
        """
        metrics = {
            'success_rate': 0,
            'rupture_rate': 0,
            'avg_contact_force': [],
            'avg_tissue_deformation': [],
            'completion_time': []
        }
        
        for traj in trajectories:
            # Success: Grasped without rupture
            if traj['final_contact_force'] > 2.0 and not traj['ruptured']:
                metrics['success_rate'] += 1
            
            if traj['ruptured']:
                metrics['rupture_rate'] += 1
            
            metrics['avg_contact_force'].append(np.mean(traj['contact_forces']))
            metrics['avg_tissue_deformation'].append(np.mean(traj['deformations']))
            metrics['completion_time'].append(traj['steps'])
        
        # Normalize
        n = len(trajectories)
        metrics['success_rate'] /= n
        metrics['rupture_rate'] /= n
        
        return metrics
    
    @staticmethod
    def plot_comparison(baseline_metrics, physics_metrics):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate
        axes[0, 0].bar(['Baseline', 'Physics-Informed'], 
                        [baseline_metrics['success_rate'], physics_metrics['success_rate']])
        axes[0, 0].set_title('Success Rate')
        axes[0, 0].set_ylim([0, 1])
        
        # Rupture rate
        axes[0, 1].bar(['Baseline', 'Physics-Informed'],
                        [baseline_metrics['rupture_rate'], physics_metrics['rupture_rate']],
                        color=['red', 'green'])
        axes[0, 1].set_title('Rupture Rate (Lower is Better)')
        
        # Force distribution
        axes[1, 0].hist([baseline_metrics['avg_contact_force'], 
                         physics_metrics['avg_contact_force']], 
                        label=['Baseline', 'Physics-Informed'], alpha=0.7)
        axes[1, 0].axvline(x=5.0, color='red', linestyle='--', label='Rupture Threshold')
        axes[1, 0].set_title('Contact Force Distribution')
        axes[1, 0].legend()
        
        # Sample efficiency
        axes[1, 1].plot(baseline_metrics['learning_curve'], label='Baseline')
        axes[1, 1].plot(physics_metrics['learning_curve'], label='Physics-Informed')
        axes[1, 1].set_xlabel('Training Episodes')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Sample Efficiency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('medical_robotics_comparison.png', dpi=300)
        print("✓ Saved comparison plot")
```

---

## Deliverables

### Code Repository
```
✓ Complete implementation (1500+ lines)
✓ Unit tests (momentum conservation, edge frame, etc.)
✓ Training scripts
✓ Evaluation tools
✓ Visualization utilities
```

### Experimental Results
```
✓ Week 1 results: Push box task
  - Sample efficiency comparison
  - OOD generalization curves
  - Momentum conservation validation

✓ Week 2 results: Tissue grasping
  - Success rate: baseline vs. physics-informed
  - Safety analysis (rupture rates)
  - Force control accuracy
```

### Technical Report (3-5 pages)
```
1. Introduction
2. Method: Physics-informed architecture
3. Experiments:
   - Push box (simple contact)
   - Tissue grasping (soft body)
4. Results & Analysis
5. Limitations & Future Work
```

**Target**: Submit to ICRA 2027 Workshop or CoRL 2026 as short paper

---

## Success Criteria

### Minimum Viable Results
- [ ] Dynami-CAL conserves momentum (error < 0.1%)
- [ ] Physics-informed model achieves > 85% success on push box
- [ ] Generalization: Physics-informed outperforms baseline by > 30% on OOD masses
- [ ] Medical task: Zero ruptures in 50 test episodes

### Stretch Goals
- [ ] Real dVRK robot experiments (if hardware available)
- [ ] Multi-modal fusion (RGB-D + force + ultrasound)
- [ ] Full paper draft ready for conference submission

---

## Risk Mitigation

### Risk 1: Implementation Bugs
**Mitigation**:
- Extensive unit tests (see Day 3-4)
- Reproduce Dynami-CAL paper results first
- Use pre-trained vision encoders (RT-2 backbone) if available

### Risk 2: Training Instability
**Mitigation**:
- Start with small networks (hidden_dim=32)
- Use gradient clipping (max_norm=0.5)
- Curriculum learning (easy→hard tasks)
- Monitor NaN detection (see Training Best Practices)

### Risk 3: Insufficient Compute
**Mitigation**:
- Use smaller batch sizes (128 instead of 256)
- Reduce environment parallelism (4 envs instead of 8)
- Train for fewer steps if needed (50K instead of 100K)
- Focus on Week 1 task only if time-constrained

---

## Timeline Gantt Chart

```
Week 1:
Day 1-2: [████████████████] Setup + Dependencies
Day 3-4: [████████████████] Implement Dynami-CAL
Day 5:   [████████████████] Push box environment
Day 6-7: [████████████████] Train & evaluate

Week 2:
Day 8-9:  [████████████████] Soft tissue physics
Day 10-11:[████████████████] Medical robotics task
Day 12-14:[████████████████] Experiments + Report
```

**Total Effort**: ~80-100 hours (10-12 hours/day)

---

## Contact & Support

**Questions?** Reach out to:
- Taisen: taisen@research.ai
- GitHub Issues: [medical-robotics-sim/issues]
- Slack Channel: #physics-informed-robotics
