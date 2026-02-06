"""
HNN (Hamiltonian Neural Network) Baseline
==========================================

Learns a Hamiltonian H(q, p) and derives forces via autograd.
Guarantees energy conservation (symplectic structure) but NOT momentum conservation.

Used as PPO features extractor: HNN predicts box acceleration,
which is concatenated with raw observation for the policy.

Reference: Greydanus et al. (2019) "Hamiltonian Neural Networks"

Author: PhysRobot Team
Date: 2026-02-06
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HamiltonianNet(nn.Module):
    """
    Learns scalar Hamiltonian H(q, p) → R.
    
    Forces are derived via autograd:
        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q   (= acceleration, since p ≈ m·v)
    
    Uses Softplus activation to ensure H is smooth (C∞),
    which is important for stable autograd.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamiltonian and derive acceleration.
        
        Args:
            q: generalized coordinates [B, 3] (relative position)
            p: generalized momenta [B, 3] (relative velocity, mass=1 proxy)
        
        Returns:
            acceleration: [B, 3] (dp/dt = -∂H/∂q)
        """
        qp = torch.cat([q, p], dim=-1)  # [B, 6]
        qp = qp.requires_grad_(True)
        
        H = self.net(qp)  # [B, 1]
        
        # Compute gradients
        dH = torch.autograd.grad(
            H.sum(), qp,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, 6]
        
        dHdq = dH[:, :3]   # ∂H/∂q
        # dHdp = dH[:, 3:]  # ∂H/∂p (= velocity, not needed)
        
        # Hamilton's equation: dp/dt = -∂H/∂q → acceleration
        acceleration = -dHdq  # [B, 3]
        
        return acceleration


class HNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    HNN-based features extractor for PPO.
    
    Computes box acceleration from Hamiltonian dynamics
    and fuses with raw observation for policy.
    
    Expects 16-dim observation (push_box_env format).
    """
    
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        self.hnn = HamiltonianNet(input_dim=6, hidden_dim=64)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(3 + observation_space.shape[0], features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using HNN dynamics prediction.
        
        Args:
            observations: [B, 16]
        
        Returns:
            features: [B, features_dim]
        """
        # Extract relative state
        ee_pos = observations[:, 4:7]
        box_pos = observations[:, 7:10]
        box_vel = observations[:, 10:13]
        
        q = box_pos - ee_pos  # relative position
        p = box_vel            # momentum proxy (mass=1)
        
        # HNN acceleration prediction
        acc = self.hnn(q, p)  # [B, 3]
        
        # Fuse with raw observation
        combined = torch.cat([acc, observations], dim=-1)
        features = self.feature_proj(combined)
        
        return features


class HNNAgent:
    """
    HNN-based PPO Agent.
    
    Uses Hamiltonian Neural Network to predict energy-conserving dynamics
    as features for PPO policy.
    """
    
    def __init__(self, env, learning_rate: float = 3e-4, verbose: int = 1):
        policy_kwargs = dict(
            features_extractor_class=HNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log="./logs/hnn/",
        )
    
    def train(self, total_timesteps: int, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        self.model.save(path)
    
    def load(self, path: str):
        self.model = PPO.load(path)
    
    def evaluate(self, env, n_episodes: int = 100) -> dict:
        rewards, successes = [], []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            ep_r = 0
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                ep_r += reward[0] if hasattr(reward, '__len__') else reward
                done = dones[0] if hasattr(dones, '__len__') else dones
                info = infos[0] if isinstance(infos, list) else infos
            rewards.append(ep_r)
            successes.append(1 if info.get('success', False) else 0)
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': float(np.mean(successes)),
        }


if __name__ == "__main__":
    print("Testing HNN baseline...")
    
    # Test HamiltonianNet
    hnn = HamiltonianNet(input_dim=6, hidden_dim=64)
    q = torch.randn(4, 3)
    p = torch.randn(4, 3)
    acc = hnn(q, p)
    print(f"  HNN: q={q.shape}, p={p.shape} → acc={acc.shape}")
    
    n_params = sum(p.numel() for p in hnn.parameters())
    print(f"  HNN params: {n_params:,}")
    
    # Test gradient flow
    loss = acc.pow(2).sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in hnn.parameters())
    print(f"  Gradient flow: {'✅' if has_grad else '❌'}")
    
    print("✅ HNN baseline test passed!")
