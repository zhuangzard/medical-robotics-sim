"""
PhysRobot: Physics-Informed Robotics (Our Method)
==================================================

Dual-stream PPO agent:
  - Policy Stream:  MLP(obs)            -> z_policy
  - Physics Stream: SV-GNN(graph)       -> z_physics  (momentum-conserving)
  - Fusion:         concat + Linear     -> features

The physics stream uses SVPhysicsCore (from sv_message_passing.py),
which guarantees  sum_i F_i = 0  by construction for ANY parameters theta.

History:
  v1: DynamiCALGraphNet (PyG MessagePassing) -- did NOT guarantee conservation
  v2: Lightweight MLP on relative geometry  -- no graph at all
  v3 (current): SVMessagePassing pipeline  -- conservation by construction
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

from environments.push_box import PushBoxEnv, make_push_box_env
from physics_core.sv_message_passing import (
    SVPhysicsCore,
    PhysRobotFeaturesExtractorV3 as _PhysRobotFeaturesExtractorV3,
)


# ────────────── SB3 Feature Extractor Adapter ──────────────

class PhysRobotSVFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible wrapper around PhysRobotFeaturesExtractorV3.

    BaseFeaturesExtractor requires (observation_space, features_dim) in __init__
    and outputs a tensor of shape [B, features_dim] from forward().
    """

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.core = _PhysRobotFeaturesExtractorV3(
            obs_dim=obs_dim,
            features_dim=features_dim,
            physics_hidden=32,
            physics_layers=1,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.core(observations)


# Backward-compatible alias
PhysRobotFeaturesExtractor = PhysRobotSVFeaturesExtractor


# ────────────── PhysRobot Agent ──────────────

class PhysRobotAgent:
    """
    PhysRobot: Physics-Informed Robotics Agent (SV-pipeline v3)

    Uses SVPhysicsCore to predict momentum-conserving forces, fused with
    a standard PPO policy stream via concatenation + Linear.
    """

    def __init__(self, env, learning_rate=3e-4, verbose=1, features_dim=64):
        """
        Initialize PhysRobot agent.

        Args:
            env: Gym environment (or VecEnv)
            learning_rate: Learning rate
            verbose: Verbosity level
            features_dim: Output dimension of the features extractor
        """
        policy_kwargs = dict(
            features_extractor_class=PhysRobotSVFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
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
            tensorboard_log="./logs/physrobot/",
        )

        self.training_history = {
            'momentum_errors': [],
            'energy_errors': [],
        }

    def train(self, total_timesteps, callback=None):
        """Train the agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

    def predict(self, obs, deterministic=True):
        """Predict action."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path):
        """Save model."""
        self.model.save(path)

    def load(self, path):
        """Load model."""
        self.model = PPO.load(path)

    def evaluate(self, env, n_episodes=100):
        """
        Evaluate agent performance.

        Works with both raw Gym envs (5-tuple step) and VecEnvs (4-tuple step).
        """
        episode_rewards = []
        success_count = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, dones, infos = step_result
                    done = dones if isinstance(dones, bool) else dones[0]
                    info = infos if isinstance(infos, dict) else infos[0]
                    reward = reward if np.isscalar(reward) else reward[0]
                episode_reward += reward

            episode_rewards.append(episode_reward)
            if isinstance(info, dict) and info.get('success', False):
                success_count += 1

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / n_episodes,
            'n_episodes': n_episodes,
        }


# ────────────── CLI Entry Point ──────────────

def main():
    """Main training script with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description='Train PhysRobot (SV-pipeline v3)')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                        help='Total timesteps for training (default: 200000)')
    parser.add_argument('--save-path', type=str, default='./models/physrobot/physrobot_sv',
                        help='Path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='./logs/physrobot/',
                        help='Directory for tensorboard logs')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--box-mass', type=float, default=1.0,
                        help='Box mass for environment')

    args = parser.parse_args()

    print("=" * 60)
    print("PhysRobot (SV-pipeline v3) Training")
    print("=" * 60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Save path: {args.save_path}")
    print("Physics: SV Message Passing (momentum-conserving)")
    print("=" * 60)

    try:
        # Create environment
        if args.n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            env = SubprocVecEnv([
                make_push_box_env(box_mass=args.box_mass)
                for _ in range(args.n_envs)
            ])
        else:
            env = DummyVecEnv([make_push_box_env(box_mass=args.box_mass)])

        # Create agent
        print("\nInitializing PhysRobot agent (SV-pipeline)...")
        agent = PhysRobotAgent(env, verbose=1)

        # Print param count
        n_params = sum(p.numel() for p in agent.model.policy.parameters())
        print(f"  Total policy parameters: {n_params:,}")

        # Train
        print(f"\nTraining for {args.total_timesteps:,} timesteps...")
        agent.train(total_timesteps=args.total_timesteps)

        # Save model
        print(f"\nSaving model to {args.save_path}...")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        agent.save(args.save_path)

        # Evaluate
        print(f"\nEvaluating on {args.eval_episodes} episodes...")
        eval_env = DummyVecEnv([make_push_box_env(box_mass=args.box_mass)])
        results = agent.evaluate(eval_env, n_episodes=args.eval_episodes)

        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
