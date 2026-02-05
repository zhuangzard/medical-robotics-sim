"""
Pure PPO Baseline (Baseline 1)
Standard PPO without physics constraints
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.push_box_env import make_push_box_env


class PurePPOAgent:
    """
    Baseline 1: Pure Proximal Policy Optimization
    - Standard MLP policy
    - No physics constraints
    - Learns purely from reward signal
    """
    
    def __init__(self, env, learning_rate=3e-4, verbose=1):
        """
        Initialize Pure PPO agent
        
        Args:
            env: Gym environment or vectorized env
            learning_rate: Learning rate for PPO
            verbose: Verbosity level
        """
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,        # Steps per update
            batch_size=64,       # Minibatch size
            n_epochs=10,         # Gradient steps per update
            gamma=0.99,          # Discount factor
            gae_lambda=0.95,     # GAE lambda
            clip_range=0.2,      # PPO clip range
            ent_coef=0.0,        # Entropy coefficient
            vf_coef=0.5,         # Value function coefficient
            max_grad_norm=0.5,   # Gradient clipping
            verbose=verbose,
            tensorboard_log="./logs/ppo_baseline/"
        )
        
        self.training_history = {
            'episodes_to_success': [],
            'episode_rewards': [],
            'success_rate': []
        }
    
    def train(self, total_timesteps, callback=None):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total timesteps to train
            callback: Optional callback for logging
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, obs, deterministic=True):
        """
        Predict action given observation
        
        Args:
            obs: Observation
            deterministic: Use deterministic policy
        
        Returns:
            action, state (state is None for stateless policy)
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        self.model = PPO.load(path)
    
    def evaluate(self, env, n_episodes=100):
        """
        Evaluate agent performance
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
        
        Returns:
            dict with evaluation metrics
        """
        episode_rewards = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            if info.get('success', False):
                success_count += 1
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / n_episodes,
            'n_episodes': n_episodes
        }


class SuccessTrackingCallback(BaseCallback):
    """
    Callback to track episodes until first success
    """
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.success_achieved = False
        self.episodes_to_success = None
        
    def _on_step(self):
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Check for success
            info = self.locals.get('infos', [{}])[0]
            if info.get('success', False) and not self.success_achieved:
                self.success_achieved = True
                self.episodes_to_success = self.episode_count
                if self.verbose > 0:
                    print(f"\n🎉 First success at episode {self.episode_count}!")
        
        return True


def train_until_success(
    env, 
    max_episodes=10000, 
    max_timesteps=500000,
    success_threshold=0.8,
    eval_freq=1000
):
    """
    Train PPO agent until reaching success threshold
    
    Args:
        env: Training environment
        max_episodes: Maximum episodes to train
        max_timesteps: Maximum timesteps to train
        success_threshold: Success rate threshold (0-1)
        eval_freq: Evaluation frequency (timesteps)
    
    Returns:
        dict with training results
    """
    agent = PurePPOAgent(env, verbose=1)
    callback = SuccessTrackingCallback(check_freq=eval_freq)
    
    print("🚀 Training Pure PPO Baseline...")
    print(f"   Max episodes: {max_episodes}")
    print(f"   Max timesteps: {max_timesteps}")
    print(f"   Success threshold: {success_threshold}")
    
    # Train
    agent.train(total_timesteps=max_timesteps, callback=callback)
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    eval_results = agent.evaluate(env, n_episodes=100)
    
    results = {
        'method': 'Pure PPO',
        'episodes_to_success': callback.episodes_to_success,
        'final_success_rate': eval_results['success_rate'],
        'final_mean_reward': eval_results['mean_reward'],
        'total_timesteps': max_timesteps
    }
    
    print(f"\n✅ Training Complete!")
    print(f"   Episodes to first success: {callback.episodes_to_success}")
    print(f"   Final success rate: {eval_results['success_rate']:.2%}")
    print(f"   Final mean reward: {eval_results['mean_reward']:.2f}")
    
    return agent, results


def main():
    """Main training script with command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Pure PPO Baseline')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                        help='Total timesteps for training (default: 200000)')
    parser.add_argument('--save-path', type=str, default='./models/ppo/ppo_baseline',
                        help='Path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='./logs/ppo/',
                        help='Directory for tensorboard logs')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--box-mass', type=float, default=1.0,
                        help='Box mass for environment')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 TRAINING PURE PPO BASELINE")
    print("="*60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Save path: {args.save_path}")
    print(f"Log directory: {args.log_dir}")
    print("="*60)
    
    try:
        # Create environment
        if args.n_envs > 1:
            env = SubprocVecEnv([
                lambda: make_push_box_env(box_mass=args.box_mass)
                for _ in range(args.n_envs)
            ])
        else:
            env = DummyVecEnv([lambda: make_push_box_env(box_mass=args.box_mass)])
        
        # Create agent
        agent = PurePPOAgent(env, verbose=1)
        
        # Train
        print("\n🏋️  Starting training...")
        callback = SuccessTrackingCallback(check_freq=1000, verbose=1)
        agent.train(total_timesteps=args.total_timesteps, callback=callback)
        
        # Save model
        print(f"\n💾 Saving model to {args.save_path}...")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        agent.save(args.save_path)
        
        # Evaluate
        print(f"\n📊 Evaluating on {args.eval_episodes} episodes...")
        eval_env = DummyVecEnv([lambda: make_push_box_env(box_mass=args.box_mass)])
        results = agent.evaluate(eval_env, n_episodes=args.eval_episodes)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Mean reward: {results['mean_reward']:.2f}")
        print(f"Std reward: {results['std_reward']:.2f}")
        if callback.episodes_to_success:
            print(f"Episodes to first success: {callback.episodes_to_success}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
