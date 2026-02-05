#!/usr/bin/env python3
"""
Complete PPO Training Script with CLI
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from environments.push_box import make_push_box_env
import numpy as np


class ProgressCallback(BaseCallback):
    """Callback for tracking training progress"""
    
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_count = 0
        
    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
        if self.num_timesteps % self.check_freq == 0:
            if self.verbose > 0:
                print(f"\n📊 Progress: {self.num_timesteps} steps completed")
                print(f"   Episodes: {self.episode_count}")
        
        return True


def train_ppo(
    total_timesteps=200000,
    n_envs=4,
    save_path='./models/ppo',
    log_dir='./logs/ppo',
    learning_rate=3e-4,
    verbose=1
):
    """
    Train PPO agent
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_path: Where to save the model
        log_dir: TensorBoard log directory
        learning_rate: Learning rate
        verbose: Verbosity level
    """
    print('='*60)
    print('🚀 Training PPO Baseline')
    print('='*60)
    print(f'Total timesteps: {total_timesteps:,}')
    print(f'Parallel envs: {n_envs}')
    print(f'Save path: {save_path}')
    print('='*60)
    
    # Create vectorized environment
    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([make_push_box_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_push_box_env])
    
    print('✅ Environment created')
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        tensorboard_log=log_dir
    )
    
    print('✅ PPO model created')
    
    # Train
    callback = ProgressCallback(check_freq=10000, verbose=verbose)
    
    print('\n🏋️  Starting training...')
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'ppo_model')
    model.save(model_file)
    
    print('\n' + '='*60)
    print('✅ Training Complete!')
    print(f'📁 Model saved to: {model_file}')
    print('='*60)
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='Total training timesteps (default: 200000)')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (default: 4)')
    parser.add_argument('--save-path', type=str, default='./models/ppo',
                       help='Model save path (default: ./models/ppo)')
    parser.add_argument('--log-dir', type=str, default='./logs/ppo',
                       help='TensorBoard log directory (default: ./logs/ppo)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    args = parser.parse_args()
    
    # Train
    train_ppo(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_path=args.save_path,
        log_dir=args.log_dir,
        learning_rate=args.learning_rate,
        verbose=args.verbose
    )
