"""
Main Training Script for Week 1 Experiments
Trains all three methods and collects sample efficiency data
"""

import numpy as np
import json
import time
from datetime import datetime
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environments.push_box_env import make_push_box_env
from baselines.ppo_baseline import PurePPOAgent, SuccessTrackingCallback
from baselines.gns_baseline import GNSAgent
from baselines.physics_informed import PhysRobotAgent


class DetailedTrackingCallback(BaseCallback):
    """
    Enhanced callback that tracks:
    - Episodes to success
    - Success rate over time
    - Episode rewards
    """
    
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=20, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self.episode_count = 0
        self.success_achieved = False
        self.episodes_to_first_success = None
        
        self.eval_history = []
        self.episode_rewards = []
        self.success_history = []
        
    def _on_step(self):
        # Track episode completion
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode', {}).get('r', 0)
            self.episode_rewards.append(episode_reward)
            
            # Check for first success
            if info.get('success', False):
                self.success_history.append(1)
                if not self.success_achieved:
                    self.success_achieved = True
                    self.episodes_to_first_success = self.episode_count
                    if self.verbose > 0:
                        print(f"\n🎉 First success at episode {self.episode_count}!")
            else:
                self.success_history.append(0)
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_results = self._evaluate()
            self.eval_history.append({
                'timestep': self.n_calls,
                'episode': self.episode_count,
                'mean_reward': eval_results['mean_reward'],
                'success_rate': eval_results['success_rate']
            })
            
            if self.verbose > 0:
                print(f"\n📊 Eval @ {self.n_calls} steps:")
                print(f"   Success rate: {eval_results['success_rate']:.2%}")
                print(f"   Mean reward: {eval_results['mean_reward']:.2f}")
        
        return True
    
    def _evaluate(self):
        """Run evaluation episodes"""
        rewards = []
        successes = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            successes.append(1 if info.get('success', False) else 0)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes)
        }


def train_single_method(
    method_name,
    agent_class,
    train_env,
    eval_env,
    max_timesteps=500000,
    save_dir="./models"
):
    """
    Train a single method and collect results
    
    Args:
        method_name: Name of method ("PPO", "GNS", "PhysRobot")
        agent_class: Agent class to instantiate
        train_env: Training environment
        eval_env: Evaluation environment
        max_timesteps: Maximum training timesteps
        save_dir: Directory to save models
    
    Returns:
        dict with training results
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training {method_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Create agent
    agent = agent_class(train_env, verbose=1)
    
    # Create callback
    callback = DetailedTrackingCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=20,
        verbose=1
    )
    
    # Train
    agent.train(total_timesteps=max_timesteps, callback=callback)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n📊 Final Evaluation for {method_name}...")
    final_eval = agent.evaluate(eval_env, n_episodes=100)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_final.zip")
    agent.save(model_path)
    
    # Compile results
    results = {
        'method': method_name,
        'episodes_to_first_success': callback.episodes_to_first_success,
        'total_episodes': callback.episode_count,
        'total_timesteps': max_timesteps,
        'training_time_seconds': training_time,
        'final_success_rate': final_eval['success_rate'],
        'final_mean_reward': final_eval['mean_reward'],
        'final_std_reward': final_eval['std_reward'],
        'eval_history': callback.eval_history,
        'model_path': model_path
    }
    
    print(f"\n✅ {method_name} Training Complete!")
    print(f"   Episodes to first success: {callback.episodes_to_first_success}")
    print(f"   Final success rate: {final_eval['success_rate']:.2%}")
    print(f"   Training time: {training_time/60:.1f} minutes")
    
    return results


def run_all_experiments(
    output_dir="./data",
    models_dir="./models",
    ppo_timesteps=200000,
    gns_timesteps=80000,
    physrobot_timesteps=16000,
    n_envs=4
):
    """
    Run complete experimental protocol
    
    Args:
        output_dir: Directory for results
        models_dir: Directory for saved models
        ppo_timesteps: Training steps for PPO
        gns_timesteps: Training steps for GNS
        physrobot_timesteps: Training steps for PhysRobot
        n_envs: Number of parallel environments
    
    Returns:
        dict with all results
    """
    print("="*60)
    print("🔬 Week 1 Training Experiments")
    print("   Goal: Generate Table 1 (Sample Efficiency)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    all_results = {}
    
    # === Experiment 1: Pure PPO ===
    print(f"\n{'='*60}")
    print("Experiment 1/3: Pure PPO Baseline")
    print(f"{'='*60}")
    
    ppo_train_env = DummyVecEnv([make_push_box_env(box_mass=1.0) for _ in range(n_envs)])
    ppo_eval_env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    
    ppo_results = train_single_method(
        method_name="Pure PPO",
        agent_class=PurePPOAgent,
        train_env=ppo_train_env,
        eval_env=ppo_eval_env,
        max_timesteps=ppo_timesteps,
        save_dir=models_dir
    )
    all_results['Pure PPO'] = ppo_results
    
    ppo_train_env.close()
    ppo_eval_env.close()
    
    # === Experiment 2: GNS ===
    print(f"\n{'='*60}")
    print("Experiment 2/3: GNS Baseline")
    print(f"{'='*60}")
    
    gns_train_env = DummyVecEnv([make_push_box_env(box_mass=1.0) for _ in range(n_envs)])
    gns_eval_env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    
    gns_results = train_single_method(
        method_name="GNS",
        agent_class=GNSAgent,
        train_env=gns_train_env,
        eval_env=gns_eval_env,
        max_timesteps=gns_timesteps,
        save_dir=models_dir
    )
    all_results['GNS'] = gns_results
    
    gns_train_env.close()
    gns_eval_env.close()
    
    # === Experiment 3: PhysRobot (Ours) ===
    print(f"\n{'='*60}")
    print("Experiment 3/3: PhysRobot (Our Method)")
    print(f"{'='*60}")
    
    physrobot_train_env = DummyVecEnv([make_push_box_env(box_mass=1.0) for _ in range(n_envs)])
    physrobot_eval_env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    
    physrobot_results = train_single_method(
        method_name="PhysRobot",
        agent_class=PhysRobotAgent,
        train_env=physrobot_train_env,
        eval_env=physrobot_eval_env,
        max_timesteps=physrobot_timesteps,
        save_dir=models_dir
    )
    all_results['PhysRobot'] = physrobot_results
    
    physrobot_train_env.close()
    physrobot_eval_env.close()
    
    # === Save Results ===
    results_path = os.path.join(output_dir, "week1_training_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ All Experiments Complete!")
    print(f"{'='*60}")
    print(f"\n📁 Results saved to: {results_path}")
    
    # Print summary table
    print_summary_table(all_results)
    
    return all_results


def print_summary_table(results):
    """
    Print Table 1: Sample Efficiency Comparison
    """
    print(f"\n{'='*60}")
    print("📊 Table 1: Sample Efficiency Comparison")
    print(f"{'='*60}\n")
    
    # Calculate relative improvement
    ppo_episodes = results['Pure PPO']['episodes_to_first_success']
    baseline = ppo_episodes if ppo_episodes else 10000
    
    print(f"{'Method':<20} {'Episodes':<15} {'Success Rate':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for method in ['Pure PPO', 'GNS', 'PhysRobot']:
        episodes = results[method]['episodes_to_first_success']
        success_rate = results[method]['final_success_rate']
        
        if episodes:
            improvement = baseline / episodes
        else:
            episodes = "N/A"
            improvement = 0.0
        
        print(f"{method:<20} {str(episodes):<15} {success_rate*100:>6.1f}%{'':<8} {improvement:>6.1f}x{'':<8}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Week 1 Training Experiments")
    parser.add_argument('--ppo-steps', type=int, default=200000, help="PPO training steps")
    parser.add_argument('--gns-steps', type=int, default=80000, help="GNS training steps")
    parser.add_argument('--physrobot-steps', type=int, default=16000, help="PhysRobot training steps")
    parser.add_argument('--n-envs', type=int, default=4, help="Number of parallel environments")
    parser.add_argument('--output-dir', type=str, default="./data", help="Output directory")
    parser.add_argument('--models-dir', type=str, default="./models", help="Models directory")
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_all_experiments(
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        ppo_timesteps=args.ppo_steps,
        gns_timesteps=args.gns_steps,
        physrobot_timesteps=args.physrobot_steps,
        n_envs=args.n_envs
    )
    
    return results


if __name__ == "__main__":
    main()
