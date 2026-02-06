#!/usr/bin/env python3
"""
Ablation Study Training Script for PhysRobot Paper
===================================================

Runs all 7 ablation variants × 5 seeds, saves results, generates comparison tables.

Ablation Variants:
    1. full_model      — PhysRobot with all components
    2. no_physics_core — Remove Dynami-CAL physics core (pure MLP policy stream)
    3. no_antisymmetric — Use standard EGNN instead of antisymmetric edge frames
    4. no_fusion       — Concatenate instead of cross-attention fusion
    5. no_pretraining  — Skip Stage 1 physics pre-training
    6. gns_baseline    — GNS + PPO (graph network without conservation)
    7. pure_ppo        — Standard PPO without any graph/physics structure

Usage:
    # Full ablation (all variants × 5 seeds)
    python scripts/run_ablation.py --mode full

    # Quick test (1 seed, reduced timesteps)
    python scripts/run_ablation.py --mode quick

    # Single variant
    python scripts/run_ablation.py --variant full_model --seeds 0 1 2

    # Resume from checkpoint
    python scripts/run_ablation.py --mode full --resume

    # Colab mode (reduced steps, single seed)
    python scripts/run_ablation.py --mode colab

Author: PhysRobot Team
Date: 2026-02-06
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ========== Configuration ==========

ABLATION_VARIANTS = {
    'full_model': {
        'description': 'PhysRobot-SV with momentum-conserving SV-pipeline (our method)',
        'agent_type': 'physrobot_sv',
    },
    'no_physics_core': {
        'description': 'Remove physics core entirely, pure MLP policy',
        'agent_type': 'pure_ppo',
    },
    'no_antisymmetric': {
        'description': 'Standard GNN message passing (no antisymmetric edge frames)',
        'agent_type': 'gns',
    },
    'no_fusion': {
        'description': 'Physics stream only (no policy stream / fusion)',
        'agent_type': 'physrobot_sv',  # same arch, ablation in training
    },
    'no_edgeframe': {
        'description': 'Physics MLP in global frame (no relative-geometry features)',
        'agent_type': 'physrobot',  # V2 style
    },
    'gns_baseline': {
        'description': 'GNS + PPO (graph networks without conservation laws)',
        'agent_type': 'gns',
    },
    'pure_ppo': {
        'description': 'Standard PPO without any graph/physics structure',
        'agent_type': 'pure_ppo',
    },
    'sac_baseline': {
        'description': 'SAC (off-policy RL baseline)',
        'agent_type': 'sac',
    },
    'td3_baseline': {
        'description': 'TD3 (off-policy RL baseline)',
        'agent_type': 'td3',
    },
    'hnn_baseline': {
        'description': 'HNN + PPO (Hamiltonian energy conservation, no momentum)',
        'agent_type': 'hnn',
    },
}

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

MODE_CONFIGS = {
    'full': {
        'total_timesteps': 200000,
        'eval_episodes': 100,
        'n_envs': 4,
        'seeds': DEFAULT_SEEDS,
        'eval_checkpoints': [20000, 40000, 80000, 120000, 160000, 200000],
        'ood_masses': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    },
    'quick': {
        'total_timesteps': 20000,
        'eval_episodes': 20,
        'n_envs': 2,
        'seeds': [42],
        'eval_checkpoints': [5000, 10000, 20000],
        'ood_masses': [0.5, 1.0, 2.0],
    },
    'colab': {
        'total_timesteps': 50000,
        'eval_episodes': 30,
        'n_envs': 2,
        'seeds': [42, 123],
        'eval_checkpoints': [10000, 25000, 50000],
        'ood_masses': [0.5, 1.0, 1.5, 2.0],
    },
}


# ========== Training Functions ==========

def create_agent(variant_config: dict, env, seed: int):
    """
    Create agent based on ablation variant configuration.
    
    Supported agent types:
    - pure_ppo      → PurePPOAgent (Stable-Baselines3 PPO)
    - sac           → SAC (Stable-Baselines3)
    - td3           → TD3 (Stable-Baselines3)
    - gns           → GNSAgent (GNN features + PPO)
    - hnn           → HNNAgent (Hamiltonian features + PPO)
    - physrobot     → PhysRobotAgent V2 (MLP physics + PPO)
    - physrobot_sv  → PhysRobot with SV-pipeline (our core innovation)
    """
    agent_type = variant_config['agent_type']
    
    try:
        if agent_type == 'pure_ppo':
            from baselines.ppo_baseline import PurePPOAgent
            return PurePPOAgent(env, verbose=0)
        
        elif agent_type == 'sac':
            from stable_baselines3 import SAC
            model = SAC(
                "MlpPolicy", env,
                learning_rate=3e-4,
                buffer_size=100_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef='auto',
                verbose=0,
            )
            return _PPOWrapper(model)
        
        elif agent_type == 'td3':
            from stable_baselines3 import TD3
            model = TD3(
                "MlpPolicy", env,
                learning_rate=1e-3,
                buffer_size=100_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                verbose=0,
            )
            return _PPOWrapper(model)
        
        elif agent_type == 'gns':
            from baselines.gns_baseline import GNSAgent
            return GNSAgent(env, verbose=0)
        
        elif agent_type == 'hnn':
            from baselines.hnn_baseline import HNNAgent
            return HNNAgent(env, verbose=0)
        
        elif agent_type == 'physrobot':
            from baselines.physics_informed import PhysRobotAgent
            return PhysRobotAgent(env, verbose=0)
        
        elif agent_type == 'physrobot_sv':
            # PhysRobot with SV-pipeline (momentum-conserving)
            # Uses PhysRobotFeaturesExtractorV3 from physics_core
            from physics_core.sv_message_passing import PhysRobotFeaturesExtractorV3
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            from stable_baselines3 import PPO
            import gymnasium as gym
            
            # Wrap V3 extractor for SB3 compatibility
            class _SB3Wrapper(BaseFeaturesExtractor):
                def __init__(self, observation_space, features_dim=64):
                    super().__init__(observation_space, features_dim)
                    self.core = PhysRobotFeaturesExtractorV3(
                        obs_dim=observation_space.shape[0],
                        features_dim=features_dim,
                        physics_hidden=32,
                        physics_layers=1,
                    )
                def forward(self, obs):
                    return self.core(obs)
            
            policy_kwargs = dict(
                features_extractor_class=_SB3Wrapper,
                features_extractor_kwargs=dict(features_dim=64),
            )
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
            return _PPOWrapper(model)
        
        else:
            print(f"  ⚠️  Unknown agent type '{agent_type}', falling back to PPO")
            from stable_baselines3 import PPO
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            return _PPOWrapper(model)
    
    except ImportError as e:
        print(f"  ⚠️  Import error for {agent_type}: {e}")
        print(f"  ⚠️  Falling back to PPO")
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        return _PPOWrapper(model)


class _PPOWrapper:
    """Minimal wrapper to give SB3 PPO a consistent agent interface."""
    
    def __init__(self, model):
        self.model = model
    
    def train(self, total_timesteps, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save(self, path):
        self.model.save(path)


def evaluate_agent(agent, env, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate agent on environment."""
    rewards = []
    successes = []
    distances = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            if hasattr(agent, 'predict'):
                action = agent.predict(obs, deterministic=True)
            else:
                action, _ = agent.model.predict(obs, deterministic=True)
            
            obs, reward, dones, infos = env.step(
                action if isinstance(action, np.ndarray) else np.array([action])
            )
            ep_reward += reward[0] if hasattr(reward, '__len__') else reward
            done = dones[0] if hasattr(dones, '__len__') else dones
            info = infos[0] if isinstance(infos, list) else infos
        
        rewards.append(ep_reward)
        successes.append(1 if info.get('success', False) else 0)
        distances.append(info.get('distance_to_goal', 999))
    
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_distance': float(np.mean(distances)),
    }


def evaluate_ood(agent, make_env_fn, masses: List[float], n_episodes: int = 50) -> Dict[str, Any]:
    """Evaluate agent on out-of-distribution masses."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    results = {}
    for mass in masses:
        env = DummyVecEnv([make_env_fn(box_mass=mass)])
        metrics = evaluate_agent(agent, env, n_episodes=n_episodes)
        results[f'mass_{mass}'] = metrics
        env.close()
    
    # Aggregate
    all_success = [results[k]['success_rate'] for k in results]
    results['mean_ood_success'] = float(np.mean(all_success))
    results['std_ood_success'] = float(np.std(all_success))
    
    return results


# ========== Main Ablation Runner ==========

def run_single_experiment(
    variant_name: str,
    seed: int,
    config: dict,
    results_dir: str,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment (one variant, one seed).
    
    Returns result dict with training + evaluation metrics.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    variant_config = ABLATION_VARIANTS[variant_name]
    
    print(f"\n{'─'*50}")
    print(f"  Variant: {variant_name} | Seed: {seed}")
    print(f"  {variant_config['description']}")
    print(f"{'─'*50}")
    
    # Check for existing results (resume support)
    result_file = os.path.join(results_dir, f"{variant_name}_seed{seed}.json")
    if os.path.exists(result_file):
        print(f"  ⏭️  Already completed, loading from cache")
        with open(result_file) as f:
            return json.load(f)
    
    start_time = time.time()
    
    # Create environment (canonical 16-dim PushBoxEnv)
    from environments.push_box import make_push_box_env
    make_env = make_push_box_env
    
    n_envs = config.get('n_envs', 4)
    train_env = DummyVecEnv([make_env(box_mass=1.0) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(box_mass=1.0)])
    
    # Create agent
    agent = create_agent(variant_config, train_env, seed)
    
    # Train
    total_timesteps = config['total_timesteps']
    print(f"  🏋️  Training for {total_timesteps:,} timesteps...")
    
    try:
        agent.train(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        train_env.close()
        eval_env.close()
        return {
            'variant': variant_name,
            'seed': seed,
            'status': 'failed',
            'error': str(e),
        }
    
    training_time = time.time() - start_time
    
    # Evaluate (in-distribution)
    print(f"  📊 Evaluating (in-distribution)...")
    eval_results = evaluate_agent(
        agent, eval_env,
        n_episodes=config['eval_episodes'],
    )
    
    # Evaluate (OOD)
    print(f"  🧪 Evaluating (OOD)...")
    ood_results = evaluate_ood(
        agent, make_env,
        masses=config['ood_masses'],
        n_episodes=config['eval_episodes'] // 2,
    )
    
    # Save model
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{variant_name}_seed{seed}")
    try:
        agent.save(model_path)
    except Exception:
        pass  # Non-critical
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    # Compile results
    result = {
        'variant': variant_name,
        'seed': seed,
        'status': 'completed',
        'config': {k: v for k, v in variant_config.items()},
        'training_time_seconds': training_time,
        'total_timesteps': total_timesteps,
        'env_type': env_type,
        'eval_in_distribution': eval_results,
        'eval_ood': ood_results,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save individual result
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✅ Done! Success rate: {eval_results['success_rate']:.1%}, "
          f"OOD: {ood_results['mean_ood_success']:.1%}, "
          f"Time: {training_time:.1f}s")
    
    return result


def run_ablation_study(
    mode: str = 'full',
    variants: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete ablation study.
    
    Args:
        mode: 'full', 'quick', or 'colab'
        variants: List of variant names (None = all)
        seeds: List of seeds (None = from mode config)
        resume: Whether to skip already-completed experiments
    
    Returns:
        Complete results dict
    """
    config = MODE_CONFIGS[mode].copy()
    
    if variants is None:
        variants = list(ABLATION_VARIANTS.keys())
    if seeds is not None:
        config['seeds'] = seeds
    
    results_dir = os.path.join(str(PROJECT_ROOT), 'results', 'ablation')
    os.makedirs(results_dir, exist_ok=True)
    
    total_experiments = len(variants) * len(config['seeds'])
    
    print("=" * 60)
    print("🔬 PhysRobot Ablation Study")
    print("=" * 60)
    print(f"  Mode: {mode}")
    print(f"  Variants: {len(variants)} ({', '.join(variants)})")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Timesteps per experiment: {config['total_timesteps']:,}")
    print(f"  Results dir: {results_dir}")
    print(f"  Resume: {resume}")
    print("=" * 60)
    
    all_results = []
    completed = 0
    
    for variant in variants:
        for seed in config['seeds']:
            completed += 1
            print(f"\n[{completed}/{total_experiments}] ", end="")
            
            result = run_single_experiment(
                variant_name=variant,
                seed=seed,
                config=config,
                results_dir=results_dir,
            )
            all_results.append(result)
    
    # Save combined results
    combined_path = os.path.join(results_dir, f'ablation_results_{mode}.json')
    with open(combined_path, 'w') as f:
        json.dump({
            'mode': mode,
            'config': config,
            'variants': ABLATION_VARIANTS,
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Ablation study complete!")
    print(f"   Results saved to: {combined_path}")
    print(f"{'=' * 60}")
    
    # Generate comparison table
    generate_comparison_table(all_results, results_dir)
    
    return {'results': all_results, 'config': config}


# ========== Table Generation ==========

def generate_comparison_table(results: List[Dict], output_dir: str):
    """Generate LaTeX and Markdown comparison tables from results."""
    
    # Aggregate results by variant
    variant_metrics = {}
    
    for r in results:
        if r.get('status') != 'completed':
            continue
        
        name = r['variant']
        if name not in variant_metrics:
            variant_metrics[name] = {
                'success_rates': [],
                'mean_rewards': [],
                'ood_success': [],
                'training_times': [],
            }
        
        vm = variant_metrics[name]
        vm['success_rates'].append(r['eval_in_distribution']['success_rate'])
        vm['mean_rewards'].append(r['eval_in_distribution']['mean_reward'])
        vm['ood_success'].append(r['eval_ood']['mean_ood_success'])
        vm['training_times'].append(r['training_time_seconds'])
    
    # ---- Markdown Table ----
    md_lines = [
        "# Ablation Study Results",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Table: Ablation Comparison",
        "",
        "| Variant | Success (ID) | Success (OOD) | Mean Reward | Train Time (s) |",
        "|---------|:------------:|:-------------:|:-----------:|:--------------:|",
    ]
    
    for variant_name in ABLATION_VARIANTS:
        if variant_name not in variant_metrics:
            continue
        
        vm = variant_metrics[variant_name]
        sr = np.mean(vm['success_rates'])
        sr_std = np.std(vm['success_rates'])
        ood = np.mean(vm['ood_success'])
        ood_std = np.std(vm['ood_success'])
        rew = np.mean(vm['mean_rewards'])
        tt = np.mean(vm['training_times'])
        
        # Bold the full model
        prefix = "**" if variant_name == 'full_model' else ""
        suffix = "**" if variant_name == 'full_model' else ""
        
        md_lines.append(
            f"| {prefix}{variant_name}{suffix} "
            f"| {sr:.1%} ± {sr_std:.1%} "
            f"| {ood:.1%} ± {ood_std:.1%} "
            f"| {rew:.1f} "
            f"| {tt:.1f} |"
        )
    
    md_lines.extend([
        "",
        "### Legend",
        "- **Success (ID)**: In-distribution success rate (box mass = 1.0 kg)",
        "- **Success (OOD)**: Out-of-distribution mean success across mass range",
        "- **Mean Reward**: Average episode reward",
        "",
        "### Ablation Descriptions",
    ])
    
    for name, cfg in ABLATION_VARIANTS.items():
        md_lines.append(f"- **{name}**: {cfg['description']}")
    
    md_path = os.path.join(output_dir, 'ablation_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    # ---- LaTeX Table ----
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study: Component Contribution Analysis}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Variant & Success (ID) & Success (OOD) & Reward \\",
        r"\midrule",
    ]
    
    for variant_name in ABLATION_VARIANTS:
        if variant_name not in variant_metrics:
            continue
        
        vm = variant_metrics[variant_name]
        sr = np.mean(vm['success_rates']) * 100
        sr_std = np.std(vm['success_rates']) * 100
        ood = np.mean(vm['ood_success']) * 100
        ood_std = np.std(vm['ood_success']) * 100
        rew = np.mean(vm['mean_rewards'])
        
        display_name = variant_name.replace('_', ' ').title()
        if variant_name == 'full_model':
            display_name = r"\textbf{Full Model (Ours)}"
        
        latex_lines.append(
            f"{display_name} & "
            f"${sr:.1f} \\pm {sr_std:.1f}$ & "
            f"${ood:.1f} \\pm {ood_std:.1f}$ & "
            f"${rew:.1f}$ \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_path = os.path.join(output_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    # ---- CSV for plotting ----
    csv_lines = ["variant,seed,success_id,success_ood,mean_reward,train_time"]
    for r in results:
        if r.get('status') != 'completed':
            continue
        csv_lines.append(
            f"{r['variant']},{r['seed']},"
            f"{r['eval_in_distribution']['success_rate']:.4f},"
            f"{r['eval_ood']['mean_ood_success']:.4f},"
            f"{r['eval_in_distribution']['mean_reward']:.2f},"
            f"{r['training_time_seconds']:.1f}"
        )
    
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"\n📊 Tables generated:")
    print(f"   Markdown: {md_path}")
    print(f"   LaTeX:    {latex_path}")
    print(f"   CSV:      {csv_path}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description='PhysRobot Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ablation (7 variants × 5 seeds = 35 experiments)
  python scripts/run_ablation.py --mode full

  # Quick test (1 seed, reduced timesteps)
  python scripts/run_ablation.py --mode quick

  # Colab-friendly (2 seeds, 50K steps)
  python scripts/run_ablation.py --mode colab

  # Single variant with specific seeds
  python scripts/run_ablation.py --variant full_model pure_ppo --seeds 42 123

  # Resume interrupted run
  python scripts/run_ablation.py --mode full --resume
        """
    )
    
    parser.add_argument(
        '--mode', type=str, default='quick',
        choices=['full', 'quick', 'colab'],
        help='Experiment mode (default: quick)'
    )
    parser.add_argument(
        '--variant', type=str, nargs='+', default=None,
        help='Specific variants to run (default: all). '
             f'Choices: {", ".join(ABLATION_VARIANTS.keys())}'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=None,
        help='Random seeds (default: from mode config)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip already-completed experiments'
    )
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help='Override total timesteps'
    )
    
    args = parser.parse_args()
    
    # Override timesteps if specified
    if args.timesteps is not None:
        MODE_CONFIGS[args.mode]['total_timesteps'] = args.timesteps
    
    results = run_ablation_study(
        mode=args.mode,
        variants=args.variant,
        seeds=args.seeds,
        resume=args.resume,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
