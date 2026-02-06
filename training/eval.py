"""
Evaluation Script for Week 1 Experiments
Tests:
1. Out-of-Distribution (OOD) Generalization (different box masses)
2. Physics Conservation Laws Validation
"""

import numpy as np
import json
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

from environments.push_box import make_push_box_env  # Canonical 16-dim env
# Legacy alias for backward compatibility
make_push_box_env_16 = make_push_box_env
make_push_box_env_10 = make_push_box_env  # NOTE: 10-dim version retired
from baselines.ppo_baseline import PurePPOAgent
from baselines.gns_baseline import GNSAgent
from baselines.physics_informed import PhysRobotAgent


def _detect_obs_dim(agent):
    """Detect the observation space dimension a loaded model expects."""
    try:
        obs_shape = agent.model.observation_space.shape
        return obs_shape[0] if obs_shape else 16
    except Exception:
        return 16  # default to 16-dim


def _make_env_for_obs_dim(obs_dim):
    """Return the correct make_push_box_env based on obs dimension."""
    if obs_dim == 10:
        return make_push_box_env_10
    else:
        return make_push_box_env_16


def evaluate_ood_generalization(
    agent, 
    agent_type,
    mass_range=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    n_episodes_per_mass=100,
    make_env_fn=None
):
    """
    Test agent on different box masses (OOD scenario)
    
    Args:
        agent: Trained agent
        agent_type: Type of agent ("PPO", "GNS", "PhysRobot")
        mass_range: List of box masses to test
        n_episodes_per_mass: Episodes per mass value
    
    Returns:
        dict with OOD results
    """
    # Auto-detect obs dimension and pick correct env factory
    if make_env_fn is None:
        obs_dim = _detect_obs_dim(agent)
        make_env_fn = _make_env_for_obs_dim(obs_dim)
        print(f"   Auto-detected obs_dim={obs_dim}, using matching env")
    
    print(f"\n🧪 Testing {agent_type} OOD Generalization")
    print(f"   Mass range: {mass_range}")
    print(f"   Episodes per mass: {n_episodes_per_mass}")
    
    results = []
    
    for mass in mass_range:
        print(f"\n  Testing mass = {mass:.2f} kg...")
        
        # Create environment with specific mass (matching model's obs space)
        env = DummyVecEnv([make_env_fn(box_mass=mass)])
        
        success_count = 0
        rewards = []
        distances = []
        
        for episode in range(n_episodes_per_mass):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                episode_reward += reward[0]
                done = dones[0]
                info = infos[0]
            
            rewards.append(episode_reward)
            distances.append(info.get('distance_to_goal', 999))
            
            if info.get('success', False):
                success_count += 1
        
        success_rate = success_count / n_episodes_per_mass
        
        results.append({
            'mass': mass,
            'success_rate': success_rate,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        })
        
        print(f"    Success rate: {success_rate:.2%}")
        print(f"    Mean distance: {np.mean(distances):.3f} m")
        
        env.close()
    
    return {
        'agent_type': agent_type,
        'mass_range': mass_range,
        'results': results
    }


def validate_conservation_laws(agent, agent_type, n_episodes=50):
    """
    Validate momentum and energy conservation
    
    Args:
        agent: Trained agent
        agent_type: Type of agent
        n_episodes: Number of episodes to test
    
    Returns:
        dict with conservation metrics
    """
    print(f"\n⚖️  Validating Conservation Laws for {agent_type}")
    print(f"   Episodes: {n_episodes}")
    
    obs_dim = _detect_obs_dim(agent)
    _make_env = _make_env_for_obs_dim(obs_dim)
    env = DummyVecEnv([_make_env(box_mass=1.0)])
    
    momentum_errors = []
    energy_errors = []
    
    for episode in range(n_episodes):
        trajectory = collect_trajectory(agent, env)
        
        # Calculate conservation errors
        momentum_error = calculate_momentum_drift(trajectory)
        energy_error = calculate_energy_drift(trajectory)
        
        momentum_errors.append(momentum_error)
        energy_errors.append(energy_error)
    
    env.close()
    
    results = {
        'agent_type': agent_type,
        'momentum_error_mean': np.mean(momentum_errors),
        'momentum_error_std': np.std(momentum_errors),
        'momentum_error_max': np.max(momentum_errors),
        'energy_error_mean': np.mean(energy_errors),
        'energy_error_std': np.std(energy_errors),
        'energy_error_max': np.max(energy_errors),
        'n_episodes': n_episodes
    }
    
    print(f"\n  📊 Conservation Metrics:")
    print(f"     Momentum error: {results['momentum_error_mean']:.4f} ± {results['momentum_error_std']:.4f}")
    print(f"     Energy error:   {results['energy_error_mean']:.4f} ± {results['energy_error_std']:.4f}")
    
    return results


def collect_trajectory(agent, env):
    """
    Collect full trajectory with physics state
    
    Returns:
        dict with positions, velocities, forces over time
    """
    obs = env.reset()
    
    trajectory = {
        'positions': [],
        'velocities': [],
        'actions': [],
        'timesteps': 0
    }
    
    done = False
    while not done:
        # Get action
        action = agent.predict(obs, deterministic=True)
        
        # Record state (from observation)
        box_pos = obs[0][7:10].copy()
        box_vel = obs[0][10:13].copy()
        
        trajectory['positions'].append(box_pos)
        trajectory['velocities'].append(box_vel)
        trajectory['actions'].append(action[0])
        
        # Step (VecEnv API: returns obs, reward, dones, infos)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        trajectory['timesteps'] += 1
    
    # Convert to numpy arrays
    trajectory['positions'] = np.array(trajectory['positions'])
    trajectory['velocities'] = np.array(trajectory['velocities'])
    trajectory['actions'] = np.array(trajectory['actions'])
    
    return trajectory


def calculate_momentum_drift(trajectory):
    """
    Calculate momentum conservation error
    
    In an isolated system, total momentum should be conserved.
    We measure the drift as the standard deviation of total momentum.
    
    Args:
        trajectory: Trajectory dict
    
    Returns:
        Momentum drift (normalized)
    """
    # Box momentum over time (mass * velocity)
    # Assuming unit mass for simplicity
    momenta = trajectory['velocities']  # [T, 3]
    
    # Total momentum magnitude at each timestep
    momentum_magnitudes = np.linalg.norm(momenta, axis=1)
    
    # Drift: std / mean
    if np.mean(momentum_magnitudes) > 1e-6:
        drift = np.std(momentum_magnitudes) / (np.mean(momentum_magnitudes) + 1e-6)
    else:
        drift = np.std(momentum_magnitudes)
    
    return drift


def calculate_energy_drift(trajectory):
    """
    Calculate energy conservation error
    
    Total energy = kinetic + potential
    E = 0.5 * m * v^2 + m * g * h
    
    Args:
        trajectory: Trajectory dict
    
    Returns:
        Energy drift (normalized)
    """
    m = 1.0  # Box mass
    g = 9.81  # Gravity
    
    positions = trajectory['positions']
    velocities = trajectory['velocities']
    
    # Kinetic energy
    v_squared = np.sum(velocities**2, axis=1)
    KE = 0.5 * m * v_squared
    
    # Potential energy (height only)
    h = positions[:, 2]  # z coordinate
    PE = m * g * h
    
    # Total energy
    total_energy = KE + PE
    
    # Drift: std / mean
    if np.mean(total_energy) > 1e-6:
        drift = np.std(total_energy) / (np.mean(total_energy) + 1e-6)
    else:
        drift = np.std(total_energy)
    
    return drift


def _load_agent(AgentClass, model_path, name):
    """Load an agent, auto-detecting obs dimension from saved model."""
    from stable_baselines3 import PPO
    # Peek at the saved model's observation space
    try:
        tmp_model = PPO.load(model_path)
        obs_dim = tmp_model.observation_space.shape[0]
        del tmp_model
    except Exception:
        obs_dim = 16  # default
    
    make_env = _make_env_for_obs_dim(obs_dim)
    env = DummyVecEnv([make_env(box_mass=1.0)])
    agent = AgentClass(env)
    agent.load(model_path)
    print(f"   ✅ Loaded {name} (obs_dim={obs_dim})")
    env.close()
    return agent


def load_trained_models(models_dir="./models"):
    """
    Load all trained models, auto-detecting observation space.
    
    Returns:
        dict with loaded agents
    """
    print(f"\n📂 Loading trained models from {models_dir}...")
    
    agents = {}
    
    # Load PPO
    ppo_path = os.path.join(models_dir, "pure_ppo_final.zip")
    if os.path.exists(ppo_path):
        print(f"   Loading Pure PPO from {ppo_path}")
        agents['Pure PPO'] = _load_agent(PurePPOAgent, ppo_path, "Pure PPO")
    else:
        print(f"   ⚠️  Pure PPO model not found")
    
    # Load GNS
    gns_path = os.path.join(models_dir, "gns_final.zip")
    if os.path.exists(gns_path):
        print(f"   Loading GNS from {gns_path}")
        agents['GNS'] = _load_agent(GNSAgent, gns_path, "GNS")
    else:
        print(f"   ⚠️  GNS model not found")
    
    # Load PhysRobot
    physrobot_path = os.path.join(models_dir, "physrobot_final.zip")
    if os.path.exists(physrobot_path):
        print(f"   Loading PhysRobot from {physrobot_path}")
        agents['PhysRobot'] = _load_agent(PhysRobotAgent, physrobot_path, "PhysRobot")
    else:
        print(f"   ⚠️  PhysRobot model not found")
    
    print(f"   ✅ Loaded {len(agents)} models")
    
    return agents


def run_ood_test(models_dir="./models", output_dir="./data"):
    """
    Run OOD generalization test on all models
    """
    print("="*60)
    print("🧪 OOD Generalization Test")
    print("="*60)
    
    agents = load_trained_models(models_dir)
    
    if not agents:
        print("❌ No trained models found!")
        return
    
    all_results = {}
    
    mass_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for name, agent in agents.items():
        results = evaluate_ood_generalization(
            agent=agent,
            agent_type=name,
            mass_range=mass_range,
            n_episodes_per_mass=100
        )
        all_results[name] = results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ood_generalization.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV for easy plotting
    csv_data = []
    for name, data in all_results.items():
        for result in data['results']:
            csv_data.append({
                'method': name,
                'mass': result['mass'],
                'success_rate': result['success_rate'],
                'mean_reward': result['mean_reward']
            })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "ood_generalization.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ OOD results saved to:")
    print(f"   {output_path}")
    print(f"   {csv_path}")


def run_conservation_test(models_dir="./models", output_dir="./data"):
    """
    Run conservation laws validation on all models
    """
    print("="*60)
    print("⚖️  Conservation Laws Validation")
    print("="*60)
    
    agents = load_trained_models(models_dir)
    
    if not agents:
        print("❌ No trained models found!")
        return
    
    all_results = {}
    
    for name, agent in agents.items():
        results = validate_conservation_laws(
            agent=agent,
            agent_type=name,
            n_episodes=50
        )
        all_results[name] = results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "conservation_validation.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Conservation results saved to: {output_path}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("📊 Conservation Comparison")
    print(f"{'='*60}\n")
    print(f"{'Method':<15} {'Momentum Error':<20} {'Energy Error':<20}")
    print("-" * 60)
    
    for name, results in all_results.items():
        mom_err = f"{results['momentum_error_mean']:.4f} ± {results['momentum_error_std']:.4f}"
        eng_err = f"{results['energy_error_mean']:.4f} ± {results['energy_error_std']:.4f}"
        print(f"{name:<15} {mom_err:<20} {eng_err:<20}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Week 1 Evaluation")
    parser.add_argument('--ood-test', action='store_true', help="Run OOD generalization test")
    parser.add_argument('--validate-physics', action='store_true', help="Validate conservation laws")
    parser.add_argument('--masses', type=str, default="0.5,0.75,1.0,1.25,1.5,2.0", 
                        help="Comma-separated box masses for OOD test")
    parser.add_argument('--models-dir', type=str, default="./models", help="Models directory")
    parser.add_argument('--output-dir', type=str, default="./data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.ood_test:
        run_ood_test(
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
    
    if args.validate_physics:
        run_conservation_test(
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
    
    if not args.ood_test and not args.validate_physics:
        print("Please specify --ood-test or --validate-physics")


if __name__ == "__main__":
    main()
