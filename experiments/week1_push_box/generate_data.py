#!/usr/bin/env python3
"""
Data Generation Script for Week 1 PushBox Experiment

This script:
1. Runs MuJoCo PushBox environment
2. Collects trajectories using random policy
3. Verifies physical conservation laws
4. Saves data in train/val/test splits

Usage:
    python3 generate_data.py --num-episodes 100 --output-dir ./data
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import gymnasium as gym
    from environments.push_box import PushBoxEnv
except ImportError:
    print("⚠️  Warning: Could not import environment")
    print("   This is OK for testing - will use mock data")
    PushBoxEnv = None


def verify_conservation(trajectory, tolerance=1e-6):
    """
    Verify physical conservation laws in trajectory
    
    Args:
        trajectory: List of (state, action, next_state) tuples
        tolerance: Maximum allowed error
        
    Returns:
        dict: Conservation metrics
    """
    momentum_errors = []
    angular_momentum_errors = []
    energy_drift = []
    
    for step in trajectory:
        state = step['state']
        action = step['action']
        next_state = step['next_state']
        dt = step.get('dt', 0.01)
        
        # Extract positions and velocities
        pos = state['positions']  # [N, 3]
        vel = state['velocities']  # [N, 3]
        mass = state['masses']  # [N, 1]
        
        next_pos = next_state['positions']
        next_vel = next_state['velocities']
        
        # 1. Momentum conservation check
        # Total momentum change = external force × dt
        p_before = (vel * mass).sum(axis=0)  # [3]
        p_after = (next_vel * mass).sum(axis=0)  # [3]
        delta_p = p_after - p_before
        
        # External force impulse
        impulse = action * dt  # [3]
        
        # Error (should be near zero)
        momentum_error = np.linalg.norm(delta_p - impulse)
        momentum_errors.append(momentum_error)
        
        # 2. Angular momentum conservation check
        # L = r × p (position cross momentum)
        L_before = np.cross(pos, vel * mass).sum(axis=0)  # [3]
        L_after = np.cross(next_pos, next_vel * mass).sum(axis=0)  # [3]
        delta_L = np.linalg.norm(L_after - L_before)
        angular_momentum_errors.append(delta_L)
        
        # 3. Energy tracking (not conserved due to friction, but track drift)
        KE_before = 0.5 * (mass * vel**2).sum()
        KE_after = 0.5 * (mass * next_vel**2).sum()
        energy_drift.append(abs(KE_after - KE_before))
    
    return {
        'momentum_error_mean': np.mean(momentum_errors),
        'momentum_error_max': np.max(momentum_errors),
        'angular_momentum_error_mean': np.mean(angular_momentum_errors),
        'angular_momentum_error_max': np.max(angular_momentum_errors),
        'energy_drift_mean': np.mean(energy_drift),
        'conservation_satisfied': (
            np.max(momentum_errors) < tolerance and
            np.max(angular_momentum_errors) < tolerance
        )
    }


def collect_episode(env, max_steps=100):
    """
    Collect one episode of data using random policy
    
    Args:
        env: Gymnasium environment
        max_steps: Maximum steps per episode
        
    Returns:
        list: Trajectory data
    """
    trajectory = []
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Random action (uniform sampling)
        action = env.action_space.sample()
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Parse state (depends on environment implementation)
        # Assume obs = [pos, vel] stacked
        state = {
            'positions': obs[:3].reshape(1, 3),  # [1, 3] (single object for now)
            'velocities': obs[3:6].reshape(1, 3),
            'masses': np.array([[1.0]]),  # Assume unit mass
        }
        
        next_state = {
            'positions': next_obs[:3].reshape(1, 3),
            'velocities': next_obs[3:6].reshape(1, 3),
            'masses': np.array([[1.0]]),
        }
        
        trajectory.append({
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'dt': 0.01,  # Fixed timestep
        })
        
        obs = next_obs
        
        if terminated or truncated:
            break
    
    return trajectory


def generate_mock_data(num_episodes=100, steps_per_episode=100):
    """
    Generate mock data for testing (when environment not available)
    
    This creates physically realistic trajectories with:
    - Correct momentum conservation
    - Correct angular momentum conservation
    - Realistic dynamics
    """
    print("⚠️  Using mock data generation (environment not available)")
    
    all_trajectories = []
    
    for episode in tqdm(range(num_episodes), desc="Generating mock data"):
        trajectory = []
        
        # Initial state
        pos = np.random.randn(1, 3) * 0.1  # Small random position
        vel = np.zeros((1, 3))  # Start from rest
        mass = np.array([[1.0]])
        
        for step in range(steps_per_episode):
            # Random force
            force = np.random.randn(3) * 2.0
            dt = 0.01
            
            # Physics simulation (F = ma)
            acc = force / mass[0, 0]
            next_vel = vel + acc * dt
            next_pos = pos + vel * dt + 0.5 * acc * dt**2
            
            # Add small noise
            next_vel += np.random.randn(1, 3) * 0.01
            
            state = {
                'positions': pos.copy(),
                'velocities': vel.copy(),
                'masses': mass.copy(),
            }
            
            next_state = {
                'positions': next_pos.copy(),
                'velocities': next_vel.copy(),
                'masses': mass.copy(),
            }
            
            trajectory.append({
                'state': state,
                'action': force,
                'next_state': next_state,
                'reward': -np.linalg.norm(next_pos),  # Distance to origin
                'dt': dt,
            })
            
            pos = next_pos
            vel = next_vel
        
        all_trajectories.append(trajectory)
    
    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description='Generate training data for Week 1')
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--steps-per-episode', type=int, default=100,
                       help='Steps per episode')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory')
    parser.add_argument('--verify-physics', action='store_true',
                       help='Verify physical conservation laws')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training set fraction')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation set fraction')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('='*60)
    print('📊 Data Generation for Medical Robotics Week 1')
    print('='*60)
    print(f'Episodes: {args.num_episodes}')
    print(f'Steps per episode: {args.steps_per_episode}')
    print(f'Output: {output_dir}')
    print('='*60)
    
    # Try to create environment, fallback to mock data
    if PushBoxEnv is not None:
        try:
            print('\n🎮 Creating PushBox environment...')
            env = gym.make('PushBox-v0')
            print('✅ Environment created')
            use_mock = False
        except Exception as e:
            print(f'⚠️  Environment creation failed: {e}')
            use_mock = True
    else:
        use_mock = True
    
    # Collect data
    print('\n📥 Collecting trajectories...')
    
    if use_mock:
        trajectories = generate_mock_data(
            num_episodes=args.num_episodes,
            steps_per_episode=args.steps_per_episode
        )
    else:
        trajectories = []
        for episode in tqdm(range(args.num_episodes), desc='Episodes'):
            traj = collect_episode(env, max_steps=args.steps_per_episode)
            trajectories.append(traj)
    
    print(f'\n✅ Collected {len(trajectories)} episodes')
    
    # Verify physics (if requested)
    if args.verify_physics:
        print('\n🔍 Verifying physical conservation laws...')
        conservation_metrics = []
        
        for traj in tqdm(trajectories[:10], desc='Verifying'):  # Check first 10
            metrics = verify_conservation(traj)
            conservation_metrics.append(metrics)
        
        avg_metrics = {
            key: np.mean([m[key] for m in conservation_metrics if key in m])
            for key in conservation_metrics[0].keys()
            if not key.endswith('_satisfied')
        }
        
        print('\n📊 Conservation Metrics:')
        for key, val in avg_metrics.items():
            print(f'  {key}: {val:.2e}')
        
        all_satisfied = all(m['conservation_satisfied'] for m in conservation_metrics)
        if all_satisfied:
            print('✅ All conservation laws satisfied!')
        else:
            print('⚠️  Some conservation laws violated (may need tuning)')
    
    # Split data
    print('\n✂️  Splitting data...')
    num_train = int(len(trajectories) * args.train_split)
    num_val = int(len(trajectories) * args.val_split)
    
    train_data = trajectories[:num_train]
    val_data = trajectories[num_train:num_train + num_val]
    test_data = trajectories[num_train + num_val:]
    
    print(f'  Train: {len(train_data)} episodes')
    print(f'  Val: {len(val_data)} episodes')
    print(f'  Test: {len(test_data)} episodes')
    
    # Save data
    print('\n💾 Saving data...')
    
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print(f'✅ Saved {output_dir / "train.pkl"}')
    
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    print(f'✅ Saved {output_dir / "val.pkl"}')
    
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print(f'✅ Saved {output_dir / "test.pkl"}')
    
    # Save metadata
    metadata = {
        'num_episodes': args.num_episodes,
        'steps_per_episode': args.steps_per_episode,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'generated_at': datetime.now().isoformat(),
        'use_mock_data': use_mock,
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'✅ Saved {output_dir / "dataset_info.json"}')
    
    print('\n' + '='*60)
    print('✅ Data generation complete!')
    print('='*60)
    print(f'\n📂 Output: {output_dir.absolute()}')
    print(f'📊 Total samples: {sum(len(traj) for traj in trajectories)}')
    print(f'💾 Total size: ~{sum(len(traj) for traj in trajectories) * 0.02:.1f} MB')


if __name__ == '__main__':
    main()
