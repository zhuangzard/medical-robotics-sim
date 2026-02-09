#!/usr/bin/env python3
"""
Complete Training Pipeline - All Three Methods
Trains PPO, GNS, and PhysRobot sequentially
"""
import argparse
import time
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo import train_ppo


def train_all(
    ppo_steps=200000,
    gns_steps=80000,
    physrobot_steps=16000,
    n_envs=4,
    save_dir='./models',
    log_dir='./logs'
):
    """
    Train all three methods sequentially
    
    Args:
        ppo_steps: PPO training steps
        gns_steps: GNS training steps  
        physrobot_steps: PhysRobot training steps
        n_envs: Parallel environments
        save_dir: Base save directory
        log_dir: Base log directory
    """
    print('='*80)
    print('🚀 COMPLETE TRAINING PIPELINE - Medical Robotics Week 1')
    print('='*80)
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'\nTraining plan:')
    print(f'  1. PPO: {ppo_steps:,} steps (~6-8 hours)')
    print(f'  2. GNS: {gns_steps:,} steps (~1-2 hours) [PLACEHOLDER]')
    print(f'  3. PhysRobot: {physrobot_steps:,} steps (~0.5-1 hour) [PLACEHOLDER]')
    print(f'\nTotal estimated time: ~10 hours')
    print('='*80)
    
    start_time = time.time()
    
    # Step 1: Train PPO
    print('\n' + '='*80)
    print('📊 Step 1/3: Training PPO Baseline')
    print('='*80)
    
    try:
        train_ppo(
            total_timesteps=ppo_steps,
            n_envs=n_envs,
            save_path=os.path.join(save_dir, 'ppo'),
            log_dir=os.path.join(log_dir, 'ppo'),
            verbose=1
        )
        print('✅ PPO training complete!')
    except Exception as e:
        print(f'❌ PPO training failed: {e}')
        raise
    
    # Step 2: Train GNS (placeholder)
    print('\n' + '='*80)
    print('📊 Step 2/3: Training GNS Baseline')
    print('='*80)
    print('⚠️  GNS training not yet implemented')
    print('   Placeholder: Would train for {:,} steps'.format(gns_steps))
    print('✅ GNS skipped')
    
    # Step 3: Train PhysRobot (placeholder)
    print('\n' + '='*80)
    print('📊 Step 3/3: Training PhysRobot')
    print('='*80)
    print('⚠️  PhysRobot training not yet implemented')
    print('   Placeholder: Would train for {:,} steps'.format(physrobot_steps))
    print('✅ PhysRobot skipped')
    
    # Summary
    duration = time.time() - start_time
    duration_hr = duration / 3600
    
    print('\n' + '='*80)
    print('🎉 TRAINING PIPELINE COMPLETE!')
    print('='*80)
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Total duration: {duration_hr:.2f} hours ({duration/60:.1f} minutes)')
    print('\nCompleted:')
    print(f'  ✅ PPO: {ppo_steps:,} steps')
    print(f'  ⏭️  GNS: Skipped (not implemented)')
    print(f'  ⏭️  PhysRobot: Skipped (not implemented)')
    print('\nModels saved to: {}'.format(save_dir))
    print('Logs saved to: {}'.format(log_dir))
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all three methods')
    parser.add_argument('--ppo-steps', type=int, default=200000,
                       help='PPO training steps (default: 200000)')
    parser.add_argument('--gns-steps', type=int, default=80000,
                       help='GNS training steps (default: 80000)')
    parser.add_argument('--physrobot-steps', type=int, default=16000,
                       help='PhysRobot training steps (default: 16000)')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Parallel environments (default: 4)')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Base save directory (default: ./models)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Base log directory (default: ./logs)')
    
    args = parser.parse_args()
    
    train_all(
        ppo_steps=args.ppo_steps,
        gns_steps=args.gns_steps,
        physrobot_steps=args.physrobot_steps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
