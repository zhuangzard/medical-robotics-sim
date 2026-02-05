"""
Quick Test Script for Week 1 Setup
Validates all components before starting full training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*70)
print("🧪 Week 1 Setup Validation")
print("="*70)

# Test 1: Import environment
print("\n[1/5] Testing PushBox Environment...")
try:
    from environments.push_box_env import PushBoxEnv, make_push_box_env
    env = PushBoxEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    print("   ✅ Environment works!")
except Exception as e:
    print(f"   ❌ Environment failed: {e}")
    sys.exit(1)

# Test 2: Import Pure PPO
print("\n[2/5] Testing Pure PPO Baseline...")
try:
    from baselines.ppo_baseline import PurePPOAgent
    env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    agent = PurePPOAgent(env, verbose=0)
    obs = env.reset()
    action = agent.predict(obs)
    env.close()
    print("   ✅ Pure PPO works!")
except Exception as e:
    print(f"   ❌ Pure PPO failed: {e}")
    sys.exit(1)

# Test 3: Import GNS
print("\n[3/5] Testing GNS Baseline...")
try:
    from baselines.gns_baseline import GNSAgent
    env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    agent = GNSAgent(env, verbose=0)
    obs = env.reset()
    action = agent.predict(obs)
    env.close()
    print("   ✅ GNS Baseline works!")
except Exception as e:
    print(f"   ❌ GNS Baseline failed: {e}")
    sys.exit(1)

# Test 4: Import PhysRobot
print("\n[4/5] Testing PhysRobot...")
try:
    from baselines.physics_informed import PhysRobotAgent
    env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    agent = PhysRobotAgent(env, verbose=0)
    obs = env.reset()
    action = agent.predict(obs)
    env.close()
    print("   ✅ PhysRobot works!")
except Exception as e:
    print(f"   ❌ PhysRobot failed: {e}")
    sys.exit(1)

# Test 5: Test training for 10 steps
print("\n[5/5] Testing mini-training (10 steps)...")
try:
    from baselines.physics_informed import PhysRobotAgent
    env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    agent = PhysRobotAgent(env, verbose=0)
    agent.train(total_timesteps=10)
    env.close()
    print("   ✅ Training loop works!")
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All Tests Passed!")
print("="*70)
print("\n🚀 Ready to start full training!")
print("\nNext steps:")
print("  1. Run full training:")
print("     python training/train.py")
print("\n  2. Or start with quick test:")
print("     python training/train.py --ppo-steps 10000 --gns-steps 5000 --physrobot-steps 2000")
print()
