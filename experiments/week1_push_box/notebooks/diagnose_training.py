"""
诊断 Colab 训练异常完成问题

问题：200,000 步应该需要 6-8 小时，但只用了 7.8 分钟
可能原因：
1. 环境每步都返回 done=True（提前终止）
2. StableBaselines3 配置问题
3. 数据收集问题
"""

import sys
import os
sys.path.insert(0, '/content/medical-robotics-sim')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.push_box import make_push_box_env

print("="*60)
print("🔍 Training Diagnostics")
print("="*60)

# Test 1: Environment sanity check
print("\n📊 Test 1: Environment Sanity Check")
print("-" * 60)

env = DummyVecEnv([make_push_box_env])
obs = env.reset()

episodes_completed = 0
total_steps = 0
max_steps = 1000

print(f"Running {max_steps} steps...")

for step in range(max_steps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_steps += 1
    
    if done[0]:
        episodes_completed += 1
        obs = env.reset()
        
    if step % 100 == 0:
        print(f"  Step {step}: {episodes_completed} episodes completed")

print(f"\n📈 Results:")
print(f"  Total steps: {total_steps}")
print(f"  Episodes completed: {episodes_completed}")
print(f"  Average episode length: {total_steps / max(episodes_completed, 1):.1f} steps")
print(f"  Expected: ~500 steps per episode")

if episodes_completed > 900:
    print("\n❌ PROBLEM: Almost every step ends the episode!")
    print("   This would cause training to finish in minutes.")
elif episodes_completed < 5:
    print("\n❌ PROBLEM: Episodes never terminate!")
    print("   This would cause training to stall.")
else:
    print("\n✅ Episode length looks normal")

# Test 2: Check trained model
print("\n📊 Test 2: Trained Model Analysis")
print("-" * 60)

model_path = "./models/ppo/ppo_baseline"
if os.path.exists(f"{model_path}.zip"):
    model = PPO.load(model_path)
    
    # Check training stats
    if hasattr(model, 'num_timesteps'):
        print(f"  Model timesteps: {model.num_timesteps}")
        print(f"  Expected: 200,000")
        
        if model.num_timesteps < 200000:
            print(f"\n❌ PROBLEM: Model trained for {model.num_timesteps} steps instead of 200,000")
        else:
            print("\n✅ Model completed full training")
    
    # Test inference
    print("\n  Testing model inference...")
    test_env = DummyVecEnv([make_push_box_env])
    obs = test_env.reset()
    
    test_rewards = []
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_rewards.append(reward[0])
        
        if done[0]:
            obs = test_env.reset()
    
    print(f"  Average reward: {np.mean(test_rewards):.2f}")
    print(f"  Reward std: {np.std(test_rewards):.2f}")
    
    if np.mean(test_rewards) < -10:
        print("\n⚠️  WARNING: Very low rewards, model may not have learned")
else:
    print(f"❌ Model not found at {model_path}")

# Test 3: Check logs
print("\n📊 Test 3: Training Logs")
print("-" * 60)

log_dir = "./logs/ppo_baseline"
if os.path.exists(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv') or f.endswith('.txt')]
    print(f"  Found {len(log_files)} log files")
    
    for log_file in log_files[:3]:
        print(f"    - {log_file}")
else:
    print("  No logs found")

print("\n" + "="*60)
print("🔍 Diagnostic Complete")
print("="*60)
