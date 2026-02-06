"""
快速测试环境奖励是否正常
"""
import sys
sys.path.insert(0, '.')

from environments.push_box import make_push_box_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

print("="*60)
print("🧪 Testing Environment Rewards")
print("="*60)

env = DummyVecEnv([make_push_box_env])
obs = env.reset()

print("\n📊 Running 10 episodes...")

all_rewards = []
all_lengths = []

for ep in range(10):
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        episode_length += 1
        
        if done[0]:
            break
    
    all_rewards.append(episode_reward)
    all_lengths.append(episode_length)
    
    print(f"  Episode {ep+1}: Length={episode_length}, Reward={episode_reward:.2f}")

print("\n📈 Summary:")
print(f"  Average Length: {np.mean(all_lengths):.0f} steps")
print(f"  Average Reward: {np.mean(all_rewards):.2f}")
print(f"  Reward Std: {np.std(all_rewards):.2f}")
print(f"  Min Reward: {np.min(all_rewards):.2f}")
print(f"  Max Reward: {np.max(all_rewards):.2f}")

if np.abs(np.mean(all_rewards)) < 0.1:
    print("\n⚠️  WARNING: Average reward very close to 0!")
    print("   This might indicate a reward function issue.")
elif np.mean(all_rewards) < -100:
    print("\n⚠️  WARNING: Very negative rewards!")
    print("   Agent might not be able to learn.")
else:
    print("\n✅ Rewards look reasonable")

print("="*60)
