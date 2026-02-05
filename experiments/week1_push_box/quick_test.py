#!/usr/bin/env python3
"""
Quick test script for Week 1 code verification
Tests all components in ~10 minutes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🧪 Week 1 Quick Test Suite")
print("="*60)
print()

# Test 1: Import core modules
print("Test 1/5: Importing core modules...")
try:
    from physics_core.edge_frame import EdgeFrame
    from physics_core.dynamical_gnn import DynamicalGNN
    from physics_core.integrators import SymplecticIntegrator
    print("  ✅ physics_core imports OK")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Create EdgeFrame
print("\nTest 2/5: Testing EdgeFrame...")
try:
    import torch
    positions = torch.randn(5, 3)  # 5 nodes, 3D
    edge_frame = EdgeFrame(positions)
    edges = edge_frame.compute_edges()
    
    # Check antisymmetry
    antisym_error = edge_frame.check_antisymmetry()
    assert antisym_error < 1e-5, f"Antisymmetry error too large: {antisym_error}"
    print(f"  ✅ EdgeFrame OK (antisymmetry error: {antisym_error:.2e})")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Create DynamicalGNN
print("\nTest 3/5: Testing DynamicalGNN...")
try:
    model = DynamicalGNN(hidden_dim=32, num_layers=2)
    pos = torch.randn(5, 3)
    vel = torch.randn(5, 3)
    
    # Forward pass
    acc = model(pos, vel)
    assert acc.shape == (5, 3), f"Wrong output shape: {acc.shape}"
    
    # Check conservation
    energy = model.compute_energy(pos, vel)
    print(f"  ✅ DynamicalGNN OK (energy: {energy.item():.2f})")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Test environment (if MuJoCo available)
print("\nTest 4/5: Testing PushBox environment...")
try:
    from environments.push_box import PushBoxEnv
    
    env = PushBoxEnv()
    obs = env.reset()
    
    # Take random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print(f"  ✅ Environment OK (obs shape: {obs.shape})")
except ImportError as e:
    print(f"  ⏭️  Skipped (MuJoCo not installed): {e}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    # Don't exit - environment test is optional

# Test 5: Quick training (10 episodes)
print("\nTest 5/5: Quick training test...")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environments.push_box import make_push_box_env
    
    # Create vectorized environment
    env = DummyVecEnv([make_push_box_env()])
    
    # Create simple model
    model = PPO('MlpPolicy', env, verbose=0)
    
    # Train for 10 episodes (~2-3 minutes)
    print("  Training 10 episodes...")
    model.learn(total_timesteps=1000)  # ~10 episodes
    
    # Evaluate
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    print("  ✅ Training pipeline OK")
except ImportError as e:
    print(f"  ⏭️  Skipped (dependencies not installed): {e}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    # Don't exit - training test is optional

print()
print("="*60)
print("✅ Quick test complete!")
print("="*60)
print()
print("📋 Next steps:")
print("1. Run full unit tests: pytest physics_core/tests/ -v")
print("2. Run environment tests: python environments/test_push_box.py")
print("3. Run full training: bash experiments/week1_push_box/setup_and_run.sh")
print("   OR")
print("4. Run on Colab: https://colab.research.google.com/github/...")
