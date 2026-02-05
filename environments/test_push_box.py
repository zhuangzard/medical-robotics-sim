"""
Test suite for PushBox environment

Tests:
1. Environment initialization
2. Random policy execution
3. Mass variation (OOD testing)
4. Rendering
5. Episode data collection

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.push_box import PushBoxEnv, make_push_box_env


def test_initialization():
    """Test 1: Environment initialization"""
    print("\n" + "="*60)
    print("TEST 1: Environment Initialization")
    print("="*60)
    
    try:
        env = PushBoxEnv()
        print("✓ Environment created successfully")
        
        # Check spaces
        print(f"✓ Observation space: {env.observation_space.shape}")
        print(f"✓ Action space: {env.action_space.shape}")
        
        # Check physics parameters
        print(f"✓ Box mass: {env.box_mass} kg")
        print(f"✓ Friction coefficient: {env.friction_coef}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_random_policy():
    """Test 2: Random policy execution"""
    print("\n" + "="*60)
    print("TEST 2: Random Policy (100 steps)")
    print("="*60)
    
    try:
        env = PushBoxEnv()
        obs, info = env.reset()
        
        print(f"✓ Initial observation shape: {obs.shape}")
        print(f"✓ Initial box position: {info['box_position'][:2]}")
        print(f"✓ Goal position: {info['goal_position']}")
        
        total_reward = 0
        contacts = 0
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            if info['contact']:
                contacts += 1
            
            if terminated or truncated:
                print(f"✓ Episode ended at step {step}")
                break
        
        print(f"✓ Completed {step+1} steps without errors")
        print(f"✓ Total reward: {total_reward:.3f}")
        print(f"✓ Contact events: {contacts}")
        print(f"✓ Final distance to goal: {info['distance_to_goal']:.3f} m")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Random policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mass_variation():
    """Test 3: Mass variation (OOD testing)"""
    print("\n" + "="*60)
    print("TEST 3: Mass Variation (OOD Testing)")
    print("="*60)
    
    try:
        env = PushBoxEnv()
        
        # Test different masses
        masses = [0.5, 1.0, 1.5, 2.0]
        
        for mass in masses:
            print(f"\nTesting with mass = {mass} kg:")
            
            obs, info = env.reset(options={'box_mass': mass})
            print(f"  ✓ Reset with mass {info['box_mass']} kg")
            
            # Run a few steps
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            print(f"  ✓ Ran {_+1} steps, final distance: {info['distance_to_goal']:.3f} m")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Mass variation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rendering():
    """Test 4: Rendering"""
    print("\n" + "="*60)
    print("TEST 4: Rendering")
    print("="*60)
    
    try:
        # Test rgb_array mode (human mode requires display)
        env = PushBoxEnv(render_mode='rgb_array')
        obs, info = env.reset()
        
        print("✓ Environment created with render_mode='rgb_array'")
        
        # Render a frame
        frame = env.render()
        
        if frame is not None:
            print(f"✓ Rendered frame shape: {frame.shape}")
        else:
            print("✗ Rendering returned None")
            return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_data():
    """Test 5: Episode data collection"""
    print("\n" + "="*60)
    print("TEST 5: Episode Data Collection")
    print("="*60)
    
    try:
        env = PushBoxEnv()
        obs, info = env.reset()
        
        # Run episode
        steps = 0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
        
        # Get episode data
        episode_data = env.get_episode_data()
        
        print(f"✓ Episode ran for {steps} steps")
        print(f"✓ States shape: {episode_data['states'].shape}")
        print(f"✓ Actions shape: {episode_data['actions'].shape}")
        print(f"✓ Rewards shape: {episode_data['rewards'].shape}")
        print(f"✓ Contacts shape: {episode_data['contacts'].shape}")
        
        # Verify data consistency
        assert episode_data['states'].shape[0] == steps
        assert episode_data['actions'].shape[0] == steps
        assert episode_data['rewards'].shape[0] == steps
        
        print("✓ Data shapes are consistent")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Episode data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_success_condition():
    """Test 6: Success condition (guided policy)"""
    print("\n" + "="*60)
    print("TEST 6: Success Condition (Simple Controller)")
    print("="*60)
    
    try:
        env = PushBoxEnv()
        
        # Set easy initial condition
        obs, info = env.reset(options={
            'box_pos': np.array([0.9, 0.5]),
            'goal_pos': np.array([1.0, 0.5])
        })
        
        print(f"✓ Initial box position: {info['box_position'][:2]}")
        print(f"✓ Goal position: {info['goal_position']}")
        print(f"✓ Initial distance: {info['distance_to_goal']:.3f} m")
        
        # Simple proportional controller
        max_steps = 100
        for step in range(max_steps):
            # Get current state
            box_pos = obs[4:6]
            goal_pos = obs[8:10]
            
            # Compute desired force direction
            direction = goal_pos - box_pos
            
            # Simple P-controller for joint angles
            # (This is simplified; proper inverse kinematics would be better)
            action = np.clip(direction * 5.0, -10, 10)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"  Step {step}: distance = {info['distance_to_goal']:.4f} m, "
                      f"success_counter = {info['success_counter']}")
            
            if terminated:
                if info['success']:
                    print(f"✓ Success! Box reached goal at step {step}")
                    return True
                else:
                    print(f"✗ Terminated without success at step {step}")
                    break
            
            if truncated:
                print(f"✗ Episode truncated at step {step}")
                break
        
        env.close()
        return False
        
    except Exception as e:
        print(f"✗ Success condition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("PUSHBOX ENVIRONMENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Initialization", test_initialization),
        ("Random Policy", test_random_policy),
        ("Mass Variation", test_mass_variation),
        ("Rendering", test_rendering),
        ("Episode Data", test_episode_data),
        ("Success Condition", test_success_condition)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
