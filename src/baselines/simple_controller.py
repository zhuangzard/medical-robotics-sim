"""
Simple baseline controllers for PushBox environment

Controllers:
1. Proportional (P) Controller - Simple position-based control
2. Random Controller - Baseline for comparison
3. Greedy Controller - Always move toward goal

Purpose: Establish baseline performance before learning

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

import numpy as np
from typing import Tuple, Dict, Any


class ProportionalController:
    """
    Simple P-controller for baseline comparison
    
    Strategy:
        1. Compute error: box_pos - goal_pos
        2. Apply proportional force toward goal
        3. Clip to action limits
    
    Expected performance:
        - Success rate: 30-50% (with good initial conditions)
        - Average steps: 100-200
        - Sample efficiency: Poor (no learning)
    """
    
    def __init__(self, kp: float = 10.0):
        """
        Initialize P-controller
        
        Args:
            kp: Proportional gain
        """
        self.kp = kp
    
    def __call__(self, obs: np.ndarray, env: Any = None) -> np.ndarray:
        """
        Compute control action
        
        Args:
            obs: State vector [joint_pos, joint_vel, box_pos, box_vel, goal_pos]
            env: Environment (optional, not used)
        
        Returns:
            action: Joint torques
        """
        # Extract positions
        box_pos = obs[4:6]
        goal_pos = obs[8:10]
        
        # Compute error
        error = goal_pos - box_pos
        
        # Proportional control
        force = self.kp * error
        
        # Clip to action limits
        action = np.clip(force, -10.0, 10.0)
        
        return action


class GreedyController:
    """
    Greedy controller - always move directly toward goal
    
    Strategy:
        1. Compute direction: (goal - box) / ||goal - box||
        2. Apply maximum force in that direction
    
    Expected performance:
        - Aggressive pushing
        - May overshoot goal
        - Fast convergence if successful
    """
    
    def __init__(self, max_force: float = 10.0):
        """
        Initialize greedy controller
        
        Args:
            max_force: Maximum force magnitude
        """
        self.max_force = max_force
    
    def __call__(self, obs: np.ndarray, env: Any = None) -> np.ndarray:
        """
        Compute control action
        
        Args:
            obs: State vector
            env: Environment (optional)
        
        Returns:
            action: Joint torques
        """
        # Extract positions
        box_pos = obs[4:6]
        goal_pos = obs[8:10]
        
        # Compute direction
        direction = goal_pos - box_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # Already at goal
            return np.zeros(2)
        
        # Normalize direction
        direction = direction / distance
        
        # Apply maximum force
        force = self.max_force * direction
        
        return force


class RandomController:
    """
    Random controller - uniformly random actions
    
    Purpose: Baseline for sample efficiency comparison
    
    Expected performance:
        - Very poor success rate (< 1%)
        - No convergence
        - Useful for data collection diversity
    """
    
    def __init__(self, action_low: float = -10.0, action_high: float = 10.0):
        """
        Initialize random controller
        
        Args:
            action_low: Minimum action value
            action_high: Maximum action value
        """
        self.action_low = action_low
        self.action_high = action_high
    
    def __call__(self, obs: np.ndarray, env: Any = None) -> np.ndarray:
        """
        Compute random action
        
        Args:
            obs: State vector (not used)
            env: Environment (optional)
        
        Returns:
            action: Random joint torques
        """
        return np.random.uniform(self.action_low, self.action_high, size=2)


class PDController:
    """
    PD-controller with velocity damping
    
    Strategy:
        1. Proportional term: kp * (goal - box_pos)
        2. Derivative term: -kd * box_vel
        3. Total: kp * error - kd * vel
    
    Expected performance:
        - Better than P-controller (less oscillation)
        - More stable convergence
        - Still limited by no learning
    """
    
    def __init__(self, kp: float = 10.0, kd: float = 2.0):
        """
        Initialize PD-controller
        
        Args:
            kp: Proportional gain
            kd: Derivative gain
        """
        self.kp = kp
        self.kd = kd
    
    def __call__(self, obs: np.ndarray, env: Any = None) -> np.ndarray:
        """
        Compute control action
        
        Args:
            obs: State vector [joint_pos, joint_vel, box_pos, box_vel, goal_pos]
            env: Environment (optional)
        
        Returns:
            action: Joint torques
        """
        # Extract state
        box_pos = obs[4:6]
        box_vel = obs[6:8]
        goal_pos = obs[8:10]
        
        # Compute error
        error = goal_pos - box_pos
        
        # PD control
        force = self.kp * error - self.kd * box_vel
        
        # Clip to action limits
        action = np.clip(force, -10.0, 10.0)
        
        return action


def evaluate_controller(
    controller,
    env,
    num_episodes: int = 10,
    max_steps: int = 500,
    render: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a controller on the environment
    
    Args:
        controller: Controller instance (callable)
        env: PushBox environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        verbose: Print progress
    
    Returns:
        Dictionary with performance metrics
    """
    results = {
        'success_rate': 0.0,
        'avg_steps': 0.0,
        'avg_reward': 0.0,
        'avg_final_distance': 0.0,
        'episodes': []
    }
    
    successes = 0
    total_steps = 0
    total_reward = 0.0
    total_final_distance = 0.0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Get action from controller
            action = controller(obs, env)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
        
        # Record results
        success = info.get('success', False)
        final_distance = info['distance_to_goal']
        
        if success:
            successes += 1
        
        total_steps += step + 1
        total_reward += episode_reward
        total_final_distance += final_distance
        
        results['episodes'].append({
            'episode': episode,
            'success': success,
            'steps': step + 1,
            'reward': episode_reward,
            'final_distance': final_distance
        })
        
        if verbose:
            status = "SUCCESS" if success else "FAIL"
            print(f"Episode {episode+1}/{num_episodes}: {status} | "
                  f"Steps: {step+1} | Reward: {episode_reward:.2f} | "
                  f"Final dist: {final_distance:.4f} m")
    
    # Compute averages
    results['success_rate'] = successes / num_episodes
    results['avg_steps'] = total_steps / num_episodes
    results['avg_reward'] = total_reward / num_episodes
    results['avg_final_distance'] = total_final_distance / num_episodes
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Final Distance: {results['avg_final_distance']:.4f} m")
    
    return results


if __name__ == "__main__":
    """Test all controllers"""
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from environments.push_box import PushBoxEnv
    
    print("="*60)
    print("BASELINE CONTROLLER EVALUATION")
    print("="*60)
    
    # Create environment
    env = PushBoxEnv(max_episode_steps=300)
    
    # Test controllers
    controllers = {
        'Random': RandomController(),
        'Proportional': ProportionalController(kp=8.0),
        'Greedy': GreedyController(max_force=10.0),
        'PD': PDController(kp=8.0, kd=2.0)
    }
    
    all_results = {}
    
    for name, controller in controllers.items():
        print(f"\n{'='*60}")
        print(f"Testing {name} Controller")
        print('='*60)
        
        results = evaluate_controller(
            controller,
            env,
            num_episodes=5,
            max_steps=300,
            render=False,
            verbose=True
        )
        
        all_results[name] = results
    
    # Print comparison
    print("\n" + "="*60)
    print("CONTROLLER COMPARISON")
    print("="*60)
    print(f"{'Controller':<15} {'Success Rate':>12} {'Avg Steps':>10} {'Avg Reward':>12}")
    print("-"*60)
    
    for name, results in all_results.items():
        print(f"{name:<15} {results['success_rate']*100:>11.1f}% "
              f"{results['avg_steps']:>10.1f} {results['avg_reward']:>12.2f}")
    
    env.close()
    print("\n✓ All controller tests completed!")
