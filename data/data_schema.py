"""
Data schema for physics-informed robotics experiments

Defines standardized data formats for:
- Episode trajectories
- Experiment results
- Performance metrics
- Physics validation

Purpose: Reproducible experiments for paper Section 4.1

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import pickle
from datetime import datetime
import os


@dataclass
class EpisodeData:
    """
    Single episode trajectory data
    
    Contains complete state-action-reward history for one episode
    """
    
    # Episode metadata
    episode_id: int
    timestamp: str
    seed: int
    
    # Environment configuration
    box_mass: float  # kg
    friction_coef: float
    initial_box_pos: np.ndarray  # [x, y]
    goal_pos: np.ndarray  # [x, y]
    
    # Trajectory data
    states: np.ndarray  # [T, state_dim] - state history
    actions: np.ndarray  # [T, action_dim] - action history
    rewards: np.ndarray  # [T] - reward history
    contacts: np.ndarray  # [T] - binary contact flags
    
    # Episode outcome
    success: bool  # Did box reach goal?
    steps: int  # Number of steps taken
    total_reward: float  # Cumulative reward
    final_distance: float  # Final distance to goal [m]
    
    # Physics metrics
    momentum_error: Optional[float] = None  # Conservation error
    energy_error: Optional[float] = None  # Conservation error
    
    def __post_init__(self):
        """Validate data shapes"""
        assert self.states.shape[0] == self.steps
        assert self.actions.shape[0] == self.steps
        assert self.rewards.shape[0] == self.steps
        assert len(self.initial_box_pos) == 2
        assert len(self.goal_pos) == 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        data = asdict(self)
        # Convert numpy arrays to lists
        data['states'] = self.states.tolist()
        data['actions'] = self.actions.tolist()
        data['rewards'] = self.rewards.tolist()
        data['contacts'] = self.contacts.tolist()
        data['initial_box_pos'] = self.initial_box_pos.tolist()
        data['goal_pos'] = self.goal_pos.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeData':
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        data['states'] = np.array(data['states'])
        data['actions'] = np.array(data['actions'])
        data['rewards'] = np.array(data['rewards'])
        data['contacts'] = np.array(data['contacts'])
        data['initial_box_pos'] = np.array(data['initial_box_pos'])
        data['goal_pos'] = np.array(data['goal_pos'])
        return cls(**data)


@dataclass
class ExperimentData:
    """
    Complete experiment data for paper Section 4.1
    
    Contains all episodes and aggregate metrics for one experimental run
    """
    
    # Experiment metadata
    experiment_name: str
    timestamp: str
    git_commit: str
    
    # Configuration
    method: str  # 'baseline', 'dynami-cal', 'mujoco', etc.
    num_episodes: int
    max_steps_per_episode: int
    
    # Environment parameters
    default_box_mass: float  # Training mass
    ood_masses: List[float]  # OOD test masses
    
    # Episode data
    episodes: List[EpisodeData]
    
    # Aggregate metrics (computed from episodes)
    success_rate: float  # Overall success rate
    avg_steps_to_goal: float  # Average steps for successful episodes
    avg_reward: float  # Average total reward
    
    # Sample efficiency metrics (for paper Figure 2)
    learning_curve: Optional[np.ndarray] = None  # [num_epochs, 2] - (episodes, success_rate)
    
    # OOD generalization metrics (for paper Table 1)
    ood_results: Optional[Dict[str, Any]] = None  # Results by mass
    
    # Physics validation
    momentum_error: Optional[float] = None  # Average momentum conservation error
    energy_error: Optional[float] = None  # Average energy conservation error
    
    # Computational metrics
    training_time: Optional[float] = None  # Total training time [s]
    inference_time_mean: Optional[float] = None  # Average inference time [ms]
    inference_time_std: Optional[float] = None  # Inference time std [ms]
    
    def __post_init__(self):
        """Validate data"""
        assert len(self.episodes) == self.num_episodes
    
    def compute_metrics(self):
        """Compute aggregate metrics from episodes"""
        if not self.episodes:
            return
        
        # Success rate
        successes = sum(1 for ep in self.episodes if ep.success)
        self.success_rate = successes / len(self.episodes)
        
        # Average steps (only successful episodes)
        successful_episodes = [ep for ep in self.episodes if ep.success]
        if successful_episodes:
            self.avg_steps_to_goal = np.mean([ep.steps for ep in successful_episodes])
        else:
            self.avg_steps_to_goal = float('inf')
        
        # Average reward
        self.avg_reward = np.mean([ep.total_reward for ep in self.episodes])
        
        # Physics errors
        momentum_errors = [ep.momentum_error for ep in self.episodes 
                          if ep.momentum_error is not None]
        if momentum_errors:
            self.momentum_error = np.mean(momentum_errors)
        
        energy_errors = [ep.energy_error for ep in self.episodes 
                        if ep.energy_error is not None]
        if energy_errors:
            self.energy_error = np.mean(energy_errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
            'method': self.method,
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'default_box_mass': self.default_box_mass,
            'ood_masses': self.ood_masses,
            'episodes': [ep.to_dict() for ep in self.episodes],
            'success_rate': self.success_rate,
            'avg_steps_to_goal': self.avg_steps_to_goal,
            'avg_reward': self.avg_reward,
            'learning_curve': self.learning_curve.tolist() if self.learning_curve is not None else None,
            'ood_results': self.ood_results,
            'momentum_error': self.momentum_error,
            'energy_error': self.energy_error,
            'training_time': self.training_time,
            'inference_time_mean': self.inference_time_mean,
            'inference_time_std': self.inference_time_std
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentData':
        """Create from dictionary"""
        # Convert episode dicts to EpisodeData objects
        data['episodes'] = [EpisodeData.from_dict(ep) for ep in data['episodes']]
        
        # Convert learning curve back to array
        if data['learning_curve'] is not None:
            data['learning_curve'] = np.array(data['learning_curve'])
        
        return cls(**data)
    
    def save(self, filepath: str):
        """
        Save experiment data to file
        
        Args:
            filepath: Output file path (.json or .pkl)
        """
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        
        elif ext == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentData':
        """
        Load experiment data from file
        
        Args:
            filepath: Input file path (.json or .pkl)
        
        Returns:
            ExperimentData instance
        """
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        
        elif ext == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


@dataclass
class ComparisonData:
    """
    Comparison data for multiple methods (paper Table 1)
    
    Used to compare Dynami-CAL vs baselines
    """
    
    # Metadata
    timestamp: str
    git_commit: str
    
    # Methods compared
    methods: List[str]  # e.g., ['baseline', 'dynami-cal']
    
    # Experiment data for each method
    experiments: Dict[str, ExperimentData]
    
    # Comparison metrics
    sample_efficiency_ratio: Optional[float] = None  # Dynami-CAL / Baseline
    ood_generalization_gap: Optional[Dict[str, float]] = None  # Gap by mass
    
    def compute_comparison(self):
        """Compute comparison metrics"""
        if 'dynami-cal' in self.experiments and 'baseline' in self.experiments:
            # Sample efficiency: episodes to reach 90% success
            baseline_exp = self.experiments['baseline']
            dynamical_exp = self.experiments['dynami-cal']
            
            # Placeholder computation (needs learning curve data)
            if baseline_exp.learning_curve is not None and dynamical_exp.learning_curve is not None:
                # Find episodes to reach 90% success
                baseline_episodes = self._episodes_to_threshold(baseline_exp.learning_curve, 0.9)
                dynamical_episodes = self._episodes_to_threshold(dynamical_exp.learning_curve, 0.9)
                
                if baseline_episodes and dynamical_episodes:
                    self.sample_efficiency_ratio = baseline_episodes / dynamical_episodes
    
    def _episodes_to_threshold(self, learning_curve: np.ndarray, threshold: float) -> Optional[int]:
        """Find episodes needed to reach success rate threshold"""
        for episodes, success_rate in learning_curve:
            if success_rate >= threshold:
                return int(episodes)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
            'methods': self.methods,
            'experiments': {k: v.to_dict() for k, v in self.experiments.items()},
            'sample_efficiency_ratio': self.sample_efficiency_ratio,
            'ood_generalization_gap': self.ood_generalization_gap
        }
    
    def save(self, filepath: str):
        """Save comparison data"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Helper functions

def create_episode_data(
    episode_id: int,
    env,
    trajectory: Dict[str, np.ndarray],
    success: bool,
    seed: int
) -> EpisodeData:
    """
    Create EpisodeData from environment trajectory
    
    Args:
        episode_id: Episode number
        env: PushBox environment
        trajectory: Dict with 'states', 'actions', 'rewards', 'contacts'
        success: Whether episode succeeded
        seed: Random seed
    
    Returns:
        EpisodeData instance
    """
    return EpisodeData(
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        box_mass=env.box_mass,
        friction_coef=env.friction_coef,
        initial_box_pos=trajectory['states'][0][4:6],
        goal_pos=trajectory['states'][0][8:10],
        states=trajectory['states'],
        actions=trajectory['actions'],
        rewards=trajectory['rewards'],
        contacts=trajectory['contacts'],
        success=success,
        steps=len(trajectory['states']),
        total_reward=np.sum(trajectory['rewards']),
        final_distance=np.linalg.norm(
            trajectory['states'][-1][4:6] - trajectory['states'][-1][8:10]
        )
    )


def create_experiment_data(
    experiment_name: str,
    method: str,
    episodes: List[EpisodeData],
    **kwargs
) -> ExperimentData:
    """
    Create ExperimentData from episode list
    
    Args:
        experiment_name: Name of experiment
        method: Method name
        episodes: List of EpisodeData
        **kwargs: Additional parameters
    
    Returns:
        ExperimentData instance
    """
    import subprocess
    
    # Get git commit
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(__file__)
        ).decode('utf-8').strip()
    except:
        git_commit = 'unknown'
    
    exp_data = ExperimentData(
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        git_commit=git_commit,
        method=method,
        num_episodes=len(episodes),
        max_steps_per_episode=max(ep.steps for ep in episodes),
        default_box_mass=episodes[0].box_mass if episodes else 1.0,
        ood_masses=kwargs.get('ood_masses', []),
        episodes=episodes,
        success_rate=0.0,  # Will be computed
        avg_steps_to_goal=0.0,
        avg_reward=0.0,
        **kwargs
    )
    
    # Compute metrics
    exp_data.compute_metrics()
    
    return exp_data


if __name__ == "__main__":
    """Test data schema"""
    print("="*60)
    print("DATA SCHEMA TEST")
    print("="*60)
    
    # Create dummy episode data
    episode = EpisodeData(
        episode_id=0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        box_mass=1.0,
        friction_coef=0.3,
        initial_box_pos=np.array([0.5, 0.0]),
        goal_pos=np.array([1.0, 0.5]),
        states=np.random.randn(100, 10),
        actions=np.random.randn(100, 2),
        rewards=np.random.randn(100),
        contacts=np.random.randint(0, 2, 100),
        success=True,
        steps=100,
        total_reward=10.5,
        final_distance=0.03
    )
    
    print("✓ Created EpisodeData")
    
    # Create experiment data
    episodes = [episode]
    experiment = create_experiment_data(
        experiment_name="test_experiment",
        method="baseline",
        episodes=episodes
    )
    
    print("✓ Created ExperimentData")
    print(f"  Success rate: {experiment.success_rate}")
    print(f"  Avg steps: {experiment.avg_steps_to_goal}")
    print(f"  Avg reward: {experiment.avg_reward}")
    
    # Test serialization
    test_file = "/tmp/test_experiment.json"
    experiment.save(test_file)
    print(f"✓ Saved to {test_file}")
    
    loaded = ExperimentData.load(test_file)
    print("✓ Loaded from file")
    
    assert loaded.experiment_name == experiment.experiment_name
    assert loaded.num_episodes == experiment.num_episodes
    
    print("\n✓ All data schema tests passed!")
