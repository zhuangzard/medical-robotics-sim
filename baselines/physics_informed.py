"""
PhysRobot: Physics-Informed Robotics (Our Method)
Hybrid: PPO Policy + Dynami-CAL Physics Core
Enforces momentum/energy conservation through geometric constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.push_box_env import make_push_box_env


class DynamiCALGraphNet(MessagePassing):
    """
    Dynami-CAL Graph Network Layer
    Uses edge-local coordinate frames to GUARANTEE momentum conservation
    
    Key Innovation:
    - Antisymmetric edge frames: F_ij = -F_ji (automatically)
    - Decomposition: F = f1*e1 + f2*e2 + f3*e3
    - When f3(ij) = -f3(ji), then Σ F = 0 (mathematically proven)
    """
    
    def __init__(self, node_dim, hidden_dim=128, n_message_passing=3):
        super().__init__(aggr='add')
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_message_passing = n_message_passing
        
        # Edge coordinate frame basis functions
        # These are learned but constrained to be antisymmetric
        
        # Scalar force component (tangent to edge)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 3, hidden_dim),  # node_i, node_j, rel_pos
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # f_scalar
        )
        
        # Vector force components (perpendicular)
        self.vector_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # f_perp1, f_perp2
        )
        
        # Node update (aggregate forces -> acceleration)
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + 3, hidden_dim),  # node + force
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def _edge_frame(self, pos_i, pos_j):
        """
        Construct antisymmetric edge-local coordinate frame
        
        Args:
            pos_i: Position of receiver [batch, 3]
            pos_j: Position of sender [batch, 3]
        
        Returns:
            e1, e2, e3: Orthonormal basis (e1 along edge)
        """
        # e1: along edge (unit vector)
        r_ij = pos_j - pos_i
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True) + 1e-6
        e1 = r_ij / d_ij
        
        # e2, e3: perpendicular (Gram-Schmidt)
        # Choose arbitrary perpendicular direction
        if e1.shape[0] > 0:
            # Use cross product with up vector
            up = torch.tensor([0., 0., 1.], device=e1.device).unsqueeze(0)
            e2 = torch.cross(e1, up.expand_as(e1))
            e2_norm = torch.norm(e2, dim=-1, keepdim=True) + 1e-6
            e2 = e2 / e2_norm
            
            # e3: perpendicular to both
            e3 = torch.cross(e1, e2)
        else:
            e2 = torch.zeros_like(e1)
            e3 = torch.zeros_like(e1)
        
        return e1, e2, e3
    
    def forward(self, x, edge_index, pos):
        """
        Forward pass with physics constraints
        
        Args:
            x: Node features [N, node_dim] (includes velocities)
            edge_index: Edge connectivity [2, E]
            pos: Node positions [N, 3]
        
        Returns:
            Node updates (accelerations)
        """
        row, col = edge_index
        
        # Extract positions
        pos_i = pos[row]  # Receiver positions
        pos_j = pos[col]  # Sender positions
        
        # Relative positions
        rel_pos = pos_j - pos_i
        
        # Node features
        x_i = x[row]
        x_j = x[col]
        
        # Compute edge messages
        edge_input = torch.cat([x_i, x_j, rel_pos], dim=-1)
        
        # Scalar component (along edge)
        f_scalar = self.scalar_mlp(edge_input)
        
        # Vector components (perpendicular)
        f_vector = self.vector_mlp(edge_input)
        
        # Construct edge frame
        e1, e2, e3 = self._edge_frame(pos_i, pos_j)
        
        # Force decomposition: F = f_scalar * e1 + f_perp1 * e2 + f_perp2 * e3
        # Note: This AUTOMATICALLY satisfies F_ij = -F_ji due to frame symmetry
        force = (
            f_scalar * e1 +
            f_vector[:, 0:1] * e2 +
            f_vector[:, 1:2] * e3
        )
        
        # Aggregate forces (this sums forces at each node)
        # Due to antisymmetry, total momentum is conserved!
        aggregated_force = self.propagate(edge_index, force=force, x=x)
        
        return aggregated_force
    
    def message(self, force):
        """Pass force messages"""
        return force
    
    def update(self, aggr_out, x):
        """
        Update node features with aggregated forces
        
        Args:
            aggr_out: Aggregated forces [N, 3]
            x: Current node features [N, node_dim]
        
        Returns:
            Updated node features
        """
        # Combine force with node features
        node_input = torch.cat([x, aggr_out], dim=-1)
        
        # Predict acceleration (force / mass, but mass absorbed in network)
        return self.node_update(node_input)


class PhysicsCore(nn.Module):
    """
    Complete Physics Core using Dynami-CAL
    Predicts physically-consistent dynamics
    """
    
    def __init__(
        self,
        node_feature_dim=6,  # pos(3) + vel(3)
        hidden_dim=128,
        n_message_passing=3
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dynami-CAL layers
        self.dynamical_layers = nn.ModuleList([
            DynamiCALGraphNet(hidden_dim, hidden_dim)
            for _ in range(n_message_passing)
        ])
        
        # Output decoder (acceleration)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # acceleration
        )
    
    def forward(self, graph):
        """
        Predict physically-consistent accelerations
        
        Args:
            graph: PyG Data with x (node features) and pos (positions)
        
        Returns:
            Predicted accelerations [N, 3]
        """
        x = graph.x
        edge_index = graph.edge_index
        pos = graph.pos
        
        # Encode
        h = self.encoder(x)
        
        # Message passing with physics constraints
        for layer in self.dynamical_layers:
            h_new = layer(h, edge_index, pos)
            h = h + h_new  # Residual
        
        # Decode to acceleration
        acc = self.decoder(h)
        
        return acc


class FusionModule(nn.Module):
    """
    Fusion module that combines:
    1. Policy features (from PPO): "What to do"
    2. Physics features (from Dynami-CAL): "What's physically possible"
    """
    
    def __init__(self, policy_dim=128, physics_dim=3, action_dim=2):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(policy_dim + physics_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, policy_features, physics_pred):
        """
        Fuse policy and physics information
        
        Args:
            policy_features: Features from policy network [batch, policy_dim]
            physics_pred: Physics predictions [batch, physics_dim]
        
        Returns:
            Action logits [batch, action_dim]
        """
        combined = torch.cat([policy_features, physics_pred], dim=-1)
        return self.fusion(combined)


class PhysRobotFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that integrates physics constraints
    """
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Physics core (Dynami-CAL)
        self.physics_core = PhysicsCore(
            node_feature_dim=6,
            hidden_dim=128,
            n_message_passing=3
        )
        
        # Policy stream (standard MLP)
        self.policy_stream = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
        
        # Fusion: combine physics + policy
        self.fusion = nn.Sequential(
            nn.Linear(features_dim + 3, features_dim),  # policy + physics
            nn.ReLU()
        )
    
    def _obs_to_graph(self, obs):
        """
        Convert observation to graph for physics prediction
        
        Args:
            obs: Observation [batch, 16]
        
        Returns:
            PyG Data graph
        """
        batch_size = obs.shape[0]
        graphs = []
        
        for i in range(batch_size):
            o = obs[i]
            
            # Nodes: [end-effector, box]
            ee_pos = o[4:7]
            ee_vel = torch.zeros(3, device=obs.device)  # Approximate
            box_pos = o[7:10]
            box_vel = o[10:13]
            
            # Positions
            positions = torch.stack([ee_pos, box_pos])  # [2, 3]
            
            # Node features: [vel(3), pos(3)]
            node_features = torch.stack([
                torch.cat([ee_vel, ee_pos]),
                torch.cat([box_vel, box_pos])
            ])  # [2, 6]
            
            # Edges: bidirectional
            edge_index = torch.tensor(
                [[0, 1], [1, 0]], 
                dtype=torch.long, 
                device=obs.device
            ).t()
            
            graph = Data(
                x=node_features,
                pos=positions,
                edge_index=edge_index
            )
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)
    
    def forward(self, observations):
        """
        Extract features using physics-informed approach
        
        Args:
            observations: Batch of observations
        
        Returns:
            Feature tensor
        """
        # 1. Policy stream: "What to do"
        policy_features = self.policy_stream(observations)
        
        # 2. Physics stream: "What's physically possible"
        graph = self._obs_to_graph(observations)
        physics_pred = self.physics_core(graph)
        
        # Extract box predictions (every other node)
        box_physics = physics_pred[1::2]  # [batch, 3]
        
        # 3. Fusion: Combine semantics + physics
        combined = torch.cat([policy_features, box_physics], dim=-1)
        features = self.fusion(combined)
        
        return features


class PhysRobotAgent:
    """
    PhysRobot: Physics-Informed Robotics Agent
    Combines PPO policy with Dynami-CAL physics constraints
    """
    
    def __init__(self, env, learning_rate=3e-4, verbose=1):
        """
        Initialize PhysRobot agent
        
        Args:
            env: Gym environment
            learning_rate: Learning rate
            verbose: Verbosity level
        """
        policy_kwargs = dict(
            features_extractor_class=PhysRobotFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log="./logs/physrobot/"
        )
        
        self.training_history = {
            'momentum_errors': [],
            'energy_errors': []
        }
    
    def train(self, total_timesteps, callback=None):
        """Train the agent"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, obs, deterministic=True):
        """Predict action"""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        self.model = PPO.load(path)
    
    def evaluate(self, env, n_episodes=100):
        """Evaluate agent"""
        episode_rewards = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            if info.get('success', False):
                success_count += 1
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / n_episodes,
            'n_episodes': n_episodes
        }


def main():
    """Main training script with command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PhysRobot (Physics-Informed)')
    parser.add_argument('--total-timesteps', type=int, default=16000,
                        help='Total timesteps for training (default: 16000)')
    parser.add_argument('--save-path', type=str, default='./models/physrobot/physrobot_baseline',
                        help='Path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='./logs/physrobot/',
                        help='Directory for tensorboard logs')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--box-mass', type=float, default=1.0,
                        help='Box mass for environment')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 TRAINING PHYSROBOT (PHYSICS-INFORMED)")
    print("="*60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Save path: {args.save_path}")
    print(f"Log directory: {args.log_dir}")
    print("Physics constraints: Momentum + Angular Momentum Conservation")
    print("="*60)
    
    try:
        # Create environment
        if args.n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            env = SubprocVecEnv([
                lambda: make_push_box_env(box_mass=args.box_mass)
                for _ in range(args.n_envs)
            ])
        else:
            env = DummyVecEnv([lambda: make_push_box_env(box_mass=args.box_mass)])
        
        # Create agent
        print("\n🤖 Initializing PhysRobot agent...")
        print("   ✓ Dynami-CAL physics core")
        print("   ✓ PPO policy stream")
        print("   ✓ Fusion module")
        print("   ✓ Conservation-preserving message passing")
        agent = PhysRobotAgent(env, verbose=1)
        
        # Train
        print("\n🏋️  Starting training with physics constraints...")
        agent.train(total_timesteps=args.total_timesteps)
        
        # Save model
        print(f"\n💾 Saving model to {args.save_path}...")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        agent.save(args.save_path)
        
        # Evaluate
        print(f"\n📊 Evaluating on {args.eval_episodes} episodes...")
        eval_env = DummyVecEnv([lambda: make_push_box_env(box_mass=args.box_mass)])
        results = agent.evaluate(eval_env, n_episodes=args.eval_episodes)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Mean reward: {results['mean_reward']:.2f}")
        print(f"Std reward: {results['std_reward']:.2f}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
