"""
GNS (Graph Network Simulator) Baseline
Based on Sanchez-Gonzalez et al. (2020)
Graph networks for dynamics prediction WITHOUT strict conservation
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


class GraphNetworkLayer(MessagePassing):
    """
    Single Graph Network Layer
    Standard message passing WITHOUT physics constraints
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__(aggr='add')
        
        # Edge model
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Node model
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass
        
        Args:
            x: Node features [N, node_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
        
        Returns:
            Updated node features
        """
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute edge messages
        
        Args:
            x_i: Receiver node features
            x_j: Sender node features
            edge_attr: Edge attributes
        """
        # Concatenate sender, receiver, edge features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Update edge features
        return self.edge_mlp(edge_input)
    
    def update(self, aggr_out, x):
        """
        Update node features
        
        Args:
            aggr_out: Aggregated messages
            x: Current node features
        """
        # Concatenate node features with aggregated messages
        node_input = torch.cat([x, aggr_out], dim=-1)
        
        # Update node features
        return self.node_mlp(node_input)


class GNSNetwork(nn.Module):
    """
    Full Graph Network Simulator
    Predicts dynamics using graph neural networks
    """
    
    def __init__(
        self, 
        node_feature_dim=6,     # pos(3) + vel(3)
        edge_feature_dim=4,     # relative_pos(3) + distance(1)
        hidden_dim=128,
        n_layers=3
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph network layers
        self.gn_layers = nn.ModuleList([
            GraphNetworkLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Decoder: predict acceleration
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # acceleration (x, y, z)
        )
    
    def forward(self, graph):
        """
        Predict accelerations from graph
        
        Args:
            graph: PyG Data object with node features and edges
        
        Returns:
            Predicted accelerations [N, 3]
        """
        x = graph.x  # Node features
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing
        for layer in self.gn_layers:
            x_new = layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual connection
        
        # Decode to accelerations
        acc = self.decoder(x)
        
        return acc


class GNSFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for PPO that uses GNS
    Converts observations to graph and predicts physics
    """
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # GNS network for physics prediction
        self.gns = GNSNetwork(
            node_feature_dim=6,
            edge_feature_dim=4,
            hidden_dim=128,
            n_layers=3
        )
        
        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(128 + 16, features_dim),  # GNS features + obs
            nn.ReLU()
        )
    
    def _obs_to_graph(self, obs):
        """
        Convert observation to graph
        
        Args:
            obs: Observation tensor [batch, 16]
                - joint_pos (2)
                - joint_vel (2)
                - ee_pos (3)
                - box_pos (3)
                - box_vel (3)
                - goal_pos (3)
        
        Returns:
            PyG Data graph
        """
        batch_size = obs.shape[0]
        graphs = []
        
        for i in range(batch_size):
            o = obs[i]
            
            # Define nodes: [end-effector, box]
            ee_pos = o[4:7]
            ee_vel = torch.zeros(3, device=obs.device)  # Approx
            box_pos = o[7:10]
            box_vel = o[10:13]
            
            # Node features: [pos(3), vel(3)]
            node_features = torch.stack([
                torch.cat([ee_pos, ee_vel]),
                torch.cat([box_pos, box_vel])
            ])  # [2, 6]
            
            # Edge: end-effector -> box
            edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=obs.device)
            
            # Edge features: relative position + distance
            rel_pos = box_pos - ee_pos
            distance = torch.norm(rel_pos).unsqueeze(0)
            edge_attr = torch.cat([rel_pos, distance]).unsqueeze(0)  # [1, 4]
            
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)
    
    def forward(self, observations):
        """
        Extract features using GNS
        
        Args:
            observations: Batch of observations
        
        Returns:
            Feature tensor
        """
        # Convert to graph
        graph = self._obs_to_graph(observations)
        
        # Predict physics with GNS
        acc_pred = self.gns(graph)
        
        # Use predictions as features (take box acceleration)
        physics_features = acc_pred[1::2]  # Box node features
        
        # Combine with raw observations
        combined = torch.cat([
            physics_features.view(observations.shape[0], -1),
            observations
        ], dim=-1)
        
        return self.feature_proj(combined)


class GNSAgent:
    """
    GNS-based PPO Agent
    Uses graph networks to learn physics, but doesn't enforce conservation
    """
    
    def __init__(self, env, learning_rate=3e-4, verbose=1):
        """
        Initialize GNS agent
        
        Args:
            env: Gym environment
            learning_rate: Learning rate
            verbose: Verbosity level
        """
        policy_kwargs = dict(
            features_extractor_class=GNSFeaturesExtractor,
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
            tensorboard_log="./logs/gns_baseline/"
        )
    
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
    
    parser = argparse.ArgumentParser(description='Train GNS Baseline')
    parser.add_argument('--total-timesteps', type=int, default=80000,
                        help='Total timesteps for training (default: 80000)')
    parser.add_argument('--save-path', type=str, default='./models/gns/gns_baseline',
                        help='Path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='./logs/gns/',
                        help='Directory for tensorboard logs')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--box-mass', type=float, default=1.0,
                        help='Box mass for environment')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 TRAINING GNS BASELINE")
    print("="*60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Save path: {args.save_path}")
    print(f"Log directory: {args.log_dir}")
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
        print("\n🤖 Initializing GNS agent...")
        agent = GNSAgent(env, verbose=1)
        
        # Train
        print("\n🏋️  Starting training...")
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
