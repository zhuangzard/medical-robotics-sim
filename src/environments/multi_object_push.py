"""
Multi-Object Push Environment
==============================

A more complex manipulation task with 3-5 pushable objects,
inter-object collisions, and complex goals (sorting, grouping).

Designed for:
- Scalability testing of Dynami-CAL GNN
- N-node graph construction
- Ablation: complexity scaling

Author: PhysRobot Team
Date: 2026-02-06
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List


class MultiObjectPushEnv(gym.Env):
    """
    Multi-object manipulation task with physics-aware graph structure.
    
    Simulates a 2D top-down workspace where a point-mass robot pushes
    multiple objects toward goal configurations.
    
    Uses simplified analytical physics (no MuJoCo dependency) for
    portability and Colab compatibility.
    
    State Space per object:
        - position (2): x, y
        - velocity (2): vx, vy
        
    Robot state:
        - position (2): x, y
        - velocity (2): vx, vy
    
    Graph Structure:
        - Nodes: [robot, obj_1, obj_2, ..., obj_N]
        - Edges: Fully connected (all pairwise)
        - Node features: [pos_x, pos_y, vel_x, vel_y, mass, is_robot]
        - Edge features: [rel_pos_x, rel_pos_y, distance, rel_vel_x, rel_vel_y]
    
    Task Modes:
        - 'push_to_goals': Each object has an assigned goal position
        - 'sort_by_mass': Sort objects left-to-right by mass
        - 'group_by_color': Group objects by color (mass proxy) into zones
        - 'stack': Push all objects to a single target zone
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }
    
    # Physics constants
    DT = 0.02           # Simulation timestep (seconds)
    SUBSTEPS = 5        # Physics substeps per env step
    FRICTION = 0.5      # Ground friction coefficient
    RESTITUTION = 0.6   # Collision restitution (bounciness)
    ROBOT_RADIUS = 0.04
    OBJECT_RADIUS = 0.05
    WORKSPACE_X = (-1.0, 1.0)
    WORKSPACE_Y = (-1.0, 1.0)
    CONTACT_THRESHOLD = 0.01  # Extra margin for contact detection
    
    def __init__(
        self,
        n_objects: int = 3,
        task_mode: str = 'push_to_goals',
        max_episode_steps: int = 500,
        mass_range: Tuple[float, float] = (0.5, 2.0),
        render_mode: Optional[str] = None,
        randomize_masses: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_objects: Number of pushable objects (3-5)
            task_mode: Task type ('push_to_goals', 'sort_by_mass', 'group_by_color', 'stack')
            max_episode_steps: Maximum steps per episode
            mass_range: (min_mass, max_mass) for objects
            render_mode: 'human' or 'rgb_array'
            randomize_masses: Whether to randomize object masses each episode
            seed: Random seed
        """
        super().__init__()
        
        assert 2 <= n_objects <= 8, f"n_objects must be in [2, 8], got {n_objects}"
        assert task_mode in ('push_to_goals', 'sort_by_mass', 'group_by_color', 'stack')
        
        self.n_objects = n_objects
        self.n_nodes = n_objects + 1  # +1 for robot
        self.task_mode = task_mode
        self.max_episode_steps = max_episode_steps
        self.mass_range = mass_range
        self.render_mode = render_mode
        self.randomize_masses = randomize_masses
        
        # ---- Observation Space ----
        # Robot: pos(2) + vel(2) = 4
        # Per object: pos(2) + vel(2) + mass(1) = 5
        # Goals: n_objects * 2 (goal positions)
        self.obs_dim = 4 + n_objects * 5 + n_objects * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        
        # ---- Action Space ----
        # Robot force: (fx, fy) in [-1, 1] (normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32
        )
        
        # ---- Internal State ----
        self.robot_pos = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.robot_mass = 1.0  # Robot mass (fixed)
        
        self.object_pos = np.zeros((n_objects, 2))
        self.object_vel = np.zeros((n_objects, 2))
        self.object_masses = np.ones(n_objects)
        
        self.goal_positions = np.zeros((n_objects, 2))
        
        self.current_step = 0
        self.success_threshold = 0.1  # meters
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to random initial state."""
        super().reset(seed=seed)
        
        # Randomize object masses
        if self.randomize_masses:
            self.object_masses = self.np_random.uniform(
                self.mass_range[0], self.mass_range[1], size=self.n_objects
            )
        else:
            self.object_masses = np.ones(self.n_objects)
        
        # Override masses if specified
        if options and 'masses' in options:
            self.object_masses = np.array(options['masses'])
        
        # Place robot at center
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_vel = np.zeros(2)
        
        # Place objects randomly (not overlapping)
        self.object_pos = self._sample_non_overlapping_positions(
            n=self.n_objects,
            min_dist=2.5 * self.OBJECT_RADIUS,
            x_range=(-0.6, 0.6),
            y_range=(-0.6, 0.6),
            avoid_center=True,
        )
        self.object_vel = np.zeros((self.n_objects, 2))
        
        # Set goal positions based on task mode
        self._generate_goals()
        
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to force
        max_force = 10.0
        force = action * max_force
        
        # Physics simulation (multiple substeps)
        for _ in range(self.SUBSTEPS):
            self._physics_step(force)
        
        # Compute reward
        reward, reward_info = self._compute_reward()
        
        # Check termination
        self.current_step += 1
        terminated = reward_info['all_success']
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated:
            reward += 50.0  # Big bonus for completing task
        
        info = self._get_info()
        info.update(reward_info)
        
        if terminated or truncated:
            info['episode'] = {
                'r': reward,
                'l': self.current_step,
            }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    # ========== Physics Engine ==========
    
    def _physics_step(self, robot_force: np.ndarray):
        """One substep of physics simulation."""
        dt = self.DT / self.SUBSTEPS
        
        # ---- Robot dynamics ----
        robot_acc = robot_force / self.robot_mass
        # Apply friction
        robot_acc -= self.FRICTION * self.robot_vel
        
        self.robot_vel += robot_acc * dt
        self.robot_pos += self.robot_vel * dt
        
        # ---- Object dynamics (friction only, no external force) ----
        for i in range(self.n_objects):
            obj_acc = -self.FRICTION * self.object_vel[i]
            self.object_vel[i] += obj_acc * dt
            self.object_pos[i] += self.object_vel[i] * dt
        
        # ---- Collision Detection & Response ----
        # Robot-Object collisions
        for i in range(self.n_objects):
            self._resolve_collision_robot_object(i)
        
        # Object-Object collisions
        for i in range(self.n_objects):
            for j in range(i + 1, self.n_objects):
                self._resolve_collision_object_object(i, j)
        
        # ---- Boundary enforcement ----
        self._enforce_boundaries()
    
    def _resolve_collision_robot_object(self, obj_idx: int):
        """Resolve elastic collision between robot and object."""
        delta = self.object_pos[obj_idx] - self.robot_pos
        dist = np.linalg.norm(delta)
        min_dist = self.ROBOT_RADIUS + self.OBJECT_RADIUS
        
        if dist < min_dist and dist > 1e-8:
            # Normal vector
            n = delta / dist
            
            # Relative velocity
            v_rel = self.robot_vel - self.object_vel[obj_idx]
            v_n = np.dot(v_rel, n)
            
            if v_n > 0:  # Approaching
                # Elastic collision with restitution
                m1 = self.robot_mass
                m2 = self.object_masses[obj_idx]
                
                j = -(1 + self.RESTITUTION) * v_n / (1/m1 + 1/m2)
                
                self.robot_vel -= (j / m1) * n
                self.object_vel[obj_idx] += (j / m2) * n
            
            # Separate overlapping bodies
            overlap = min_dist - dist
            separation = n * (overlap / 2 + 0.001)
            self.robot_pos -= separation
            self.object_pos[obj_idx] += separation
    
    def _resolve_collision_object_object(self, i: int, j: int):
        """Resolve elastic collision between two objects."""
        delta = self.object_pos[j] - self.object_pos[i]
        dist = np.linalg.norm(delta)
        min_dist = 2 * self.OBJECT_RADIUS
        
        if dist < min_dist and dist > 1e-8:
            n = delta / dist
            v_rel = self.object_vel[i] - self.object_vel[j]
            v_n = np.dot(v_rel, n)
            
            if v_n > 0:  # Approaching
                m1 = self.object_masses[i]
                m2 = self.object_masses[j]
                
                j_imp = -(1 + self.RESTITUTION) * v_n / (1/m1 + 1/m2)
                
                self.object_vel[i] -= (j_imp / m1) * n
                self.object_vel[j] += (j_imp / m2) * n
            
            overlap = min_dist - dist
            separation = n * (overlap / 2 + 0.001)
            self.object_pos[i] -= separation
            self.object_pos[j] += separation
    
    def _enforce_boundaries(self):
        """Keep all bodies within workspace."""
        # Robot
        self.robot_pos = np.clip(
            self.robot_pos,
            [self.WORKSPACE_X[0] + self.ROBOT_RADIUS,
             self.WORKSPACE_Y[0] + self.ROBOT_RADIUS],
            [self.WORKSPACE_X[1] - self.ROBOT_RADIUS,
             self.WORKSPACE_Y[1] - self.ROBOT_RADIUS],
        )
        # Zero velocity at boundary
        for d in range(2):
            lo = [self.WORKSPACE_X[0], self.WORKSPACE_Y[0]][d] + self.ROBOT_RADIUS
            hi = [self.WORKSPACE_X[1], self.WORKSPACE_Y[1]][d] - self.ROBOT_RADIUS
            if self.robot_pos[d] <= lo or self.robot_pos[d] >= hi:
                self.robot_vel[d] = 0.0
        
        # Objects
        for i in range(self.n_objects):
            self.object_pos[i] = np.clip(
                self.object_pos[i],
                [self.WORKSPACE_X[0] + self.OBJECT_RADIUS,
                 self.WORKSPACE_Y[0] + self.OBJECT_RADIUS],
                [self.WORKSPACE_X[1] - self.OBJECT_RADIUS,
                 self.WORKSPACE_Y[1] - self.OBJECT_RADIUS],
            )
            for d in range(2):
                lo = [self.WORKSPACE_X[0], self.WORKSPACE_Y[0]][d] + self.OBJECT_RADIUS
                hi = [self.WORKSPACE_X[1], self.WORKSPACE_Y[1]][d] - self.OBJECT_RADIUS
                if self.object_pos[i, d] <= lo or self.object_pos[i, d] >= hi:
                    self.object_vel[i, d] *= -self.RESTITUTION
    
    # ========== Reward ==========
    
    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Compute task reward."""
        if self.task_mode == 'push_to_goals':
            return self._reward_push_to_goals()
        elif self.task_mode == 'sort_by_mass':
            return self._reward_sort_by_mass()
        elif self.task_mode == 'group_by_color':
            return self._reward_group_by_color()
        elif self.task_mode == 'stack':
            return self._reward_stack()
        else:
            raise ValueError(f"Unknown task mode: {self.task_mode}")
    
    def _reward_push_to_goals(self) -> Tuple[float, Dict[str, Any]]:
        """Each object has an assigned goal."""
        distances = []
        successes = []
        
        for i in range(self.n_objects):
            d = np.linalg.norm(self.object_pos[i] - self.goal_positions[i])
            distances.append(d)
            successes.append(d < self.success_threshold)
        
        # Distance reward (negative total distance)
        reward = -sum(distances)
        
        # Per-object success bonus
        reward += sum(successes) * 5.0
        
        # Control cost (encourage efficiency)
        reward -= 0.01 * np.sum(self.robot_vel ** 2)
        
        return reward, {
            'distances': distances,
            'successes': successes,
            'all_success': all(successes),
            'n_success': sum(successes),
            'mean_distance': np.mean(distances),
        }
    
    def _reward_sort_by_mass(self) -> Tuple[float, Dict[str, Any]]:
        """Objects should be sorted left-to-right by increasing mass."""
        # Get mass-sorted order
        mass_order = np.argsort(self.object_masses)
        
        # Desired x positions: evenly spaced from -0.5 to 0.5
        target_x = np.linspace(-0.5, 0.5, self.n_objects)
        target_y = 0.0
        
        distances = []
        for rank, obj_idx in enumerate(mass_order):
            target = np.array([target_x[rank], target_y])
            d = np.linalg.norm(self.object_pos[obj_idx] - target)
            distances.append(d)
        
        reward = -sum(distances)
        all_close = all(d < self.success_threshold for d in distances)
        
        # Update goal_positions for visualization
        for rank, obj_idx in enumerate(mass_order):
            self.goal_positions[obj_idx] = np.array([target_x[rank], target_y])
        
        return reward, {
            'distances': distances,
            'all_success': all_close,
            'mean_distance': np.mean(distances),
            'successes': [d < self.success_threshold for d in distances],
            'n_success': sum(d < self.success_threshold for d in distances),
        }
    
    def _reward_group_by_color(self) -> Tuple[float, Dict[str, Any]]:
        """Group objects by mass category into zones."""
        # Categorize by mass: light (<1.0), medium (1.0-1.5), heavy (>1.5)
        zones = {
            'light': np.array([-0.5, 0.0]),
            'medium': np.array([0.0, 0.0]),
            'heavy': np.array([0.5, 0.0]),
        }
        
        distances = []
        for i in range(self.n_objects):
            m = self.object_masses[i]
            if m < 1.0:
                target = zones['light']
            elif m < 1.5:
                target = zones['medium']
            else:
                target = zones['heavy']
            
            self.goal_positions[i] = target
            d = np.linalg.norm(self.object_pos[i] - target)
            distances.append(d)
        
        reward = -sum(distances)
        all_close = all(d < self.success_threshold * 2 for d in distances)
        
        return reward, {
            'distances': distances,
            'all_success': all_close,
            'mean_distance': np.mean(distances),
            'successes': [d < self.success_threshold * 2 for d in distances],
            'n_success': sum(d < self.success_threshold * 2 for d in distances),
        }
    
    def _reward_stack(self) -> Tuple[float, Dict[str, Any]]:
        """Push all objects to a single target zone."""
        target = np.array([0.5, 0.5])
        
        distances = []
        for i in range(self.n_objects):
            self.goal_positions[i] = target
            d = np.linalg.norm(self.object_pos[i] - target)
            distances.append(d)
        
        reward = -sum(distances)
        all_close = all(d < self.success_threshold for d in distances)
        
        return reward, {
            'distances': distances,
            'all_success': all_close,
            'mean_distance': np.mean(distances),
            'successes': [d < self.success_threshold for d in distances],
            'n_success': sum(d < self.success_threshold for d in distances),
        }
    
    # ========== Goal Generation ==========
    
    def _generate_goals(self):
        """Generate goal positions based on task mode."""
        if self.task_mode == 'push_to_goals':
            self.goal_positions = self._sample_non_overlapping_positions(
                n=self.n_objects,
                min_dist=2.5 * self.OBJECT_RADIUS,
                x_range=(-0.8, 0.8),
                y_range=(-0.8, 0.8),
                avoid_center=False,
            )
        elif self.task_mode == 'sort_by_mass':
            target_x = np.linspace(-0.5, 0.5, self.n_objects)
            mass_order = np.argsort(self.object_masses)
            for rank, obj_idx in enumerate(mass_order):
                self.goal_positions[obj_idx] = np.array([target_x[rank], 0.0])
        elif self.task_mode == 'group_by_color':
            for i in range(self.n_objects):
                m = self.object_masses[i]
                if m < 1.0:
                    self.goal_positions[i] = np.array([-0.5, 0.0])
                elif m < 1.5:
                    self.goal_positions[i] = np.array([0.0, 0.0])
                else:
                    self.goal_positions[i] = np.array([0.5, 0.0])
        elif self.task_mode == 'stack':
            self.goal_positions[:] = np.array([0.5, 0.5])
    
    # ========== Observation & Info ==========
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        obs_parts = [
            self.robot_pos,                          # 2
            self.robot_vel,                          # 2
        ]
        for i in range(self.n_objects):
            obs_parts.extend([
                self.object_pos[i],                  # 2
                self.object_vel[i],                  # 2
                np.array([self.object_masses[i]]),   # 1
            ])
        for i in range(self.n_objects):
            obs_parts.append(self.goal_positions[i]) # 2
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return episode info."""
        distances = [
            np.linalg.norm(self.object_pos[i] - self.goal_positions[i])
            for i in range(self.n_objects)
        ]
        return {
            'step': self.current_step,
            'n_objects': self.n_objects,
            'task_mode': self.task_mode,
            'mean_distance': np.mean(distances),
            'object_masses': self.object_masses.copy(),
            'success': all(d < self.success_threshold for d in distances),
        }
    
    # ========== Graph Construction (for Dynami-CAL) ==========
    
    def build_graph(self) -> Dict[str, Any]:
        """
        Construct PyG-compatible graph from current state.
        
        Returns dict with:
            - x: Node features [N_nodes, 6] (pos_x, pos_y, vel_x, vel_y, mass, is_robot)
            - pos: Node positions [N_nodes, 2]
            - edge_index: Fully connected edges [2, N_edges]
            - edge_attr: Edge features [N_edges, 5] (rel_pos_x, rel_pos_y, dist, rel_vel_x, rel_vel_y)
        
        Requires torch. Returns numpy arrays if torch is not available.
        """
        import torch
        # ---- Node features ----
        # Node 0: robot
        node_features = [np.concatenate([
            self.robot_pos, self.robot_vel,
            [self.robot_mass], [1.0]  # is_robot=1
        ])]
        positions = [self.robot_pos.copy()]
        velocities = [self.robot_vel.copy()]
        
        # Nodes 1..N: objects
        for i in range(self.n_objects):
            node_features.append(np.concatenate([
                self.object_pos[i], self.object_vel[i],
                [self.object_masses[i]], [0.0]  # is_robot=0
            ]))
            positions.append(self.object_pos[i].copy())
            velocities.append(self.object_vel[i].copy())
        
        x = np.stack(node_features)  # [N, 6]
        pos = np.stack(positions)     # [N, 2]
        vel = np.stack(velocities)    # [N, 2]
        
        # ---- Fully connected edges (no self-loops) ----
        N = self.n_nodes
        src, tgt = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    tgt.append(j)
        edge_index = np.array([src, tgt])  # [2, E]
        
        # ---- Edge features ----
        edge_attrs = []
        for e in range(edge_index.shape[1]):
            i, j = edge_index[0, e], edge_index[1, e]
            rel_pos = pos[j] - pos[i]
            dist = np.linalg.norm(rel_pos)
            rel_vel = vel[j] - vel[i]
            edge_attrs.append(np.concatenate([rel_pos, [dist], rel_vel]))
        edge_attr = np.stack(edge_attrs)  # [E, 5]
        
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'pos': torch.tensor(pos, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
        }
    
    def build_graph_3d(self) -> Dict[str, Any]:
        """
        Construct 3D-compatible graph for physics_core modules.
        
        Pads 2D positions/velocities to 3D (z=0) for compatibility with
        EdgeFrame and DynamicalGNN which expect 3D inputs.
        
        Returns dict with:
            - positions: [N, 3]
            - velocities: [N, 3]
            - edge_index: [2, E]
            - masses: [N]
        
        Requires torch.
        """
        import torch
        N = self.n_nodes
        
        # Pad to 3D
        positions_3d = np.zeros((N, 3))
        velocities_3d = np.zeros((N, 3))
        masses = np.zeros(N)
        
        # Robot
        positions_3d[0, :2] = self.robot_pos
        velocities_3d[0, :2] = self.robot_vel
        masses[0] = self.robot_mass
        
        # Objects
        for i in range(self.n_objects):
            positions_3d[i + 1, :2] = self.object_pos[i]
            velocities_3d[i + 1, :2] = self.object_vel[i]
            masses[i + 1] = self.object_masses[i]
        
        # Fully connected edges
        src, tgt = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    tgt.append(j)
        edge_index = np.array([src, tgt])
        
        return {
            'positions': torch.tensor(positions_3d, dtype=torch.float32),
            'velocities': torch.tensor(velocities_3d, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'masses': torch.tensor(masses, dtype=torch.float32),
        }
    
    # ========== Utilities ==========
    
    def _sample_non_overlapping_positions(
        self,
        n: int,
        min_dist: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        avoid_center: bool = False,
        max_attempts: int = 1000,
    ) -> np.ndarray:
        """Sample n non-overlapping positions."""
        positions = []
        
        for _ in range(n):
            for attempt in range(max_attempts):
                pos = np.array([
                    self.np_random.uniform(x_range[0], x_range[1]),
                    self.np_random.uniform(y_range[0], y_range[1]),
                ])
                
                if avoid_center and np.linalg.norm(pos) < 0.15:
                    continue
                
                # Check overlap with existing positions
                valid = True
                for existing in positions:
                    if np.linalg.norm(pos - existing) < min_dist:
                        valid = False
                        break
                
                if valid:
                    positions.append(pos)
                    break
            else:
                # Fallback: place randomly (may overlap)
                positions.append(np.array([
                    self.np_random.uniform(x_range[0], x_range[1]),
                    self.np_random.uniform(y_range[0], y_range[1]),
                ]))
        
        return np.stack(positions)
    
    def render(self):
        """Render the environment (matplotlib-based for portability)."""
        if self.render_mode is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(self.WORKSPACE_X[0] - 0.1, self.WORKSPACE_X[1] + 0.1)
        ax.set_ylim(self.WORKSPACE_Y[0] - 0.1, self.WORKSPACE_Y[1] + 0.1)
        ax.set_aspect('equal')
        ax.set_title(f'MultiObjectPush (step {self.current_step})')
        
        # Draw workspace boundary
        ax.add_patch(patches.Rectangle(
            (self.WORKSPACE_X[0], self.WORKSPACE_Y[0]),
            self.WORKSPACE_X[1] - self.WORKSPACE_X[0],
            self.WORKSPACE_Y[1] - self.WORKSPACE_Y[0],
            fill=False, edgecolor='black', linewidth=2,
        ))
        
        # Draw goal positions
        for i in range(self.n_objects):
            ax.add_patch(patches.Circle(
                self.goal_positions[i], self.success_threshold,
                fill=True, alpha=0.2, facecolor='green', edgecolor='green',
            ))
            ax.annotate(f'G{i}', self.goal_positions[i],
                       ha='center', va='center', fontsize=8, color='green')
        
        # Draw objects (size proportional to mass)
        cmap = plt.cm.RdYlBu
        mass_norm = (self.object_masses - self.mass_range[0]) / (
            self.mass_range[1] - self.mass_range[0] + 1e-6)
        
        for i in range(self.n_objects):
            color = cmap(mass_norm[i])
            r = self.OBJECT_RADIUS * (0.8 + 0.4 * mass_norm[i])
            ax.add_patch(patches.Circle(
                self.object_pos[i], r,
                fill=True, facecolor=color, edgecolor='black',
            ))
            ax.annotate(f'{i}\n{self.object_masses[i]:.1f}',
                       self.object_pos[i], ha='center', va='center', fontsize=7)
        
        # Draw robot
        ax.add_patch(patches.Circle(
            self.robot_pos, self.ROBOT_RADIUS,
            fill=True, facecolor='blue', edgecolor='darkblue',
        ))
        ax.annotate('R', self.robot_pos, ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        
        # Draw robot velocity arrow
        if np.linalg.norm(self.robot_vel) > 0.01:
            ax.arrow(self.robot_pos[0], self.robot_pos[1],
                    self.robot_vel[0] * 0.2, self.robot_vel[1] * 0.2,
                    head_width=0.02, head_length=0.01, fc='blue', ec='blue')
        
        plt.tight_layout()
        
        if self.render_mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        elif self.render_mode == 'human':
            plt.show(block=False)
            plt.pause(0.01)
            plt.close(fig)
            return None
    
    def close(self):
        """Clean up resources."""
        pass


# ========== Factory Functions ==========

def make_multi_push_env(
    n_objects: int = 3,
    task_mode: str = 'push_to_goals',
    **kwargs,
):
    """Factory function for creating MultiObjectPushEnv."""
    def _init():
        return MultiObjectPushEnv(
            n_objects=n_objects,
            task_mode=task_mode,
            **kwargs,
        )
    return _init


# ========== Quick Test ==========

if __name__ == '__main__':
    print("Testing MultiObjectPushEnv...")
    
    for n_obj in [3, 4, 5]:
        for mode in ['push_to_goals', 'sort_by_mass', 'group_by_color', 'stack']:
            env = MultiObjectPushEnv(n_objects=n_obj, task_mode=mode)
            obs, info = env.reset(seed=42)
            
            print(f"\n  n_objects={n_obj}, mode={mode}")
            print(f"  obs_dim={obs.shape[0]}, expected={env.obs_dim}")
            assert obs.shape[0] == env.obs_dim
            
            # Run a few steps
            total_reward = 0
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"  steps={step+1}, reward={total_reward:.2f}, "
                  f"mean_dist={info['mean_distance']:.3f}")
            
            # Test graph construction
            graph = env.build_graph()
            expected_nodes = n_obj + 1
            expected_edges = expected_nodes * (expected_nodes - 1)
            assert graph['x'].shape == (expected_nodes, 6), \
                f"x shape: {graph['x'].shape}"
            assert graph['edge_index'].shape == (2, expected_edges), \
                f"edge_index shape: {graph['edge_index'].shape}"
            assert graph['edge_attr'].shape == (expected_edges, 5)
            print(f"  graph: {expected_nodes} nodes, {expected_edges} edges ✅")
            
            # Test 3D graph
            graph_3d = env.build_graph_3d()
            assert graph_3d['positions'].shape == (expected_nodes, 3)
            assert graph_3d['velocities'].shape == (expected_nodes, 3)
            print(f"  3D graph: positions {graph_3d['positions'].shape} ✅")
            
            env.close()
    
    print("\n✅ All MultiObjectPushEnv tests passed!")
