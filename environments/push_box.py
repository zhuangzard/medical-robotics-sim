"""
PushBox Environment - Simple rigid body manipulation task for Dynami-CAL validation

Task: Robot arm pushes a box to a goal position
Physics: MuJoCo for rigid body dynamics + contact physics
Purpose: Validate 12.5x sample efficiency and OOD generalization (paper Section 4.1)

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os
from typing import Optional, Tuple, Dict, Any
import time


class PushBoxEnv(gym.Env):
    """
    Simple rigid body manipulation task for physics learning validation
    
    State Space:
        - robot_joint_pos (2): shoulder, elbow angles [rad]
        - robot_joint_vel (2): joint velocities [rad/s]
        - box_pos (2): x, y position [m]
        - box_vel (2): x, y velocity [m/s]
        - goal_pos (2): x, y target position [m]
        Total: 10 dimensions
    
    Action Space:
        - joint_torques (2): shoulder, elbow torques [Nm]
        Range: [-10, 10] Nm
    
    Reward:
        - distance_reward: -||box_pos - goal_pos||
        - contact_bonus: +0.1 when robot contacts box
        - control_cost: -0.01 * ||action||^2
        - success_bonus: +10 when box reaches goal
    
    Physics Parameters:
        - Box mass: 0.5-2.0 kg (default 1.0 kg for training)
        - Friction coefficient: μ = 0.3
        - Contact model: Spring-damper (MuJoCo default)
        - Timestep: 0.002 s
    
    Success Criteria:
        Box within 0.05m of goal for 10 consecutive timesteps
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        box_mass: float = 1.0,
        friction_coef: float = 0.3,
        success_threshold: float = 0.05,
        success_duration: int = 10
    ):
        """
        Initialize PushBox environment
        
        Args:
            render_mode: 'human' for visualization, 'rgb_array' for video
            max_episode_steps: Maximum steps per episode
            box_mass: Mass of the box [kg]
            friction_coef: Friction coefficient
            success_threshold: Distance threshold for success [m]
            success_duration: Number of steps to maintain success
        """
        super().__init__()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.success_duration = success_duration
        self.render_mode = render_mode
        
        # Physics parameters
        self.box_mass = box_mass
        self.friction_coef = friction_coef
        
        # Load MuJoCo model
        xml_path = os.path.join(
            os.path.dirname(__file__),
            'assets',
            'push_box.xml'
        )
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set physics parameters
        self._set_physics_params()
        
        # Rendering
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        # Define spaces
        # State: [joint_pos(2), joint_vel(2), box_pos(2), box_vel(2), goal_pos(2)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        # Action: [shoulder_torque, elbow_torque]
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.success_counter = 0
        self.goal_position = np.array([1.0, 0.5])  # Will be randomized
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'contacts': []
        }
        
    def _set_physics_params(self):
        """Set physics parameters in MuJoCo model"""
        # Find box body
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        
        # Set mass (modify inertia matrix)
        # MuJoCo stores mass in body_mass array
        if box_id >= 0:
            # Mass is set via geom mass which contributes to body
            # We'll modify it during reset for OOD testing
            pass
        
        # Set friction
        if box_geom_id >= 0:
            self.model.geom_friction[box_geom_id] = [
                self.friction_coef,  # sliding friction
                0.005,               # torsional friction
                0.0001               # rolling friction
            ]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options (e.g., 'box_mass', 'goal_pos')
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial positions
        if options and 'box_pos' in options:
            box_pos = options['box_pos']
        else:
            # Random box position in [0.3, 0.7] x [-0.3, 0.3]
            box_pos = np.array([
                self.np_random.uniform(0.3, 0.7),
                self.np_random.uniform(-0.3, 0.3)
            ])
        
        if options and 'goal_pos' in options:
            self.goal_position = options['goal_pos']
        else:
            # Random goal position in [0.8, 1.2] x [-0.5, 0.5]
            self.goal_position = np.array([
                self.np_random.uniform(0.8, 1.2),
                self.np_random.uniform(-0.5, 0.5)
            ])
        
        # Set box mass if specified (for OOD testing)
        if options and 'box_mass' in options:
            self.set_box_mass(options['box_mass'])
        
        # Set initial state
        # Box position (freejoint has 7 DOFs: 3 pos + 4 quat)
        box_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        ]
        self.data.qpos[box_qpos_addr:box_qpos_addr+3] = [box_pos[0], box_pos[1], 0.05]
        self.data.qpos[box_qpos_addr+3:box_qpos_addr+7] = [1, 0, 0, 0]  # identity quaternion
        
        # Random arm configuration
        self.data.qpos[0] = self.np_random.uniform(-np.pi/4, np.pi/4)  # shoulder
        self.data.qpos[1] = self.np_random.uniform(-np.pi/4, np.pi/4)  # elbow
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        # Update goal marker position
        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.model.site_pos[goal_site_id] = [self.goal_position[0], self.goal_position[1], 0.05]
        
        # Forward dynamics to compute dependent quantities
        mujoco.mj_forward(self.model, self.data)
        
        # Reset counters
        self.current_step = 0
        self.success_counter = 0
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'contacts': []
        }
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Joint torques [shoulder, elbow]
        
        Returns:
            observation: Current state
            reward: Step reward
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply control
        self.data.ctrl[:] = action
        
        # Step simulation (multiple substeps for stability)
        substeps = 10
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_obs()
        
        # Compute reward
        reward, reward_info = self._compute_reward(action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Success: box at goal for required duration
        box_pos = self._get_box_position()
        dist_to_goal = np.linalg.norm(box_pos[:2] - self.goal_position)  # Only x, y
        
        if dist_to_goal < self.success_threshold:
            self.success_counter += 1
            if self.success_counter >= self.success_duration:
                terminated = True
                reward += 10.0  # Success bonus
        else:
            self.success_counter = 0
        
        # Failure: box falls off table (z < 0)
        if box_pos[2] < 0:
            terminated = True
            reward -= 5.0  # Failure penalty
        
        # Max steps
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Store episode data
        self.episode_data['states'].append(observation)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        self.episode_data['contacts'].append(reward_info['contact'])
        
        # Get info
        info = self._get_info()
        info.update(reward_info)
        info['success'] = self.success_counter >= self.success_duration
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            State vector [10]: [joint_pos, joint_vel, box_pos, box_vel, goal_pos]
        """
        # Joint positions and velocities
        joint_pos = self.data.qpos[:2]  # shoulder, elbow
        joint_vel = self.data.qvel[:2]
        
        # Box position and velocity (freejoint)
        box_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        ]
        box_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        ]
        
        box_pos = self.data.qpos[box_qpos_addr:box_qpos_addr+2]  # x, y only
        box_vel = self.data.qvel[box_qvel_addr:box_qvel_addr+2]  # vx, vy only
        
        # Concatenate observation
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            box_pos,
            box_vel,
            self.goal_position
        ])
        
        return obs.astype(np.float32)
    
    def _get_box_position(self) -> np.ndarray:
        """Get 3D position of box center"""
        box_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        ]
        return self.data.qpos[box_qpos_addr:box_qpos_addr+3].copy()
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute step reward
        
        Returns:
            reward: Total reward
            info: Breakdown of reward components
        """
        # Distance to goal
        box_pos = self._get_box_position()[:2]  # x, y only
        dist_to_goal = np.linalg.norm(box_pos - self.goal_position)
        distance_reward = -dist_to_goal
        
        # Contact bonus (check if robot touches box)
        contact = self._check_contact()
        contact_bonus = 0.1 if contact else 0.0
        
        # Control cost
        control_cost = -0.01 * np.sum(action ** 2)
        
        # Total reward
        reward = distance_reward + contact_bonus + control_cost
        
        info = {
            'distance_reward': distance_reward,
            'contact_bonus': contact_bonus,
            'control_cost': control_cost,
            'distance_to_goal': dist_to_goal,
            'contact': contact
        }
        
        return reward, info
    
    def _check_contact(self) -> bool:
        """Check if robot end-effector is in contact with box"""
        # Get contact information from MuJoCo
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Use correct MuJoCo API to get geometry names
            geom1_name = mujoco.mj_id2name(
                self.model, 
                mujoco.mjtObj.mjOBJ_GEOM, 
                contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                self.model, 
                mujoco.mjtObj.mjOBJ_GEOM, 
                contact.geom2
            )
            
            # Check if contact involves arm and box
            arm_geoms = ['upper_arm_geom', 'forearm_geom']
            box_geom = 'box_geom'
            
            if (geom1_name in arm_geoms and geom2_name == box_geom) or \
               (geom2_name in arm_geoms and geom1_name == box_geom):
                return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        box_pos = self._get_box_position()
        
        return {
            'step': self.current_step,
            'box_position': box_pos,
            'goal_position': self.goal_position,
            'success_counter': self.success_counter,
            'box_mass': self.box_mass
        }
    
    def set_box_mass(self, mass: float):
        """
        Set box mass (for OOD testing)
        
        Args:
            mass: New mass [kg]
        """
        self.box_mass = mass
        
        # Update geom mass
        box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        if box_geom_id >= 0:
            # Get geom size (box half-extents)
            size = self.model.geom_size[box_geom_id]
            volume = 8 * size[0] * size[1] * size[2]
            density = mass / volume
            
            # Set mass directly
            # Note: This is a workaround; proper way is to set density and recompile
            # For quick testing, we modify the inertia
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            if box_body_id >= 0:
                # Scale inertia proportionally
                old_mass = self.model.body_mass[box_body_id]
                scale = mass / max(old_mass, 0.001)
                
                self.model.body_mass[box_body_id] = mass
                self.model.body_inertia[box_body_id] *= scale
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
            
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def get_episode_data(self) -> Dict[str, np.ndarray]:
        """
        Get collected episode data
        
        Returns:
            Dictionary with states, actions, rewards, contacts
        """
        return {
            'states': np.array(self.episode_data['states']),
            'actions': np.array(self.episode_data['actions']),
            'rewards': np.array(self.episode_data['rewards']),
            'contacts': np.array(self.episode_data['contacts'])
        }


# Factory function for easy creation
def make_push_box_env(
    render_mode: Optional[str] = None,
    **kwargs
) -> PushBoxEnv:
    """
    Factory function to create PushBoxEnv
    
    Args:
        render_mode: Render mode
        **kwargs: Additional environment parameters
    
    Returns:
        PushBoxEnv instance
    """
    return PushBoxEnv(render_mode=render_mode, **kwargs)
