"""
PushBox Environment for Physics-Informed Robotics Research
Week 1: 2-DOF Robot Arm + Box Pushing Task
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os


class PushBoxEnv(gym.Env):
    """
    2-DOF robot arm must push a box to a goal position.
    
    Observation Space:
        - Joint positions (2)
        - Joint velocities (2)
        - End-effector position (3)
        - Box position (3)
        - Box velocity (3)
        - Goal position (3)
        Total: 16-dimensional
    
    Action Space:
        - Joint torques (2): [-10, 10] Nm
    
    Reward:
        - Dense: -distance_to_goal
        - Sparse: +100 if success (box within 0.1m of goal)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, box_mass=1.0):
        super().__init__()
        
        # Load MuJoCo model
        xml_path = os.path.join(
            os.path.dirname(__file__), 
            "assets", 
            "push_box.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Configurable box mass (for OOD testing)
        self.box_mass = box_mass
        self._set_box_mass(box_mass)
        
        # Spaces
        self.action_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,), 
            dtype=np.float32
        )
        
        # Goal position (fixed for training, can randomize later)
        self.goal_pos = np.array([1.0, 0.5, 0.05])
        
        # Episode tracking
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Success threshold
        self.success_threshold = 0.1  # meters
        
        # Render mode
        self.render_mode = render_mode
        self.viewer = None
        
    def _set_box_mass(self, mass):
        """Set the box mass (for OOD generalization testing)"""
        # Find box body index
        box_body_id = mujoco.mj_name2id(
            self.model, 
            mujoco.mjtObj.mjOBJ_BODY, 
            "box"
        )
        # Set mass
        self.model.body_mass[box_body_id] = mass
        
    def set_box_mass(self, mass):
        """Public interface to change box mass"""
        self.box_mass = mass
        self._set_box_mass(mass)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial positions slightly for robustness
        if seed is not None:
            np.random.seed(seed)
        
        # Robot arm: random initial pose
        self.data.qpos[0] = np.random.uniform(-0.5, 0.5)  # shoulder
        self.data.qpos[1] = np.random.uniform(-0.5, 0.5)  # elbow
        
        # Box: start at random position near robot
        self.data.qpos[2] = np.random.uniform(0.4, 0.6)  # x
        self.data.qpos[3] = np.random.uniform(-0.2, 0.2)  # y
        self.data.qpos[4] = 0.05  # z (on ground)
        # Quaternion for box orientation (no rotation)
        self.data.qpos[5:9] = [1, 0, 0, 0]
        
        # Reset velocities
        self.data.qvel[:] = 0.0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        """Get observation vector"""
        # Joint positions and velocities
        joint_pos = self.data.qpos[:2].copy()
        joint_vel = self.data.qvel[:2].copy()
        
        # End-effector position
        ee_site_id = mujoco.mj_name2id(
            self.model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            "endeffector"
        )
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        
        # Box position and velocity
        box_pos = self.data.qpos[2:5].copy()
        box_vel = self.data.qvel[2:5].copy()
        
        # Goal position (relative to box for better learning)
        goal_pos = self.goal_pos.copy()
        
        obs = np.concatenate([
            joint_pos,      # 2
            joint_vel,      # 2
            ee_pos,         # 3
            box_pos,        # 3
            box_vel,        # 3
            goal_pos        # 3
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """Get episode info"""
        box_pos = self.data.qpos[2:5]
        distance_to_goal = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
        success = distance_to_goal < self.success_threshold
        
        return {
            'distance_to_goal': distance_to_goal,
            'success': success,
            'box_mass': self.box_mass,
            'timestep': self.current_step
        }
    
    def step(self, action):
        """Execute one timestep"""
        # Apply action (joint torques)
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward
        box_pos = self.data.qpos[2:5]
        distance_to_goal = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
        
        # Dense reward: negative distance
        reward = -distance_to_goal
        
        # Sparse bonus for success
        success = distance_to_goal < self.success_threshold
        if success:
            reward += 100.0
        
        # Check termination
        self.current_step += 1
        terminated = success
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        if self.viewer is None and self.render_mode == "human":
            # Initialize viewer
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.model, 
                self.data
            )
        
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        
        # For rgb_array, return rendered frame
        # (requires offscreen rendering setup)
        return None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Factory function for vectorized environments
def make_push_box_env(box_mass=1.0):
    """Factory function for creating PushBox environments"""
    def _init():
        return PushBoxEnv(box_mass=box_mass)
    return _init
