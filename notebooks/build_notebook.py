#!/usr/bin/env python3
"""Build the Week 1 training notebook programmatically"""
import json, os

cells = []

def md(text):
    cells.append({"cell_type":"markdown","metadata":{},"source": [l+"\n" for l in text.strip().split("\n")]})

def code(text):
    lines = text.strip().split("\n")
    src = [l+"\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src})

# ============ CELL 1: Title ============
md("""# 🤖 Physics-Informed Robotics — Week 1 Training

**Target**: ICRA 2027 / CoRL 2026  
**Task**: PushBox — 2-DOF robot arm pushes box to goal  
**Methods**: Pure PPO (200K) vs GNS (80K) vs PhysRobot (16K)  
**Runtime**: ~2-4 hours on V100/A100

---""")

# ============ CELL 2: GPU + Deps ============
code("""# Cell 1: GPU Check & Dependencies
!nvidia-smi
!pip install mujoco gymnasium stable-baselines3 torch-geometric -q

import torch
print(f"\\n✅ PyTorch {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU — training will be slow!")""")

# ============ CELL 3: Drive Mount ============
code("""# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/medical-robotics-sim'
for d in ['models', 'results', 'logs']:
    os.makedirs(f'{SAVE_DIR}/{d}', exist_ok=True)
print(f"✅ Save dir: {SAVE_DIR}")""")

# ============ CELL 4: MuJoCo XML + Environment ============
code("""# Cell 3: Environment — PushBoxEnv (16-dim observation)
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os, tempfile

# Inline MuJoCo XML
PUSH_BOX_XML = '''<mujoco model="push_box">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture builtin="checker" height="100" name="texplane" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.3" shininess="0.5" specular="0.5" texrepeat="3 3" texture="texplane"/>
  </asset>
  <default>
    <joint armature="0.01" damping="0.1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" friction="0.3 0.005 0.0001" margin="0.001" rgba="0.8 0.6 0.4 1"/>
  </default>
  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" material="MatPlane"/>
    <body name="arm_base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.05 0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="upper_arm" pos="0 0 0.02">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-180 180" damping="0.5"/>
        <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" rgba="0.5 0.5 0.8 1"/>
        <body name="forearm" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 0 1" range="-180 180" damping="0.5"/>
          <geom name="forearm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" rgba="0.5 0.5 0.8 1"/>
          <site name="endeffector" pos="0.3 0 0" size="0.02" rgba="1 0.5 0 0.8"/>
        </body>
      </body>
    </body>
    <body name="box" pos="0.5 0 0.05">
      <freejoint name="box_freejoint"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="1.0" rgba="0.2 0.8 0.2 1" friction="0.3 0.005 0.0001"/>
      <site name="box_center" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>
    </body>
    <site name="goal" pos="1.0 0.5 0.05" size="0.06" rgba="1 0 0 0.4" type="sphere"/>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1.0" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="elbow_motor" joint="elbow" gear="1.0" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>'''

# Write XML to temp file
XML_PATH = '/tmp/push_box.xml'
with open(XML_PATH, 'w') as f:
    f.write(PUSH_BOX_XML)

class PushBoxEnv(gym.Env):
    """16-dim obs: [joint_pos(2), joint_vel(2), ee_pos(3), box_pos(3), box_vel(3), goal_pos(3)]"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, box_mass=1.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.box_mass = box_mass
        self._set_box_mass(box_mass)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.goal_pos = np.array([1.0, 0.5, 0.05])
        self.max_episode_steps = 500
        self.current_step = 0
        self.success_threshold = 0.1
        self.render_mode = render_mode

    def _set_box_mass(self, mass):
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        if box_body_id >= 0:
            self.model.body_mass[box_body_id] = mass

    def set_box_mass(self, mass):
        self.box_mass = mass
        self._set_box_mass(mass)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.random.uniform(-0.5, 0.5)
        self.data.qpos[1] = np.random.uniform(-0.5, 0.5)
        self.data.qpos[2] = np.random.uniform(0.4, 0.6)
        self.data.qpos[3] = np.random.uniform(-0.2, 0.2)
        self.data.qpos[4] = 0.05
        self.data.qpos[5:9] = [1, 0, 0, 0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        joint_pos = self.data.qpos[:2].copy()
        joint_vel = self.data.qvel[:2].copy()
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        box_pos = self.data.qpos[2:5].copy()
        box_vel = self.data.qvel[2:5].copy()
        return np.concatenate([joint_pos, joint_vel, ee_pos, box_pos, box_vel, self.goal_pos]).astype(np.float32)

    def _get_info(self):
        box_pos = self.data.qpos[2:5]
        dist = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
        return {'distance_to_goal': dist, 'success': dist < self.success_threshold,
                'box_mass': self.box_mass, 'timestep': self.current_step}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        box_pos = self.data.qpos[2:5]
        dist = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
        reward = -dist
        success = dist < self.success_threshold
        if success:
            reward += 100.0
        self.current_step += 1
        terminated = success
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

def make_push_box_env(box_mass=1.0):
    def _init():
        return PushBoxEnv(box_mass=box_mass)
    return _init

print("✅ PushBoxEnv defined (16-dim obs)")""")

# ============ CELL 5: Test Env ============
code("""# Cell 4: Test Environment
env = PushBoxEnv()
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (16,)
print(f"Observation: {obs}")
for i in range(10):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
print(f"Reward: {r:.4f}, Distance: {info['distance_to_goal']:.4f}")
env.close()
print("✅ Environment works!")""")

# ============ CELL 6: Baselines ============
code(r"""# Cell 5: All 3 Methods — PPO, GNS, PhysRobot
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ torch_geometric not available, GNS/PhysRobot will use simplified versions")

# ===== Tracking Callback =====
class TrainingCallback(BaseCallback):
    def __init__(self, name="", eval_env_fn=None, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.name = name
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.episode_count = 0
        self.success_count = 0
        self.first_success_ep = None
        self.eval_history = []

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                info = infos[i] if i < len(infos) else {}
                if info.get('success', False):
                    self.success_count += 1
                    if self.first_success_ep is None:
                        self.first_success_ep = self.episode_count
                        print(f"\n🎉 [{self.name}] First success at episode {self.episode_count}!")
        if self.n_calls % self.eval_freq == 0 and self.eval_env_fn:
            sr = self._quick_eval()
            self.eval_history.append({'step': self.n_calls, 'ep': self.episode_count, 'sr': sr})
            print(f"  📊 [{self.name}] Step {self.n_calls}: success_rate={sr:.1%}")
        return True

    def _quick_eval(self, n=20):
        env = DummyVecEnv([self.eval_env_fn])
        succ = 0
        for _ in range(n):
            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                done = dones[0]
            if infos[0].get('success', False):
                succ += 1
        env.close()
        return succ / n

# ===== Method 1: Pure PPO =====
class PurePPOAgent:
    def __init__(self, env, lr=3e-4, verbose=0):
        self.model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=2048,
                         batch_size=64, n_epochs=10, gamma=0.99, verbose=verbose)
    def train(self, steps, callback=None):
        self.model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a
    def save(self, path): self.model.save(path)

# ===== GNS Features Extractor (simplified if no PyG) =====
if HAS_PYG:
    class GraphNetLayer(MessagePassing):
        def __init__(self, node_dim, edge_dim, hidden=128):
            super().__init__(aggr='add')
            self.edge_mlp = nn.Sequential(nn.Linear(2*node_dim+edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, edge_dim))
            self.node_mlp = nn.Sequential(nn.Linear(node_dim+edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, node_dim))
        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        def message(self, x_i, x_j, edge_attr):
            return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], -1))
        def update(self, aggr_out, x):
            return self.node_mlp(torch.cat([x, aggr_out], -1))

    class GNSFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            self.node_enc = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128))
            self.edge_enc = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128))
            self.gn = GraphNetLayer(128, 128)
            self.decoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
            self.proj = nn.Sequential(nn.Linear(3+16, features_dim), nn.ReLU())

        def forward(self, obs):
            bs = obs.shape[0]
            graphs = []
            for i in range(bs):
                o = obs[i]
                ee_pos, box_pos, box_vel = o[4:7], o[7:10], o[10:13]
                x = torch.stack([torch.cat([torch.zeros(3,device=obs.device), ee_pos]),
                                 torch.cat([box_vel, box_pos])])
                ei = torch.tensor([[0],[1]], dtype=torch.long, device=obs.device)
                rp = box_pos - ee_pos
                ea = torch.cat([rp, torch.norm(rp).unsqueeze(0)]).unsqueeze(0)
                graphs.append(Data(x=x, edge_index=ei, edge_attr=ea))
            batch = Batch.from_data_list(graphs)
            h = self.node_enc(batch.x)
            ea = self.edge_enc(batch.edge_attr)
            h = h + self.gn(h, batch.edge_index, ea)
            acc = self.decoder(h)
            box_acc = acc[1::2]
            return self.proj(torch.cat([box_acc, obs], -1))
else:
    class GNSFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            self.net = nn.Sequential(nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, features_dim))
        def forward(self, obs):
            return self.net(obs)

class GNSAgent:
    def __init__(self, env, lr=3e-4, verbose=0):
        pk = dict(features_extractor_class=GNSFeaturesExtractor, features_extractor_kwargs=dict(features_dim=128))
        self.model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=2048,
                         batch_size=64, n_epochs=10, gamma=0.99, policy_kwargs=pk, verbose=verbose)
    def train(self, steps, callback=None):
        self.model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a
    def save(self, path): self.model.save(path)

# ===== PhysRobot Features Extractor =====
if HAS_PYG:
    class DynamiCALLayer(MessagePassing):
        def __init__(self, node_dim, hidden=128):
            super().__init__(aggr='add')
            self.scalar_mlp = nn.Sequential(nn.Linear(2*node_dim+3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.vector_mlp = nn.Sequential(nn.Linear(2*node_dim+3, hidden), nn.ReLU(), nn.Linear(hidden, 2))
            self.node_update = nn.Sequential(nn.Linear(node_dim+3, hidden), nn.ReLU(), nn.Linear(hidden, node_dim))
        def forward(self, x, edge_index, pos):
            row, col = edge_index
            pos_i, pos_j = pos[row], pos[col]
            rel = pos_j - pos_i
            x_i, x_j = x[row], x[col]
            inp = torch.cat([x_i, x_j, rel], -1)
            fs = self.scalar_mlp(inp)
            fv = self.vector_mlp(inp)
            d = torch.norm(rel, dim=-1, keepdim=True) + 1e-6
            e1 = rel / d
            up = torch.tensor([0.,0.,1.], device=e1.device).unsqueeze(0).expand_as(e1)
            e2 = torch.cross(e1, up); e2 = e2 / (torch.norm(e2,-1,True)+1e-6)
            e3 = torch.cross(e1, e2)
            force = fs*e1 + fv[:,0:1]*e2 + fv[:,1:2]*e3
            return self.propagate(edge_index, force=force, x=x)
        def message(self, force): return force
        def update(self, aggr_out, x):
            return self.node_update(torch.cat([x, aggr_out], -1))

    class PhysRobotFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            self.enc = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128))
            self.layers = nn.ModuleList([DynamiCALLayer(128) for _ in range(3)])
            self.dec = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
            self.policy = nn.Sequential(nn.Linear(observation_space.shape[0], 128), nn.ReLU(), nn.Linear(128, features_dim))
            self.fuse = nn.Sequential(nn.Linear(features_dim+3, features_dim), nn.ReLU())

        def forward(self, obs):
            pf = self.policy(obs)
            bs = obs.shape[0]
            graphs = []
            for i in range(bs):
                o = obs[i]
                ee_pos, box_pos, box_vel = o[4:7], o[7:10], o[10:13]
                positions = torch.stack([ee_pos, box_pos])
                nf = torch.stack([torch.cat([torch.zeros(3,device=obs.device), ee_pos]),
                                  torch.cat([box_vel, box_pos])])
                ei = torch.tensor([[0,1],[1,0]], dtype=torch.long, device=obs.device)
                graphs.append(Data(x=nf, pos=positions, edge_index=ei))
            batch = Batch.from_data_list(graphs)
            h = self.enc(batch.x)
            for layer in self.layers:
                h = h + layer(h, batch.edge_index, batch.pos)
            acc = self.dec(h)
            box_acc = acc[1::2]
            return self.fuse(torch.cat([pf, box_acc], -1))
else:
    class PhysRobotFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            self.net = nn.Sequential(nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, features_dim))
        def forward(self, obs): return self.net(obs)

class PhysRobotAgent:
    def __init__(self, env, lr=3e-4, verbose=0):
        pk = dict(features_extractor_class=PhysRobotFeaturesExtractor, features_extractor_kwargs=dict(features_dim=128))
        self.model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=2048,
                         batch_size=64, n_epochs=10, gamma=0.99, policy_kwargs=pk, verbose=verbose)
    def train(self, steps, callback=None):
        self.model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a
    def save(self, path): self.model.save(path)

print(f"✅ All agents defined (PyG: {HAS_PYG})")""")

# ============ CELL 7: Training ============
code(r"""# Cell 6: Train All 3 Methods
import time, json, traceback

results = {}
agents = {}

def evaluate_agent(agent, env_fn, n_episodes=100):
    env = DummyVecEnv([env_fn])
    rewards, successes = [], 0
    for _ in range(n_episodes):
        obs = env.reset(); done = False; ep_r = 0
        while not done:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            ep_r += r[0]; done = dones[0]
        rewards.append(ep_r)
        if infos[0].get('success', False): successes += 1
    env.close()
    return {'mean_reward': float(np.mean(rewards)), 'std_reward': float(np.std(rewards)),
            'success_rate': successes/n_episodes}

env_fn = make_push_box_env(box_mass=1.0)

# ===== 1. Pure PPO (200K steps) =====
print("="*60)
print("🚀 Training 1/3: Pure PPO (200K steps)")
print("="*60)
try:
    t0 = time.time()
    train_env = DummyVecEnv([make_push_box_env(1.0) for _ in range(4)])
    agent_ppo = PurePPOAgent(train_env)
    cb_ppo = TrainingCallback("PPO", eval_env_fn=env_fn, eval_freq=20000)
    agent_ppo.train(200_000, callback=cb_ppo)
    train_env.close()
    eval_ppo = evaluate_agent(agent_ppo, env_fn)
    agent_ppo.save(f"{SAVE_DIR}/models/ppo_final")
    results['PPO'] = {**eval_ppo, 'first_success_ep': cb_ppo.first_success_ep,
                      'time_min': (time.time()-t0)/60, 'eval_history': cb_ppo.eval_history}
    agents['PPO'] = agent_ppo
    print(f"\n✅ PPO done: SR={eval_ppo['success_rate']:.1%}, reward={eval_ppo['mean_reward']:.1f}, "
          f"first_success={cb_ppo.first_success_ep}, time={results['PPO']['time_min']:.1f}min")
except Exception as e:
    print(f"❌ PPO failed: {e}")
    traceback.print_exc()

# ===== 2. GNS (80K steps) =====
print("\n" + "="*60)
print("🚀 Training 2/3: GNS (80K steps)")
print("="*60)
try:
    t0 = time.time()
    train_env = DummyVecEnv([make_push_box_env(1.0) for _ in range(4)])
    agent_gns = GNSAgent(train_env)
    cb_gns = TrainingCallback("GNS", eval_env_fn=env_fn, eval_freq=10000)
    agent_gns.train(80_000, callback=cb_gns)
    train_env.close()
    eval_gns = evaluate_agent(agent_gns, env_fn)
    agent_gns.save(f"{SAVE_DIR}/models/gns_final")
    results['GNS'] = {**eval_gns, 'first_success_ep': cb_gns.first_success_ep,
                      'time_min': (time.time()-t0)/60, 'eval_history': cb_gns.eval_history}
    agents['GNS'] = agent_gns
    print(f"\n✅ GNS done: SR={eval_gns['success_rate']:.1%}, reward={eval_gns['mean_reward']:.1f}, "
          f"first_success={cb_gns.first_success_ep}, time={results['GNS']['time_min']:.1f}min")
except Exception as e:
    print(f"❌ GNS failed: {e}")
    traceback.print_exc()

# ===== 3. PhysRobot (16K steps) =====
print("\n" + "="*60)
print("🚀 Training 3/3: PhysRobot — Our Method (16K steps)")
print("="*60)
try:
    t0 = time.time()
    train_env = DummyVecEnv([make_push_box_env(1.0) for _ in range(4)])
    agent_pr = PhysRobotAgent(train_env)
    cb_pr = TrainingCallback("PhysRobot", eval_env_fn=env_fn, eval_freq=4000)
    agent_pr.train(16_000, callback=cb_pr)
    train_env.close()
    eval_pr = evaluate_agent(agent_pr, env_fn)
    agent_pr.save(f"{SAVE_DIR}/models/physrobot_final")
    results['PhysRobot'] = {**eval_pr, 'first_success_ep': cb_pr.first_success_ep,
                            'time_min': (time.time()-t0)/60, 'eval_history': cb_pr.eval_history}
    agents['PhysRobot'] = agent_pr
    print(f"\n✅ PhysRobot done: SR={eval_pr['success_rate']:.1%}, reward={eval_pr['mean_reward']:.1f}, "
          f"first_success={cb_pr.first_success_ep}, time={results['PhysRobot']['time_min']:.1f}min")
except Exception as e:
    print(f"❌ PhysRobot failed: {e}")
    traceback.print_exc()

# Save intermediate results
with open(f"{SAVE_DIR}/results/training_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n💾 Results saved to {SAVE_DIR}/results/training_results.json")""")

# ============ CELL 8: Results Table ============
code(r"""# Cell 7: Results Summary — Table 1
print("="*70)
print("📊 Table 1: Sample Efficiency Comparison")
print("="*70)
print(f"{'Method':<15} {'Steps':<10} {'Success%':<12} {'First Success':<15} {'Time(min)':<10}")
print("-"*70)
for method, steps in [('PPO', '200K'), ('GNS', '80K'), ('PhysRobot', '16K')]:
    if method in results:
        r = results[method]
        fs = r.get('first_success_ep', 'N/A')
        tm = f"{r.get('time_min', 0):.1f}"
        print(f"{method:<15} {steps:<10} {r['success_rate']*100:>6.1f}%     {str(fs):<15} {tm:<10}")
    else:
        print(f"{method:<15} {steps:<10} {'FAILED':<12}")
print("="*70)

# Improvement ratio
if 'PPO' in results and 'PhysRobot' in results:
    ppo_fs = results['PPO'].get('first_success_ep')
    pr_fs = results['PhysRobot'].get('first_success_ep')
    if ppo_fs and pr_fs and pr_fs > 0:
        ratio = ppo_fs / pr_fs
        print(f"\n🎯 PhysRobot sample efficiency: {ratio:.1f}x better than PPO!")
    print(f"\n📈 Step reduction: 200K → 16K = {200000/16000:.1f}x fewer steps")""")

# ============ CELL 9: Plots ============
code(r"""# Cell 8: Learning Curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Success rate over training
ax = axes[0]
colors = {'PPO': '#e74c3c', 'GNS': '#3498db', 'PhysRobot': '#2ecc71'}
for method in ['PPO', 'GNS', 'PhysRobot']:
    if method in results and results[method].get('eval_history'):
        hist = results[method]['eval_history']
        steps = [h['step'] for h in hist]
        sr = [h['sr'] for h in hist]
        ax.plot(steps, sr, 'o-', label=method, color=colors[method], linewidth=2)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Success Rate')
ax.set_title('Learning Curves — Success Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Bar chart of final results
ax = axes[1]
methods = [m for m in ['PPO', 'GNS', 'PhysRobot'] if m in results]
srs = [results[m]['success_rate']*100 for m in methods]
bars = ax.bar(methods, srs, color=[colors[m] for m in methods], edgecolor='black')
for bar, sr in zip(bars, srs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{sr:.1f}%', ha='center', fontweight='bold')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Final Performance Comparison')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/results/learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Plot saved to {SAVE_DIR}/results/learning_curves.png")""")

# ============ CELL 10: OOD Test ============
code(r"""# Cell 9: OOD Generalization Test
print("="*60)
print("🧪 Out-of-Distribution Generalization Test")
print("="*60)

masses = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ood_results = {}

for method_name, agent in agents.items():
    print(f"\nTesting {method_name}...")
    method_results = []
    for mass in masses:
        env = DummyVecEnv([make_push_box_env(box_mass=mass)])
        succ = 0
        for _ in range(50):
            obs = env.reset(); done = False
            while not done:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                done = dones[0]
            if infos[0].get('success', False): succ += 1
        env.close()
        sr = succ / 50
        method_results.append(sr)
        print(f"  mass={mass:.2f}kg → SR={sr:.0%}")
    ood_results[method_name] = method_results

# Plot OOD
fig, ax = plt.subplots(figsize=(10, 6))
for method in ood_results:
    ax.plot(masses, [s*100 for s in ood_results[method]], 'o-', label=method,
            color=colors.get(method, 'gray'), linewidth=2, markersize=8)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Training mass')
ax.set_xlabel('Box Mass (kg)', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('OOD Generalization: Performance vs Box Mass', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/results/ood_generalization.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ OOD plot saved")""")

# ============ CELL 11: Save All ============
code(r"""# Cell 10: Save All Results
final_results = {
    'training': results,
    'ood': {m: dict(zip([str(x) for x in masses], v)) for m, v in ood_results.items()},
    'config': {
        'ppo_steps': 200000, 'gns_steps': 80000, 'physrobot_steps': 16000,
        'n_envs': 4, 'box_mass_train': 1.0, 'ood_masses': masses
    }
}
with open(f'{SAVE_DIR}/results/week1_complete_results.json', 'w') as f:
    json.dump(final_results, f, indent=2, default=str)

print("="*60)
print("🎉 Week 1 Training Complete!")
print("="*60)
print(f"\n📁 All files saved to: {SAVE_DIR}")
print(f"   ├── models/ppo_final.zip")
print(f"   ├── models/gns_final.zip")
print(f"   ├── models/physrobot_final.zip")
print(f"   ├── results/training_results.json")
print(f"   ├── results/week1_complete_results.json")
print(f"   ├── results/learning_curves.png")
print(f"   └── results/ood_generalization.png")
print(f"\n🚀 Ready for paper writing!")""")

# ============ BUILD NOTEBOOK ============
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "V100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), "week1_training_v3.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

# Validate
with open(out_path) as f:
    nb = json.load(f)
print(f"✅ Generated {len(nb['cells'])} cells → {out_path}")
print(f"   File size: {os.path.getsize(out_path):,} bytes")
for i, c in enumerate(nb['cells']):
    t = c['cell_type']
    s = len(''.join(c['source']))
    print(f"   Cell {i}: {t} ({s} chars)")
