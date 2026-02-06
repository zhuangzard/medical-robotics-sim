"""
Test suite: Agent Training Smoke Tests
========================================

Verifies that all 3 agent types can be created and trained
for a small number of steps without crashing.

These are NOT performance tests -- just crash/shape/API tests.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Skip all tests if MuJoCo not available
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    import torch
    import stable_baselines3
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

pytestmark = pytest.mark.skipif(
    not (HAS_MUJOCO and HAS_SB3),
    reason="Requires mujoco + stable-baselines3"
)


@pytest.fixture
def vec_env():
    """Create a simple DummyVecEnv for testing."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environments.push_box import make_push_box_env
    env = DummyVecEnv([make_push_box_env(box_mass=1.0)])
    yield env
    env.close()


@pytest.fixture
def raw_env():
    """Create a raw (non-vectorized) PushBoxEnv for evaluate()."""
    from environments.push_box import PushBoxEnv
    env = PushBoxEnv(box_mass=1.0)
    yield env
    env.close()


class TestPurePPO:
    """Pure PPO agent smoke test."""

    def test_create_and_train(self, vec_env):
        from baselines.ppo_baseline import PurePPOAgent
        agent = PurePPOAgent(vec_env, verbose=0)
        # Train for 1 PPO update (n_steps=2048)
        agent.train(total_timesteps=2048)
        # Predict should return an action
        obs = vec_env.reset()
        action = agent.predict(obs, deterministic=True)
        assert action.shape == (1, 2), f"Unexpected action shape: {action.shape}"

    def test_save_load(self, vec_env, tmp_path):
        from baselines.ppo_baseline import PurePPOAgent
        agent = PurePPOAgent(vec_env, verbose=0)
        path = str(tmp_path / "ppo_test")
        agent.save(path)
        agent2 = PurePPOAgent(vec_env, verbose=0)
        agent2.load(path)
        # Both should predict the same action
        obs = vec_env.reset()
        a1 = agent.predict(obs, deterministic=True)
        a2 = agent2.predict(obs, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)


class TestGNSAgent:
    """GNS agent smoke test."""

    def test_create_and_train(self, vec_env):
        from baselines.gns_baseline import GNSAgent
        agent = GNSAgent(vec_env, verbose=0)
        agent.train(total_timesteps=2048)
        obs = vec_env.reset()
        action = agent.predict(obs, deterministic=True)
        assert action.shape == (1, 2)


class TestPhysRobotAgent:
    """PhysRobot (SV-pipeline) agent smoke test."""

    def test_create_and_train(self, vec_env):
        from baselines.physics_informed import PhysRobotAgent
        agent = PhysRobotAgent(vec_env, verbose=0)
        agent.train(total_timesteps=2048)
        obs = vec_env.reset()
        action = agent.predict(obs, deterministic=True)
        assert action.shape == (1, 2)

    def test_features_extractor_output(self, vec_env):
        """Verify the SV features extractor produces correct shape."""
        import torch
        from baselines.physics_informed import PhysRobotSVFeaturesExtractor
        obs_space = vec_env.observation_space
        ext = PhysRobotSVFeaturesExtractor(obs_space, features_dim=64)
        obs = torch.randn(4, obs_space.shape[0])
        out = ext(obs)
        assert out.shape == (4, 64)

    def test_physics_core_in_agent_conserves(self, vec_env):
        """Extract the SVPhysicsCore from the agent and verify conservation."""
        import torch
        from baselines.physics_informed import PhysRobotAgent
        agent = PhysRobotAgent(vec_env, verbose=0)
        # Reach into the policy to get the physics core
        extractor = agent.model.policy.features_extractor
        physics_core = extractor.core.physics_core

        pos = torch.randn(3, 3)
        vel = torch.randn(3, 3)
        src, dst = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    src.append(i)
                    dst.append(j)
        ei = torch.tensor([src, dst], dtype=torch.long)
        F = physics_core(pos, vel, ei)
        err = F.sum(dim=0).norm().item()
        assert err < 1e-4, f"Conservation violated in agent: ||sum F|| = {err:.2e}"


class TestTraining1000Steps:
    """Train each agent for 1000 env steps without crash."""

    @pytest.mark.parametrize("agent_name", ["PPO", "PhysRobot"])
    def test_1000_steps_no_crash(self, vec_env, agent_name):
        if agent_name == "PPO":
            from baselines.ppo_baseline import PurePPOAgent
            agent = PurePPOAgent(vec_env, verbose=0)
        else:
            from baselines.physics_informed import PhysRobotAgent
            agent = PhysRobotAgent(vec_env, verbose=0)

        # n_steps default is 2048, so 2048 timesteps = 1 PPO update
        # This tests that the full pipeline (env -> obs -> extractor -> policy) works
        agent.train(total_timesteps=2048)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
