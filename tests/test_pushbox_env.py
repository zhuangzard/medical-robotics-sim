"""
Test suite: PushBoxEnv Consistency
===================================

Validates the canonical PushBoxEnv (16-dim obs, 2-dim action).
Tests:
  - Observation/action space shapes
  - Reset determinism with seed
  - Step reward range
  - Success detection
  - OOD mass variation
  - Factory function
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.push_box import PushBoxEnv, make_push_box_env


class TestSpaces:
    """Observation and action space validation."""

    def test_obs_space_shape(self):
        env = PushBoxEnv()
        assert env.observation_space.shape == (16,)
        env.close()

    def test_action_space_shape(self):
        env = PushBoxEnv()
        assert env.action_space.shape == (2,)
        env.close()

    def test_action_bounds(self):
        env = PushBoxEnv()
        assert env.action_space.low[0] == -10.0
        assert env.action_space.high[0] == 10.0
        env.close()

    def test_obs_dtype(self):
        env = PushBoxEnv()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        env.close()


class TestResetDeterminism:
    """Same seed -> same initial state."""

    def test_same_seed_same_obs(self):
        env = PushBoxEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_different_seeds_different_obs(self):
        env = PushBoxEnv()
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)
        env.close()


class TestStep:
    """Step mechanics."""

    def test_step_returns_5_tuple(self):
        env = PushBoxEnv()
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, f"step() returned {len(result)} values, expected 5"
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (16,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_episode_truncates(self):
        """Episode should truncate at max_episode_steps."""
        env = PushBoxEnv()
        env.reset(seed=0)
        for i in range(env.max_episode_steps + 10):
            _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            if terminated or truncated:
                assert i < env.max_episode_steps + 1
                break
        env.close()

    def test_reward_is_finite(self):
        """Reward should never be NaN or Inf."""
        env = PushBoxEnv()
        env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated or truncated:
                env.reset()
        env.close()


class TestInfo:
    """Info dict contents."""

    def test_info_keys(self):
        env = PushBoxEnv()
        _, info = env.reset(seed=0)
        assert 'distance_to_goal' in info
        assert 'success' in info
        assert 'box_mass' in info
        assert 'timestep' in info
        env.close()

    def test_distance_positive(self):
        env = PushBoxEnv()
        _, info = env.reset(seed=0)
        assert info['distance_to_goal'] >= 0
        env.close()


class TestMassVariation:
    """OOD mass setting."""

    @pytest.mark.parametrize("mass", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_different_masses(self, mass):
        env = PushBoxEnv(box_mass=mass)
        obs, info = env.reset(seed=0)
        assert info['box_mass'] == mass
        # Should be able to step without crash
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward)
            if term or trunc:
                break
        env.close()

    def test_set_box_mass(self):
        env = PushBoxEnv(box_mass=0.5)
        env.set_box_mass(3.0)
        assert env.box_mass == 3.0
        env.close()


class TestFactory:
    """Factory function tests."""

    def test_make_push_box_env_returns_callable(self):
        factory = make_push_box_env(box_mass=1.0)
        assert callable(factory)
        env = factory()
        assert isinstance(env, PushBoxEnv)
        env.close()

    def test_factory_mass_propagated(self):
        factory = make_push_box_env(box_mass=2.5)
        env = factory()
        assert env.box_mass == 2.5
        env.close()


class TestObsLayout:
    """Verify the 16-dim observation layout is consistent."""

    def test_obs_layout_documented(self):
        """
        Layout: [joint_pos(2), joint_vel(2), ee_pos(3), box_pos(3), box_vel(3), goal(3)]
        """
        env = PushBoxEnv()
        obs, _ = env.reset(seed=0)

        # Goal should be the last 3 elements and match env.goal_pos
        goal_from_obs = obs[13:16]
        np.testing.assert_allclose(goal_from_obs, env.goal_pos, atol=1e-5)

        # Box position should be at indices 7:10
        # Verify it's within reasonable bounds (arm reach ~0.7m)
        box_pos = obs[7:10]
        assert np.linalg.norm(box_pos[:2]) < 2.0, "Box too far from origin"

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
