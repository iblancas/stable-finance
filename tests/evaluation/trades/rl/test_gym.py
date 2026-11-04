"""The Gymnasium adapter: spaces and scale around the same environment."""
from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env
from helpers import quoted_world

from stable_finance import BacktestConfig, ExecutionEnv, rollout
from stable_finance.evaluation.trades.rl.gym import GymExecutionEnv


def _env(**kwargs):
    world = quoted_world(seed=3)
    config = BacktestConfig(cost_model="cross", annualization="decisions")
    return ExecutionEnv(world["target"], world["realized"], world["market"],
                        days=world["days"], config=config, episode_days=1, **kwargs)


@pytest.mark.filterwarnings("ignore:.*infinity")  # features are unbounded weights and returns
def test_it_passes_gymnasiums_own_checks():
    check_env(GymExecutionEnv(_env()), skip_render_check=True)
    check_env(GymExecutionEnv(_env(), rates=(0.0, 0.5, 1.0)), skip_render_check=True)


def test_the_reward_is_the_net_return_in_basis_points():
    gym_env = GymExecutionEnv(_env())
    gym_env.reset(seed=4)
    rewards, nets = [], []
    ended = False
    while not ended:
        _, reward, terminated, truncated, info = gym_env.step(np.full(5, 2.0, dtype=np.float32))
        rewards.append(reward)
        nets.append(info["net"])
        ended = terminated or truncated
    np.testing.assert_allclose(rewards, np.array(nets) * 1e4)
    # Rates above one are clipped to one: the same episode as trading to the target.
    expected = rollout(_env(), lambda observation: 1.0, seed=4)
    np.testing.assert_allclose(nets, expected.net)
    np.testing.assert_allclose(gym_env.env.episode().net, expected.net)


def test_a_discrete_action_picks_a_rate():
    gym_env = GymExecutionEnv(_env(), rates=(0.0, 0.5, 1.0))
    gym_env.reset(seed=4)
    ended = False
    while not ended:
        _, _, terminated, truncated, _ = gym_env.step(np.full(5, 1))
        ended = terminated or truncated
    expected = rollout(_env(), lambda observation: 0.5, seed=4)
    np.testing.assert_allclose(gym_env.env.episode().net, expected.net)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="reward_scale"):
        GymExecutionEnv(_env(), reward_scale=0.0)
    with pytest.raises(ValueError, match="rates"):
        GymExecutionEnv(_env(), rates=(0.5,))
    with pytest.raises(ValueError, match="rates"):
        GymExecutionEnv(_env(), rates=(0.0, 1.5))
