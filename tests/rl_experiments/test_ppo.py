"""Basic PPO: its two formulas by hand, known answers, and the harness on what it plays."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.rl.coupling import NetLimit
from experiments.rl.ppo import advantages, clipped_objective, train
from experiments.rl.study import ConstantRate, compare, make_env, synthetic_panels
from stable_finance import BacktestConfig, rollout, run_backtest

CROSS = BacktestConfig(cost_model="cross", efq=1.0, annualization="decisions")
MID = BacktestConfig(cost_model="mid", annualization="decisions")
SMALL = {"n_assets": 40, "train_days": 12, "test_days": 6, "per_day": 39}
FAST = {"episodes": 120, "days": 4, "warmup": 3, "hidden": 32, "minibatch": 1024,
        "learning_rate": 1e-3}


def test_an_advantage_is_worked_backward_through_the_surprises():
    rewards = np.array([[1.0], [2.0]])
    values = np.array([[0.5], [1.0]])
    final = np.array([3.0])
    # Last step: 2 + 0.9 * 3 - 1 = 3.7. First: 1 + 0.9 * 1 - 0.5 = 1.4, plus 0.9 * 0.5 * 3.7.
    estimate = advantages(rewards, values, final, alive=True, gamma=0.9, lam=0.5)
    np.testing.assert_allclose(estimate, [[1.4 + 0.45 * 3.7], [3.7]])


def test_only_the_panels_end_drops_the_future():
    rewards, values, final = np.array([[2.0]]), np.array([[1.0]]), np.array([3.0])
    assert advantages(rewards, values, final, alive=False, gamma=0.9, lam=0.5)[0, 0] == 1.0
    assert advantages(rewards, values, final, alive=True, gamma=0.9, lam=0.5)[0, 0] == 3.7


def test_the_clip_stops_a_good_action_being_pushed_too_far():
    ratio, clip = torch.tensor([1.5, 1.1, 0.5, 1.5]), 0.2
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(clipped_objective(ratio, advantage, clip),
                               torch.tensor([1.2, 1.1, -0.8, -1.5]))


def test_with_no_skill_it_trades_far_less():
    train_panel, test = synthetic_panels(1, ic=0.0, **SMALL)
    agent = train(make_env(train_panel, CROSS, episode_days=1), **FAST)
    env = make_env(test, CROSS)
    traded = np.abs(rollout(env, agent.policy(env)).fills.quantity).sum()
    assert traded < 0.2 * np.abs(rollout(env, ConstantRate(1.0)).fills.quantity).sum()


def test_at_no_cost_a_true_signal_is_chased():
    train_panel, test = synthetic_panels(2, ic=0.5, **SMALL)
    agent = train(make_env(train_panel, MID, episode_days=1), **FAST)
    env = make_env(test, MID)
    observation, _ = env.reset()
    gap, goal = (observation["names"][:, env.name_features.index(f)] for f in ("gap", "target"))
    # Small targets lag, their advantage being small, and catch up with more days.
    wanted = (np.sign(gap) == np.sign(goal)) & (np.abs(goal) > np.median(np.abs(goal)))
    assert agent.policy(env)(observation)[wanted].mean() > 0.9


def test_a_seed_fixes_the_agent():
    train_panel, test = synthetic_panels(3, **SMALL)

    def net(seed):
        agent = train(make_env(train_panel, CROSS, episode_days=1), seed=seed,
                      **{**FAST, "episodes": 8})
        env = make_env(test, CROSS)
        return rollout(env, agent.policy(env)).net

    np.testing.assert_array_equal(net(1), net(1))
    assert not np.array_equal(net(1), net(2))


def test_what_it_plays_is_charged_by_the_harness():
    train_panel, test = synthetic_panels(4, **SMALL)
    agent = train(NetLimit(make_env(train_panel, CROSS, episode_days=1), 0.02),
                  **{**FAST, "episodes": 8})
    env = NetLimit(make_env(test, CROSS), 0.02)
    episode = rollout(env, agent.policy(env))
    report = run_backtest(episode.weights, test.realized, market=test.market, days=test.days,
                          config=CROSS, with_series=True)
    np.testing.assert_allclose(episode.net, report.series.net, rtol=1e-12, atol=1e-15)


def test_an_agent_refuses_other_features():
    train_panel, _ = synthetic_panels(3, **SMALL)
    agent = train(make_env(train_panel, CROSS, episode_days=1), **{**FAST, "episodes": 4})
    other = make_env(train_panel, BacktestConfig(cost_model="impact", equity=1e6, execution_days=0))
    with pytest.raises(ValueError, match="other features"):
        agent.policy(other)


def test_the_study_runs_every_learned_arm():
    train_panel, test = synthetic_panels(5, n_assets=20, train_days=4, test_days=2, per_day=20)
    arms = compare(train_panel, test, CROSS, seeds=1, episodes=2, dqn_episodes=4, limit=0.02,
                   log=lambda text: None)
    assert {"double DQN", "double DQN, band in buffer", "PPO", "limit: independent",
            "limit: team (VDN)", "limit: team + intention", "limit: PPO"} <= set(arms)
    assert arms["PPO"]["seeds"] == arms["limit: PPO"]["seeds"] == 1
