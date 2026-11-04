"""Double DQN and its cooperative forms: known answers, and the harness on what they play."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.rl.coupling import NetLimit
from experiments.rl.dqn import Agent, Inputs, train
from experiments.rl.study import ConstantRate, NoTradeBand, make_env, synthetic_panels
from stable_finance import (
    BacktestConfig,
    Transitions,
    name_rows,
    rollout,
    row_features,
    run_backtest,
)

CROSS = BacktestConfig(cost_model="cross", efq=1.0, annualization="decisions")
MID = BacktestConfig(cost_model="mid", annualization="decisions")
SMALL = {"n_assets": 40, "train_days": 12, "test_days": 6, "per_day": 39}
FAST = {"episodes": 40, "warmup": 6, "updates": 30, "batch": 16, "memory": 600, "hidden": 32}


def _traded(episode):
    return np.abs(episode.fills.quantity).sum()


def test_with_no_skill_it_learns_to_stop_trading():
    train_panel, test = synthetic_panels(1, ic=0.0, **SMALL)
    agent = train(make_env(train_panel, CROSS, episode_days=1), **FAST)
    env = make_env(test, CROSS)
    assert _traded(rollout(env, agent.policy(env))) < 0.2 * _traded(rollout(env, ConstantRate(1.0)))


def test_at_no_cost_a_true_signal_is_chased():
    train_panel, test = synthetic_panels(2, ic=0.5, **SMALL)
    agent = train(make_env(train_panel, MID, episode_days=1), **FAST)
    env = make_env(test, MID)
    observation, _ = env.reset()
    gap, goal = (observation["names"][:, env.name_features.index(f)] for f in ("gap", "target"))
    under_held = np.sign(gap) == np.sign(goal)
    assert agent.policy(env)(observation)[under_held].mean() > 0.7


def test_a_seed_fixes_the_agent():
    train_panel, test = synthetic_panels(3, **SMALL)

    def net(seed):
        agent = train(make_env(train_panel, CROSS, episode_days=1), seed=seed,
                      **{**FAST, "episodes": 4})
        env = make_env(test, CROSS)
        return rollout(env, agent.policy(env)).net

    np.testing.assert_array_equal(net(1), net(1))
    assert not np.array_equal(net(1), net(2))


def test_the_buffer_says_who_played():
    train_panel, _ = synthetic_panels(3, **SMALL)
    env = make_env(train_panel, CROSS, episode_days=1)
    agent = train(env, experts={"band": NoTradeBand(env, 1.0)}, **{**FAST, "episodes": 2})
    assert agent.buffer.sources == ("random", "band", "self")
    assert set(agent.buffer.sample(500, source="band").source) == {"band"}


@pytest.mark.parametrize("form", [{}, {"team": True}, {"team": True, "intention": True}],
                         ids=["independent", "team", "intention"])
def test_what_every_form_plays_is_charged_by_the_harness(form):
    train_panel, test = synthetic_panels(4, **SMALL)
    agent = train(NetLimit(make_env(train_panel, CROSS, episode_days=1), 0.02), **form,
                  **{**FAST, "episodes": 6})
    env = NetLimit(make_env(test, CROSS), 0.02)
    episode = rollout(env, agent.policy(env))
    report = run_backtest(episode.weights, test.realized, market=test.market, days=test.days,
                          config=CROSS, with_series=True)
    np.testing.assert_allclose(episode.net, report.series.net, rtol=1e-12, atol=1e-15)


def test_intention_tells_every_name_what_the_book_would_do():
    train_panel, _ = synthetic_panels(5, **SMALL)
    env = make_env(train_panel, CROSS, episode_days=1)
    agent = train(env, team=True, intention=True, **{**FAST, "episodes": 2})
    observation, _ = env.reset(seed=0)
    rows = name_rows(env, observation)
    intent = agent._intent(agent.inputs(rows), rows)
    assert intent.shape == (len(env.assets), 1)
    assert torch.all(intent == intent[0])          # one number for the whole book
    assert agent.online[0].in_features == len(row_features(env)) + 1


def test_a_long_and_a_short_are_read_as_one_case():
    columns = ("held", "target", "gap", "half_spread")
    rows = np.array([[0.01, 0.02, 0.01, 2e-4], [-0.01, -0.02, -0.01, 2e-4], [0.0, 0.01, 0.01, 3e-4]])
    features = Inputs(columns, rows)(rows)
    torch.testing.assert_close(features[0], features[1])


def test_an_agent_refuses_other_features():
    train_panel, _ = synthetic_panels(3, **SMALL)
    agent = train(make_env(train_panel, CROSS, episode_days=1), **{**FAST, "episodes": 1})
    other = make_env(train_panel, BacktestConfig(cost_model="impact", equity=1e6, execution_days=0))
    with pytest.raises(ValueError, match="other features"):
        agent.policy(other)


def _rigged(team=False, gamma=0.9):
    """An agent whose networks give fixed values, so an update's target can be worked by hand."""
    columns = ("held", "target", "gap", "gross_target")
    agent = Agent(Inputs(columns, np.eye(4)), (0.0, 1.0), team=team, gamma=gamma)
    return agent


def _batch(reward, *, terminated=False, n_names=1):
    rows = np.zeros((1, n_names, 4), dtype=np.float32)
    return Transitions(rows, np.zeros((1, n_names)), np.asarray(reward, dtype=float).reshape(1, -1),
                       rows, np.full((1, n_names), terminated), np.ones((1, n_names), dtype=bool),
                       np.zeros((1, n_names), dtype=int), np.full((1, n_names), "a"))


def _rig(agent, now, online_next, slow_next):
    """The online network says ``now`` then ``online_next``; the slow one says ``slow_next``."""
    online = iter([torch.tensor(now, requires_grad=True) * 1.0, torch.tensor(online_next)])
    agent._values = lambda network, rows: (next(online) if network is agent.online
                                           else torch.tensor(slow_next))


def test_one_network_picks_the_next_rate_and_the_other_values_it():
    """The double in double Q-learning: the slow network's best (10) is not what is read."""
    agent = _rigged()
    _rig(agent, [[[1.0, 0.0]]], [[[1.0, 2.0]]], [[[10.0, 0.5]]])
    # The online network picks rate 1, the slow one values it 0.5: the goal is 0.55 + 0.45.
    assert agent.update(_batch(0.55)) == pytest.approx(0.0, abs=1e-6)


def test_the_panels_end_stops_bootstrapping():
    agent = _rigged()
    _rig(agent, [[[1.0, 0.0]]], [[[1.0, 2.0]]], [[[10.0, 0.5]]])
    assert agent.update(_batch(1.0, terminated=True)) == pytest.approx(0.0, abs=1e-6)


def test_a_team_is_judged_on_the_sum_and_a_name_on_its_own():
    """Opposite errors cancel in the book's value and do not in each name's."""
    values = {"now": [[[1.0, 0.0], [1.0, 0.0]]], "online_next": [[[1.0, 0.0], [1.0, 0.0]]],
              "slow_next": [[[0.0, 0.0], [0.0, 0.0]]]}
    team, alone = _rigged(team=True), _rigged(team=False)
    _rig(team, **values)
    _rig(alone, **values)
    assert team.update(_batch([0.0, 2.0], n_names=2)) == pytest.approx(0.0, abs=1e-6)
    assert alone.update(_batch([0.0, 2.0], n_names=2)) == pytest.approx(0.5)


def test_the_intention_is_the_net_drift_of_what_each_name_would_do():
    columns = ("held", "target", "gap", "gross_target")
    agent = Agent(Inputs(columns, np.eye(4)), (0.0, 1.0), intention=True)
    rows = np.array([[0.0, 0.2, 0.2, 2.0], [0.0, 0.3, 0.3, 2.0]])
    features = agent.inputs(rows)
    # Everyone would hold still: all 0.5 of the gap stays, over a gross of 2.
    agent.online = lambda x: torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(agent._intent(features, rows), torch.full((2, 1), -0.25))
    # Everyone would trade fully: nothing is left to drift.
    agent.online = lambda x: torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    torch.testing.assert_close(agent._intent(features, rows), torch.zeros(2, 1))
