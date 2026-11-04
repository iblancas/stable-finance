"""The replay buffer: every row priced by the environment, whoever played it."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import PER_DAY, quoted_world

from stable_finance import (
    BacktestConfig,
    ExecutionEnv,
    FollowPath,
    PortfolioWeights,
    RandomRate,
    RandomWalkRate,
    RatePerEpisode,
    ReplayBuffer,
    collect,
    implementation_shortfall,
    name_rows,
    rollout,
    row_features,
    run_backtest,
)

CROSS = BacktestConfig(cost_model="cross", efq=0.7, annualization="decisions")
CONFIGS = [
    CROSS,
    BacktestConfig(cost_model="mid", annualization="decisions"),
    BacktestConfig(cost_model="book", equity=1e6, annualization="decisions"),
    BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5, annualization="decisions"),
    BacktestConfig(cost_model="impact", equity=1e6, execution_days=0, annualization="decisions"),
]
N_ASSETS = 5


def _env(world, config=CROSS, **kwargs):
    return ExecutionEnv(world["target"], world["realized"], world["market"],
                        days=world["days"], config=config, **kwargs)


def _buffer(env, capacity=100_000, seed=0):
    return ReplayBuffer(capacity, len(row_features(env)), seed=seed)


def _add(buffer, n, *, value=0.0, day=0, source="a"):
    rows = np.full((n, buffer.n_features), value)
    buffer.add(rows, np.zeros(n), np.full(n, value), rows, terminated=False,
               tradable=np.ones(n, dtype=bool), day=day, source=source)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.cost_model}-{c.execution_days}")
def test_the_rows_of_an_episode_add_up_to_the_harness(config):
    """THE POINT OF THE MODULE: nothing enters a buffer that the harness did not charge."""
    world = quoted_world(seed=3)
    env = _env(world, config)
    net, shortfall = _buffer(env), _buffer(env)
    collect(env, RandomRate(), net, seed=1, reward="net")
    episode = env.episode()
    collect(env, RandomRate(), shortfall, seed=1, reward="shortfall")

    by_decision = net.contents().reward.reshape(-1, N_ASSETS).sum(axis=1)
    report = run_backtest(episode.weights, world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    np.testing.assert_allclose(by_decision, report.series.net, rtol=1e-12, atol=1e-15)
    scorecard, = implementation_shortfall(episode.target, episode.fills, episode.realized)
    np.testing.assert_allclose(-shortfall.contents().reward.reshape(-1, N_ASSETS).sum(axis=1),
                               scorecard.shortfall, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.cost_model}-{c.execution_days}")
def test_an_experts_path_is_retraced_and_charged_by_the_harness(config):
    world = quoted_world(seed=4)
    env = _env(world, config)
    played = rollout(env, RandomRate(), seed=2)
    expert = FollowPath(env, played.weights)
    buffer = _buffer(env)
    collect(env, expert, buffer, source="expert", reward="net")
    assert expert.unexpressed == pytest.approx(0.0, abs=1e-12)
    retraced = env.episode()
    np.testing.assert_allclose(retraced.weights.values, played.weights.values, atol=1e-12)
    np.testing.assert_allclose(buffer.contents().reward.reshape(-1, N_ASSETS).sum(axis=1),
                               played.net, rtol=1e-9, atol=1e-13)
    assert set(buffer.contents().source) == {"expert"}


def _steady_world():
    """Quotes everywhere and a target that never moves."""
    world = quoted_world(seed=1, gaps=False)
    target = world["target"]
    steady = np.broadcast_to(target.values[:1], target.values.shape)
    world["target"] = PortfolioWeights(steady, target.decisions, target.assets, target.horizons)
    return world


def test_what_a_rate_cannot_express_is_clipped_and_counted():
    """Twice the target: the first trade gets half way, and every later one nowhere."""
    world = _steady_world()
    env = _env(world, BacktestConfig(cost_model="mid", annualization="decisions"))
    target = world["target"]
    path = PortfolioWeights(2 * target.values, target.decisions, target.assets, target.horizons)
    expert = FollowPath(env, path)
    episode = rollout(env, expert)
    np.testing.assert_allclose(episode.weights.values, target.values, atol=1e-15)
    n_decisions = len(target.decisions)
    assert expert.unexpressed == pytest.approx(n_decisions / (n_decisions + 1))


def test_a_path_that_is_not_on_the_grid_is_refused():
    world = quoted_world(seed=1)
    env = _env(world)
    target = world["target"]
    with pytest.raises(TypeError, match="PortfolioWeights"):
        FollowPath(env, world["realized"])
    shorter = PortfolioWeights(target.values[:-1], target.decisions[:-1], target.assets,
                               target.horizons)
    with pytest.raises(ValueError, match="decisions"):
        FollowPath(env, shorter)
    renamed = PortfolioWeights(target.values, target.decisions, [f"X{i}" for i in range(5)],
                               target.horizons)
    with pytest.raises(ValueError, match="assets"):
        FollowPath(env, renamed)
    other = PortfolioWeights(target.values, target.decisions, target.assets, [900])
    with pytest.raises(ValueError, match="horizon"):
        FollowPath(env, other)


@pytest.mark.parametrize("behaviour", [RandomRate, RandomWalkRate, RatePerEpisode])
def test_one_seed_replays_the_days_and_the_rates(behaviour):
    world = quoted_world(seed=6)
    env = _env(world, episode_days=1)
    first, again, other = _buffer(env), _buffer(env), _buffer(env)
    collect(env, behaviour(), first, episodes=4, seed=11)
    collect(env, behaviour(), again, episodes=4, seed=11)
    collect(env, behaviour(), other, episodes=4, seed=12)
    np.testing.assert_array_equal(first.contents().rate, again.contents().rate)
    np.testing.assert_array_equal(first.contents().reward, again.contents().reward)
    np.testing.assert_array_equal(first.contents().day, again.contents().day)
    assert not np.array_equal(first.contents().rate, other.contents().rate)


def _rates(world, behaviour, **kwargs):
    """Rates played, as (episode, decision, name), one day an episode."""
    env = _env(world, episode_days=1)
    buffer = _buffer(env)
    collect(env, behaviour, buffer, seed=3, **kwargs)
    return buffer.contents().rate.reshape(-1, PER_DAY, N_ASSETS)


def test_each_behaviour_moves_as_it_says():
    world = quoted_world(seed=2)
    independent = _rates(world, RandomRate())
    assert np.all((independent >= 0) & (independent <= 1))
    assert np.all(np.abs(np.diff(independent, axis=1)).max(axis=1) > 0.1)

    walk = _rates(world, RandomWalkRate(step=0.01))
    assert np.all((walk >= 0) & (walk <= 1))
    assert 0 < np.abs(np.diff(walk, axis=1)).max() < 0.06

    held = _rates(world, RatePerEpisode(low=0.05))
    assert np.all((held >= 0.05) & (held <= 1))
    np.testing.assert_array_equal(np.diff(held, axis=1), 0.0)
    assert not np.array_equal(held[0, 0], held[1, 0])


def test_a_rate_held_all_day_reaches_furthest_from_a_slow_target():
    """Why RatePerEpisode is the behaviour to start from: it visits the large gaps."""
    rng = np.random.default_rng(0)
    world = quoted_world(seed=5, n_days=30, n_assets=20, gaps=False)
    target = world["target"]
    slow = np.zeros(target.values.shape)
    slow[0] = rng.normal(size=slow[0].shape)
    for t in range(1, len(slow)):
        slow[t] = 0.9 * slow[t - 1] + np.sqrt(1 - 0.9**2) * rng.normal(size=slow[t].shape)
    slow -= slow.mean(axis=1, keepdims=True)
    slow /= np.abs(slow).sum(axis=1, keepdims=True)
    world["target"] = PortfolioWeights(slow, target.decisions, target.assets, target.horizons)
    size = np.abs(slow).mean()

    def share_far(behaviour):
        env = _env(world, episode_days=1)
        buffer = _buffer(env, capacity=200_000)
        collect(env, behaviour, buffer, seed=1)
        gap = buffer.contents().rows[:, env.name_features.index("gap")]
        return (np.abs(gap) > 2 * size).mean()

    assert share_far(RatePerEpisode()) > share_far(RandomWalkRate()) > share_far(RandomRate())


def test_the_action_stored_is_the_action_played():
    """A rate is moved to the grid before it trades, so its reward belongs to it."""
    world = quoted_world(seed=3)
    env = _env(world)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    buffer = _buffer(env)
    collect(env, RandomRate(), buffer, seed=1, reward="net", rates=grid)
    played = buffer.contents().rate.reshape(-1, N_ASSETS)
    assert set(played.ravel()) == set(grid)
    steps = iter(played)
    np.testing.assert_allclose(rollout(env, lambda observation: next(steps)).net,
                               buffer.contents().reward.reshape(-1, N_ASSETS).sum(axis=1),
                               rtol=1e-12, atol=1e-15)


def test_any_callable_is_a_policy():
    world = quoted_world(seed=3)
    env = _env(world)
    buffer = _buffer(env)
    assert collect(env, lambda observation: 0.5, buffer) == 48 * N_ASSETS
    np.testing.assert_array_equal(buffer.contents().rate, 0.5)


def test_a_row_is_the_name_then_the_book_without_progress():
    """Progress would read a day in training and the whole test period in evaluation."""
    world = quoted_world(seed=8)
    env = _env(world)
    assert row_features(env) == env.name_features + ("time_of_day", "gross_held", "gross_target",
                                                     "gross_gap")
    observation, _ = env.reset()
    observation, *_ = env.step(0.5)
    rows = name_rows(env, observation)
    assert rows.shape == (N_ASSETS, len(row_features(env)))
    n = len(env.name_features)
    np.testing.assert_array_equal(rows[:, :n], observation["names"])
    np.testing.assert_array_equal(rows[:, n:], np.tile(observation["book"][1:], (N_ASSETS, 1)))


def test_only_the_end_of_the_panel_stops_bootstrapping():
    world = quoted_world(seed=8)
    env = _env(world, episode_days=1)
    buffer = _buffer(env)
    collect(env, lambda observation: 0.5, buffer)
    rows = buffer.contents()
    assert set(rows.day) == set(range(6))
    ended = np.zeros(48, dtype=bool)
    ended[-1] = True
    np.testing.assert_array_equal(rows.terminated.reshape(48, N_ASSETS),
                                  np.tile(ended[:, None], (1, N_ASSETS)))
    # Across a day end the next row is the next decision's, not a repeat of this one.
    goal = env.name_features.index("target")
    close = (PER_DAY - 1) * N_ASSETS
    np.testing.assert_allclose(rows.next_rows[close:close + N_ASSETS, goal],
                               world["target"].values[PER_DAY, :, 0], rtol=1e-6)


def test_the_mask_is_kept_beside_the_rows():
    world = quoted_world(seed=8)
    env = _env(world)
    buffer = _buffer(env)
    collect(env, lambda observation: 1.0, buffer)
    tradable = buffer.contents().tradable.reshape(48, N_ASSETS)
    np.testing.assert_array_equal(tradable, np.isfinite(world["market"].get("half_spread")))
    assert not tradable.all()


def test_episodes_are_each_played_once_unless_drawn():
    world = quoted_world(seed=8)
    env = _env(world, episode_days=1)
    every, drawn = _buffer(env), _buffer(env)
    collect(env, lambda observation: 1.0, every)
    np.testing.assert_array_equal(every.contents().day, np.repeat(np.arange(6), PER_DAY * N_ASSETS))
    assert collect(env, lambda observation: 1.0, drawn, episodes=20, seed=1) == 20 * PER_DAY * N_ASSETS
    visits = np.bincount(drawn.contents().day, minlength=6)
    assert visits.max() > visits.min()


def test_the_oldest_rows_go_first():
    buffer = ReplayBuffer(10, 2)
    for k in range(3):
        _add(buffer, 4, value=k, day=k)
    assert len(buffer) == 10
    np.testing.assert_array_equal(buffer.contents().day, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(buffer.contents().reward, buffer.contents().day)


def test_sampling_is_seeded_and_can_ask_for_a_source():
    def filled(seed):
        buffer = ReplayBuffer(100, 2, seed=seed)
        _add(buffer, 30, value=1.0, source="random")
        _add(buffer, 10, value=2.0, source="expert")
        return buffer

    first, again, other = filled(0).sample(50), filled(0).sample(50), filled(1).sample(50)
    np.testing.assert_array_equal(first.reward, again.reward)
    assert not np.array_equal(first.reward, other.reward)
    assert first.rows.dtype == np.float32 and first.reward.dtype == np.float64

    buffer = filled(0)
    assert buffer.sources == ("random", "expert")
    drawn = buffer.sample(50, source="expert")
    assert set(drawn.source) == {"expert"}
    np.testing.assert_array_equal(drawn.reward, 2.0)
    with pytest.raises(ValueError, match="human"):
        buffer.sample(5, source="human")
    with pytest.raises(ValueError, match="no rows"):
        ReplayBuffer(5, 2).sample(1)


def test_whole_decisions_can_be_drawn_together():
    world = quoted_world(seed=3)
    env = _env(world)
    buffer = _buffer(env, capacity=30 * N_ASSETS, seed=1)
    collect(env, RandomRate(), buffer, seed=1)
    drawn = buffer.sample_steps(8, N_ASSETS)
    assert drawn.rows.shape == (8, N_ASSETS, buffer.n_features)
    assert drawn.reward.shape == (8, N_ASSETS)
    # A decision is one moment: every name's row shows the same clock and day.
    clock = row_features(env).index("time_of_day")
    np.testing.assert_array_equal(drawn.rows[:, :, clock], drawn.rows[:, :1, clock].repeat(N_ASSETS, 1))
    np.testing.assert_array_equal(drawn.day, drawn.day[:, :1].repeat(N_ASSETS, 1))
    with pytest.raises(ValueError, match="whole decisions"):
        buffer.sample_steps(2, 7)
    with pytest.raises(ValueError, match="no rows"):
        ReplayBuffer(10, 2).sample_steps(1, 5)


def test_bad_settings_are_refused():
    world = quoted_world(seed=0)
    env = _env(world)
    for capacity, width in ((0, 2), (5, 0), (2.5, 2)):
        with pytest.raises(ValueError, match="positive integer"):
            ReplayBuffer(capacity, width)
    with pytest.raises(ValueError, match="shape"):
        _buffer(env).add(np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)),
                         terminated=False, tradable=np.ones(3, dtype=bool), day=0, source="a")
    with pytest.raises(ValueError, match="do not fit"):
        _add(ReplayBuffer(3, 2), 4)
    with pytest.raises(ValueError, match="reward"):
        collect(env, RandomRate(), _buffer(env), reward="sharpe")
    with pytest.raises(ValueError, match="episodes"):
        collect(env, RandomRate(), _buffer(env), episodes=0)
    with pytest.raises(ValueError, match="rates"):
        collect(env, RandomRate(), _buffer(env), rates=(0.5,))
    with pytest.raises(ValueError, match="rates"):
        collect(env, RandomRate(), _buffer(env), rates=(0.0, 1.5))
    with pytest.raises(ValueError, match="step"):
        RandomWalkRate(step=0.0)
    with pytest.raises(ValueError, match="low"):
        RatePerEpisode(low=0.0)
