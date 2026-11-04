"""The trading environment: the harness's fills and net return, one decision at a time."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import PER_DAY, quoted_world

from stable_finance import (
    BacktestConfig,
    ExecutionEnv,
    ForwardReturns,
    MarketData,
    MissingRolesError,
    Payoff,
    PortfolioWeights,
    implementation_shortfall,
    rollout,
    run_backtest,
)

CONFIGS = [
    BacktestConfig(cost_model="cross", efq=0.7, annualization="decisions"),
    BacktestConfig(cost_model="mid", annualization="decisions"),
    BacktestConfig(cost_model="book", equity=1e6, annualization="decisions"),
    BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5, annualization="decisions"),
    BacktestConfig(cost_model="impact", equity=1e6, execution_days=0, annualization="decisions"),
]


def _env(world, config, **kwargs):
    return ExecutionEnv(world["target"], world["realized"], world["market"],
                        days=world["days"], config=config, **kwargs)


def _random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda observation: rng.uniform(0, 1, size=len(observation["mask"]))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.cost_model}-{c.execution_days}")
def test_trading_to_the_target_is_the_harness(config):
    """THE POINT OF THE MODULE: eta = 1 everywhere is run_backtest on the target.

    A learned policy is only comparable with the rules if its reward is the
    number they are scored on. If this drifted, a policy could win in the
    environment and lose in the harness.
    """
    world = quoted_world(seed=3)
    episode = rollout(_env(world, config), lambda observation: 1.0)
    report = run_backtest(world["target"], world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    np.testing.assert_allclose(episode.net, report.series.net, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(episode.cost, report.series.cost, rtol=1e-12, atol=1e-15)
    shortfall, = implementation_shortfall(episode.target, episode.fills, episode.realized)
    np.testing.assert_allclose(shortfall.net, episode.net, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.cost_model}-{c.execution_days}")
def test_any_policy_is_scored_by_the_harness_on_its_own_path(config):
    """Whatever the rates, the path traded toward reproduces the rewards in run_backtest."""
    world = quoted_world(seed=4)
    episode = rollout(_env(world, config), _random_policy(0))
    report = run_backtest(episode.weights, world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    np.testing.assert_allclose(episode.net, report.series.net, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(episode.gross - episode.cost, episode.net, atol=1e-15)


def test_a_constant_rate_is_partial_adjustment():
    world = quoted_world(seed=1, gaps=False)
    config = BacktestConfig(cost_model="cross", annualization="decisions")
    episode = rollout(_env(world, config), lambda observation: 0.25)
    expected, held = [], np.zeros(5)
    for goal in world["target"].values[:, :, 0]:
        held = 0.75 * held + 0.25 * goal
        expected.append(held)
    np.testing.assert_allclose(episode.weights.values[:, :, 0], expected, atol=1e-15)


def test_slicing_a_trade_saves_nothing_within_the_schedule():
    """A concave cost charged slice by slice would pay an agent to slice forever.

    Within the impact model's schedule every order in a name is one parent,
    so four slices of a trade cost what the trade costs; filled at the
    decision, each slice is charged alone, and slicing looks cheaper.
    """
    n_decisions = 3 * PER_DAY
    decisions = np.arange(n_decisions) * 300
    goal = np.tile([[0.5], [-0.5]], (n_decisions, 1, 1))
    target = PortfolioWeights(goal, decisions, ["A", "B"], [300])
    realized = ForwardReturns(np.zeros(goal.shape), decisions, ["A", "B"], [300])
    market = MarketData({"mid": np.full((n_decisions, 2), 100.0),
                         "adv": np.full((n_decisions, 2), 5e7)}, decisions, ["A", "B"])
    days = np.repeat(np.arange(3), PER_DAY)

    def cost(execution_days, rates):
        config = BacktestConfig(cost_model="impact", equity=1e6, execution_days=execution_days,
                                annualization="decisions")
        env = ExecutionEnv(target, realized, market, days=days, config=config)
        env.reset()
        for t in range(n_decisions):
            env.step(rates[t] if t < len(rates) else 1.0)
        return env.episode().cost.sum()

    at_once, in_four = [1.0], [1 / 4, 1 / 3, 1 / 2, 1.0]          # equal quarters of the gap
    assert cost(1, in_four) == pytest.approx(cost(1, at_once), rel=1e-9)
    assert cost(0, in_four) < 0.9 * cost(0, at_once)


def test_the_policy_sees_nothing_after_the_decision():
    """Corrupting the future, or the arrival quotes and window at any time, changes no observation."""
    world = quoted_world(seed=2)
    market = world["market"]
    n_decisions = market.shape[0]
    rng = np.random.default_rng(9)
    after = {"exec_bid_price": market.get("bid_price") * 0.999,
             "exec_ask_price": market.get("ask_price") * 1.001,
             "window_best_buy": market.get("mid"), "window_best_sell": market.get("mid")}
    honest = MarketData({**market.columns, **after}, market.decisions, market.assets)
    cut = 20

    def corrupt(values):
        values = np.array(values, copy=True)
        values[cut + 1:] = rng.uniform(50, 150, size=values[cut + 1:].shape)
        return values

    columns = {k: corrupt(v) for k, v in market.columns.items()}
    columns.update({k: rng.uniform(50, 150, size=v.shape) for k, v in after.items()})
    leaky = MarketData(columns, market.decisions, market.assets)
    forecast = world["realized"]
    future = {
        "target": PortfolioWeights(corrupt(world["target"].values), *_axes(world)),
        "realized": ForwardReturns(corrupt(world["realized"].values), *_axes(world)),
        "forecast": ForwardReturns(corrupt(forecast.values), *_axes(world)),
    }
    config = BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5)
    seen = []
    for target, realized, data, mu in (
            (world["target"], world["realized"], honest, forecast),
            (future["target"], future["realized"], leaky, future["forecast"])):
        env = ExecutionEnv(target, realized, data, days=world["days"], config=config, forecast=mu)
        observation, _ = env.reset()
        policy = _random_policy(5)
        trail = [observation]
        for _ in range(n_decisions):
            observation, *_ = env.step(policy(observation))
            trail.append(observation)
        seen.append(trail)
    for t in range(cut + 1):
        for key in ("names", "book", "mask"):
            np.testing.assert_array_equal(seen[0][t][key], seen[1][t][key])
    assert not np.array_equal(seen[0][cut + 1]["names"], seen[1][cut + 1]["names"])


def _axes(world):
    target = world["target"]
    return target.decisions, target.assets, target.horizons


def test_a_seeded_episode_replays_exactly():
    world = quoted_world(seed=6)
    config = BacktestConfig(cost_model="cross", annualization="decisions")
    env = _env(world, config, episode_days=1)
    first = rollout(env, _random_policy(1), seed=7)
    again = rollout(env, _random_policy(1), seed=7)
    np.testing.assert_array_equal(first.net, again.net)
    np.testing.assert_array_equal(first.fills.quantity, again.fills.quantity)
    drawn = {rollout(env, _random_policy(1), seed=s).target.decisions[0] for s in range(20)}
    assert len(drawn) > 1


def test_an_episode_is_whole_consecutive_days():
    world = quoted_world(seed=6)
    config = BacktestConfig(cost_model="cross", annualization="decisions")
    env = _env(world, config, episode_days=2)
    assert env.n_episodes == 5
    episode = rollout(env, lambda observation: 1.0, options={"start_day": 3})
    np.testing.assert_array_equal(episode.target.decisions, world["target"].decisions[24:40])
    # A fresh episode starts flat, so its first trade is the whole target.
    np.testing.assert_allclose(episode.weights.values[0], world["target"].values[24])
    with pytest.raises(ValueError, match="start_day"):
        env.reset(options={"start_day": 5})


def test_the_observation_names_its_columns():
    world = quoted_world(seed=8)
    cross = _env(world, BacktestConfig(cost_model="cross"))
    assert cross.name_features == ("held", "target", "gap", "half_spread", "imbalance",
                                   "last_return", "volatility")
    assert cross.book_features == ("progress", "time_of_day", "gross_held", "gross_target",
                                   "gross_gap")
    observation, _ = cross.reset()
    assert observation["names"].shape == (5, len(cross.name_features))
    quoted = np.isfinite(world["market"].get("half_spread")[0])
    np.testing.assert_array_equal(observation["mask"], quoted)
    observation, *_ = cross.step(1.0)
    held = cross.episode().fills.quantity[0, :, 0]
    np.testing.assert_allclose(observation["names"][:, 0], held)
    assert observation["book"][0] == pytest.approx(1 / 48)
    assert observation["book"][1] == pytest.approx(1 / PER_DAY)

    impact = _env(world, BacktestConfig(cost_model="impact", equity=1e6))
    assert impact.name_features[:4] == ("held", "target", "gap", "in_flight")
    assert "adv_scale" in impact.name_features
    impact.reset()
    observation, *_ = impact.step(1.0)
    # The first order is worked over a day, so most of it is still in flight.
    in_flight = observation["names"][:, 3]
    np.testing.assert_allclose(in_flight + observation["names"][:, 0],
                               world["target"].values[0, :, 0])
    assert np.all(np.abs(in_flight) > 0)


def test_bad_actions_and_configurations_are_refused():
    world = quoted_world(seed=0)
    env = _env(world, BacktestConfig(cost_model="cross"))
    env.reset()
    for action in (1.5, -0.1, np.nan, np.ones(4)):
        with pytest.raises(ValueError):
            env.step(action)
    for _ in range(48):
        env.step(0.0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0.0)

    with pytest.raises(ValueError, match="every_decision"):
        _env(world, BacktestConfig(rebalance="daily"))
    with pytest.raises(ValueError, match="rebalanced"):
        _env(world, BacktestConfig(holding="drift"))
    with pytest.raises(TypeError, match="PortfolioWeights"):
        ExecutionEnv(world["realized"], world["realized"], world["market"])
    with pytest.raises(MissingRolesError):
        ExecutionEnv(world["target"], world["realized"],
                     MarketData({"mid": world["market"].get("mid")}, *_axes(world)[:2]))
    with pytest.raises(ValueError, match="days"):
        ExecutionEnv(world["target"], world["realized"], world["market"],
                     config=BacktestConfig(cost_model="impact", equity=1e6))
    with pytest.raises(ValueError, match="days"):
        ExecutionEnv(world["target"], world["realized"], world["market"], episode_days=1)
    with pytest.raises(ValueError, match="longer than"):
        _env(world, BacktestConfig(), episode_days=7)


UNSCHEDULED = [c for c in CONFIGS if c.cost_model != "impact" or c.execution_days == 0]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.cost_model}-{c.execution_days}")
def test_the_payoff_by_name_is_the_books_reward(config):
    """The reward splits exactly by name, which is what lets a name be an agent."""
    world = quoted_world(seed=5)
    env = _env(world, config)
    observation, _ = env.reset()
    policy = _random_policy(2)
    shortfall, ended = [], False
    while not ended:
        observation, net, terminated, truncated, info = env.step(policy(observation))
        payoff = info["payoff"]
        assert payoff.earned.sum() == pytest.approx(info["gross"], abs=1e-15)
        assert payoff.paid.sum() == pytest.approx(info["cost"], abs=1e-15)
        assert payoff.reward("net").sum() == pytest.approx(net, abs=1e-15)
        np.testing.assert_array_equal(payoff.held, observation["names"][:, 0])
        shortfall.append(-payoff.reward("shortfall").sum())
        ended = terminated or truncated
    episode = env.episode()
    scorecard, = implementation_shortfall(episode.target, episode.fills, episode.realized)
    np.testing.assert_allclose(shortfall, scorecard.shortfall, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="reward"):
        payoff.reward("sharpe")
    np.testing.assert_array_equal(payoff.reward(lambda p: -p.paid), -payoff.paid)
    np.testing.assert_allclose(payoff.gap, episode.target.values[-1, :, 0] - payoff.held)


def _played(world, config, n_steps, **kwargs):
    """A fresh environment after the same ``n_steps`` of a random policy."""
    env = _env(world, config, **kwargs)
    observation, _ = env.reset()
    policy = _random_policy(3)
    for _ in range(n_steps):
        observation, *_ = env.step(policy(observation))
    return env


def _assert_same_payoff(left, right):
    for name in left._fields:
        np.testing.assert_allclose(getattr(left, name), getattr(right, name),
                                   rtol=1e-9, atol=1e-15, err_msg=name)


@pytest.mark.parametrize("config", UNSCHEDULED, ids=lambda c: c.cost_model)
def test_parallel_alternates_is_the_step_not_taken(config):
    """A candidate pays in parallel_alternates exactly what it pays played, partial fills and gaps included."""
    world = quoted_world(seed=7)
    rates = np.array([0.0, 0.3, 1.0])
    for n_steps in (0, 5, 20):
        asked = _played(world, config, n_steps).parallel_alternates(rates)
        assert asked.held.shape == (3, 5)
        for j, rate in enumerate(rates):
            info = _played(world, config, n_steps).step(rate)[-1]
            _assert_same_payoff(Payoff(*(x[j] for x in asked)), info["payoff"])


@pytest.mark.parametrize("config", UNSCHEDULED, ids=lambda c: c.cost_model)
def test_parallel_alternates_from_any_position_at_any_decision(config):
    world = quoted_world(seed=8)
    rng = np.random.default_rng(1)
    positions, rates, day = rng.normal(scale=0.1, size=(2, 5)), np.array([0.2, 0.9]), 3
    asked = _env(world, config).parallel_alternates(rates, held=positions, decision=day * PER_DAY)
    assert asked.held.shape == (2, 2, 5)
    for b, position in enumerate(positions):
        for j, rate in enumerate(rates):
            env = _env(world, config, initial=position, episode_days=1)
            env.reset(options={"start_day": day})
            _assert_same_payoff(Payoff(*(x[b, j] for x in asked)), env.step(rate)[-1]["payoff"])
    one = _env(world, config).parallel_alternates(rates, held=positions[0], decision=day * PER_DAY)
    _assert_same_payoff(one, Payoff(*(x[0] for x in asked)))


def test_parallel_alternates_changes_nothing():
    world = quoted_world(seed=9)
    config = BacktestConfig(cost_model="book", equity=1e6, annualization="decisions")
    plain = rollout(_env(world, config), _random_policy(4))
    env = _env(world, config)
    observation, _ = env.reset()
    policy = _random_policy(4)
    ended = False
    while not ended:
        env.parallel_alternates([0.0, 1.0])
        env.parallel_alternates([0.5], held=np.ones(5), decision=0)
        observation, _, ended, _, _ = env.step(policy(observation))
    np.testing.assert_array_equal(env.episode().net, plain.net)


def test_parallel_alternates_refuses_what_it_cannot_answer():
    world = quoted_world(seed=0)
    scheduled = _env(world, BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5))
    assert scheduled.schedule > 0
    with pytest.raises(ValueError, match="schedule"):
        scheduled.parallel_alternates([0.5])
    env = _env(world, BacktestConfig(cost_model="cross"))
    for rates in (0.5, [], [1.5], [np.nan], [[0.5]]):
        with pytest.raises(ValueError, match="rates"):
            env.parallel_alternates(rates)
    for held in (np.ones(4), np.ones((2, 2, 5)), np.full(5, np.nan)):
        with pytest.raises(ValueError, match="held"):
            env.parallel_alternates([0.5], held=held)
    for decision in (-1, 48):
        with pytest.raises(ValueError, match="decision"):
            env.parallel_alternates([0.5], decision=decision)


def test_a_day_end_is_not_the_end():
    """The book carries overnight, so only the panel running out stops a learner bootstrapping."""
    world = quoted_world(seed=6)
    config = BacktestConfig(cost_model="cross", annualization="decisions")
    env = _env(world, config, episode_days=1)
    goal = env.name_features.index("target")

    env.reset(options={"start_day": 2})
    for k in range(PER_DAY):
        assert env.decision_index == 2 * PER_DAY + k
        observation, _, terminated, truncated, _ = env.step(1.0)
    assert (terminated, truncated) == (False, True)
    np.testing.assert_array_equal(observation["names"][:, goal],
                                  world["target"].values[3 * PER_DAY, :, 0])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(1.0)

    env.reset(options={"start_day": 5})
    for _ in range(PER_DAY):
        observation, _, terminated, truncated, _ = env.step(1.0)
    assert (terminated, truncated) == (True, False)
    np.testing.assert_array_equal(observation["names"][:, goal],
                                  world["target"].values[-1, :, 0])


def test_a_starting_book_is_one_weight_per_name():
    world = quoted_world(seed=6)
    start = np.linspace(-0.1, 0.1, 5)
    for initial in (start, start[:, None]):
        env = _env(world, BacktestConfig(cost_model="cross"), initial=initial)
        observation, _ = env.reset()
        np.testing.assert_array_equal(observation["names"][:, 0], start)
    with pytest.raises(ValueError, match="initial"):
        _env(world, BacktestConfig(cost_model="cross"), initial=np.zeros(4))


def test_a_learner_reads_what_each_decision_could_see():
    world = quoted_world(seed=2)
    env = _env(world, BacktestConfig(cost_model="cross"))
    np.testing.assert_array_equal(env.known("target"), world["target"].values[:, :, 0])
    observation, _ = env.reset()
    np.testing.assert_array_equal(env.known("mask")[0], observation["mask"])
    column = env.name_features.index("half_spread")
    np.testing.assert_array_equal(env.known("half_spread")[0], observation["names"][:, column])
    with pytest.raises(ValueError, match="known"):
        env.known("realized")
