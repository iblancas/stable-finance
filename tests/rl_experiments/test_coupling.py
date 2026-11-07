"""The net-exposure limit: what couples the names, charged by the harness."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import quoted_world

from experiments.rl.coupling import NetLimit
from stable_finance import (
    BacktestConfig,
    ExecutionEnv,
    ReplayBuffer,
    collect,
    rollout,
    run_backtest,
)
from stable_finance.evaluation.trades.rl import RandomRate, row_features

MID = BacktestConfig(cost_model="mid", annualization="decisions")
CROSS = BacktestConfig(cost_model="cross", efq=0.7, annualization="decisions")


def _env(world, config=MID, **kwargs):
    return ExecutionEnv(world["target"], world["realized"], world["market"], days=world["days"],
                        config=config, **kwargs)


def _observation(gap, mask=None):
    gap = np.asarray(gap, dtype=float)
    names = np.zeros((len(gap), 3))
    names[:, 2] = gap
    return {"names": names, "mask": np.ones(len(gap), dtype=bool) if mask is None else mask}


def test_only_the_orders_pushing_past_the_limit_are_cut_and_evenly():
    env = NetLimit(_env(quoted_world(seed=1)), 0.1)
    # On the target's net now. Buying 0.6 and selling nothing leaves 0.6 net, 0.5 over the
    # limit, so the buys keep a sixth and the name not trading is left alone.
    rates = env.allowed([1.0, 1.0, 0.0], _observation([0.4, 0.2, -0.6]))
    np.testing.assert_allclose(rates, [1 / 6, 1 / 6, 0.0])
    # Trading every gap in full lands on the target, so nothing is cut however large it is.
    np.testing.assert_array_equal(env.allowed(np.ones(3), _observation([0.4, 0.2, -0.6])),
                                  np.ones(3))


def test_rates_within_the_limit_pass_untouched():
    env = NetLimit(_env(quoted_world(seed=1)), 0.5)
    rates = np.array([0.3, 0.9, 0.1])
    np.testing.assert_array_equal(env.allowed(rates, _observation([0.4, 0.2, -0.2])), rates)


def test_a_name_that_cannot_fill_does_not_count():
    env = NetLimit(_env(quoted_world(seed=1)), 0.1)
    mask = np.array([True, False, True])
    # Only the first buy can fill: 0.3 net, 0.2 over, so it keeps a third. Counting the
    # silent name's order too would have cut it to a sixth.
    rates = env.allowed([1.0, 1.0, 0.0], _observation([0.3, 0.3, -0.6], mask))
    np.testing.assert_allclose(rates, [1 / 3, 1.0, 0.0])


def test_an_order_can_be_cut_but_never_added():
    env = NetLimit(_env(quoted_world(seed=1)), 0.0)
    # The book is short of its target with nothing buying: no buy can be conjured.
    rates = env.allowed(np.zeros(2), _observation([0.3, 0.3]))
    np.testing.assert_array_equal(rates, np.zeros(2))


@pytest.mark.parametrize("config", [MID, CROSS], ids=lambda c: c.cost_model)
def test_the_harness_still_charges_what_was_traded(config):
    world = quoted_world(seed=2)
    env = NetLimit(_env(world, config), 0.02)
    episode = rollout(env, RandomRate(), seed=3)
    report = run_backtest(episode.weights, world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    np.testing.assert_allclose(episode.net, report.series.net, rtol=1e-12, atol=1e-15)


def _lopsided(observation):
    """Longs trade fast and shorts slowly. One rate for every name could not drift at all."""
    return np.where(observation["names"][:, 1] > 0, 0.9, 0.1)


def test_one_rate_for_every_name_never_drifts():
    """A dollar-neutral target moved toward evenly stays dollar neutral, so nothing is cut."""
    world = quoted_world(seed=3, gaps=False)
    env = NetLimit(_env(world), 0.0)
    env.reset()
    assert env.step(0.3)[-1]["cut"] == pytest.approx(0.0, abs=1e-12)


def test_the_limit_binds_and_keeps_the_book_balanced():
    world = quoted_world(seed=3, gaps=False)
    free = rollout(_env(world), _lopsided)
    env = NetLimit(_env(world), 0.01)
    observation, _ = env.reset()
    cuts, ended = [], False
    while not ended:
        observation, _, terminated, truncated, info = env.step(_lopsided(observation))
        cuts.append(info["cut"])
        ended = terminated or truncated
    assert max(cuts) > 0
    limited = env.episode()
    target_net = world["target"].values[:, :, 0].sum(axis=1)

    def drift(episode):
        return np.abs(episode.weights.values[:, :, 0].sum(axis=1) - target_net).mean()

    assert drift(limited) < drift(free)


def test_a_buffer_keeps_the_rate_wanted_and_the_reward_paid():
    """A learner's action is what it asked for; the cut is the world answering."""
    world = quoted_world(seed=4, gaps=False)
    env = NetLimit(_env(world), 0.0)
    buffer = ReplayBuffer(10_000, len(row_features(env)))
    grid = (0.0, 0.5, 1.0)
    collect(env, RandomRate(), buffer, seed=1, rates=grid, reward="net")
    assert set(buffer.contents().rate) == set(grid)
    np.testing.assert_allclose(buffer.contents().reward.reshape(-1, 5).sum(axis=1),
                               env.episode().net, rtol=1e-12, atol=1e-15)


def test_a_bad_limit_is_refused():
    env = _env(quoted_world(seed=1))
    for limit in (-0.1, np.nan):
        with pytest.raises(ValueError, match="limit"):
            NetLimit(env, limit)
