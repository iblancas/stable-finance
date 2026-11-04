"""Tabular Q-learning: each learner against answers known in advance."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import PER_DAY, quoted_world

from experiments.rl.qlearning import (
    GAP_EDGES,
    GAP_GRID,
    RATES,
    SIZE_EDGES,
    Cells,
    TablePolicy,
    explore,
    sweep,
)
from experiments.rl.study import (
    ConstantRate,
    make_env,
    require_no_leak,
    synthetic_panels,
)
from stable_finance import (
    BacktestConfig,
    ExecutionEnv,
    ForwardReturns,
    MarketData,
    PortfolioWeights,
    rollout,
)

CROSS = BacktestConfig(cost_model="cross", efq=1.0, annualization="decisions")
MID = BacktestConfig(cost_model="mid", annualization="decisions")
SMALL = {"n_assets": 40, "train_days": 12, "test_days": 6, "per_day": 39}


def _traded(episode):
    return np.abs(episode.fills.quantity).sum()


@pytest.mark.parametrize(("learn", "share"), [(sweep, 0.05),
                                              (lambda env: explore(env, episodes=150), 0.35)],
                         ids=["sweep", "explore"])
def test_with_no_skill_there_is_nothing_to_trade_for(learn, share):
    """A forecast with no skill makes every trade a pure cost, so a learner stops trading.

    The sweep all but does. Exploring keeps trading on noise for longer: it sees
    each rate at different moments, and the best of several noisy averages
    always looks like a gain. The share falls only slowly with more episodes.
    """
    train, test = synthetic_panels(1, ic=0.0, **SMALL)
    policy = learn(make_env(train, CROSS, episode_days=1))
    env = make_env(test, CROSS)
    assert _traded(rollout(env, policy)) < share * _traded(rollout(env, ConstantRate(1.0)))


def test_at_no_cost_a_true_signal_is_chased_all_the_way():
    """With nothing to pay and a real edge, an under-held name trades fully."""
    train, _ = synthetic_panels(2, ic=0.5, **SMALL)
    policy = sweep(make_env(train, MID))
    n_gap = len(GAP_EDGES) + 1
    best = policy.rates[policy.best].reshape(len(SIZE_EDGES) + 1, n_gap, -1)
    under_held = np.digitize(0.35, GAP_EDGES)                  # a clear gap toward the target
    tradable = slice(0, policy.cells.n_cost - 1)
    assert np.all(best[1:, under_held:, tradable] == 1.0)


def _quadratic_world(n_assets=60, n_decisions=16):
    """A fixed target, no returns, and an impact cost of exactly ``q**2`` a name."""
    decisions = np.arange(n_decisions) * 300
    assets = [f"N{i}" for i in range(n_assets)]
    goal = np.where(np.arange(n_assets) % 2 == 0, 1.0, -1.0) / n_assets
    target = PortfolioWeights(np.broadcast_to(goal[None, :, None], (n_decisions, n_assets, 1)),
                              decisions, assets, [300])
    realized = ForwardReturns(np.zeros(target.values.shape), decisions, assets, [300])
    market = MarketData({"mid": np.full((n_decisions, n_assets), 100.0),
                         "adv": np.full((n_decisions, n_assets), 1e6)}, decisions, assets)
    # Per unit of weight the cost is b * 100 |q| equity / adv in basis points.
    config = BacktestConfig(cost_model="impact", equity=1e6, execution_days=0,
                            impact=(0.0, 100.0, 0.0), annualization="decisions")
    return ExecutionEnv(target, realized, market, days=np.arange(n_decisions) // 8, config=config)


def test_quadratic_cost_gives_the_closed_form_rate():
    """The textbook problem: min sum gamma^t (lam q^2 + kappa gap^2), with gap_t+1 = gap_t - q.

    Its value is P gap^2 with P = lam (kappa + gamma P) / (lam + kappa + gamma P), and the
    best rate is the same at every gap: eta = (kappa + gamma P) / (lam + kappa + gamma P).
    """
    env = _quadratic_world()
    q = np.full(len(env.assets), 0.01)
    lam = env.parallel_alternates([1.0], held=env.known("target")[0] - q).paid[0, 0] / q[0]**2
    assert lam == pytest.approx(1.0)
    kappa, gamma = 0.5, 0.9
    p = 1.0
    for _ in range(2000):
        p = lam * (kappa + gamma * p) / (lam + kappa + gamma * p)
    exact = (kappa + gamma * p) / (lam + kappa + gamma * p)

    def preference(payoff):
        return -payoff.paid - kappa * payoff.gap**2

    rates = np.linspace(0.0, 1.0, 21)
    steps = np.geomspace(0.02, 8.0, 60)
    middles = np.r_[steps[0] / 2, np.sqrt(steps[:-1] * steps[1:]), steps[-1] * 1.25]
    policy = sweep(env, rates=rates, reward=preference, gamma=gamma,
                   gap_edges=np.r_[-steps[::-1], steps], gaps=np.r_[-middles[::-1], middles])
    chosen = policy.rates[policy.best][policy.visits.sum(axis=1) > 0]
    moved = chosen[chosen > 0]              # the cell holding a gap of about zero has no best rate
    assert np.median(moved) == pytest.approx(exact, abs=0.05)

    # The default cells are coarse, and it shows: the rate comes out too slow.
    coarse = sweep(env, rates=rates, reward=preference, gamma=gamma)
    assert np.median(coarse.rates[coarse.best][coarse.best > 0]) < exact - 0.05


def test_the_synthetic_prices_hold_no_future():
    """A price built from the return it is scored on hands a policy the answer."""
    train, _ = synthetic_panels(6, **SMALL)
    env = make_env(train, CROSS)
    assert abs(require_no_leak(env, train)) < 0.05
    # The last move is the return earned over the decision before, and only that.
    earned = train.realized.values[:-1, :, 0]
    np.testing.assert_allclose(env.known("last_return")[1:], earned, atol=1e-10)


def test_a_panel_that_leaks_is_refused():
    train, _ = synthetic_panels(6, **SMALL)
    env = make_env(train, CROSS)
    realized = train.realized
    told = ForwardReturns(env.known("last_return")[:, :, None], realized.decisions,
                          realized.assets, realized.horizons)
    with pytest.raises(ValueError, match="contain the future"):
        require_no_leak(env, train._replace(realized=told))


def test_a_sweep_must_try_every_gap_cell():
    train, _ = synthetic_panels(4, **SMALL)
    with pytest.raises(ValueError, match="every gap cell"):
        sweep(make_env(train, CROSS), gaps=GAP_GRID[:4])


def test_the_sweep_is_the_same_every_run():
    train, _ = synthetic_panels(3, **SMALL)
    first, again = sweep(make_env(train, CROSS)), sweep(make_env(train, CROSS))
    np.testing.assert_array_equal(first.values, again.values)
    assert first.visits.sum() > 0


def test_exploring_is_fixed_by_its_seed():
    train, _ = synthetic_panels(3, **SMALL)

    def learn(seed):
        return explore(make_env(train, CROSS, episode_days=1), episodes=5, seed=seed).values

    np.testing.assert_array_equal(learn(1), learn(1))
    assert not np.array_equal(learn(1), learn(2))


def test_the_sweep_needs_fills_that_depend_only_on_the_decision():
    world = quoted_world(seed=1)
    config = BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5)
    env = ExecutionEnv(world["target"], world["realized"], world["market"], days=world["days"],
                       config=config, episode_days=1)
    with pytest.raises(ValueError, match="schedule"):
        sweep(env)
    assert explore(env, episodes=2).visits.sum() == 2 * PER_DAY * 5     # exploring still runs


def test_a_long_and_a_short_share_a_cell():
    train, _ = synthetic_panels(4, **SMALL)
    env = make_env(train, CROSS)
    cells = Cells(env)
    target, gap = np.array([0.02, -0.02]), np.array([0.01, -0.01])
    long, short = cells.index(target, gap, 2e-4, True, 0.02)
    assert long == short
    over, under = cells.index(0.02, np.array([-0.01, 0.01]), 2e-4, True, 0.02)
    assert over != under


def test_a_name_that_cannot_trade_has_a_cell_of_its_own():
    train, _ = synthetic_panels(4, **SMALL)
    cells = Cells(make_env(train, CROSS))
    tradable = cells.index(0.02, 0.01, 2e-4, True, 0.02)
    silent = cells.index(0.02, 0.01, 2e-4, False, 0.02)
    assert silent != tradable
    assert silent == cells.index(0.02, 0.01, 9e-4, False, 0.02)


def test_a_cell_never_tried_holds_still():
    train, _ = synthetic_panels(5, **SMALL)
    policy = sweep(make_env(train, CROSS))
    never = policy.visits.sum(axis=1) == 0
    assert never.any()
    assert np.all(policy.rates[policy.best[never]] == 0.0)


def test_a_rate_never_tried_does_not_win_by_default():
    """Rewards here are mostly negative, so an untried rate's zero would beat every real value."""
    train, _ = synthetic_panels(4, **SMALL)
    cells = Cells(make_env(train, CROSS))
    values = np.array([[-1.0, 0.0, -2.0]])
    visits = np.array([[5.0, 0.0, 5.0]])
    assert TablePolicy(cells, values, visits, np.array([0.0, 0.5, 1.0])).best[0] == 0


def test_rates_must_ascend():
    train, _ = synthetic_panels(4, **SMALL)
    with pytest.raises(ValueError, match="ascending"):
        sweep(make_env(train, CROSS), rates=(1.0, 0.0))
    assert RATES == tuple(sorted(RATES))
