"""Implementation shortfall: the paper portfolio against the fills."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance import (
    BacktestConfig,
    ForwardReturns,
    MarketData,
    PortfolioWeights,
    execute,
    fill_path,
    implementation_shortfall,
    orders_from_weights,
    run_backtest,
)

AXES = ([1, 2], ["A", "B"], [300])


def _two_decisions():
    target = PortfolioWeights(np.array([[[0.5], [-0.5]], [[0.0], [1.0]]]), *AXES)
    realized = ForwardReturns(np.array([[[0.10], [-0.10]], [[0.20], [0.03]]]), *AXES)
    return target, realized


def _world(seed=0, n_decisions=48, n_assets=5):
    rng = np.random.default_rng(seed)
    decisions = np.arange(n_decisions) * 300
    assets = [f"N{i}" for i in range(n_assets)]
    days = np.repeat(np.arange(n_decisions // 8), 8)
    weights = rng.normal(scale=0.3, size=(n_decisions, n_assets))
    weights -= weights.mean(axis=1, keepdims=True)
    realized = rng.normal(scale=0.004, size=(n_decisions, n_assets))
    spread = np.abs(rng.normal(3e-4, 1e-4, size=(n_decisions, n_assets)))
    spread[rng.random(spread.shape) < 0.1] = np.nan            # names that stop quoting
    mid = 100 * np.exp(np.cumsum(rng.normal(scale=0.002, size=(n_decisions, n_assets)), axis=0))
    market = MarketData({"half_spread": spread, "mid": mid,
                         "adv": np.full((n_decisions, n_assets), 5e7)}, decisions, assets)
    return {
        "target": PortfolioWeights(weights[:, :, None], decisions, assets, [300]),
        "realized": ForwardReturns(realized[:, :, None], decisions, assets, [300]),
        "market": market, "days": days, "spread": spread,
    }


def test_full_fills_at_the_touch_cost_only_the_spread():
    target, realized = _two_decisions()
    spread = np.full((2, 2), 0.001)
    fills = fill_path(target, MarketData({"half_spread": spread}, *AXES[:2]), efq=1.0)
    result, = implementation_shortfall(target, fills, realized, half_spread=spread)

    np.testing.assert_allclose(result.paper, [0.10, 0.03])
    np.testing.assert_allclose(result.gross, result.paper)
    np.testing.assert_allclose(result.opportunity, [0.0, 0.0])
    # One unit traded, then 0.5 + 1.5: each pays the half-spread.
    np.testing.assert_allclose(result.execution, [0.001, 0.002])
    np.testing.assert_allclose(result.traded, [1.0, 2.0])
    np.testing.assert_allclose(result.net, [0.099, 0.028])
    np.testing.assert_allclose(result.shortfall, result.paper - result.net)
    assert result.fill_rate == 1.0
    assert result.realized_efq == pytest.approx(1.0)
    assert result.execution_bps == pytest.approx(10.0)


def test_an_unquoted_name_is_opportunity_cost_not_execution_cost():
    target, realized = _two_decisions()
    spread = np.full((2, 2), 0.001)
    spread[0, 1] = np.nan                       # B cannot trade at the first decision
    fills = fill_path(target, MarketData({"half_spread": spread}, *AXES[:2]), efq=1.0)
    result, = implementation_shortfall(target, fills, realized, half_spread=spread)

    # The paper book was short B into a -10% move; the real one was flat.
    np.testing.assert_allclose(result.opportunity, [0.05, 0.0])
    np.testing.assert_allclose(result.gross, [0.05, 0.03])
    np.testing.assert_allclose(result.execution, [0.0005, 0.0015])
    np.testing.assert_allclose(result.gap, [0.5, 0.0])
    # Gaps open: 0.5 + 0.5, then 0.5 + 1.0. Closed: 0.5, then all 1.5.
    assert result.fill_rate == pytest.approx(2.0 / 2.5)
    np.testing.assert_allclose(result.shortfall, result.paper - result.net)


@pytest.mark.parametrize("config", [
    BacktestConfig(cost_model="cross", efq=0.7, annualization="decisions"),
    BacktestConfig(cost_model="cross", efq=1.3, annualization="decisions", holding="drift"),
    BacktestConfig(cost_model="mid", annualization="decisions"),
])
def test_gross_cost_and_net_are_the_harnesss_own(config):
    """THE POINT OF THE MODULE: a shortfall reconciles with the Sharpe beside it.

    The scorecard re-marks the same fills the harness charged. If its gross,
    cost or net drifted from ``run_backtest``'s, a shortfall reported next to a
    net Sharpe would explain a number it was not computed from.
    """
    world = _world(seed=3)
    report = run_backtest(world["target"], world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    drift = config.holding == "drift"
    fills = fill_path(world["target"], world["market"], model=config.cost_model, efq=config.efq,
                      drift=world["realized"].values if drift else None)
    result, = implementation_shortfall(world["target"], fills, world["realized"],
                                       half_spread=world["market"], drift=drift)

    np.testing.assert_allclose(result.gross, report.series.gross, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.execution, report.series.cost, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.net, report.series.net, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.paper - result.net, result.shortfall, atol=1e-15)
    if config.cost_model == "mid":
        assert result.execution_bps == 0.0
    else:
        assert result.realized_efq == pytest.approx(config.efq)


def test_an_order_worked_over_a_schedule_lags_its_target_as_opportunity_cost():
    world = _world(seed=5)
    # A target set at each day's first decision and held all day, so that a
    # schedule shorter than a day can reach it.
    first = np.flatnonzero(np.r_[True, world["days"][1:] != world["days"][:-1]])
    daily = world["target"].values[first][world["days"]]
    target = PortfolioWeights(daily, world["target"].decisions, world["target"].assets, [300])
    config = BacktestConfig(cost_model="impact", equity=1e6, execution_days=0.5,
                            annualization="decisions")
    report = run_backtest(target, world["realized"], market=world["market"],
                          days=world["days"], config=config, with_series=True)
    # Half of an eight-decision day: each order fills in four slices.
    fills = execute(orders_from_weights(target), world["market"], model="impact",
                    equity=1e6, schedule=4)
    result, = implementation_shortfall(target, fills, world["realized"])

    np.testing.assert_allclose(result.net, report.series.net, rtol=1e-12, atol=1e-15)
    assert np.abs(result.opportunity).sum() > 0
    # A quarter of the gap closes at each slice, so the pooled rate sits between.
    assert 0.2 < result.fill_rate < 1
    # The gap opens at each day's first decision and is closed four slices later.
    assert np.all(result.gap[first] > 0)
    np.testing.assert_allclose(result.gap[first + 4], 0.0, atol=1e-12)
    assert np.isnan(result.realized_efq)        # no quotes were passed


def test_a_target_that_moves_faster_than_its_schedule_is_never_reached():
    """Slices of an old order land after the target has moved on.

    Redrawn at every decision, the target is mostly somewhere the outstanding
    slices are not heading, and the fills close about as much gap as they open.
    """
    world = _world(seed=5)
    fills = execute(orders_from_weights(world["target"]), world["market"], model="impact",
                    equity=1e6, schedule=4)
    result, = implementation_shortfall(world["target"], fills, world["realized"])
    assert abs(result.fill_rate) < 0.1


def test_orders_are_scored_against_every_order_filled():
    """A cancelled order is never retried, so the paper book keeps its position."""
    target, realized = _two_decisions()
    orders = orders_from_weights(target)
    spread = np.full((2, 2), 0.001)
    spread[0, 1] = np.nan
    fills = execute(orders, MarketData({"half_spread": spread}, *AXES[:2]), efq=1.0)
    result, = implementation_shortfall(orders, fills, realized)

    np.testing.assert_allclose(result.paper, [0.10, 0.03])     # the target, every order filled
    # B's opening sale never filled and is not retried, so after its +1.5 order
    # the real book is long 1.5 where the paper book is long 1.0.
    np.testing.assert_allclose(result.opportunity, [0.05, (1.0 - 1.5) * 0.03])
    np.testing.assert_allclose(result.gap, [0.5, 0.5])


def _scored():
    world = _world()
    fills = fill_path(world["target"], world["market"], model="cross", efq=0.7)
    return world["target"], fills, world["realized"]


def test_fees_are_a_line_of_their_own():
    """They come off the net, and neither execution nor opportunity moves."""
    target, fills, realized = _scored()
    plain, = implementation_shortfall(target, fills, realized)
    fees = np.linspace(1e-5, 2e-5, len(plain.net))
    billed, = implementation_shortfall(target, fills, realized, fees=fees)
    np.testing.assert_array_equal(billed.execution, plain.execution)
    np.testing.assert_array_equal(billed.opportunity, plain.opportunity)
    np.testing.assert_array_equal(billed.fees, fees)
    np.testing.assert_allclose(billed.net, plain.net - fees)
    np.testing.assert_allclose(billed.shortfall, billed.paper - billed.net, atol=1e-15)
    np.testing.assert_array_equal(plain.fees, 0.0)


def test_fees_that_cannot_be_read_are_refused():
    target, fills, realized = _scored()
    with pytest.raises(ValueError, match="shape"):
        implementation_shortfall(target, fills, realized, fees=np.zeros(2))
    with pytest.raises(ValueError, match="non-negative"):
        implementation_shortfall(target, fills, realized, fees=-np.ones(len(target.decisions)))


def test_market_data_and_an_array_give_the_same_efq():
    world = _world(seed=1)
    fills = fill_path(world["target"], world["market"], efq=0.55)
    from_market, = implementation_shortfall(world["target"], fills, world["realized"],
                                            half_spread=world["market"])
    from_array, = implementation_shortfall(world["target"], fills, world["realized"],
                                           half_spread=world["spread"])
    assert from_market.realized_efq == from_array.realized_efq == pytest.approx(0.55)


def test_a_book_with_nothing_to_trade_has_no_fill_rate():
    _, realized = _two_decisions()
    flat = PortfolioWeights(np.zeros((2, 2, 1)), *AXES)
    result, = implementation_shortfall(flat, fill_path(flat, model="mid"), realized)
    assert np.isnan(result.fill_rate) and np.isnan(result.execution_bps)
    np.testing.assert_allclose(result.shortfall, [0.0, 0.0])


def test_misaligned_inputs_are_refused():
    target, realized = _two_decisions()
    fills = fill_path(target, model="mid")
    other = ForwardReturns(realized.values, [1, 3], ["A", "B"], [300])
    with pytest.raises(ValueError, match="decisions are not aligned"):
        implementation_shortfall(target, fills, other)
    with pytest.raises(ValueError, match="half_spread must have shape"):
        implementation_shortfall(target, fills, realized, half_spread=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="cannot be negative"):
        implementation_shortfall(target, fills, realized, half_spread=-np.ones((2, 2)))


def test_the_function_declares_the_quotes_it_reads():
    assert implementation_shortfall.requires == ("half_spread",)
