"""The study's baselines and scorecard, which every learner is judged against."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.rl.study import (
    ConstantRate,
    NoTradeBand,
    _pool,
    compare,
    make_env,
    score,
    synthetic_panels,
    tuned_band,
    tuned_rate,
)
from stable_finance import BacktestConfig, rollout

CROSS = BacktestConfig(cost_model="cross", efq=1.0, annualization="decisions")
MID = BacktestConfig(cost_model="mid", annualization="decisions")
SMALL = {"n_assets": 30, "train_days": 6, "test_days": 4, "per_day": 39}


def test_a_band_holds_inside_and_trades_to_its_edge_outside():
    train, _ = synthetic_panels(1, **SMALL)
    env = make_env(train, MID)
    band = NoTradeBand(env, 1.0)
    observation, _ = env.reset()
    gap = env.name_features.index("gap")
    gross = env.book_features.index("gross_target")
    for _ in range(5):
        before = observation["names"][:, gap].copy()
        edge = observation["book"][gross] / len(before)
        observation, _, _, _, info = env.step(band(observation))
        after = info["payoff"].gap
        inside = np.abs(before) <= edge
        np.testing.assert_allclose(after[inside], before[inside])
        np.testing.assert_allclose(np.abs(after[~inside]), edge)


def test_a_band_of_no_width_trades_to_the_target():
    train, _ = synthetic_panels(1, **SMALL)
    env = make_env(train, CROSS)
    np.testing.assert_allclose(rollout(env, NoTradeBand(env, 0.0)).net,
                               rollout(env, ConstantRate(1.0)).net)


def test_tuning_reads_the_cost():
    """With no skill the best rule barely trades; at no cost with skill it trades briskly."""
    no_skill, _ = synthetic_panels(2, ic=0.0, **SMALL)
    env = make_env(no_skill, CROSS)
    assert tuned_rate(env).rate <= 0.05
    assert tuned_band(env).width >= 3.0
    skill, _ = synthetic_panels(2, ic=0.5, **SMALL)
    assert tuned_rate(make_env(skill, MID)).rate == 1.0


def test_the_scorecard_adds_up():
    train, _ = synthetic_panels(3, **SMALL)
    env = make_env(train, CROSS)
    slow, fast = rollout(env, ConstantRate(0.2)), rollout(env, ConstantRate(1.0))
    row = score(slow, fast, train.days)
    assert row["net_bp_day"] == pytest.approx(
        row["paper_bp_day"] - row["execution_bp_day"] - row["opportunity_bp_day"])
    assert row["gain_bp_day"] == pytest.approx(
        row["net_bp_day"] - score(fast, fast, train.days)["net_bp_day"])
    assert score(fast, fast, train.days)["gain_bp_day"] == 0.0
    assert row["gross_held"] == pytest.approx(
        np.abs(np.cumsum(slow.fills.quantity[:, :, 0], axis=0)).sum(axis=1).mean())


def test_seeds_are_pooled_with_their_range():
    pooled = _pool([{"gain_bp_day": 1.0, "net_bp_day": 3.0}, {"gain_bp_day": 3.0, "net_bp_day": 5.0}])
    assert pooled == {"gain_bp_day": 2.0, "net_bp_day": 4.0, "seeds": 2,
                      "gain_low": 1.0, "gain_high": 3.0}


def test_every_arm_is_judged_against_the_same_band():
    train, test = synthetic_panels(4, **SMALL)
    arms = compare(train, test, CROSS, seeds=2, episodes=3, dqn=False, limit=0.02,
                   log=lambda text: None)
    assert arms["no-trade band"]["gain_bp_day"] == 0.0
    assert arms["limit: no-trade band"]["gain_bp_day"] == 0.0
    assert arms["epsilon-greedy table"]["seeds"] == 2
    assert {"trade to target", "best constant rate", "sweep"} <= set(arms)
