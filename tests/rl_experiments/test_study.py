"""The study's baselines and scorecard, which every learner is judged against."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.rl.study import (
    RATE_GRID,
    ConstantRate,
    NoTradeBand,
    _pool,
    compare,
    implied_risk,
    main,
    make_env,
    score,
    synthetic_panels,
    tuned_band,
    tuned_rate,
)
from stable_finance import BacktestConfig, no_trade_band, partial_adjustment, rollout

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


def test_the_band_is_the_librarys_rule_played_a_decision_at_a_time():
    """One band, not two: where every order fills, the two follow the same path."""
    train, _ = synthetic_panels(2, **SMALL)
    env = make_env(train, MID)
    played = rollout(env, NoTradeBand(env, 1.5)).weights.values
    mean_target = np.abs(train.target.values[:, :, 0]).mean(axis=1, keepdims=True)
    rule = no_trade_band(train.target, 1.5, cost_belief=mean_target).values
    np.testing.assert_allclose(played, rule, atol=1e-12)


def test_a_constant_rate_is_the_librarys_partial_adjustment():
    train, _ = synthetic_panels(2, **SMALL)
    played = rollout(make_env(train, MID), ConstantRate(0.2)).weights.values
    np.testing.assert_allclose(played, partial_adjustment(train.target, 0.2).values, atol=1e-12)


def test_the_implied_risk_aversion_makes_the_targets_size_the_best_one():
    train, _ = synthetic_panels(3, **SMALL)
    risk = implied_risk(train)
    weights, mu = train.target.values[:, :, 0], train.forecast.values[:, :, 0]

    def utility(k):
        return (k * weights * mu).sum(axis=1).mean() - risk.penalty(k * weights).mean()

    sizes = np.linspace(0.5, 1.5, 101)
    assert sizes[np.argmax([utility(k) for k in sizes])] == pytest.approx(1.0)


def test_a_forecast_with_no_edge_implies_no_risk_aversion():
    train, _ = synthetic_panels(3, ic=0.0, **SMALL)
    with pytest.raises(ValueError, match="no edge"):
        implied_risk(train)


def test_tuning_under_risk_reads_the_aversion():
    """No aversion is the net return; a crushing one holds as little as the grid allows."""
    train, _ = synthetic_panels(3, **SMALL)
    env = make_env(train, CROSS)
    risk = implied_risk(train)
    assert tuned_rate(env, risk=risk._replace(aversion=0.0)).rate == tuned_rate(env).rate
    assert tuned_rate(env, risk=risk._replace(aversion=1e12)).rate == min(RATE_GRID)


def test_an_arms_utility_is_judged_against_the_bands_like_its_net():
    train, _ = synthetic_panels(3, **SMALL)
    env = make_env(train, CROSS)
    risk = implied_risk(train)
    slow, fast = rollout(env, ConstantRate(0.2)), rollout(env, ConstantRate(1.0))
    row, base = score(slow, fast, train.days, risk), score(fast, fast, train.days, risk)
    assert row["utility_gain_bp_day"] == pytest.approx(
        row["utility_bp_day"] - base["utility_bp_day"])
    assert row["utility_gain_se"] > 0
    assert base["utility_gain_bp_day"] == 0.0


def test_doing_nothing_is_on_the_table_and_scores_nothing():
    train, test = synthetic_panels(2, **SMALL)
    arms = compare(train, test, CROSS, seeds=1, episodes=2, dqn=False, risk=implied_risk(train),
                   log=lambda _: None)
    idle = arms["no trading"]
    assert idle["net_bp_day"] == idle["utility_bp_day"] == idle["gross_held"] == 0.0


def test_every_arm_can_want_the_same_utility():
    train, test = synthetic_panels(2, **SMALL)
    risk = implied_risk(train)
    arms = compare(train, test, CROSS, seeds=1, episodes=2, dqn=False, risk=risk, averse=True,
                   log=lambda _: None)
    assert all("utility_bp_day" in row for row in arms.values())
    with pytest.raises(ValueError, match="risk"):
        compare(train, test, CROSS, dqn=False, averse=True)


def test_every_learner_is_handed_the_same_reward(monkeypatch):
    """A learner left on the old reward would look like a result about risk."""
    pytest.importorskip("torch")
    from experiments.rl import dqn, ppo, study

    seen = []

    class Learned:
        def policy(self, env):
            return ConstantRate(0.5)

    def learner(env, *, reward, **_):
        seen.append(reward)
        return Learned()

    def table(env, *, reward, **_):
        seen.append(reward)
        return ConstantRate(0.5)

    monkeypatch.setattr(study, "explore", table)
    monkeypatch.setattr(study, "sweep", table)
    monkeypatch.setattr(dqn, "train", learner)
    monkeypatch.setattr(ppo, "train", learner)
    train, test = synthetic_panels(2, **SMALL)
    study.compare(train, test, CROSS, seeds=1, limit=0.05, risk=implied_risk(train), averse=True,
                  log=lambda _: None)
    assert len(seen) == 9
    assert callable(seen[0]) and all(reward is seen[0] for reward in seen)


def test_real_data_is_told_its_months():
    with pytest.raises(SystemExit):
        main([])


#: A study small enough that a check which should have refused it fails in seconds.
TINY = ["--synthetic", "--no-dqn", "--seeds", "1", "--episodes", "2"]


@pytest.fixture
def small_world(monkeypatch):
    from experiments.rl import study

    made = study.synthetic_panels
    monkeypatch.setattr(study, "synthetic_panels", lambda ic: made(2, ic=ic, **SMALL))
    return made


def test_the_default_reward_has_no_aversion_to_scale(small_world, tmp_path):
    with pytest.raises(SystemExit):
        main([*TINY, "--aversion-scales", "0.5", "--out", str(tmp_path)])


@pytest.mark.parametrize("scales", [["-1"], ["nan"], ["inf"], ["0.5", "0.5"]])
def test_aversion_scales_are_distinct_finite_and_not_negative(scales, small_world, tmp_path):
    with pytest.raises(SystemExit):
        main([*TINY, "--reward", "risk", "--aversion-scales", *scales, "--out", str(tmp_path)])


def test_each_risk_aversion_trains_and_scores_its_own_arms(small_world, monkeypatch, tmp_path):
    """Both panels at every multiple of the one implied aversion, each in its own file."""
    from experiments.rl import study

    seen, compared = [], study.compare

    def spy(*args, risk, averse, **kwargs):
        seen.append((risk.aversion, averse))
        return compared(*args, risk=risk, averse=averse, **kwargs)

    monkeypatch.setattr(study, "compare", spy)
    assert main([*TINY, "--reward", "risk", "--aversion-scales", "1", "0.25",
                 "--out", str(tmp_path)]) == 0
    implied = implied_risk(small_world(2, ic=0.03, **SMALL)[0]).aversion
    assert seen == pytest.approx([(implied, True), (implied / 4, True)] * 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "study-synthetic-risk-x0.25.json", "study-synthetic-risk.json"]
    whole, quarter = (json.loads((tmp_path / f"study-synthetic-risk{s}.json").read_text())
                      for s in ("", "-x0.25"))
    assert list(whole) == list(quarter) == ["skill", "no skill"]
    for panel in whole:
        full, part = whole[panel]["trade to target"], quarter[panel]["trade to target"]
        assert part["net_bp_day"] == pytest.approx(full["net_bp_day"])
        assert part["net_bp_day"] - part["utility_bp_day"] == pytest.approx(
            (full["net_bp_day"] - full["utility_bp_day"]) / 4)


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
    pooled = _pool([{"gain_bp_day": 1.0, "net_exposure": 0.1},
                    {"gain_bp_day": 3.0, "net_exposure": 0.5}])
    assert pooled == pytest.approx({"gain_bp_day": 2.0, "net_exposure": 0.3, "seeds": 2,
                                    "gain_low": 1.0, "gain_high": 3.0, "exposure_high": 0.5})


def test_every_arm_is_judged_against_the_same_band():
    train, test = synthetic_panels(4, **SMALL)
    arms = compare(train, test, CROSS, seeds=2, episodes=3, dqn=False, limit=0.02,
                   log=lambda text: None)
    assert arms["no-trade band"]["gain_bp_day"] == 0.0
    assert arms["limit: no-trade band"]["gain_bp_day"] == 0.0
    assert arms["epsilon-greedy table"]["seeds"] == 2
    assert {"trade to target", "best constant rate", "sweep"} <= set(arms)
