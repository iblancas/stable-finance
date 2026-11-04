"""Learned trading policies: where reinforcement-learning methods go.

The rules in :mod:`stable_finance.evaluation.trades.rules` and the allocator
in :mod:`stable_finance.evaluation.trades.rebalance` are hand-written
policies for the same decision a learned one makes: given the portfolio held,
the target, the forecast and what trading costs, how far to move. A policy
here takes that state and returns the same ``PortfolioWeights`` (or
``Orders``) the rules do, so the harness in
:mod:`stable_finance.evaluation.backtest` and the execution models in
:mod:`stable_finance.evaluation.execution` score it unchanged.

:class:`~stable_finance.evaluation.trades.rl.parametric.ParametricPolicy` is
the first: a parametric portfolio policy with transaction costs, fit
deterministically on net-of-cost utility.
:class:`~stable_finance.evaluation.trades.rl.env.ExecutionEnv` is the
environment a policy is learned in, step by step: the harness's own fills
and net return behind a Gymnasium-style ``reset`` and ``step``, with an
optional adapter in :mod:`~stable_finance.evaluation.trades.rl.gym`.
:mod:`~stable_finance.evaluation.trades.rl.replay` stores what is played
there for learners that train on stored steps, from random behaviours or an
expert's path, with every reward priced by the environment.
"""

from stable_finance.evaluation.trades.rl.env import (
    ExecutionEnv,
    Payoff,
    Rollout,
    rollout,
)
from stable_finance.evaluation.trades.rl.parametric import ParametricPolicy, to_weights
from stable_finance.evaluation.trades.rl.replay import (
    FollowPath,
    RandomRate,
    RandomWalkRate,
    RatePerEpisode,
    ReplayBuffer,
    Transitions,
    collect,
    name_rows,
    row_features,
)

__all__ = ["ExecutionEnv", "FollowPath", "ParametricPolicy", "Payoff", "RandomRate",
           "RandomWalkRate", "RatePerEpisode", "ReplayBuffer", "Rollout", "Transitions",
           "collect", "name_rows", "rollout", "row_features", "to_weights"]
