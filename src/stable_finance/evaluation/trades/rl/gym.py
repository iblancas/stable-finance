"""A Gymnasium view of :class:`~stable_finance.evaluation.trades.rl.env.ExecutionEnv`.

The environment itself depends on nothing but numpy, so the library never
imports a reinforcement-learning stack. This module is the one place that
does, for training a policy with a library that expects a ``gymnasium.Env``
(Stable-Baselines3, say); it needs the ``rl`` extra
(``pip install stable-finance[rl]``) and is not imported by the package.

It changes three things, none of them what the book is charged:

* SPACES. The observation is a ``Dict`` of ``names`` (one row of features
  per name), ``book`` and ``mask``, in float32 as most learners expect. The
  action is a ``Box`` of one trading rate per name in ``[0, 1]``, clipped to
  it, or with ``rates`` a ``MultiDiscrete`` choice among those rates, the
  discrete action of the Q-learning papers.
* REWARD SCALE. The reward is the net return times ``reward_scale``
  (default ``1e4``, basis points of equity), since a return of a few
  hundredths of a percent per step is too small for most learners' defaults.
  ``info["net"]`` keeps the unscaled net return, and
  :meth:`~stable_finance.evaluation.trades.rl.env.ExecutionEnv.episode`
  is what to score.
* SEEDING. ``reset(seed=...)`` seeds both Gymnasium's generator and the
  environment's own episode draw, so a seeded run replays exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import gymnasium
import numpy as np
from gymnasium import spaces

from stable_finance.evaluation.trades.rl.env import ExecutionEnv

__all__ = ["GymExecutionEnv"]


class GymExecutionEnv(gymnasium.Env):
    """:class:`ExecutionEnv` as a ``gymnasium.Env``; see the module docstring."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, env: ExecutionEnv, *, reward_scale: float = 1e4,
                 rates: Sequence[float] | None = None) -> None:
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.env = env
        self.reward_scale = float(reward_scale)
        n_assets = len(env.assets)
        self.observation_space = spaces.Dict({
            "names": spaces.Box(-np.inf, np.inf, (n_assets, len(env.name_features)), np.float32),
            "book": spaces.Box(-np.inf, np.inf, (len(env.book_features),), np.float32),
            "mask": spaces.MultiBinary(n_assets),
        })
        if rates is None:
            self.rates = None
            self.action_space = spaces.Box(0.0, 1.0, (n_assets,), np.float32)
        else:
            self.rates = np.asarray(rates, dtype=np.float64)
            if (self.rates.ndim != 1 or len(self.rates) < 2 or not np.all(np.isfinite(self.rates))
                    or np.any((self.rates < 0) | (self.rates > 1))):
                raise ValueError("rates must be at least two trading rates in [0, 1]")
            self.action_space = spaces.MultiDiscrete(np.full(n_assets, len(self.rates)))

    @staticmethod
    def _observation(observation: dict) -> dict:
        return {"names": observation["names"].astype(np.float32),
                "book": observation["book"].astype(np.float32),
                "mask": observation["mask"].astype(np.int8)}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed, options=options)
        return self._observation(observation), info

    def step(self, action):
        if self.rates is None:
            rate = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        else:
            rate = self.rates[np.asarray(action, dtype=np.int64)]
        observation, net, terminated, truncated, info = self.env.step(rate)
        info = {**info, "net": net}
        return (self._observation(observation), net * self.reward_scale, terminated,
                truncated, info)
