"""What makes the names one problem: a limit on the book's net exposure.

Without a coupling a name's reward depends on no other name, and "multi-agent"
is one policy applied many times. Fang et al. (2023, "Learning multi-agent
intention-aware communication for optimal multi-order execution in finance",
KDD) couple their orders through a shared cash balance: buying spends what
selling raises, and when it runs out the environment cuts the buys evenly.

:class:`NetLimit` is that for a long-short book, where the shared resource is
balance: the longs and shorts of a dollar-neutral target must move together,
and a book whose long leg races ahead carries market risk the target never
asked for. If the rates asked for would leave the book's net exposure more
than ``limit`` from the target's, the orders pushing it there are cut evenly
until they do not. What one name gets to trade now depends on the others.

The cut is made to the rates before they are played, so the fills are the
harness's and an episode's path still reproduces its net return in
``run_backtest``. Unlike Fang et al. there is no penalty on top: the only
reward is what the harness charges. Orders can only be cut, not added, so a
book already past the limit with nothing pushing it further stays there until
the rates bring it back.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance import ExecutionEnv


class NetLimit:
    """``env`` with every step's rates cut to keep the net exposure within ``limit``.

    Anything else is the environment's, so it can be played by ``rollout`` and
    filled by ``collect``. ``info["cut"]`` is the share of the wanted trading
    the limit removed at that step.
    """

    def __init__(self, env: ExecutionEnv, limit: float) -> None:
        if not np.isfinite(limit) or limit < 0:
            raise ValueError("limit must be finite and not negative")
        self.env = env
        self.limit = float(limit)
        self._gap = env.name_features.index("gap")

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._observation, info = self.env.reset(seed=seed, options=options)
        return self._observation, info

    def allowed(self, rates: ArrayLike, observation: dict) -> NDArray[np.float64]:
        """``rates`` with the orders that breach the limit cut evenly."""
        gap = observation["names"][:, self._gap]
        rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), gap.shape)
        order = np.where(observation["mask"], rates * gap, 0.0)
        drift = order.sum() - gap.sum()                      # net of held less target, after
        excess = abs(drift) - self.limit
        pushing = np.sign(order) == np.sign(drift)
        push = np.abs(order[pushing]).sum()
        if excess <= 0 or push == 0:
            return rates
        return np.where(pushing, rates * max(0.0, 1.0 - excess / push), rates)

    def step(self, action: ArrayLike):
        rates = self.allowed(action, self._observation)
        gap = self._observation["names"][:, self._gap]
        wanted = np.abs(np.broadcast_to(action, gap.shape) * gap).sum()
        self._observation, net, terminated, truncated, info = self.env.step(rates)
        info["cut"] = 1.0 - np.abs(rates * gap).sum() / wanted if wanted > 0 else 0.0
        return self._observation, net, terminated, truncated, info
