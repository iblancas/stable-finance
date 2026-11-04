"""Experience for learners that train on stored steps: a replay buffer and how to fill it.

A ROW IS A NAME AT A DECISION, not the book at a decision. The net return is
exactly a sum over names (see :mod:`~stable_finance.evaluation.trades.rl.env`)
and one policy is applied to every name, so an episode of ``T`` decisions
gives ``T * n_assets`` transitions of one small problem, where a row of the
whole book would be ``T`` transitions of a problem with an action per name.
:func:`name_rows` builds a row: the name's own features, then the book's. It
leaves out ``progress``, which counts through whatever the episode is, a day
in training and the whole test period in evaluation, so a policy that learned
from it would read the wrong clock; ``time_of_day`` is the one that means the
same in both.

THE HARNESS PRICES EVERY ROW. :func:`collect` is the only way in: it plays a
policy in the environment and stores what the environment reports. An expert
therefore supplies what it would trade and never what that paid, and the rows
of an episode add up to the net return the harness books for it.

WHO PLAYS. Deep Q-learning fills its buffer from its own epsilon-greedy play,
as Ning, Lin and Jaimungal (2018, "Double deep Q-learning for optimal
execution", arXiv:1812.06600) do; demonstrations are an addition to that, not
a requirement, as in Hester et al. (2018, "Deep Q-learning from
demonstrations", AAAI). Either fits here, since a policy is any callable:

* an expert, as that callable, or as a weight path through :class:`FollowPath`;
* a random behaviour, for coverage. Independent rates average out to a brisk
  partial adjustment, so on a target that moves slowly the book stays close
  to it and the large gaps a learner must also value are rarely visited.
  :class:`RandomWalkRate` does a little better and :class:`RatePerEpisode`,
  which lets some names trade slowly all day, does most, so it is the one to
  start from. (A target with no persistence reverses this: the gap is then
  mostly the target's own jump, and a brisk book chases it hardest.)

These play the real market, not a simulated one: the randomness is in the
rates, and a row's price and return are what happened.

``rates`` gives :func:`collect` a grid, for learners with a discrete action.
A rate is moved to the grid BEFORE it is played, so the action stored is the
action that earned the reward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.trades.rl.env import (
    REWARDS,
    ExecutionEnv,
    _start_episode,
)

__all__ = ["FollowPath", "RandomRate", "RandomWalkRate", "RatePerEpisode", "ReplayBuffer",
           "Transitions", "collect", "name_rows", "row_features"]


def row_features(env: ExecutionEnv) -> tuple[str, ...]:
    """The columns of :func:`name_rows`."""
    return env.name_features + tuple(f for f in env.book_features if f != "progress")


def name_rows(env: ExecutionEnv, observation: dict) -> NDArray[np.float64]:
    """One row per name: its own features, then the book's without ``progress``."""
    names = observation["names"]
    book = observation["book"][[i for i, f in enumerate(env.book_features) if f != "progress"]]
    return np.concatenate([names, np.broadcast_to(book, (len(names), len(book)))], axis=1)


class Transitions(NamedTuple):
    """Rows of a :class:`ReplayBuffer`, one per name and decision.

    ``rate`` is the rate played and ``reward`` what it earned, in fractions
    of equity; ``next_rows`` is the state the name moved to. ``terminated``
    is True only where no next decision exists, so a learner stops
    bootstrapping there and nowhere else. ``tradable`` is the observation's
    mask: False rows are kept so an episode's rewards still add up, though
    the rate had no effect. ``day`` and ``source`` say where the row came from.
    """

    rows: NDArray[np.float32]
    rate: NDArray[np.float64]
    reward: NDArray[np.float64]
    next_rows: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    tradable: NDArray[np.bool_]
    day: NDArray[np.int64]
    source: NDArray[np.str_]


class ReplayBuffer:
    """The latest ``capacity`` transitions, sampled uniformly.

    ``n_features`` is the width of a row, ``len(row_features(env))``, and
    ``seed`` fixes what :meth:`sample` draws. Features are held in float32,
    as a network takes them; rewards stay float64, being small and summed.
    """

    def __init__(self, capacity: int, n_features: int, *, seed: int | None = None) -> None:
        for name, value in (("capacity", capacity), ("n_features", n_features)):
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.capacity, self.n_features = int(capacity), int(n_features)
        self._rng = np.random.default_rng(seed)
        self._rows = np.zeros((self.capacity, self.n_features), dtype=np.float32)
        self._next_rows = np.zeros_like(self._rows)
        self._rate = np.zeros(self.capacity)
        self._reward = np.zeros(self.capacity)
        self._terminated = np.zeros(self.capacity, dtype=bool)
        self._tradable = np.zeros(self.capacity, dtype=bool)
        self._day = np.zeros(self.capacity, dtype=np.int64)
        self._source = np.zeros(self.capacity, dtype=np.int64)
        self._sources: list[str] = []
        self._write = self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def sources(self) -> tuple[str, ...]:
        """Every source added so far, whether or not its rows are still held."""
        return tuple(self._sources)

    def add(self, rows: ArrayLike, rate: ArrayLike, reward: ArrayLike, next_rows: ArrayLike, *,
            terminated: bool, tradable: ArrayLike, day: int, source: str) -> None:
        """Store one decision's rows, overwriting the oldest once full."""
        rows, next_rows = np.asarray(rows), np.asarray(next_rows)
        n = len(rows)
        if rows.shape != (n, self.n_features) or next_rows.shape != rows.shape:
            raise ValueError(f"rows must have shape (n, {self.n_features})")
        if n > self.capacity:
            raise ValueError(f"{n} rows do not fit a buffer of {self.capacity}")
        if source not in self._sources:
            self._sources.append(source)
        at = (self._write + np.arange(n)) % self.capacity
        self._rows[at], self._next_rows[at] = rows, next_rows
        self._rate[at], self._reward[at] = rate, reward
        self._terminated[at], self._tradable[at] = terminated, tradable
        self._day[at], self._source[at] = day, self._sources.index(source)
        self._write = int(at[-1] + 1) % self.capacity if n else self._write
        self._size = min(self._size + n, self.capacity)

    def _read(self, at: NDArray[np.int64]) -> Transitions:
        return Transitions(self._rows[at], self._rate[at], self._reward[at], self._next_rows[at],
                           self._terminated[at], self._tradable[at], self._day[at],
                           np.asarray(self._sources, dtype=str)[self._source[at]])

    def contents(self) -> Transitions:
        """Every row held, oldest first."""
        return self._read((self._write - self._size + np.arange(self._size)) % self.capacity)

    def sample_steps(self, n: int, n_rows: int) -> Transitions:
        """Draw ``n`` whole decisions of ``n_rows`` names, each field shaped ``(n, n_rows, ...)``.

        For learners that need every name of a decision together. Every
        :meth:`add` must have been of ``n_rows`` rows and ``capacity`` must be
        a multiple of it, so a decision never straddles the end of the buffer.
        """
        if self.capacity % n_rows or self._write % n_rows:
            raise ValueError(f"the buffer does not hold whole decisions of {n_rows} rows")
        n_steps = self._size // n_rows
        if n_steps == 0:
            raise ValueError("there are no rows to sample")
        return self._read(self._rng.integers(n_steps, size=int(n))[:, None] * n_rows
                          + np.arange(n_rows))

    def sample(self, n: int, *, source: str | None = None) -> Transitions:
        """Draw ``n`` rows at random, with replacement, from one ``source`` or all."""
        pool = np.arange(self._size)
        if source is not None:
            code = self._sources.index(source) if source in self._sources else -1
            pool = pool[self._source[:self._size] == code]
        if len(pool) == 0:
            raise ValueError("there are no rows to sample" + (f" from {source!r}" if source else ""))
        return self._read(self._rng.choice(pool, size=int(n)))


class RandomRate:
    """Each name's rate drawn afresh, uniform on ``[0, 1]``, at every decision."""

    def start(self, rng: np.random.Generator, n_assets: int) -> None:
        self._rng, self._n_assets = rng, n_assets

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        return self._rng.uniform(0.0, 1.0, self._n_assets)


class RandomWalkRate:
    """Each name's rate wandering in ``[0, 1]`` by ``step`` a decision, from a uniform start."""

    def __init__(self, step: float = 0.05) -> None:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        self.step = float(step)

    def start(self, rng: np.random.Generator, n_assets: int) -> None:
        self._rng, self._rate = rng, rng.uniform(0.0, 1.0, n_assets)

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        moved = self._rate + self._rng.normal(0.0, self.step, self._rate.shape)
        self._rate = np.clip(moved, 0.0, 1.0)
        return self._rate


class RatePerEpisode:
    """One rate per name for a whole episode, log-uniform on ``[low, 1]``."""

    def __init__(self, low: float = 0.01) -> None:
        if not 0 < low < 1:
            raise ValueError("low must be in (0, 1)")
        self.low = float(low)

    def start(self, rng: np.random.Generator, n_assets: int) -> None:
        self._rate = np.exp(rng.uniform(np.log(self.low), 0.0, n_assets))

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        return self._rate


class FollowPath:
    """The rates that retrace a weight path, as far as rates can.

    ``weights`` is what an expert aimed at, on the environment's decisions,
    assets and horizon; NaN is flat, as in a target. At each decision the
    rate is the share of the gap to the target that reaches the path,

        eta = (path - current) / (target - current),

    so a path the environment took is retraced exactly. A rate only moves
    toward the target and never past it, so a path that moves away from the
    target, overshoots it, or trades where there is no gap cannot be
    expressed: the rate is clipped and :attr:`unexpressed` says how much of
    the path's trading was lost so. Moving rates to a grid in :func:`collect`
    loses more, which that does not count. Each episode starts from the
    environment's ``initial``, so with ``episode_days`` the path is rejoined
    at every open, at a cost the expert never paid: to retrace a path whole,
    play it as one episode.
    """

    def __init__(self, env: ExecutionEnv, weights: PortfolioWeights) -> None:
        if not isinstance(weights, PortfolioWeights):
            raise TypeError(f"weights must be PortfolioWeights, not {type(weights).__name__}")
        for axis in ("decisions", "assets"):
            if not np.array_equal(getattr(weights, axis), getattr(env, axis)):
                raise ValueError(f"{axis} of the path and the environment are not aligned")
        horizon = np.flatnonzero(np.asarray(weights.horizons) == env.axis[0])
        if len(horizon) == 0:
            raise ValueError(f"the path has no horizon {env.axis[0]}; "
                             f"it has {list(weights.horizons)}")
        self.env = env
        self._path = np.nan_to_num(weights.values[:, :, horizon[0]], nan=0.0)
        self._target = env.name_features.index("target")
        self._gap = env.name_features.index("gap")
        self._wanted = self._missed = 0.0

    @property
    def unexpressed(self) -> float:
        """The share of the path's trading so far that no rate in ``[0, 1]`` could do."""
        return self._missed / self._wanted if self._wanted > 0 else 0.0

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        goal = observation["names"][:, self._target]
        gap = observation["names"][:, self._gap]
        move = self._path[self.env.decision_index] - (goal - gap)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.clip(np.where(gap != 0, move / gap, 0.0), 0.0, 1.0)
        self._wanted += float(np.abs(move).sum())
        self._missed += float(np.abs(move - rate * gap).sum())
        return rate


def _grid(rates: Sequence[float]) -> NDArray[np.float64]:
    grid = np.asarray(rates, dtype=np.float64)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
            or np.any((grid < 0) | (grid > 1))):
        raise ValueError("rates must be at least two trading rates in [0, 1]")
    return grid


def collect(env: ExecutionEnv, policy: Callable[[dict], ArrayLike], buffer: ReplayBuffer, *,
            episodes: int | None = None, seed: int | None = None, source: str = "policy",
            reward: str | Callable = "shortfall", rates: Sequence[float] | None = None) -> int:
    """Play ``policy`` in ``env`` and store every name's every step; return the rows added.

    By default each of the environment's episodes is played once, in order;
    ``episodes`` instead draws that many at random. ``seed`` fixes the draw
    and any behaviour played. ``source`` tags the rows, and ``reward`` is the
    kind of :meth:`~stable_finance.evaluation.trades.rl.env.Payoff.reward` to
    store. ``rates`` is the grid each rate is moved to before it is played.
    """
    if not callable(reward) and reward not in REWARDS:
        raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
    if episodes is not None and (int(episodes) != episodes or episodes < 1):
        raise ValueError("episodes must be a positive integer")
    grid = None if rates is None else _grid(rates)
    plan = range(env.n_episodes) if episodes is None else [None] * int(episodes)
    n_assets = len(env.assets)
    added = 0
    for number, start_day in enumerate(plan):
        options = None if start_day is None else {"start_day": start_day}
        observation, _ = env.reset(seed=seed if number == 0 else None, options=options)
        _start_episode(policy, env)
        ended = False
        while not ended:
            rate = np.broadcast_to(np.asarray(policy(observation), dtype=np.float64), (n_assets,))
            if grid is not None:
                rate = grid[np.abs(rate[:, None] - grid).argmin(axis=1)]
            rows, tradable = name_rows(env, observation), observation["mask"]
            observation, _, terminated, truncated, info = env.step(rate)
            buffer.add(rows, rate, info["payoff"].reward(reward), name_rows(env, observation),
                       terminated=terminated, tradable=tradable, day=info["day"], source=source)
            added += n_assets
            ended = terminated or truncated
    return added
