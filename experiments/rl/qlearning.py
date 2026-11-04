"""Tabular Q-learning for the trading rate: one table shared by every name.

The reward splits exactly by name, so a table need only describe one name: how
big its target is, how far it stands from it, and what trading it costs now.
:class:`Cells` turns a name's state into a row of the table, and a column is a
rate. Two ways of filling the same table:

* :func:`explore` plays the market, epsilon-greedily, one rate per visit. The
  step size is ``1 / visits``, so each value is the running average of what
  was seen in that cell. A constant step size does much worse here: a single
  return is far larger than the cost being learned, and a value that keeps
  moving keeps chasing the last return.
* :func:`sweep` plays nothing. The market is replayed and does not respond to
  the policy, so every rate from every position can be asked at every decision
  of the data, which is how Nevmyvaka, Feng and Kearns (2006, "Reinforcement
  learning for optimized trade execution", ICML) avoid exploring. All the
  rates of a decision face the same return, so the comparison between them is
  not at the mercy of it, and the table is the same on every run. Their
  problem has a deadline and sweeps backward once; a book has no deadline, so
  this pass gathers each cell's average reward and where each rate leads, and
  value iteration solves that for the discounted fixed point.

:func:`sweep` needs fills that depend only on the decision, so it cannot run
under an impact schedule. :func:`explore` runs under any cost model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from stable_finance import ExecutionEnv

#: The rates of a table, in the units of Fang et al.'s discrete action.
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Edges of the size of a target, over the book's mean absolute target.
SIZE_EDGES = np.array([0.5, 1.0, 1.5, 2.5])
#: Edges of the gap along the target's sign, in the same units: positive is
#: under-held and negative over-held, which is what makes a long and a short
#: the same cell.
GAP_EDGES = np.array([-2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0])
#: Gaps a sweep tries, at least one in every gap cell, so every cell is visited.
GAP_GRID = np.array([-2.5, -1.5, -0.75, -0.35, -0.1, 0.1, 0.35, 0.75, 1.5, 2.5])
#: How many quantiles of the cost feature a table tells apart.
COST_LEVELS = 3


def _cost_feature(env: ExecutionEnv) -> str | None:
    """What sets a name's cost: its liquidity under impact, else its spread."""
    wanted = "adv_scale" if "adv_scale" in env.name_features else "half_spread"
    return wanted if wanted in env.name_features else None


class Cells:
    """The rows of a table: a name's target size, gap and cost, as one index.

    The cost's edges are quantiles over ``env``, which is the training panel:
    a table learned there reads a test panel on the same edges. A name that
    cannot trade has a cost cell of its own, since no rate does anything there.
    Coarse gap cells bias the rate: a value that is flat across a cell rewards
    the rate that only just reaches the next one. ``gap_edges`` can be finer.
    """

    def __init__(self, env: ExecutionEnv, gap_edges: NDArray = GAP_EDGES) -> None:
        self.gap_edges = np.asarray(gap_edges, dtype=np.float64)
        self._target = env.name_features.index("target")
        self._gap = env.name_features.index("gap")
        self.cost = _cost_feature(env)
        self._cost = None if self.cost is None else env.name_features.index(self.cost)
        self._gross = env.book_features.index("gross_target")
        self.edges = np.zeros(0)
        if self.cost is not None:
            levels = env.known(self.cost)[env.known("mask")]
            quantiles = np.quantile(levels, np.arange(1, COST_LEVELS) / COST_LEVELS)
            self.edges = np.unique(quantiles)
        self.n_cost = len(self.edges) + 2                    # the levels and "cannot trade"
        self.n_cells = (len(SIZE_EDGES) + 1) * (len(self.gap_edges) + 1) * self.n_cost

    def index(self, target, gap, cost, tradable, scale) -> NDArray[np.int64]:
        """The cell of each name; the arguments broadcast."""
        scale = np.where(scale > 0, scale, 1.0)
        along = np.where(target >= 0, gap, -gap)
        size = np.digitize(np.abs(target) / scale, SIZE_EDGES)
        side = np.digitize(along / scale, self.gap_edges)
        level = np.where(tradable, np.digitize(cost, self.edges), self.n_cost - 1)
        return (size * (len(self.gap_edges) + 1) + side) * self.n_cost + level

    def of(self, observation: dict) -> NDArray[np.int64]:
        """The cell of each name in an observation."""
        names = observation["names"]
        cost = 0.0 if self._cost is None else names[:, self._cost]
        scale = observation["book"][self._gross] / len(names)
        return self.index(names[:, self._target], names[:, self._gap], cost,
                          observation["mask"], scale)


class TablePolicy:
    """The best rate in each name's cell; a cell never tried holds still.

    ``values`` and ``visits`` are ``(n_cells, n_rates)``, with ``rates`` in
    ascending order, so a tie goes to the slower rate.
    """

    def __init__(self, cells: Cells, values: NDArray, visits: NDArray, rates: NDArray) -> None:
        self.cells, self.values, self.visits, self.rates = cells, values, visits, rates
        self.best = np.where(visits > 0, values, -np.inf).argmax(axis=1)

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        return self.rates[self.best[self.cells.of(observation)]]


def _rates(rates: Sequence[float]) -> NDArray[np.float64]:
    grid = np.asarray(rates, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("rates must be at least two rates in ascending order")
    return grid


def explore(env: ExecutionEnv, *, rates: Sequence[float] = RATES,
            reward: str | Callable = "shortfall", gamma: float = 0.9, episodes: int = 1000,
            seed: int = 0, start: float = 0.5, floor: float = 0.05) -> TablePolicy:
    """Learn by playing, with epsilon falling from ``start`` to ``floor``."""
    grid = _rates(rates)
    cells = Cells(env)
    n_rates = len(grid)
    size = cells.n_cells * n_rates
    values, visits = np.zeros(size), np.zeros(size)
    rng = np.random.default_rng(seed)
    n_assets = len(env.assets)
    for episode in range(episodes):
        epsilon = max(floor, start * (1 - episode / (0.7 * episodes)))
        observation, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ended = False
        while not ended:
            here = cells.of(observation)
            greedy = values.reshape(-1, n_rates)[here].argmax(axis=1)
            action = np.where(rng.uniform(size=n_assets) < epsilon,
                              rng.integers(n_rates, size=n_assets), greedy)
            observation, _, terminated, truncated, info = env.step(grid[action])
            ended = terminated or truncated
            future = 0.0 if terminated else values.reshape(-1, n_rates)[cells.of(observation)].max(1)
            target = info["payoff"].reward(reward) + gamma * future
            at = here * n_rates + action
            count = np.bincount(at, minlength=size)
            visits += count
            error = np.bincount(at, target - values[at], size)
            values += np.where(count > 0, error / np.maximum(visits, 1), 0.0)
    shape = (cells.n_cells, n_rates)
    return TablePolicy(cells, values.reshape(shape), visits.reshape(shape), grid)


def sweep(env: ExecutionEnv, *, rates: Sequence[float] = RATES,
          reward: str | Callable = "shortfall", gamma: float = 0.9,
          gap_edges: NDArray = GAP_EDGES, gaps: NDArray = GAP_GRID,
          max_iterations: int = 2000) -> TablePolicy:
    """Learn by asking every rate from every position at every decision of ``env``.

    ``gaps`` are the positions tried, in the units of ``gap_edges``, at least
    one in every gap cell.
    """
    grid = _rates(rates)
    gaps = np.asarray(gaps, dtype=np.float64)
    if len(np.unique(np.digitize(gaps, gap_edges))) != len(gap_edges) + 1:
        raise ValueError("gaps must try at least one position in every gap cell")
    cells = Cells(env, gap_edges)
    n_rates = len(grid)
    n_pairs = cells.n_cells * n_rates
    total, visits = np.zeros(n_pairs), np.zeros(n_pairs)
    moves = np.zeros(n_pairs * cells.n_cells)
    target, mask = env.known("target"), env.known("mask")
    cost = env.known(cells.cost) if cells.cost else np.zeros(target.shape)
    scale = np.abs(target).mean(axis=1)
    action = np.arange(n_rates)[None, :, None]
    for t in range(len(target) - 1):
        along = np.where(target[t] >= 0, 1.0, -1.0)
        gap = along * gaps[:, None] * scale[t]
        alternates = env.parallel_alternates(grid, held=target[t] - gap, decision=t)
        here = cells.index(target[t], gap, cost[t], mask[t], scale[t])
        there = cells.index(target[t + 1], target[t + 1] - alternates.held, cost[t + 1],
                            mask[t + 1], scale[t + 1])
        at = (here[:, None, :] * n_rates + action).ravel()
        total += np.bincount(at, alternates.reward(reward).ravel(), n_pairs)
        visits += np.bincount(at, minlength=n_pairs)
        moves += np.bincount(at * cells.n_cells + there.ravel(), minlength=len(moves))

    seen = np.maximum(visits, 1)
    mean = total / seen
    chance = moves.reshape(n_pairs, cells.n_cells) / seen[:, None]
    values = np.zeros(n_pairs)
    tolerance = 1e-9 * max(np.abs(mean).max(), np.finfo(float).tiny)
    for _ in range(max_iterations):
        updated = mean + gamma * chance @ values.reshape(-1, n_rates).max(axis=1)
        done = np.abs(updated - values).max() < tolerance
        values = updated
        if done:
            break
    shape = (cells.n_cells, n_rates)
    return TablePolicy(cells, values.reshape(shape), visits.reshape(shape), grid)
