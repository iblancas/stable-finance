"""Double DQN for the trading rate: one network for every name, in three forms.

A row of the buffer is a name at a decision, and every name shares the network,
so a day of a few hundred names is tens of thousands of transitions of one
small problem. The update is double Q-learning (van Hasselt, Guez and Silver,
2016, "Deep reinforcement learning with double Q-learning", AAAI), as Ning, Lin
and Jaimungal (2018, arXiv:1812.06600) use for execution: the online network
picks the next rate and a slower copy values it, which matters here because
taking the maximum of noisy values overstates all of them. The buffer starts
with random behaviour, for the large gaps the agent's own play seldom visits,
and any experts, and then fills with the agent's own epsilon-greedy play.

The forms differ in how names cooperate, which only matters under a coupling
such as :class:`~experiments.rl.coupling.NetLimit`:

* INDEPENDENT. Each name learns from its own reward. Without a coupling this is
  exact, the reward being a sum over names; with one, a name does not bear what
  its order costs the others.
* ``team=True``. Value decomposition (Sunehag et al., 2018, "Value-decomposition
  networks for cooperative multi-agent learning", AAMAS): the book's value is
  the sum of the names', and the error is taken on that sum, against the book's
  reward. Each name acts on its own value, so the action is still per name.
* ``intention=True``. One round of the communication of Fang et al. (2023, KDD):
  every name first says what it would do alone, the book works out the net
  drift those intentions add up to, and every name decides again knowing it.
  Fang et al. use several rounds, attention and PPO; this keeps the idea and
  stays a Q-learner.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from experiments.rl.qlearning import RATES
from stable_finance import (
    RatePerEpisode,
    ReplayBuffer,
    Transitions,
    collect,
    name_rows,
    row_features,
)

#: Features that turn over with the side of the book, so a long and a short are one case.
SIGNED = ("held", "target", "gap", "in_flight", "imbalance", "last_return", "forecast")


class Inputs:
    """Rows as a network reads them: turned to the long side, then standardised."""

    def __init__(self, columns: Sequence[str], rows: NDArray) -> None:
        self.columns = tuple(columns)
        self._target = self.columns.index("target")
        self._signed = [i for i, name in enumerate(self.columns) if name in SIGNED]
        turned = self._turn(rows)
        self.mean = turned.mean(axis=0)
        spread = turned.std(axis=0)
        self.spread = np.where(spread > 0, spread, 1.0)

    def _turn(self, rows: NDArray) -> NDArray[np.float64]:
        turned = np.array(rows, dtype=np.float64)
        side = np.where(turned[..., self._target] < 0, -1.0, 1.0)
        turned[..., self._signed] *= side[..., None]
        return turned

    def __call__(self, rows: NDArray) -> torch.Tensor:
        return torch.as_tensor((self._turn(rows) - self.mean) / self.spread, dtype=torch.float32)


class Agent:
    """The networks and the update; :func:`train` is what makes one."""

    def __init__(self, inputs: Inputs, rates: Sequence[float], *, team: bool = False,
                 intention: bool = False, gamma: float = 0.9, hidden: int = 64,
                 learning_rate: float = 1e-3, sync_every: int = 200,
                 reward_scale: float = 1.0) -> None:
        self.inputs = inputs
        self.rates = np.asarray(rates, dtype=np.float64)
        self.team, self.intention, self.gamma = team, intention, gamma
        self.sync_every, self.reward_scale = sync_every, reward_scale
        self.online = nn.Sequential(nn.Linear(len(inputs.columns) + int(intention), hidden),
                                    nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(self.rates)))
        self.slow = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self._gap = inputs.columns.index("gap")
        self._gross = inputs.columns.index("gross_target")
        self._updates = 0

    def _intent(self, features: torch.Tensor, rows: NDArray) -> torch.Tensor:
        """The net drift of what every name would do alone, over the book's gross target."""
        alone = torch.cat([features, torch.zeros_like(features[..., :1])], dim=-1)
        with torch.no_grad():
            rate = self.rates[self.online(alone).argmax(dim=-1).numpy()]
        drift = ((rate - 1.0) * rows[..., self._gap]).sum(axis=-1)
        gross = np.maximum(rows[..., 0, self._gross], np.finfo(np.float32).tiny)
        shape = features.shape[:-1] + (1,)
        return torch.as_tensor(np.broadcast_to((drift / gross)[..., None, None], shape).copy(),
                               dtype=torch.float32)

    def _values(self, network: nn.Module, rows: NDArray) -> torch.Tensor:
        """Each name's value of each rate; ``rows`` are the names of whole decisions."""
        features = self.inputs(rows)
        if self.intention:
            features = torch.cat([features, self._intent(features, rows)], dim=-1)
        return network(features)

    def greedy(self, rows: NDArray) -> NDArray[np.int64]:
        """The best rate's index for each name of one decision."""
        with torch.no_grad():
            return self._values(self.online, rows).argmax(dim=-1).numpy()

    def update(self, batch: Transitions) -> float:
        """One gradient step on whole decisions, shaped ``(n_decisions, n_assets, ...)``."""
        action = np.abs(batch.rate[..., None] - self.rates).argmin(axis=-1)
        action = torch.as_tensor(action)[..., None]
        reward = torch.as_tensor(batch.reward * self.reward_scale, dtype=torch.float32)
        alive = torch.as_tensor(~batch.terminated, dtype=torch.float32)
        chosen = self._values(self.online, batch.rows).gather(-1, action)[..., 0]
        with torch.no_grad():
            best = self._values(self.online, batch.next_rows).argmax(dim=-1, keepdim=True)
            future = self._values(self.slow, batch.next_rows).gather(-1, best)[..., 0]
            goal = reward + self.gamma * alive * future
        if self.team:
            loss = nn.functional.smooth_l1_loss(chosen.mean(dim=1), goal.mean(dim=1))
        else:
            tradable = torch.as_tensor(batch.tradable)
            loss = nn.functional.smooth_l1_loss(chosen[tradable], goal[tradable])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._updates += 1
        if self._updates % self.sync_every == 0:
            self.slow.load_state_dict(self.online.state_dict())
        return loss.item()

    def policy(self, env) -> Callable[[dict], NDArray[np.float64]]:
        """The greedy policy, for an environment with the same features."""
        if row_features(env) != self.inputs.columns:
            raise ValueError("this agent was trained on other features: "
                             f"{self.inputs.columns} against {row_features(env)}")
        return lambda observation: self.rates[self.greedy(name_rows(env, observation))]


class EpsilonGreedy:
    """The agent's own choice, or a random rate with chance ``epsilon``, per name."""

    def __init__(self, agent: Agent, env, epsilon: float) -> None:
        self.agent, self.env, self.epsilon = agent, env, epsilon

    def start(self, rng: np.random.Generator, n_assets: int) -> None:
        self._rng, self._n_assets = rng, n_assets

    def __call__(self, observation: dict) -> NDArray[np.float64]:
        action = self.agent.greedy(name_rows(self.env, observation))
        wander = self._rng.integers(len(self.agent.rates), size=self._n_assets)
        return self.agent.rates[np.where(self._rng.uniform(size=self._n_assets) < self.epsilon,
                                         wander, action)]


def train(env, *, rates: Sequence[float] = RATES, reward: str | Callable = "shortfall",
          team: bool = False, intention: bool = False, experts: Mapping | None = None,
          episodes: int = 200, warmup: int = 30, batch: int = 32, updates: int = 50,
          memory: int = 4000, gamma: float = 0.9, hidden: int = 64, learning_rate: float = 1e-3,
          sync_every: int = 200, start: float = 0.3, floor: float = 0.02,
          settle: float = 1.0, reward_scale: float | None = None, seed: int = 0) -> Agent:
    """Train an agent in ``env``; the buffer it learned from is ``agent.buffer``.

    ``warmup`` episodes of random behaviour open the buffer, then ``experts``,
    a mapping from a name to a policy, play every episode once. Each of the
    ``episodes`` then adds one episode of the agent's own play, with epsilon
    falling from ``start`` to ``floor``, and takes ``updates`` gradient steps
    of ``batch`` whole decisions, with the learning rate falling by the share
    ``settle`` over the run. ``memory`` is the buffer's size in decisions.
    ``reward_scale`` defaults to basis points of the book, about one a decision.
    """
    torch.manual_seed(seed)
    n_assets = len(env.assets)
    columns = row_features(env)
    buffer = ReplayBuffer(memory * n_assets, len(columns), seed=seed)
    collect(env, RatePerEpisode(), buffer, episodes=warmup, seed=seed, source="random",
            reward=reward, rates=rates)
    for name, expert in (experts or {}).items():
        collect(env, expert, buffer, source=name, reward=reward, rates=rates)
    agent = Agent(Inputs(columns, buffer.contents().rows), rates, team=team, intention=intention,
                  gamma=gamma, hidden=hidden, learning_rate=learning_rate, sync_every=sync_every,
                  reward_scale=1e4 * n_assets if reward_scale is None else reward_scale)
    agent.buffer = buffer
    for episode in range(episodes):
        epsilon = max(floor, start * (1 - episode / (0.7 * episodes)))
        collect(env, EpsilonGreedy(agent, env, epsilon), buffer, episodes=1, source="self",
                reward=reward, rates=rates)
        for group in agent.optimizer.param_groups:
            group["lr"] = learning_rate * (1 - settle * episode / episodes)
        for _ in range(updates):
            agent.update(buffer.sample_steps(batch, n_assets))
    return agent
