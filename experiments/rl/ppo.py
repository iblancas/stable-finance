"""Basic PPO for the trading rate: one stochastic policy shared by every name.

Where the Q-learners learn the value of each rate and take the best, this
learns the policy itself, as Lin and Beling (2020, "An end-to-end optimal trade
execution framework based on proximal policy optimization", IJCAI) and Fang et
al. (2023, KDD) do for execution. Proximal policy optimization (Schulman et
al., 2017, arXiv:1707.06347) plays a few days with the current policy, scores
each action by how much better it did than the critic expected, and takes
several gradient steps on those days, clipped so the policy cannot move far
from the one that played them. It needs no replay buffer: what it learns from
is thrown away after each update.

The setting is the Q-learners': a name is an agent, the action is one of the
same rates, the input is the same row, and the reward is the name's own. Each
name's advantage is estimated along its own path (Schulman et al., 2016,
"High-dimensional continuous control using generalized advantage estimation",
ICLR). A day's end is a truncation, so the critic's value of the next decision
stands in for what follows it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from experiments.rl.dqn import Inputs
from experiments.rl.qlearning import RATES
from stable_finance import ReplayBuffer, collect
from stable_finance.evaluation.trades.rl import RatePerEpisode, name_rows, row_features


def advantages(rewards: NDArray, values: NDArray, final: NDArray, *, alive: bool,
               gamma: float, lam: float) -> NDArray[np.float64]:
    """Generalised advantage estimates along an episode, ``(n_steps, n_assets)``.

    ``values`` are the critic's at each step and ``final`` its value of the
    state after the last. ``alive`` is False where no state follows, and then
    nothing is added for the future.
    """
    following = np.concatenate([values[1:], final[None]])
    if not alive:
        following[-1] = 0.0
    surprise = rewards + gamma * following - values
    estimate = np.zeros_like(surprise)
    running = np.zeros(surprise.shape[1:])
    for t in range(len(surprise) - 1, -1, -1):
        running = surprise[t] + gamma * lam * running
        estimate[t] = running
    return estimate


def clipped_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    """PPO's objective: the lesser of the ratio's gain and its gain clipped near one."""
    return torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)


def _network(n_inputs: int, hidden: int, n_outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                         nn.Tanh(), nn.Linear(hidden, n_outputs))


class Agent:
    """An actor over the rates and a critic of the state; :func:`train` makes one."""

    def __init__(self, inputs: Inputs, rates: Sequence[float], *, hidden: int = 64,
                 learning_rate: float = 3e-4, clip: float = 0.2, entropy: float = 0.01) -> None:
        self.inputs = inputs
        self.rates = np.asarray(rates, dtype=np.float64)
        self.clip, self.entropy = clip, entropy
        self.actor = _network(len(inputs.columns), hidden, len(self.rates))
        self.critic = _network(len(inputs.columns), hidden, 1)
        self.parameters = [*self.actor.parameters(), *self.critic.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)

    def act(self, rows: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Draw each name's rate; return its index, its log chance and the state's value."""
        features = self.inputs(rows)
        with torch.no_grad():
            choice = torch.distributions.Categorical(logits=self.actor(features))
            action = choice.sample()
            return (action.numpy(), choice.log_prob(action).numpy(),
                    self.critic(features)[..., 0].numpy().astype(np.float64))

    def value(self, rows: NDArray) -> NDArray[np.float64]:
        with torch.no_grad():
            return self.critic(self.inputs(rows))[..., 0].numpy().astype(np.float64)

    def update(self, features: torch.Tensor, action: torch.Tensor, chance: torch.Tensor,
               advantage: torch.Tensor, goal: torch.Tensor) -> float:
        """One gradient step on a minibatch of rows."""
        choice = torch.distributions.Categorical(logits=self.actor(features))
        ratio = (choice.log_prob(action) - chance).exp()
        gain = clipped_objective(ratio, advantage, self.clip).mean()
        error = nn.functional.smooth_l1_loss(self.critic(features)[..., 0], goal)
        loss = -gain + 0.5 * error - self.entropy * choice.entropy().mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, 0.5)
        self.optimizer.step()
        return loss.item()

    def policy(self, env) -> Callable[[dict], NDArray[np.float64]]:
        """Each name's likeliest rate, for an environment with the same features."""
        if row_features(env) != self.inputs.columns:
            raise ValueError("this agent was trained on other features: "
                             f"{self.inputs.columns} against {row_features(env)}")

        def act(observation: dict) -> NDArray[np.float64]:
            with torch.no_grad():
                logits = self.actor(self.inputs(name_rows(env, observation)))
            return self.rates[logits.argmax(dim=-1).numpy()]

        return act


def _play(env, agent: Agent, reward: str | Callable, scale: float, gamma: float,
          lam: float) -> tuple[NDArray, ...]:
    """One episode of the current policy, as rows with their advantages."""
    observation, _ = env.reset()
    rows, actions, chances, values, rewards, tradable = [], [], [], [], [], []
    ended = terminated = False
    while not ended:
        row = name_rows(env, observation)
        action, chance, value = agent.act(row)
        rows.append(row)
        actions.append(action)
        chances.append(chance)
        values.append(value)
        tradable.append(observation["mask"])
        observation, _, terminated, truncated, info = env.step(agent.rates[action])
        rewards.append(info["payoff"].reward(reward) * scale)
        ended = terminated or truncated
    values = np.stack(values)
    advantage = advantages(np.stack(rewards), values, agent.value(name_rows(env, observation)),
                           alive=not terminated, gamma=gamma, lam=lam)
    return (np.concatenate(rows), np.concatenate(actions), np.concatenate(chances),
            advantage.ravel(), (advantage + values).ravel(), np.concatenate(tradable))


def train(env, *, rates: Sequence[float] = RATES, reward: str | Callable = "shortfall",
          episodes: int = 200, days: int = 4, epochs: int = 4, minibatch: int = 4096,
          warmup: int = 10, gamma: float = 0.9, lam: float = 0.95, clip: float = 0.2,
          entropy: float = 0.01, hidden: int = 64, learning_rate: float = 3e-4,
          reward_scale: float | None = None, seed: int = 0) -> Agent:
    """Train an agent in ``env``.

    ``episodes`` is the days played in all, the same budget as the Q-learners,
    ``days`` of them before each update of ``epochs`` passes in ``minibatch``
    rows. ``warmup`` days of random behaviour set how inputs are standardised.
    ``reward_scale`` defaults to basis points of the book.
    """
    torch.manual_seed(seed)
    n_assets = len(env.assets)
    columns = row_features(env)
    opening = ReplayBuffer(len(env.decisions) * n_assets, len(columns), seed=seed)
    collect(env, RatePerEpisode(), opening, episodes=warmup, seed=seed, reward=reward)
    agent = Agent(Inputs(columns, opening.contents().rows), rates, hidden=hidden,
                  learning_rate=learning_rate, clip=clip, entropy=entropy)
    scale = 1e4 * n_assets if reward_scale is None else reward_scale
    for _ in range(max(1, episodes // days)):
        played = [_play(env, agent, reward, scale, gamma, lam) for _ in range(days)]
        rows, action, chance, advantage, goal, tradable = (np.concatenate(x) for x in zip(*played))
        rows, action, chance = rows[tradable], action[tradable], chance[tradable]
        advantage, goal = advantage[tradable], goal[tradable]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        tensors = (agent.inputs(rows), torch.as_tensor(action), torch.as_tensor(chance),
                   torch.as_tensor(advantage, dtype=torch.float32),
                   torch.as_tensor(goal, dtype=torch.float32))
        for _ in range(epochs):
            order = torch.randperm(len(rows))
            for start in range(0, len(rows), minibatch):
                at = order[start:start + minibatch]
                agent.update(*(x[at] for x in tensors))
    return agent
