"""A trading environment for learned execution policies, charged exactly as the harness charges.

The trades stage decides how far to move toward a target given what trading
costs. A reinforcement-learning policy makes that decision one step at a
time, so it needs the harness as an environment: show the policy the state at
a decision, take its trade, fill it, pay for it, and report what the book
earned. :class:`ExecutionEnv` is that loop, with the reset/step interface of
Gymnasium and no dependency on it (an adapter is in
:mod:`~stable_finance.evaluation.trades.rl.gym`).

THE ACTION is a trading rate per name, ``eta`` in ``[0, 1]``: the fraction of
the gap to the target to trade now,

    aim_t = (1 - eta) * current + eta * target_t,

where ``current`` is the position held (or, under the impact models, the
position committed to once the slices in flight have filled). ``eta = 1``
trades to the target, ``eta = 0`` holds, and one ``eta`` for every name is
the partial-adjustment rule of Garleanu and Pedersen (2013, "Dynamic trading
with predictable returns and transaction costs", JF 68(6)), whose rate is
then the policy's only parameter. A rate per name can express what one rate
cannot: a no-trade band (Constantinides, 1986), trading the names that are
cheap now and waiting on the dear ones, or the front-loaded schedule of
Almgren and Chriss (2000, "Optimal execution of portfolio transactions",
Journal of Risk 3(2)). This is the action of the execution-RL literature that
needs no order book, the quantity to trade per period (Ning, Lin and
Jaimungal, 2018, arXiv:1812.06600), and because one policy is applied to
every name it is the parameter-shared form multi-order execution takes
(Fang et al., 2023, "Learning multi-agent intention-aware communication for
optimal multi-order execution in finance", KDD). A target is a weight path
from the portfolio stage; NaN in it is flat, as the harness reads it.

THE REWARD is the harness's net return for the decision, in fractions of
equity: what the book held after the decision's fills earned over the
realized return, less what those fills paid over their reference price and
any fees the configuration bills on them,

    net_t = sum_i held_t,i * r_t,i - sum_i q_t,i * (p_t,i - m_t,i) / m_t,i - fees_t.

The fills come from :func:`~stable_finance.evaluation.execution.execute`
under the configuration's own cost model and the fees from
:func:`~stable_finance.evaluation.execution.fee_cost`, so only what the
harness would bill is charged; no cost a policy believes in ever enters what
it is charged. A fee is not shown to the policy, which learns it from the
reward; a per-share fee, which differs with a name's price, it can learn only
on average. The policy
never sees the realized return, the arrival quotes or the execution window
(see :mod:`~stable_finance.evaluation.trades.rl.features`).

THE REWARD SPLITS BY NAME. Every cost model charges a name on its own fills,
so the net return is exactly a sum over names, and :meth:`ExecutionEnv.step`
reports the parts as ``info["payoff"]``, a :class:`Payoff`. A learner that
treats each name as an agent trains on those. :meth:`Payoff.reward` also
gives them as ``"shortfall"``, the net return less what the target earned on
paper: the target does not depend on the policy, so the two rewards share
their best policy, and taking the paper return out removes some of the
market's noise. The reward :meth:`~ExecutionEnv.step` returns stays the net.

WHAT ANOTHER RATE WOULD HAVE DONE is a question the data can answer, because
the market is replayed and does not respond to the policy.
:meth:`ExecutionEnv.parallel_alternates` puts it to the same execution model,
for any position at any decision, without playing it. This is the assumption
Nevmyvaka, Feng and Kearns (2006, "Reinforcement learning for optimized trade
execution", ICML) use to try every action in every state instead of
exploring. Under a schedule the impact models charge an order by the orders
before it, a history the question does not give, so it refuses. It and
``info["payoff"]`` carry the realized return: they are for training, and a
policy being scored must not read them at a decision.

THE ENVIRONMENT IS THE HARNESS. An episode's :class:`Rollout` carries the
weight path the policy traded toward, and handed to
:func:`~stable_finance.run_backtest` at the weights level that path gives
the same fills and the same net return, decision by decision, for any
policy: the trades are chased from what is held under ``mid``, ``cross`` and
``book``, exactly as :func:`~stable_finance.evaluation.execution.fill_path`
chases them, and under the impact models each step's order is the move in
the committed path, exactly as
:func:`~stable_finance.evaluation.execution.orders_from_weights` makes it. A
policy that always plays ``eta = 1`` therefore reproduces ``run_backtest``
on the target itself. That is what lets a learned policy be compared with
the hand-written rules on one scale.

Under the impact models an order is worked over the configuration's
schedule, and every order in a name within that schedule is charged as ONE
parent order, as the model is calibrated (Frazzini, Israel and Moskowitz,
2018). That matters more here than anywhere: charged one slice at a time, a
concave cost rewards splitting a trade into ever smaller pieces, and a
learning agent finds that at once. So the environment fills each step by
re-running the impact model over the episode so far, which keeps every
parent intact and costs time quadratic in the episode's length -- trivial
over a day, the natural episode for an order worked over a day. As in the
harness, the committed position is the running sum of the orders sent and
``in_flight`` is that less the held position; an impact order is never
retried, so one sent where the model cannot price the name (no mid or no
ADV at the decision) is dropped there, and ``in_flight`` overstates what is
still to come by it.

AN EPISODE is a block of ``episode_days`` consecutive trading days, read
from ``days``, drawn at :meth:`ExecutionEnv.reset` from the ``seed`` (or
named with ``options={"start_day": k}``); without ``episode_days`` it is the
whole panel. Each episode starts from ``initial`` (default flat), with
nothing in flight. An episode that stops where the panel goes on is
TRUNCATED, not terminated: the book is not liquidated at the close, so the
position still has a future, and the last observation is the next decision's
for a learner to bootstrap from. Only the panel's last decision terminates,
having no next decision to show. The configuration must rebalance every
decision and hold ``holding="rebalanced"``: when to trade is the policy's
decision, and drift holding is left for later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import ADV, MarketData, require_market_aligned
from stable_finance.evaluation._grid import require_aligned
from stable_finance.evaluation.config import DEFAULT_BACKTEST, BacktestConfig
from stable_finance.evaluation.execution import (
    Fills,
    Orders,
    execute,
    fee_cost,
    initial_positions,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.preparation import (
    check_axes,
    impact_schedule,
    prepare,
    share_price,
)
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.trades.rl.features import decision_features, observable

__all__ = ["BOOK_FEATURES", "REWARDS", "ExecutionEnv", "Payoff", "Rollout", "rollout",
           "start_episode"]

_IMPACT = ("impact", "impact_stock")

#: Features of the whole book, in the order :attr:`ExecutionEnv.book_features`
#: lists them; ``time_of_day`` is present only with ``days``.
BOOK_FEATURES = ("progress", "time_of_day", "gross_held", "gross_target", "gross_gap")

#: What :meth:`Payoff.reward` can report.
REWARDS = ("net", "shortfall")


class Payoff(NamedTuple):
    """What a decision did, name by name, in fractions of equity.

    ``held`` is the position after the decision's fills, ``earned`` what that
    position earned over the realized return, ``paid`` what the fills paid
    over their reference price and in fees, ``paper`` what the target would
    have earned, and ``gap`` what is left to the target. Summed over names,
    ``earned`` and ``paid`` are the gross return and the cost the harness
    books for the decision.
    """

    held: NDArray[np.float64]
    earned: NDArray[np.float64]
    paid: NDArray[np.float64]
    paper: NDArray[np.float64]
    gap: NDArray[np.float64]

    def reward(self, kind: str | Callable[[Payoff], NDArray] = "shortfall") -> NDArray[np.float64]:
        """Each name's reward.

        ``"net"``, or ``"shortfall"``, the net less the paper return, or any
        function of the payoff, for a preference the harness does not charge.
        """
        if callable(kind):
            return kind(self)
        if kind not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {kind!r}")
        net = self.earned - self.paid
        return net if kind == "net" else net - self.paper


@dataclass(frozen=True)
class Rollout:
    """One episode, in the contracts the harness and the scorecards take.

    ``weights`` is the path the policy traded toward, ``target`` the path it
    was given and ``fills`` what executed, all on the episode's decisions,
    with ``realized`` the returns the episode was scored on. ``net``,
    ``gross`` and ``cost`` are the rewards and their two parts, one per
    decision, and ``fees`` the part of ``cost`` billed beside the fills.
    ``run_backtest(weights, realized, ...)`` under the episode's configuration
    reproduces ``net``, and
    ``implementation_shortfall(target, fills, realized, fees=fees)`` splits
    what the policy gave up against the target.
    """

    target: PortfolioWeights
    weights: PortfolioWeights
    fills: Fills
    realized: ForwardReturns
    net: NDArray[np.float64]
    gross: NDArray[np.float64]
    cost: NDArray[np.float64]
    fees: NDArray[np.float64]


def _rows(market: MarketData, rows: slice) -> MarketData:
    """The market at a run of consecutive decisions."""
    return MarketData({k: v[rows] for k, v in market.columns.items()},
                      market.decisions[rows], market.assets, market.estimated)


class ExecutionEnv:
    """Trade toward a target one decision at a time, filled and charged by the harness.

    See the module docstring for the action, the reward and the episode.
    ``target`` is the weight path to trade toward and ``realized`` the
    return each name earns from each decision, on the same grid;
    ``horizon_index`` picks the horizon both are read at. ``market`` is the
    observable market on that grid, ``days`` labels each decision's trading
    day, and ``config`` supplies the cost model and its settings
    (``cost_model``, ``efq``, ``equity``, ``execution_days``, ``impact``),
    each refused as :func:`~stable_finance.run_backtest` refuses it.
    ``forecast``, on the same grid, is shown to the policy and never used to
    charge it. ``lookback`` is the volatility window, in decisions.

    :meth:`reset` and :meth:`step` follow Gymnasium's signatures. An
    observation is a dict of ``"names"``, ``(n_assets,
    len(name_features))``; ``"book"``, ``(len(book_features),)``; and
    ``"mask"``, ``(n_assets,)``, True where the name looks tradable as quoted
    at the decision. The mask only informs: an order in a name the execution
    model cannot fill is left unfilled there, as in the harness.
    """

    def __init__(
        self,
        target: PortfolioWeights,
        realized: ForwardReturns,
        market: MarketData | None = None,
        *,
        days: ArrayLike | None = None,
        config: BacktestConfig = DEFAULT_BACKTEST,
        forecast: ForwardReturns | None = None,
        horizon_index: int = 0,
        initial: ArrayLike | None = None,
        episode_days: int | None = None,
        lookback: int = 12,
    ) -> None:
        if not isinstance(target, PortfolioWeights):
            raise TypeError(f"target must be PortfolioWeights, not {type(target).__name__}")
        require_aligned(target, realized)
        if config.rebalance != "every_decision":
            raise ValueError("the policy decides when to trade, so the configuration must "
                             "rebalance='every_decision'")
        if config.holding != "rebalanced":
            raise ValueError("the environment supports holding='rebalanced' only")
        market = MarketData.empty(target.decisions, target.assets) if market is None else market
        require_market_aligned(market, target)
        check_axes(config, market, target)
        context = prepare(target, realized, market, days, horizon_index)
        self.schedule = impact_schedule(context, config) if config.cost_model in _IMPACT else 0

        self.config = config
        self.market = market
        self.decisions = target.decisions
        self.assets = target.assets
        self.axis = context["axis"]
        n_decisions, n_assets = context["n_decisions"], context["n_assets"]
        self._target = np.nan_to_num(target.values[:, :, horizon_index], nan=0.0)
        self._outcome = context["outcome_values"]
        self._share_price = share_price(context, config)
        self._initial = initial_positions(None if initial is None else np.reshape(initial, (-1, 1)),
                                           (n_assets, 1))[:, 0]
        self._held = self._initial.copy()
        self._committed = self._initial.copy()

        mu = None
        if forecast is not None:
            require_aligned(forecast, target)
            mu = forecast.values[:, :, horizon_index]
        market_features = decision_features(market, forecast=mu, equity=config.equity,
                                            lookback=lookback)
        self._features = market_features.values
        self._market_features = market_features.names
        impact = config.cost_model in _IMPACT
        state = ("held", "target", "gap") + (("in_flight",) if impact else ())
        self.name_features = state + market_features.names
        self.book_features = tuple(f for f in BOOK_FEATURES
                                   if f != "time_of_day" or context["days_given"])
        self._mask = self._tradable(market, config)

        day = self._day = context["day"]
        if context["days_given"]:
            starts = np.r_[True, day[1:] != day[:-1]]
            index = np.arange(n_decisions)
            self._within = index - np.maximum.accumulate(np.where(starts, index, 0))
            self._per_day = float(np.median(np.bincount(day)))
        if episode_days is None:
            self._blocks = [(0, n_decisions)]
        else:
            if not context["days_given"]:
                raise ValueError("episode_days needs days labelling each decision's trading day")
            if int(episode_days) != episode_days or episode_days < 1:
                raise ValueError("episode_days must be a positive integer")
            if np.any(np.diff(day) < 0):
                raise ValueError("an episode is consecutive days, so the decisions must be "
                                 "in time order")
            n_days = int(day.max()) + 1
            if episode_days > n_days:
                raise ValueError(f"episode_days={episode_days} is longer than the "
                                 f"{n_days} days of data")
            first = np.searchsorted(day, np.arange(n_days + 1))
            self._blocks = [(int(first[k]), int(first[k + int(episode_days)]))
                            for k in range(n_days - int(episode_days) + 1)]
        self.np_random = np.random.default_rng()
        self._start = self._end = self._t = 0

    @property
    def n_episodes(self) -> int:
        """How many distinct episodes :meth:`reset` can draw."""
        return len(self._blocks)

    @property
    def decision_index(self) -> int:
        """The row of the panel the next :meth:`step` decides at."""
        return self._t

    def known(self, name: str) -> NDArray:
        """``"target"``, ``"mask"`` or a market feature, ``(n_decisions, n_assets)``.

        Each row is what its own decision could see, whatever is held, so a
        learner working across the panel reads no further than a policy could.
        """
        if name == "target":
            return self._target
        if name == "mask":
            return self._mask
        if name not in self._market_features:
            raise ValueError(f"{name!r} is not known here; there is target, mask and "
                             f"{self._market_features}")
        return self._features[:, :, self._market_features.index(name)]

    @staticmethod
    def _tradable(market: MarketData, config: BacktestConfig) -> NDArray[np.bool_]:
        """Where a name looks fillable from what was quoted at the decision."""
        ok = np.ones(market.shape, dtype=bool)
        for quantity in ("mid", "close", "vwap"):
            price = observable(market, quantity)
            if price is not None:
                ok &= np.isfinite(price) & (price > 0)
                break
        if config.cost_model == "cross" and config.efq:
            spread = observable(market, "half_spread")
            if spread is not None:
                ok &= np.isfinite(spread)
        if config.cost_model in _IMPACT:
            adv = observable(market, ADV)
            if adv is not None:
                ok &= np.isfinite(adv) & (adv > 0)
        return ok

    # ── the Gymnasium interface ─────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start an episode; return its first observation and an info dict."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        if "start_day" in options:
            k = int(options["start_day"])
            if not 0 <= k < len(self._blocks):
                raise ValueError(f"start_day must be in [0, {len(self._blocks)})")
        else:
            k = int(self.np_random.integers(len(self._blocks)))
        self._start, self._end = self._blocks[k]
        self._t = self._start
        n_steps, n_assets = self._end - self._start, len(self.assets)
        self._held = self._initial.copy()
        self._committed = self._initial.copy()
        self._aim = np.zeros((n_steps, n_assets))
        self._orders = np.zeros((n_steps, n_assets))
        self._quantity = np.zeros((n_steps, n_assets))
        self._price = np.full((n_steps, n_assets), np.nan)
        self._reference = np.full((n_steps, n_assets), np.nan)
        self._net, self._gross, self._cost = (np.zeros(n_steps) for _ in range(3))
        return self._observe(), {"decision": self.decisions[self._t], "episode": k}

    def step(self, action: ArrayLike):
        """Trade ``action`` of the gap in each name; return obs, reward, terminated, truncated, info.

        The reward is the net return. ``info`` has the decision and its
        ``day``, the book's ``gross``, ``cost``, ``traded`` and remaining
        ``gap``, and the ``payoff`` by name.
        """
        if self._t >= self._end:
            raise RuntimeError("the episode is over; call reset()")
        rate = self._rates(action)
        t, k = self._t, self._t - self._start
        goal = self._target[t]
        impact = self.config.cost_model in _IMPACT
        current = self._committed if impact else self._held
        # (1 - eta) w + eta g rather than w + eta (g - w): at eta 1 it is the
        # target exactly, as the harness trades to it.
        aim = (1.0 - rate) * current + rate * goal
        self._aim[k] = aim
        self._orders[k] = aim - current
        q, price, reference = self._fill(t, k)
        self._held = self._held + q
        self._committed = aim if impact else self._held
        self._quantity[k], self._price[k], self._reference[k] = q, price, reference

        payoff = self._payoff(t, self._held, q, price, reference)
        gross, cost = float(payoff.earned.sum()), float(payoff.paid.sum())
        self._net[k], self._gross[k], self._cost[k] = gross - cost, gross, cost
        self._t += 1
        ended = self._t >= self._end
        terminated = ended and self._t >= len(self.decisions)
        info = {"decision": self.decisions[t], "day": int(self._day[t]), "gross": gross,
                "cost": cost, "traded": float(np.abs(q).sum()),
                "gap": float(np.abs(goal - self._held).sum()), "payoff": payoff}
        return self._observe(), gross - cost, terminated, ended and not terminated, info

    def parallel_alternates(self, rates: ArrayLike, *, held: ArrayLike | None = None,
                decision: int | None = None) -> Payoff:
        """What each of ``rates`` would do at a decision, without playing it.

        ``rates`` is a 1-D array of candidates, each applied to every name.
        ``held`` is the position to trade from, ``(n_assets,)`` or
        ``(n_positions, n_assets)``, and ``decision`` the row of the panel;
        left out, they are the state the next :meth:`step` starts from. The
        :class:`Payoff` has shape ``held.shape[:-1] + (len(rates), n_assets)``
        and comes from the execution model :meth:`step` calls, so a candidate
        pays here exactly what it would pay played. Nothing in the episode
        changes. It reads the realized return, so it is for training: see the
        module docstring.
        """
        if self.schedule:
            raise ValueError("parallel_alternates needs fills that depend only on the decision, "
                             "and an order worked over a schedule is charged by those before it")
        t = self._t if decision is None else int(decision)
        if not 0 <= t < len(self.decisions):
            raise ValueError(f"decision must be in [0, {len(self.decisions)}), got {t}")
        rates = np.asarray(rates, dtype=np.float64)
        if (rates.ndim != 1 or len(rates) == 0 or not np.all(np.isfinite(rates))
                or np.any((rates < 0) | (rates > 1))):
            raise ValueError("rates must be a 1-D array of trading rates in [0, 1]")
        n_assets = len(self.assets)
        position, current = self._held, self._committed
        if held is not None:
            position = current = np.asarray(held, dtype=np.float64)
            if (position.ndim not in (1, 2) or position.shape[-1] != n_assets
                    or not np.all(np.isfinite(position))):
                raise ValueError(f"held must be finite with shape ({n_assets},) or "
                                 f"(n_positions, {n_assets}), got {position.shape}")
        rate, from_here = rates[:, None], current[..., None, :]
        order = (1.0 - rate) * from_here + rate * self._target[t] - from_here
        # Each candidate is one horizon of a one-decision order, which the
        # execution models fill independently.
        batch = order.reshape(-1, n_assets).T[None]
        rows = slice(t, t + 1)
        fills = self._execute(Orders(batch, self.decisions[rows], self.assets,
                                     np.arange(1, batch.shape[2] + 1)), rows)
        q, price, reference = (x[0].T.reshape(order.shape)
                               for x in (fills.quantity, fills.price, fills.mid_price))
        return self._payoff(t, position[..., None, :] + q, q, price, reference)

    # ── internals ────────────────────────────────────────────────────────────

    def _rates(self, action: ArrayLike) -> NDArray[np.float64]:
        rate = np.asarray(action, dtype=np.float64)
        if rate.ndim == 0:
            rate = np.full(len(self.assets), float(rate))
        if rate.shape != (len(self.assets),):
            raise ValueError(f"action must be a scalar or have shape ({len(self.assets)},), "
                             f"got {rate.shape}")
        if not np.all(np.isfinite(rate)) or np.any((rate < 0) | (rate > 1)):
            raise ValueError("every trading rate must be finite and in [0, 1]")
        return rate

    def _execute(self, orders: Orders, rows: slice) -> Fills:
        """Fill orders on these rows of the market, with the model the harness would call."""
        config = self.config
        market = _rows(self.market, rows)
        if config.cost_model in _IMPACT:
            return execute(orders, market, model=config.cost_model, equity=config.equity,
                           schedule=self.schedule, impact=config.impact)
        return execute(orders, market, model=config.cost_model, efq=config.efq,
                       equity=config.equity)

    def _fill(self, t: int, k: int):
        """This decision's fills."""
        if self.config.cost_model in _IMPACT:
            # Every order of the episode so far, so each parent is charged whole.
            rows = slice(self._start, t + 1)
            values = self._orders[:k + 1, :, None]
        else:
            rows = slice(t, t + 1)
            values = self._orders[k][None, :, None]
        fills = self._execute(Orders(values, self.decisions[rows], self.assets, self.axis), rows)
        return fills.quantity[-1, :, 0], fills.price[-1, :, 0], fills.mid_price[-1, :, 0]

    def _payoff(self, t: int, held, q, price, reference) -> Payoff:
        """Read a decision's fills by name; ``held`` is the position after them."""
        returns = self._outcome[t]
        finite = np.isfinite(returns)
        with np.errstate(invalid="ignore", divide="ignore"):
            slippage = np.where(q != 0, (price - reference) / reference, 0.0)
        goal = self._target[t]
        paper = np.where(finite, goal * returns, 0.0)
        return Payoff(held, np.where(finite, held * returns, 0.0),
                      q * slippage + self._fees(t, q),
                      np.broadcast_to(paper, held.shape), goal - held)

    def _fees(self, rows: int | slice, q: NDArray) -> NDArray[np.float64]:
        """The fees trades ``q`` at ``rows`` of the panel pay, by name, as the harness bills them."""
        if not self.config.fees:
            return np.zeros_like(q)
        price = None if self._share_price is None else self._share_price[rows]
        return fee_cost(q, price, **self.config.fees)

    def _observe(self) -> dict:
        """The state before the decision at ``_t``; past the panel, the last decision's market."""
        t = min(self._t, len(self.decisions) - 1)
        goal = self._target[t]
        state = [self._held, goal, goal - self._committed]
        if self.config.cost_model in _IMPACT:
            state.append(self._committed - self._held)
        names = np.concatenate([np.stack(state, axis=1), self._features[t]], axis=1)
        book = {"progress": (self._t - self._start) / (self._end - self._start),
                "gross_held": float(np.abs(self._held).sum()),
                "gross_target": float(np.abs(goal).sum()),
                "gross_gap": float(np.abs(goal - self._committed).sum())}
        if "time_of_day" in self.book_features:
            book["time_of_day"] = self._within[t] / self._per_day
        return {"names": names,
                "book": np.array([book[f] for f in self.book_features]),
                "mask": self._mask[t].copy()}

    def episode(self) -> Rollout:
        """The decisions of the current episode played so far, as a :class:`Rollout`."""
        n = self._t - self._start
        rows = slice(self._start, self._t)
        axes = (self.decisions[rows], self.assets, self.axis)
        return Rollout(
            target=PortfolioWeights(self._target[rows, :, None], *axes),
            weights=PortfolioWeights(self._aim[:n, :, None], *axes),
            fills=Fills(self._quantity[:n, :, None], self._price[:n, :, None],
                        self._reference[:n, :, None], *axes),
            realized=ForwardReturns(self._outcome[rows, :, None], *axes),
            net=self._net[:n].copy(), gross=self._gross[:n].copy(), cost=self._cost[:n].copy(),
            fees=self._fees(rows, self._quantity[:n]).sum(axis=1),
        )


def rollout(env: ExecutionEnv, policy, *, seed: int | None = None,
            options: dict | None = None) -> Rollout:
    """Play one episode with ``policy``, a callable from an observation to an action.

    A policy with a ``start(rng, n_assets)`` method is told when the episode
    begins, with the environment's generator, so one seed replays both.
    """
    observation, _ = env.reset(seed=seed, options=options)
    start_episode(policy, env)
    ended = False
    while not ended:
        observation, _, terminated, truncated, _ = env.step(policy(observation))
        ended = terminated or truncated
    return env.episode()


def start_episode(policy, env: ExecutionEnv) -> None:
    """Tell a policy that keeps state that an episode has begun, if it wants to know."""
    start = getattr(policy, "start", None)
    if start is not None:
        start(env.np_random, len(env.assets))
