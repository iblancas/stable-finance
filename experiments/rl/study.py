"""Do learned execution policies beat simple rules, on days they never saw?

Every learner trains on one panel and is scored on the next, as one episode,
by the harness: the same fills, the same net return and the same shortfall
split as any other trading policy. The bar is not trading to the target but
the best simple rule, a no-trade band tuned on the training panel, because a
learner that only rediscovers a band has added nothing.

Two controls keep the comparison honest. With a forecast of no skill there is
nothing to gain, so a learner should stop trading and no arm should beat the
band; one that does has found a leak or is overfitting noise. And an arm's
gain is shown beside the gross book it ran and its daily Sharpe, because
holding more of a good position is not execution.

The default reward is risk-neutral, so nothing stops a learner holding more
than the target. ``--reward risk`` trains on the same reward less the book's
variance, and tunes the band and the constant rate on the same utility, so
every arm wants the same thing. The target is sized to a gross, not by a risk
aversion, so the study uses the one at which the target's size is the best
one (:func:`implied_risk`). Every arm is scored on the harness's net return
either way, with the utility beside it, and a book that never trades is shown
because it scores zero on both: an arm below it did worse than doing nothing.

    uv run --extra hf --extra mds --extra rl-experiments python -m experiments.rl.study \\
        --months 2020-07 2020-08
    uv run --extra rl-experiments python -m experiments.rl.study --synthetic
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: a multithreaded BLAS sums in an
# order that depends on the thread count, which moves the last digits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

import stable_finance as sf
from experiments.rl.coupling import NetLimit
from experiments.rl.qlearning import explore, sweep
from stable_finance.evaluation.trades.rl import mean_variance_reward

OUT = Path(__file__).resolve().parent.parent / "results" / "rl"
HORIZON, EFQ, HALF_LIFE = 300, 0.7, 6
RATE_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
#: Wide enough to reach a book that barely trades, so that a band tuned on a
#: utility can get to its real best and the bar is not set by the end of the grid.
BAND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 25.0, 50.0)


class Panel(NamedTuple):
    """What an environment is built from."""

    target: sf.PortfolioWeights
    realized: sf.ForwardReturns
    market: sf.MarketData
    days: np.ndarray
    forecast: sf.ForwardReturns


class Risk(NamedTuple):
    """What a book's variance costs: one decision's covariance and a risk aversion."""

    covariance: np.ndarray
    aversion: float

    def penalty(self, held: np.ndarray) -> np.ndarray:
        """Half the risk aversion times the variance of ``held``, decision by decision."""
        return 0.5 * self.aversion * np.einsum("ti,ij,tj->t", held, self.covariance, held)


def implied_risk(panel: Panel) -> Risk:
    """The panel's covariance, and the risk aversion at which the target is the right size.

    The target is scaled to a gross, so it has no risk aversion to reuse. Scaled
    by ``k`` it has utility ``k edge - k^2 aversion variance / 2``, which peaks
    at ``k = 1`` when ``aversion = edge / variance``. A forecast with no edge
    implies none, so read this from a panel with skill.
    """
    covariance = sf.estimate_covariance(panel.realized)[0]
    weights = np.nan_to_num(panel.target.values[:, :, 0])
    edge = float(np.nansum(weights * panel.forecast.values[:, :, 0], axis=1).mean())
    if edge <= 0:
        raise ValueError("a forecast with no edge implies no risk aversion")
    variance = float(np.einsum("ti,ij,tj->t", weights, covariance, weights).mean())
    return Risk(covariance, edge / variance)


def make_env(panel: Panel, config: sf.BacktestConfig, **kwargs) -> sf.ExecutionEnv:
    return sf.ExecutionEnv(panel.target, panel.realized, panel.market, days=panel.days,
                           config=config, **kwargs)


def synthetic_panels(seed: int = 0, *, n_assets: int = 100, train_days: int = 30,
                     test_days: int = 20, ic: float = 0.03, per_day: int = 78,
                     volatility: float = 15e-4, half_spread: float = 2e-4,
                     keep: float = 0.9) -> tuple[Panel, Panel]:
    """A train and a test panel with a known signal: the ``target`` is what earns ``ic``.

    The target is a dollar-neutral AR(1) signal with persistence ``keep`` per
    decision, and each name's return has a mean of ``ic * volatility`` times
    its signal. Alpha is therefore proportional to the target by construction,
    which real forecasts are not: this world can show that a learner works,
    not how much it would earn. Spreads wander with a half-life of a few hours.
    ``per_day`` is how long this world's days are, there being no data to read
    it from; nothing else in the study assumes a length of day.
    """
    rng = np.random.default_rng(seed)
    n_decisions = (train_days + test_days) * per_day
    decisions = np.arange(n_decisions) * HORIZON
    assets = [f"S{i}" for i in range(n_assets)]
    signal = np.zeros((n_decisions, n_assets))
    level = np.zeros((n_decisions, n_assets))
    state, width = rng.normal(size=n_assets), rng.normal(size=n_assets)
    for t in range(n_decisions):
        state = keep * state + np.sqrt(1 - keep**2) * rng.normal(size=n_assets)
        width = 0.98 * width + np.sqrt(1 - 0.98**2) * rng.normal(size=n_assets)
        signal[t], level[t] = state, width
    signal -= signal.mean(axis=1, keepdims=True)
    weights = signal / np.abs(signal).sum(axis=1, keepdims=True)
    returns = rng.normal(0.0, volatility, signal.shape) + ic * volatility * signal
    # A return is earned AFTER its decision, so the price at a decision holds
    # only the returns before it. Otherwise last_return is the answer.
    moved = np.cumsum(returns, axis=0) - returns
    mid = 100 * np.exp(moved)
    half = mid * half_spread * np.exp(0.3 * level)
    size = rng.uniform(1e3, 1e4, signal.shape)
    columns = {"bid_price": mid - half, "ask_price": mid + half, "bid_size": size,
               "ask_size": size[::-1], "adv": np.full(signal.shape, 5e7)}

    def panel(rows: slice) -> Panel:
        grid = (decisions[rows], assets)
        n_rows = len(decisions[rows])
        return Panel(sf.PortfolioWeights(weights[rows, :, None], *grid, [HORIZON]),
                     sf.ForwardReturns(returns[rows, :, None], *grid, [HORIZON]),
                     sf.MarketData({k: v[rows] for k, v in columns.items()}, *grid),
                     np.repeat(np.arange(n_rows // per_day), per_day),
                     sf.ForwardReturns(ic * volatility * signal[rows, :, None], *grid, [HORIZON]))

    split = train_days * per_day
    return panel(slice(0, split)), panel(slice(split, None))


def liquid_names(month: str, others: Sequence[str], n_names: int) -> np.ndarray:
    """The ``n_names`` tightest names quoted nearly always, present in every month."""
    from experiments.month_cache import load_month

    data = load_month(month)
    assets = data["realized"].assets.astype(str)
    spread = data["half_spread"]
    quoted = np.isfinite(spread).mean(axis=0) > 0.95
    shared = set(assets)
    for other in others:
        shared &= set(load_month(other)["realized"].assets.astype(str))
    candidates = np.flatnonzero(quoted & np.isin(assets, sorted(shared)))
    tightest = np.argsort(np.nanmedian(spread[:, candidates], axis=0))[:n_names]
    return np.sort(assets[candidates[tightest]])


def month_panel(month: str, assets: np.ndarray, *, ic: float, seed: int = 0) -> Panel:
    """A Market-1T month with an oracle forecast of known skill as the target's source."""
    from experiments.month_cache import load_month
    from experiments.policy import signal_book

    data = load_month(month)
    realized = data["realized"]
    index = {a: i for i, a in enumerate(realized.assets.astype(str))}
    take = [index[a] for a in assets]
    horizon = int(np.flatnonzero(realized.horizons == HORIZON)[0])
    returns = sf.ForwardReturns(realized.values[:, take, horizon:horizon + 1],
                                realized.decisions, assets, np.asarray([HORIZON]))
    # No skill has no natural scale, and the book is normalised to its gross anyway.
    oracle = sf.copula_forecast(returns, np.asarray([ic]), half_life=HALF_LIFE,
                                days=data["days"], seed=seed, scale=None if ic else 1.0)
    target = sf.PortfolioWeights(signal_book(oracle.values)[:, :, None], realized.decisions,
                                 assets, np.asarray([HORIZON]))
    market = sf.MarketData({q: data["market"].get(q)[:, take]
                            for q in ("bid_price", "ask_price", "exec_bid_price",
                                      "exec_ask_price", "bid_size", "ask_size")},
                           realized.decisions, assets)
    return Panel(target, returns, market, data["days"], oracle)


def require_no_leak(env: sf.ExecutionEnv, panel: Panel, *, most: float = 0.5) -> float:
    """Refuse a panel whose last move already tells the return about to be earned.

    The environment keeps the future out of what it shows, but it can only
    show the prices it is given, and a price built from the return it is
    scored on hands a policy the answer. No real market lets a past return
    explain the next one this well. Returns the correlation.
    """
    realized = panel.realized.values[1:, :, 0]
    last = env.known("last_return")[1:]
    both = np.isfinite(realized) & (last != 0)
    correlation = float(np.corrcoef(last[both], realized[both])[0, 1])
    if abs(correlation) > most:
        raise ValueError(f"last_return has correlation {correlation:.3f} with the realized "
                         "return: the prices contain the future")
    return correlation


class ConstantRate:
    """:func:`~stable_finance.partial_adjustment`, step by step: the action is already its rate."""

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def __call__(self, observation: dict) -> float:
        return self.rate


class NoTradeBand:
    """:func:`~stable_finance.no_trade_band`, one decision at a time, from the book really held.

    The library rule follows a path from where it says the book is. A limit
    can cut an order, so here each decision starts from the position the
    environment reports. ``width`` is in mean targets: the band's half-width in
    weight is ``width`` times the target's mean absolute weight at the decision.
    """

    def __init__(self, env: sf.ExecutionEnv, width: float) -> None:
        self.width = float(width)
        self._target = env.name_features.index("target")
        self._gap = env.name_features.index("gap")
        self._gross = env.book_features.index("gross_target")
        self._grid = (env.decisions[:1], env.assets, env.axis)

    def __call__(self, observation: dict) -> np.ndarray:
        goal = observation["names"][:, self._target]
        gap = observation["names"][:, self._gap]
        current = goal - gap
        band = self.width * observation["book"][self._gross] / len(goal)
        moved = sf.no_trade_band(sf.PortfolioWeights(goal[None, :, None], *self._grid), band,
                                 initial=current[:, None]).values[0, :, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(np.where(gap != 0, (moved - current) / gap, 0.0), 0.0, 1.0)


def _held(episode: sf.Rollout) -> np.ndarray:
    return np.cumsum(episode.fills.quantity[:, :, 0], axis=0)


def _utility(episode: sf.Rollout, risk: Risk) -> np.ndarray:
    """The net return less what the book's variance costs, decision by decision."""
    return episode.net - risk.penalty(_held(episode))


def _best(env, policies: dict, risk: Risk | None) -> tuple:
    played = {key: sf.rollout(env, policy) for key, policy in policies.items()}
    scored = {key: (episode.net if risk is None else _utility(episode, risk)).mean()
              for key, episode in played.items()}
    return max(scored, key=scored.get)


def tuned_rate(env: sf.ExecutionEnv, rates: Sequence[float] = RATE_GRID, *,
               risk: Risk | None = None) -> ConstantRate:
    """The constant rate with the best net return on ``env``, or the best utility under ``risk``."""
    return ConstantRate(_best(env, {r: ConstantRate(r) for r in rates}, risk))


def tuned_band(env: sf.ExecutionEnv, widths: Sequence[float] = BAND_GRID, *,
               risk: Risk | None = None) -> NoTradeBand:
    """The band with the best net return on ``env``, or the best utility under ``risk``."""
    return NoTradeBand(env, _best(env, {w: NoTradeBand(env, w) for w in widths}, risk))


def _by_day(values: np.ndarray, days: np.ndarray) -> np.ndarray:
    _, index = np.unique(days, return_inverse=True)
    return np.bincount(index, values)


def score(episode: sf.Rollout, base: sf.Rollout, days: np.ndarray,
          risk: Risk | None = None) -> dict:
    """What the harness says of an episode, in basis points, beside its gain over ``base``.

    Under ``risk`` it also says what the episode was worth to an investor who
    minds variance. Nothing is charged: the net return is the same either way.
    """
    card, = sf.implementation_shortfall(episode.target, episode.fills, episode.realized,
                                        fees=episode.fees)
    daily = _by_day(episode.net, days)
    gain = daily - _by_day(base.net, days)
    held = _held(episode)
    utility = {}
    if risk is not None:
        worth = _by_day(_utility(episode, risk), days)
        over = worth - _by_day(_utility(base, risk), days)
        utility = {"utility_bp_day": float(worth.mean() * 1e4),
                   "utility_gain_bp_day": float(over.mean() * 1e4),
                   "utility_gain_se": float(over.std(ddof=1) / np.sqrt(len(over)) * 1e4)}
    return {
        "net_bp_day": float(daily.mean() * 1e4),
        "paper_bp_day": float(_by_day(card.paper, days).mean() * 1e4),
        "execution_bp_day": float(_by_day(card.execution, days).mean() * 1e4),
        "opportunity_bp_day": float(_by_day(card.opportunity, days).mean() * 1e4),
        "gain_bp_day": float(gain.mean() * 1e4),
        "gain_se": float(gain.std(ddof=1) / np.sqrt(len(gain)) * 1e4),
        "sharpe_day": float(daily.mean() / daily.std(ddof=1)) if daily.std() > 0 else 0.0,
        "gross_held": float(np.abs(held).sum(axis=1).mean()),
        "net_exposure": float(np.abs(held.sum(axis=1)).mean()),
        **utility,
    }


def _pool(rows: list[dict]) -> dict:
    """Seeds of one arm: each number's mean, the gain's range, and the worst seed's net exposure.

    The worst seed is kept because a mean of exposures can be one seed's runaway.
    """
    pooled = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    gains = [row["gain_bp_day"] for row in rows]
    return {**pooled, "seeds": len(rows), "gain_low": min(gains), "gain_high": max(gains),
            "exposure_high": max(row["net_exposure"] for row in rows)}


def compare(train: Panel, test: Panel, config: sf.BacktestConfig, *, seeds: int = 3,
            episodes: int = 1000, dqn: bool = True, dqn_episodes: int = 200,
            limit: float | None = None, risk: Risk | None = None, averse: bool = False,
            log: Callable = print) -> dict:
    """Every arm on one train and test panel, against the tuned band.

    ``risk`` adds each arm's utility to the table. ``averse`` makes that utility
    what every arm is after: the learners train on it and the rules are tuned on it.
    """
    if averse and risk is None:
        raise ValueError("averse needs a risk")
    wants = risk if averse else None
    reward = (mean_variance_reward(risk.covariance, risk.aversion) if averse else "shortfall")
    day_env = make_env(train, config, episode_days=1)
    whole = make_env(train, config)
    rate, band_width = tuned_rate(whole, risk=wants), tuned_band(whole, risk=wants).width
    test_env = make_env(test, config)
    correlation = require_no_leak(test_env, test)
    log(f"  tuned on train: rate {rate.rate}, band {band_width}; "
        f"last_return against the return earned: {correlation:+.3f}")

    def scored(policy, env=None, base=None):
        episode = sf.rollout(env or test_env, policy)
        return score(episode, episode if base is None else base, test.days, risk)

    base = sf.rollout(test_env, NoTradeBand(test_env, band_width))
    arms = {"trade to target": scored(ConstantRate(1.0), base=base),
            "no trading": scored(ConstantRate(0.0), base=base),
            "best constant rate": scored(rate, base=base),
            "no-trade band": scored(NoTradeBand(test_env, band_width), base=base)}
    if test_env.schedule == 0:
        start = time.perf_counter()
        arms["sweep"] = scored(sweep(whole, reward=reward), base=base)
        log(f"  sweep {time.perf_counter() - start:.0f}s {arms['sweep']['gain_bp_day']:+.2f}")
    start = time.perf_counter()
    arms["epsilon-greedy table"] = _pool([
        scored(explore(day_env, reward=reward, episodes=episodes, seed=s), base=base)
        for s in range(seeds)])
    log(f"  explore {time.perf_counter() - start:.0f}s")
    if dqn:
        from experiments.rl.dqn import train as train_dqn
        from experiments.rl.ppo import train as train_ppo

        def agents(learn, **kwargs):
            start = time.perf_counter()
            rows = []
            for s in range(seeds):
                agent = learn(day_env, reward=reward, episodes=dqn_episodes, seed=s, **kwargs)
                rows.append(scored(agent.policy(test_env), base=base))
            log(f"  {learn.__module__} {sorted(kwargs)} {time.perf_counter() - start:.0f}s")
            return _pool(rows)

        arms["double DQN"] = agents(train_dqn)
        band_expert = NoTradeBand(day_env, band_width)
        arms["double DQN, band in buffer"] = agents(train_dqn, experts={"band": band_expert})
        arms["PPO"] = agents(train_ppo)
    if limit is not None:
        arms.update(_coupled(train, test, config, limit, band_width, rate, seeds, dqn,
                             dqn_episodes, risk, reward, log))
    return arms


def _coupled(train, test, config, limit, band_width, rate, seeds, dqn, dqn_episodes, risk, reward,
             log) -> dict:
    """The arms under a net-exposure limit, against the band under the same limit."""
    limited = NetLimit(make_env(test, config), limit)

    def scored(policy, base):
        episode = sf.rollout(limited, policy)
        return score(episode, episode if base is None else base, test.days, risk)

    base = sf.rollout(limited, NoTradeBand(limited, band_width))
    arms = {"limit: best constant rate": scored(rate, base),
            "limit: no-trade band": scored(NoTradeBand(limited, band_width), base)}
    if not dqn:
        return arms
    from experiments.rl.dqn import train as train_dqn
    from experiments.rl.ppo import train as train_ppo

    training = NetLimit(make_env(train, config, episode_days=1), limit)
    for name, learn, kwargs in (("independent", train_dqn, {}),
                                ("team (VDN)", train_dqn, {"team": True}),
                                ("team + intention", train_dqn, {"team": True, "intention": True}),
                                ("PPO", train_ppo, {})):
        start = time.perf_counter()
        rows = [scored(learn(training, reward=reward, episodes=dqn_episodes, seed=s,
                             **kwargs).policy(limited), base) for s in range(seeds)]
        arms[f"limit: {name}"] = _pool(rows)
        log(f"  {name} {time.perf_counter() - start:.0f}s")
    return arms


def show(title: str, arms: dict) -> None:
    print(f"\n{title}")
    print(f"{'arm':30s}{'net':>8s}{'exec':>8s}{'opp':>8s}{'vs band':>9s}{'+/-':>7s}"
          f"{'seeds lo..hi':>17s}{'Sharpe/d':>9s}{'gross':>7s}{'|net|':>7s}{'utility':>9s}"
          f"{'vs band':>9s}{'+/-':>7s}")
    for name, row in arms.items():
        spread = (f"{row['gain_low']:+.1f}..{row['gain_high']:+.1f}" if "gain_low" in row else "")
        print(f"{name:30s}{row['net_bp_day']:8.1f}{row['execution_bp_day']:8.1f}"
              f"{row['opportunity_bp_day']:8.1f}{row['gain_bp_day']:9.2f}{row['gain_se']:7.2f}"
              f"{spread:>17s}{row['sharpe_day']:9.2f}{row['gross_held']:7.2f}"
              f"{row['net_exposure']:7.3f}{row.get('utility_bp_day', float('nan')):9.1f}"
              f"{row.get('utility_gain_bp_day', float('nan')):9.2f}"
              f"{row.get('utility_gain_se', float('nan')):7.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="no data needed")
    parser.add_argument("--months", nargs=2, metavar=("TRAIN", "TEST"),
                        help="YYYY-MM, needed unless --synthetic")
    parser.add_argument("--names", type=int, default=300)
    parser.add_argument("--ic", type=float, default=0.03)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--dqn-episodes", type=int, default=200)
    parser.add_argument("--limit", type=float, default=0.05)
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument("--reward", choices=("shortfall", "risk"), default="shortfall",
                        help="what every arm is after: see the module docstring")
    args = parser.parse_args(argv)
    if not args.synthetic and args.months is None:
        parser.error("--months TRAIN TEST is needed unless --synthetic")
    OUT.mkdir(parents=True, exist_ok=True)
    config = sf.BacktestConfig(cost_model="cross", efq=EFQ, annualization="decisions")
    results, risk = {}, None
    for name, ic in (("skill", args.ic), ("no skill", 0.0)):
        print(f"\n== {name}, IC {ic}", flush=True)
        if args.synthetic:
            train, test = synthetic_panels(ic=ic)
        else:
            assets = liquid_names(args.months[0], args.months[1:], args.names)
            train = month_panel(args.months[0], assets, ic=ic)
            test = month_panel(args.months[1], assets, ic=ic, seed=1)
        # From the panel with skill, and kept: a preference does not change
        # because the forecast lost its edge, and no edge implies no aversion.
        risk = risk or implied_risk(train)
        print(f"  risk aversion {risk.aversion:.0f}", flush=True)
        results[name] = compare(train, test, config, seeds=args.seeds, episodes=args.episodes,
                                dqn=not args.no_dqn, dqn_episodes=args.dqn_episodes,
                                limit=args.limit, risk=risk, averse=args.reward == "risk",
                                log=lambda text: print(text, flush=True))
        show(f"{name}: bp of equity a day, held out", results[name])
    tag = "synthetic" if args.synthetic else "-".join(args.months)
    tag += "-risk" if args.reward == "risk" else ""
    (OUT / f"study-{tag}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
