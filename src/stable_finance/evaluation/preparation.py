"""What a run needs before any configuration acts on it.

:func:`~stable_finance.run_backtest` and
:class:`~stable_finance.evaluation.trades.rl.env.ExecutionEnv` must read the
same inputs the same way, or a policy could score one number in the environment
and another in the harness. So both start here: the checks on a configuration,
the grid and outcomes in a context, how a span of time becomes decisions, and
the price a per-share fee is converted at.

This sits BELOW both of them. The backtest imports the trades stage, and the
environment is part of that stage, so neither can import the other.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.config import BacktestConfig, check_config
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["check_axes", "day_index", "decision_seconds", "decisions_spanning",
           "impact_schedule", "prepare", "share_price"]


def check_axes(config: BacktestConfig, market: MarketData, source) -> None:
    """Check the axes a level uses; below the forecast only trading remains."""
    if isinstance(source, ForwardReturns):
        check_config(config, market)
    else:
        if config.factor_limit is not None:
            raise ValueError("factor_limit builds a target from forecasts; for weights or "
                             "orders pass with_exposure to report held exposures")
        check_config(config.replace(universe="all", alpha=None, alpha_ic=None,
                                    neutralize=None), market)


def day_index(days: ArrayLike | None, n_decisions: int) -> NDArray[np.int64]:
    """Map each decision to a day, defaulting to one day per decision."""
    if days is None:
        return np.arange(n_decisions, dtype=np.int64)
    labels = np.asarray(days).ravel()
    if len(labels) != n_decisions:
        raise ValueError(f"days must have length {n_decisions}, got {len(labels)}")
    _, index = np.unique(labels, return_inverse=True)
    return index.astype(np.int64)


def prepare(like, realized, market: MarketData, days, horizon_index) -> dict:
    """Everything downstream needs that no configuration can change.

    ``like`` is the input on the grid: a forecast, whose values are ``mu``, or
    weights or orders, which have no forecast and so no ``mu``.
    """
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(like, axis), getattr(realized, axis)):
            raise ValueError(f"{axis} of the input and the realized returns are not aligned")
    n_decisions, n_assets, _ = like.values.shape
    # None, not NaN, when the market has no spread: NaN is a name that was not
    # quoting, and reading a missing column that way would screen out the
    # whole universe instead of trading it frictionlessly.
    spread = market.get("half_spread") if market.has("half_spread") else None
    horizon = int(like.horizons[horizon_index])
    axis = np.asarray([horizon])
    outcome_values = realized.values[:, :, horizon_index]
    return {
        "mu": (like.values[:, :, horizon_index] if isinstance(like, ForwardReturns)
               else None),
        "market": market,
        "outcome_values": outcome_values,
        "outcome": ForwardReturns(outcome_values[:, :, None], realized.decisions,
                                  realized.assets, axis),
        "spread": spread, "axis": axis, "horizon": horizon,
        "decisions": like.decisions, "assets": like.assets,
        "day": day_index(days, n_decisions),
        # what the caller called each decision's day (else the decision itself), for the P&L series
        "day_labels": np.asarray(days).ravel() if days is not None else np.asarray(like.decisions),
        "days_given": days is not None,
        "n_decisions": n_decisions, "n_assets": n_assets,
    }


def decision_seconds(context: dict) -> float:
    """The spacing between consecutive decisions on one day, in seconds.

    Read from the decisions themselves: epoch seconds, as
    :func:`~stable_finance.evaluation.inputs.evaluation_inputs` takes them, or
    the ``"date|anchor"`` labels the Market-1T readers write, whose anchor is
    seconds past the session's origin. The median gap within a day is used,
    so overnight gaps and an occasional missing decision do not count.
    """
    decisions = np.asarray(context["decisions"])
    try:
        stamps = decisions.astype(np.float64)
    except (TypeError, ValueError):
        parts = np.char.partition(decisions.astype(str), "|")
        if not np.all(parts[:, 1] == "|"):
            raise ValueError("this needs the decisions' times: pass decisions as epoch "
                             "seconds or 'date|anchor' labels") from None
        stamps = parts[:, 2].astype(np.float64)
    gaps = np.diff(stamps)
    if context["days_given"]:
        day = context["day"]
        gaps = gaps[day[1:] == day[:-1]]
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        raise ValueError("need two decisions on one day to measure their spacing; "
                         "pass days labelling each decision's trading day")
    return float(np.median(gaps))


def decisions_spanning(context: dict, *, seconds: float | None = None,
                        days: float | None = None) -> int:
    """How many decisions a span of time covers: the one time-to-decisions rule.

    A span in ``days`` is that many times the decisions the data has in a
    day, read from the ``days`` labels (the impact model's schedule). A span
    in ``seconds`` is that over the spacing of the decisions (a held book's
    horizon). Both round to the nearest whole decision.
    """
    if days is not None:
        if not context["days_given"]:
            raise ValueError("a span in days needs days labelling each decision's trading day")
        per_day = float(np.median(np.bincount(context["day"])))
        return int(round(days * per_day))
    return max(round(float(seconds) / decision_seconds(context)), 1)


def impact_schedule(context: dict, config: BacktestConfig) -> int:
    """How many decisions the impact model works an order over.

    One trading day is however many decisions the data has in a day, read
    from the ``days`` labels; a run without them cannot say what a day is.
    The default is two days, about the horizon of the paper's own trades.
    """
    days = 2.0 if config.execution_days is None else config.execution_days
    if days == 0:
        return 0
    if not context["days_given"]:
        raise ValueError(f"cost_model={config.cost_model!r} works orders over execution_days, so it "
                         "needs days labelling each decision's trading day")
    return decisions_spanning(context, days=days)


def share_price(context: dict, config: BacktestConfig) -> NDArray[np.float64] | None:
    """The share price a per-share fee is converted at: the mid, else not needed."""
    return context["market"].get("mid") if config.fee_per_share is not None else None
