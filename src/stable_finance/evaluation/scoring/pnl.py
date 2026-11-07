"""What a portfolio held and earned: P&L series, gross and after costs.

Weights or fills in, one return series per horizon out, with the costs that
were charged. Positions are tracked from fills, so a missed fill shows up as a
position that lags its target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation._grid import require_aligned
from stable_finance.evaluation.execution import Fills, initial_positions
from stable_finance.dataset.market import (
    MarketData,
    market_quantity,
    requires,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio


@dataclass(frozen=True)
class BacktestResult:
    """One periodic return series and its summary statistics."""

    gross_returns: NDArray[np.float64]
    costs: NDArray[np.float64]
    net_returns: NDArray[np.float64]
    sharpe: float


@requires("half_spread")
def evaluate_weights(
    weights: PortfolioWeights,
    realized: ForwardReturns,
    *,
    half_spread: ArrayLike | MarketData | None = None,
    periods_per_year: float = 252.0,
) -> tuple[BacktestResult, ...]:
    """Evaluate weights independently at every horizon.

    ``half_spread`` is the entry half-spread divided by mid price and must have
    shape ``(n_decisions, n_assets)``. Turnover is ``abs(w_t - w_{t-1})``;
    the first portfolio is traded from cash. This models crossing the quoted
    spread and deliberately excludes market impact and commissions.

    ``half_spread=None`` is the frictionless result, stated explicitly.
    :class:`MarketData` in its place must carry a spread (or both quotes): a
    market handed over to be charged is never silently charged nothing.
    """
    require_aligned(weights, realized)
    half_spread = market_quantity(half_spread, "half_spread", stage=evaluate_weights,
                                  like=weights)
    n_decisions, n_assets, n_horizons = weights.values.shape
    if half_spread is None:
        spread = np.zeros((n_decisions, n_assets), dtype=np.float64)
    else:
        spread = np.asarray(half_spread, dtype=np.float64)
        if spread.shape != (n_decisions, n_assets):
            raise ValueError(
                "half_spread must have shape "
                f"{(n_decisions, n_assets)}, got {spread.shape}"
            )
        if np.any(np.isfinite(spread) & (spread < 0)):
            raise ValueError("half_spread cannot be negative")

    results = []
    for horizon in range(n_horizons):
        weight = weights.values[:, :, horizon]
        outcome = realized.values[:, :, horizon]
        valid = np.isfinite(weight) & np.isfinite(outcome)
        gross = np.where(valid, weight * outcome, 0.0).sum(axis=1)
        gross[~valid.any(axis=1)] = np.nan

        previous = np.vstack([np.zeros((1, n_assets)), weight[:-1]])
        turnover = np.abs(weight - previous)
        costs = np.where(np.isfinite(turnover) & np.isfinite(spread),
                         turnover * spread, 0.0).sum(axis=1)
        costs[~np.isfinite(weight).any(axis=1)] = np.nan
        net = gross - costs
        results.append(BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            sharpe=sharpe_ratio(net, periods_per_year=periods_per_year),
        ))
    return tuple(results)


def positions_from_fills(
    fills: Fills, *, initial: ArrayLike | None = None, drift: ArrayLike | None = None
) -> PortfolioWeights:
    """Weights actually held after each decision's fills.

    Without ``drift`` a position is the running sum of its fills
    (``holding="rebalanced"``). With ``drift`` -- each name's realized return
    from each decision to the next, ``(decision, asset, horizon)`` -- what is
    held moves with that return between decisions before the next fills land
    (``holding="drift"``).
    """
    n_decisions, n_assets, n_horizons = fills.quantity.shape
    start = initial_positions(initial, (n_assets, n_horizons))
    if drift is None:
        positions = start[None] + np.cumsum(fills.quantity, axis=0)
    else:
        growth = 1.0 + np.nan_to_num(np.asarray(drift, dtype=np.float64), nan=0.0)
        if growth.shape != fills.quantity.shape:
            raise ValueError(f"drift must have shape {fills.quantity.shape}, got {growth.shape}")
        positions = np.empty(fills.quantity.shape)
        held = start
        for t in range(n_decisions):
            held = (held * growth[t - 1] if t else held) + fills.quantity[t]
            positions[t] = held
    return PortfolioWeights(positions, fills.decisions, fills.assets, fills.horizons)


def evaluate_fills(
    fills: Fills,
    realized_mid_returns: ForwardReturns,
    *,
    initial: ArrayLike | None = None,
    periods_per_year: float = 252.0,
    drift: bool = False,
    fees: ArrayLike | None = None,
) -> tuple[BacktestResult, ...]:
    """Mark the realized position path at mid and charge each fill's cost.

    ``drift=True`` carries positions with their returns between decisions
    (``holding="drift"``); the returns must then run decision to decision.

    Gross return is the position held after the decision's fills times the
    mid-to-mid forward return. Execution cost is the filled quantity times the
    fill price's distance from mid, so buys above mid and sells below mid both
    cost money. One result is returned per horizon.

    ``fees`` is what each fill paid on top, as a fraction of equity, on the
    fills' grid (it broadcasts to ``fills.quantity``) -- typically
    :func:`~stable_finance.evaluation.execution.fee_cost` of the fills'
    quantity -- and is added to the cost. Fees are kept out of the fill price,
    so they are passed here rather than read from the fills.
    """
    require_aligned(fills, realized_mid_returns)
    positions = positions_from_fills(
        fills, initial=initial, drift=realized_mid_returns.values if drift else None).values
    traded = fills.quantity != 0
    slippage = np.where(
        traded, (fills.price - fills.mid_price) / fills.mid_price, 0.0
    )
    per_asset_cost = fills.quantity * slippage
    if fees is not None:
        try:
            paid = np.broadcast_to(np.asarray(fees, dtype=np.float64), per_asset_cost.shape)
        except ValueError:
            raise ValueError(f"fees must broadcast to {per_asset_cost.shape}") from None
        if not np.isfinite(paid).all() or (paid < 0).any():
            raise ValueError("fees must be finite and non-negative")
        per_asset_cost = per_asset_cost + paid
    results = []
    for horizon in range(len(fills.horizons)):
        position = positions[:, :, horizon]
        outcome = realized_mid_returns.values[:, :, horizon]
        valid = np.isfinite(outcome)
        gross = np.where(valid, position * outcome, 0.0).sum(axis=1)
        gross[~valid.any(axis=1)] = np.nan
        costs = per_asset_cost[:, :, horizon].sum(axis=1)
        net = gross - costs
        results.append(BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            sharpe=sharpe_ratio(net, periods_per_year=periods_per_year),
        ))
    return tuple(results)
