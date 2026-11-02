"""Implementation shortfall: what reaching a book cost, apart from what the book was.

A net Sharpe mixes two questions: was the target portfolio any good, and how
well was it reached? A forecast that is excellent but traded badly and one
that is mediocre but traded well can report the same net return. Perold
(1988, "The implementation shortfall: paper versus reality", Journal of
Portfolio Management 14(3):4-9) separates them by comparing the book against a
PAPER PORTFOLIO: one that holds the target at every decision, filled in full
and for free at the price the realized return is measured from. The
implementation shortfall is the paper return minus the real one, and it
splits exactly in two:

    shortfall_t   = paper_t - net_t = execution_t + opportunity_t
    execution_t   = sum_i q_t,i * (p_t,i - m_t,i) / m_t,i
    opportunity_t = sum_i (target_t,i - held_t,i) * r_t,i

EXECUTION is what the fills paid over the price they are measured against:
the spread at the touch the order met under ``cross`` (and the drift while it
travelled there), the depth walked under ``book``, the impact under
``impact``, a resting order's limit. OPPORTUNITY is the return the paper
portfolio earned on the part of the target the real book did not hold: an
order that did not fill, an impact slice still being worked, a name that
stopped quoting. Under the impact model, the price move while an order is
worked lands here, because the position lags the target while it moves.

THE ACCOUNTING IS THE HARNESS'S. Positions come from
:func:`~stable_finance.evaluation.scoring.pnl.positions_from_fills`, each fill
is charged ``q (p - m) / m`` as in :func:`~stable_finance.run_backtest`, and a
decision at which no name has a realized return earns nothing. So ``net`` here
is the harness's net return for the same fills, decision by decision, and a
shortfall always reconciles with the Sharpe it is reported beside. Only what
the fills paid is charged; a cost a policy assumed when choosing its trades
never enters.

The scorecard takes the target the fills were chasing, not a forecast, so it
scores the trades stage and the execution model alone. That is the number an
execution policy, learned or hand-written, is judged on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData, market_quantity, requires
from stable_finance.evaluation._grid import require_aligned
from stable_finance.evaluation.execution import Fills, Orders, _initial_positions
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.pnl import positions_from_fills

__all__ = ["ImplementationShortfall", "implementation_shortfall"]


@dataclass(frozen=True)
class ImplementationShortfall:
    """One horizon's shortfall, decision by decision, with its summaries.

    Every series has one value per decision, in fractions of equity:
    ``paper`` is what the target earned, ``gross`` what the held book earned,
    ``execution`` what the fills paid and ``opportunity`` what the gap to the
    target lost. ``traded`` is the two-way traded weight, the sum over names of
    the absolute fill, and ``gap`` the gross weight still away from the target
    after each decision's fills.

    ``fill_rate`` is the fraction of the gap to the target open at each
    decision that its fills closed, pooled over the run: 1 when every
    decision ends at its target, 0 when nothing trades. For an order path
    chased from what is held it is the filled share of what was sent; a fill
    that moves a name away from its target counts against it. NaN when the
    book never had a gap to close.

    ``realized_efq`` is the effective half-spread paid over the half-spread
    quoted at the decision, pooled over every fill with a quote: the
    execution quality the fills actually achieved, on the same scale as the
    ``efq`` a configuration assumes. NaN without quotes.
    """

    paper: NDArray[np.float64]
    gross: NDArray[np.float64]
    execution: NDArray[np.float64]
    opportunity: NDArray[np.float64]
    traded: NDArray[np.float64]
    gap: NDArray[np.float64]
    fill_rate: float
    realized_efq: float

    @property
    def net(self) -> NDArray[np.float64]:
        """The held book's return after what its fills paid."""
        return self.gross - self.execution

    @property
    def shortfall(self) -> NDArray[np.float64]:
        """The paper return minus the net one: execution plus opportunity."""
        return self.execution + self.opportunity

    @property
    def execution_bps(self) -> float:
        """Execution cost per unit of weight traded, in basis points; NaN if nothing traded."""
        traded = float(self.traded.sum())
        return float(self.execution.sum() / traded * 1e4) if traded > 0 else float("nan")


def _paper_path(target: PortfolioWeights | Orders,
                initial: NDArray[np.float64]) -> NDArray[np.float64]:
    """The weights the paper portfolio holds after each decision.

    A target path is held as given, NaN as flat, as :func:`fill_path` chases
    it. Orders are the path they imply when every one fills: the running sum
    of the deltas from ``initial``.
    """
    values = np.nan_to_num(np.asarray(target.values, dtype=np.float64), nan=0.0)
    if isinstance(target, Orders):
        return initial[None] + np.cumsum(values, axis=0)
    return values


@requires("half_spread")
def implementation_shortfall(
    target: PortfolioWeights | Orders,
    fills: Fills,
    realized: ForwardReturns,
    *,
    half_spread: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    drift: bool = False,
) -> tuple[ImplementationShortfall, ...]:
    """Split the gap between a target and its fills into execution and opportunity.

    ``target`` is what the fills were chasing: the weight path handed to
    :func:`~stable_finance.evaluation.execution.fill_path`, or the
    :class:`~stable_finance.evaluation.execution.Orders` handed to
    :func:`~stable_finance.evaluation.execution.execute`, whose paper path is
    every order filled. ``realized`` is the return each name earned from each
    decision, the one the book is scored on. ``initial`` is the book held
    before the first decision (default cash), for both the paper and the real
    portfolio. ``drift=True`` is ``holding="drift"``: the real book moves with
    its returns between decisions, which must then run decision to decision;
    the paper book is the target at every decision either way.

    ``half_spread`` (a ``(n_decisions, n_assets)`` array, or market data
    carrying one) is the quoted half-spread at each decision, read only for
    ``realized_efq``; without it that is NaN.

    One result is returned per horizon, as by
    :func:`~stable_finance.evaluation.scoring.pnl.evaluate_fills`.
    """
    require_aligned(fills, target)
    require_aligned(fills, realized)
    spread = market_quantity(half_spread, "half_spread", stage=implementation_shortfall,
                             like=fills)
    n_decisions, n_assets, n_horizons = fills.quantity.shape
    if spread is not None:
        spread = np.asarray(spread, dtype=np.float64)
        if spread.shape != (n_decisions, n_assets):
            raise ValueError(f"half_spread must have shape {(n_decisions, n_assets)}, "
                             f"got {spread.shape}")
        if np.any(np.isfinite(spread) & (spread < 0)):
            raise ValueError("half_spread cannot be negative")

    start = _initial_positions(initial, (n_assets, n_horizons))
    aim = _paper_path(target, start)
    held = positions_from_fills(
        fills, initial=start, drift=realized.values if drift else None).values
    # What was held just before each decision's fills: under drift, the
    # previous book after it moved with its returns.
    before = held - fills.quantity
    traded = fills.quantity != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        slippage = np.where(traded, (fills.price - fills.mid_price) / fills.mid_price, 0.0)
    charged = fills.quantity * slippage

    results = []
    for h in range(n_horizons):
        outcome = realized.values[:, :, h]
        valid = np.isfinite(outcome)
        r = np.where(valid, outcome, 0.0)
        paper = (aim[:, :, h] * r).sum(axis=1)
        gross = (held[:, :, h] * r).sum(axis=1)
        opportunity = ((aim[:, :, h] - held[:, :, h]) * r).sum(axis=1)

        quantity = fills.quantity[:, :, h]
        opened = np.abs(aim[:, :, h] - before[:, :, h])
        left = np.abs(aim[:, :, h] - held[:, :, h])
        open_total = opened.sum()
        fill_rate = (float((opened - left).sum() / open_total) if open_total > 0
                     else float("nan"))

        realized_efq = float("nan")
        if spread is not None:
            quoted = traded[:, :, h] & np.isfinite(spread) & (spread > 0)
            paid = np.where(quoted, np.abs(quantity) * np.where(quoted, spread, 0.0), 0.0).sum()
            if paid > 0:
                realized_efq = float(np.where(quoted, charged[:, :, h], 0.0).sum() / paid)

        results.append(ImplementationShortfall(
            paper=paper,
            gross=gross,
            execution=charged[:, :, h].sum(axis=1),
            opportunity=opportunity,
            traded=np.abs(quantity).sum(axis=1),
            gap=left.sum(axis=1),
            fill_rate=fill_rate,
            realized_efq=realized_efq,
        ))
    return tuple(results)
