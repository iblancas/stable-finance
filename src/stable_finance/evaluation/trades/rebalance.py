"""How far to move from the portfolio held toward the target, given what trading costs.

The portfolio stage solves for the holding it would want with no frictions:
for mean-variance, ``w* = Sigma^-1 mu / risk_aversion``. Trading there costs
money, so this stage decides how much of the move is worth making. Each
rebalance chooses the weights that minimise

    (risk_aversion / 2) (w - w*)' Sigma (w - w*) + sum_i cost_i |w_i - w_prev_i|

-- tracking error against the target, under the target's own risk model,
against the cost of the trade from ``w_prev``. ``cost_i`` is the cost per
unit of weight traded, for instance the quoted half-spread as a fraction of
mid. Without it every rebalance jumps straight to the target.

This is the cost-aware mean-variance problem, not an approximation of it:

    mu' w - (risk_aversion / 2) w' Sigma w
        = -(risk_aversion / 2) (w - w*)' Sigma (w - w*) + const

so :func:`rebalance` toward the mean-variance target gives the weights
:func:`allocate` gets by solving the whole objective at once. ``allocate`` and
:class:`MeanVarianceAllocator` remain for callers who hold a forecast rather
than a target. The problem is convex and is solved by accelerated proximal
gradient descent, whose proximal step is the trade rule itself: move toward
the target unless the cost of the move exceeds what it buys.

``risk_aversion`` may be one scalar or one value per decision. A per-decision
``lambda_t`` lets a caller calibrate the trade-off from data available at each
decision instead of from the whole panel; each decision's solve then uses its
own ``lambda_t``, step size included, and a scalar is the constant special case.

There are two solvers, and the choice is numerical, never a modelling one: both
aim at the same unique optimum (the objective is strongly convex whenever
``Sigma`` is positive definite).

The default, ``solver="restart"``, runs to the step tolerance. It is
accelerated proximal gradient with the adaptive restart of O'Donoghue and
Candes (2015, "Adaptive restart for accelerated gradient schemes", Found.
Comput. Math. 15:715-732): whenever the step just taken points against the
generalized gradient, the momentum has carried the iterate past the valley
floor and is reset. That one test recovers the linear rate an accelerated
method has on a strongly convex problem without anyone having to supply the
strong-convexity constant, which a shrunk covariance does not advertise. It
also works in a metric scaled by each name's variance -- a diagonal (Jacobi)
preconditioner, so each name takes a step sized to its own curvature rather
than to the riskiest direction in the book. The scaled problem's condition
number is that of the correlation matrix, smaller than the covariance's, and
the accelerated rate improves with its square root. The proximal step stays
the per-name soft threshold, with each name's own step. Its iteration cap,
:data:`RESTART_MAX_ITER` or ``max_iter`` if larger, is a safety net and not a
budget: a run-to-tolerance solver capped low is the capped solver again. A
solve that reaches it raises a :class:`~sklearn.exceptions.ConvergenceWarning`,
and ``return_n_iter=True`` reports every decision's iteration count.

``solver="fista"`` is plain FISTA stopped after ``max_iter`` iterations. On a
real cross-section of several hundred names that budget runs out at every
decision: FISTA's momentum makes it oscillate around the optimum of a strongly
convex problem rather than settle, so the capped answer is an approximation
whose error carries forward, because each decision starts from the last. It is
kept so that the cost of a fixed budget can be measured
(``experiments.mv_solver``), not as a way to run the allocator.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from stable_finance.dataset.market import MarketData, market_quantity, requires
from stable_finance.evaluation.config import REBALANCE_SOLVERS
from stable_finance.evaluation.portfolio.construction import _per_decision_risk_aversion
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.risk import estimate_covariance
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "DEFAULT_SOLVER",
    "REBALANCE_SOLVERS",
    "RESTART_MAX_ITER",
    "MeanVarianceAllocator",
    "allocate",
    "rebalance",
]

#: The solver every rebalance uses unless told otherwise.
DEFAULT_SOLVER = "restart"

#: The least iteration cap the ``"restart"`` solver runs under. On the real
#: month it converges in a few hundred iterations per decision; the cap exists
#: only so a pathological input cannot loop for ever.
RESTART_MAX_ITER = 20000


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _rebalance(
    mu: np.ndarray,
    sigma: np.ndarray,
    previous: np.ndarray,
    cost: np.ndarray,
    *,
    risk_aversion: float,
    step: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Solve one rebalance with FISTA; the cost term is handled by its prox.

    Returns the weights and the number of iterations run, which is
    ``max_iter`` when the budget ran out.
    """
    weights = previous.copy()
    momentum = previous.copy()
    t = 1.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        gradient = risk_aversion * (sigma @ momentum) - mu
        proposal = momentum - step * gradient
        # Proximal step for cost * |w - previous|: shrink the trade toward
        # zero. An infinite cost freezes the asset at its previous weight.
        updated = previous + _soft_threshold(proposal - previous, step * cost)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = updated + ((t - 1.0) / t_next) * (updated - weights)
        converged = np.max(np.abs(updated - weights)) < tol
        weights, t = updated, t_next
        if converged:
            break
    return weights, iterations


def _rebalance_restart(
    mu: np.ndarray,
    sigma: np.ndarray,
    previous: np.ndarray,
    cost: np.ndarray,
    *,
    risk_aversion: float,
    step: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int, bool]:
    """Solve one rebalance to tolerance: accelerated proximal gradient with restart.

    ``step`` holds one step per name, ``1 / (lambda * L_D * d_i)`` with ``d``
    the variance scaling and ``L_D`` the largest eigenvalue of the scaled
    covariance, so ``diag(1 / step)`` dominates the Hessian ``lambda * Sigma``
    and the quadratic model each step minimises lies above the objective,
    which is what the convergence argument needs. In that metric the cost's
    proximal step is still a soft threshold, at ``step_i * cost_i`` per name.

    The restart test is O'Donoghue and Candes' gradient scheme: reset the
    momentum when the generalized gradient ``(momentum - updated) / step``
    has a positive inner product with the step just taken. It is chosen over
    their function-value scheme because it needs no extra product with
    ``Sigma`` and because, near the optimum, it compares quantities of the size
    of the step rather than objective values that agree to machine precision.

    Returns the weights, the iterations run, and whether the step fell below
    ``tol`` within ``max_iter``.
    """
    weights = previous.copy()
    momentum = previous.copy()
    t = 1.0
    for iteration in range(1, max_iter + 1):
        gradient = risk_aversion * (sigma @ momentum) - mu
        proposal = momentum - step * gradient
        updated = previous + _soft_threshold(proposal - previous, step * cost)
        change = updated - weights
        if np.max(np.abs(change)) < tol:
            return updated, iteration, True
        if np.dot((momentum - updated) / step, change) > 0.0:
            # The momentum overshot: restart from the current iterate.
            t, momentum = 1.0, updated
        else:
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = updated + ((t - 1.0) / t_next) * change
            t = t_next
        weights = updated
    return weights, max_iter, False


def _check_solver(solver: str) -> None:
    if solver not in REBALANCE_SOLVERS:
        raise ValueError(f"solver={solver!r} is not one of {REBALANCE_SOLVERS}")


def _variance_scaled_curvature(sigma: np.ndarray, largest: float) -> tuple[np.ndarray, float]:
    """The restart solver's metric: each name's variance, and the scaled curvature.

    A name with no variance keeps the unscaled metric (``largest``); any
    positive scale is valid, since the step is derived from the scaled
    matrix's own largest eigenvalue.
    """
    variance = np.diag(sigma)
    scale = np.where(variance > 0, variance, largest)
    root = np.sqrt(scale)
    curvature = float(np.linalg.eigvalsh(sigma / np.outer(root, root)).max())
    return scale, curvature


def _broadcast_cost(
    cost: ArrayLike | None, shape: tuple[int, int]
) -> np.ndarray:
    if cost is None:
        return np.zeros(shape, dtype=np.float64)
    array = np.asarray(cost, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"cost must broadcast to {shape}, got shape {array.shape}"
        ) from error
    if np.any(np.isfinite(array) & (array < 0)):
        raise ValueError("cost cannot be negative")
    # A missing cost means the asset cannot be priced, so it cannot be traded.
    return np.where(np.isfinite(array), array, np.inf)


def _broadcast_initial(
    initial: ArrayLike | None, shape: tuple[int, int]
) -> np.ndarray:
    if initial is None:
        return np.zeros(shape, dtype=np.float64)
    array = np.asarray(initial, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"initial must broadcast to {shape}, got shape {array.shape}"
        ) from error
    if not np.isfinite(array).all():
        raise ValueError("initial weights must be finite")
    return array


def _track(signal: np.ndarray, covariance: ArrayLike, *, risk_aversion: float | ArrayLike,
           cost, initial, max_iter: int, tol: float, solver: str = DEFAULT_SOLVER
           ) -> tuple[np.ndarray, np.ndarray]:
    """Every decision's rebalance, in order, from ``signal = lambda_t * Sigma @ w*``.

    ``signal`` is ``(n_decisions, n_assets, n_horizons)``: the forecast itself
    for :func:`allocate`, and the target's implied forecast for
    :func:`rebalance`. Decisions are solved in order, each from the weights
    chosen at the previous one. ``risk_aversion`` is a scalar or one
    ``lambda_t`` per decision; the proximal step ``1 / (lambda_t * L)``, with
    ``L`` the covariance's largest eigenvalue, is that decision's own, since
    the gradient's Lipschitz constant scales with ``lambda_t``.

    The default solver scales that step per name by its variance (see
    :func:`_rebalance_restart`); ``"fista"`` takes the unscaled step within
    the ``max_iter`` budget. Returns the weights and each decision's
    iteration count, ``(n_decisions, n_horizons)``.
    """
    _check_solver(solver)
    n_decisions, n_assets, n_horizons = signal.shape
    aversion = _per_decision_risk_aversion(risk_aversion, n_decisions)
    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.shape != (n_horizons, n_assets, n_assets):
        raise ValueError(
            f"covariance must have shape {(n_horizons, n_assets, n_assets)}, "
            f"got {sigma.shape}"
        )
    if not np.isfinite(sigma).all():
        raise ValueError("covariance must be finite")
    costs = _broadcast_cost(cost, (n_decisions, n_assets))
    previous = _broadcast_initial(initial, (n_assets, n_horizons))
    weights = np.empty_like(signal)
    iterations = np.zeros((n_decisions, n_horizons), dtype=np.int64)
    cap = max(int(max_iter), RESTART_MAX_ITER)
    unconverged = 0
    for horizon in range(n_horizons):
        largest = float(np.linalg.eigvalsh(sigma[horizon]).max())
        if largest <= 0:
            raise ValueError("covariance must have positive variance")
        if solver == "restart":
            scale, curvature = _variance_scaled_curvature(sigma[horizon], largest)
        held = previous[:, horizon]
        for decision in range(n_decisions):
            lam = float(aversion[decision])
            if solver == "restart":
                held, used, converged = _rebalance_restart(
                    signal[decision, :, horizon], sigma[horizon], held, costs[decision],
                    risk_aversion=lam, step=1.0 / (lam * curvature * scale),
                    max_iter=cap, tol=tol,
                )
                unconverged += not converged
            else:
                held, used = _rebalance(
                    signal[decision, :, horizon], sigma[horizon], held, costs[decision],
                    risk_aversion=lam, step=1.0 / (lam * largest),
                    max_iter=max_iter, tol=tol,
                )
            weights[decision, :, horizon] = held
            iterations[decision, horizon] = used
    if unconverged:
        warnings.warn(
            f"the restart solver reached its cap of {cap} iterations before the step "
            f"fell below tol={tol} on {unconverged} of {n_decisions * n_horizons} "
            "decisions; their weights are approximate. Raise max_iter or tol.",
            ConvergenceWarning, stacklevel=3,
        )
    return weights, iterations


@requires("half_spread")
def rebalance(
    target: PortfolioWeights,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    solver: str = DEFAULT_SOLVER,
    return_n_iter: bool = False,
) -> PortfolioWeights | tuple[PortfolioWeights, np.ndarray]:
    """The weights actually held, moving toward ``target`` only as far as cost allows.

    ``target`` is the frictionless holding at each decision, such as
    :func:`~stable_finance.evaluation.portfolio.construction.mean_variance_target`.
    ``covariance`` (``(n_horizons, n_assets, n_assets)``) prices the tracking
    error and ``risk_aversion`` its weight against cost: one positive scalar,
    or a 1-D array with one ``lambda_t`` per decision, which should match the
    one the target was built with. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing target is a
    target of zero. ``cost=None`` moves straight to the target.

    ``solver="restart"`` (the default) runs accelerated proximal gradient with
    adaptive restart until the largest weight change in an iteration falls
    below ``tol``, under a cap of ``max(max_iter, RESTART_MAX_ITER)``;
    ``solver="fista"`` is FISTA stopped after ``max_iter`` iterations or once
    that change falls below ``tol``, whichever is first (see the module
    docstring). With ``return_n_iter=True`` the result is ``(weights,
    n_iter)``, ``n_iter`` holding the iterations each decision's solve ran,
    ``(n_decisions, n_horizons)``: under ``"fista"`` a count equal to
    ``max_iter`` means the budget, not the tolerance, ended the solve.
    """
    _check_solver(solver)
    cost = market_quantity(cost, "half_spread", stage=rebalance, like=target)
    sigma = np.asarray(covariance, dtype=np.float64)
    goal = np.nan_to_num(target.values, nan=0.0)
    if sigma.shape[0] != goal.shape[2]:
        raise ValueError("covariance must have one matrix per horizon")
    aversion = _per_decision_risk_aversion(risk_aversion, goal.shape[0])
    signal = aversion[:, None, None] * np.einsum("hij,djh->dih", sigma, goal)
    weights, n_iter = _track(signal, sigma, risk_aversion=aversion, cost=cost,
                             initial=initial, max_iter=max_iter, tol=tol, solver=solver)
    held = PortfolioWeights(weights, target.decisions, target.assets, target.horizons)
    return (held, n_iter) if return_n_iter else held


@requires("half_spread")
def allocate(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    solver: str = DEFAULT_SOLVER,
    return_n_iter: bool = False,
) -> PortfolioWeights | tuple[PortfolioWeights, np.ndarray]:
    """Forecasts to held weights in one solve: the mean-variance target, rebalanced toward.

    Equivalent to ``rebalance(mean_variance_target(forecast, ...), ...)``,
    solved from the forecast directly. Decisions are rebalanced in order, each
    starting from the weights chosen at the previous decision, so the whole
    path reflects the transaction-cost trade-off. ``covariance`` has shape
    ``(n_horizons, n_assets, n_assets)``. ``risk_aversion`` is one positive
    scalar or a 1-D array with one ``lambda_t`` per decision. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing forecast is
    treated as a zero expected return: the asset stays in the problem and is
    traded only as far as risk and cost justify.

    ``cost=None`` allocates frictionlessly. :class:`MarketData` in its place
    charges the quoted half-spread (EFQ 1) and must carry one; scale it
    yourself, or pass ``trading_cost(...)``, for any other execution model.

    ``solver`` and ``return_n_iter`` are as in :func:`rebalance`.
    """
    _check_solver(solver)
    cost = market_quantity(cost, "half_spread", stage=allocate, like=forecast)
    signal = np.nan_to_num(forecast.values, nan=0.0)
    weights, n_iter = _track(signal, covariance, risk_aversion=risk_aversion, cost=cost,
                             initial=initial, max_iter=max_iter, tol=tol, solver=solver)
    held = PortfolioWeights(weights, forecast.decisions, forecast.assets, forecast.horizons)
    return (held, n_iter) if return_n_iter else held


class MeanVarianceAllocator(BaseEstimator):
    """Forecast-to-weights adapter with a proportional transaction cost.

    ``fit`` estimates one asset covariance per horizon from a realized-return
    history; ``predict`` solves the cost-aware mean-variance problem at every
    decision of a forecast panel, starting from the current portfolio. Bring
    your own covariance through :func:`allocate` to skip fitting. ``solver``
    picks the numerical method, as in :func:`allocate`.
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        *,
        shrinkage: float = 0.1,
        min_samples: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-10,
        solver: str = DEFAULT_SOLVER,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.shrinkage = shrinkage
        self.min_samples = min_samples
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

    def fit(self, returns: ForwardReturns, y: None = None) -> "MeanVarianceAllocator":
        """Estimate the per-horizon covariance from realized returns."""
        self.covariance_ = estimate_covariance(
            returns, shrinkage=self.shrinkage, min_samples=self.min_samples
        )
        self.assets_ = returns.assets.copy()
        self.horizons_ = returns.horizons.copy()
        return self

    def predict(
        self,
        forecast: ForwardReturns,
        *,
        cost: ArrayLike | MarketData | None = None,
        initial: ArrayLike | None = None,
    ) -> PortfolioWeights:
        """Allocate every decision of ``forecast``; see :func:`allocate`."""
        if not hasattr(self, "covariance_"):
            raise NotFittedError("fit the MeanVarianceAllocator before calling predict")
        if not np.array_equal(forecast.assets, self.assets_):
            raise ValueError("assets are not aligned with the fitted covariance")
        if not np.array_equal(forecast.horizons, self.horizons_):
            raise ValueError("horizons are not aligned with the fitted covariance")
        return allocate(
            forecast,
            self.covariance_,
            risk_aversion=self.risk_aversion,
            cost=cost,
            initial=initial,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
