"""Every decision between a forward-return forecast and a Sharpe ratio.

A Sharpe ratio is not a property of a forecast. It is a property of a forecast
plus a trading harness, and the harness contributes decisions the model being
evaluated has no opinion about: which names are tradable, how much of the
cross-section to hold, how risk is forecast, how weights follow from it, how
often to rebalance, what a trade costs, and how a periodic return is
annualized. This module makes that harness an explicit, enumerable object.

There is a DEFAULT so that the common case is one call and the library has an
opinion. There is a config SPACE so that the sensitivity of a result to the
harness can be measured rather than assumed, which is the only way to tell a
finding about a model from a finding about a backtest.

    from stable_finance import DEFAULT_BACKTEST, enumerate_configs

    report = run_backtest(forecast, realized, half_spread=spread)   # default
    space = enumerate_configs()                                     # all of it
    space = enumerate_configs(efq=0.70)                             # a slice
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields, replace

import numpy as np

from stable_finance.dataset.schema import MissingRolesError
from stable_finance.evaluation.execution import (
    COST_MODEL_REQUIRES,
    COST_MODELS,
    FEE_REQUIRES,
    FEE_SETTINGS,
    _fee_setting,
)
from stable_finance.evaluation.portfolio.construction import (
    SELECTION_REQUIRES,
    SELECTION_RULES,
    WEIGHTING_REQUIRES,
    WEIGHTING_RULES,
)
from stable_finance.evaluation.portfolio.risk import (
    RISK_MODEL_REQUIRES,
    RISK_MODELS,
    UNIVERSE_REQUIRES,
    UNIVERSE_SCREENS,
)
from stable_finance.evaluation.portfolio.signals import (
    ALPHA_RULES,
    NEUTRALIZATIONS,
    SIGNAL_REQUIRES,
)

#: The solvers, by name. ``"restart"`` is accelerated proximal gradient with
#: adaptive restart in a variance-scaled metric, run until the step falls
#: below ``tol``; ``"fista"`` is FISTA stopped at the ``max_iter`` budget.
#: Here rather than beside the solvers so the configuration needs nothing from
#: the trades stage, whose environment reads the configuration.
REBALANCE_SOLVERS = ("restart", "fista")

__all__ = [
    "ANNUALIZATIONS",
    "OPTIONAL_SETTINGS",
    "EFQ_GRID",
    "BacktestConfig",
    "CONFIG_SPACE",
    "DEFAULT_BACKTEST",
    "HOLDING_RULES",
    "REBALANCE_RULES",
    "check_config",
    "enumerate_configs",
]

#: The execution qualities swept, as EFQ (effective / quoted half-spread).
#: 0 is a midpoint fill and 1 pays the whole quoted spread; 0.55 and 0.88 are
#: Levy (2022)'s measured direct-market-access and NYSE figures, and 0.70 sits
#: between them. EFQ REPLACES the old ``cost_model`` sweep as the cost axis,
#: because it is one continuous measured quantity rather than five named
#: models, and because a result reported against it can be read by anyone who
#: knows their own execution.
EFQ_GRID = (0.0, 0.25, 0.50, 0.70, 0.88, 1.00)

#: How a periodic return series is annualized. ``decisions`` treats every
#: decision as an independent period, which for an intraday anchor grid
#: annualizes a signal that is in the market for part of the session as though
#: it were in the market all year; ``daily`` first sums each day's decisions
#: into one daily return and annualizes at 252. Neither is wrong, they answer
#: different questions, and the gap between them is large enough that leaving
#: it implicit would be the mistake.
ANNUALIZATIONS = ("decisions", "daily")

#: When the book is allowed to change. ``every_decision`` re-solves at every
#: anchor; ``daily`` sets the book at each day's first decision and holds it,
#: which cuts turnover by roughly the number of anchors in a day and is the
#: difference between a signal that pays for its spread and one that does not.
#: ``horizon`` sets the book once per forecast horizon (restarting each day),
#: holds it for the horizon, and books its forward return ONCE: the way to
#: score an h-horizon forecast on a grid of decisions closer together than h,
#: where booking every decision's forward return counts each move about
#: h / spacing times. The harness reads the horizon from the forecast and the
#: spacing from the decisions.
REBALANCE_RULES = ("every_decision", "daily", "horizon")

#: How a position is carried from one decision to the next. ``rebalanced``
#: holds each name at its weight: every decision's book is a fresh bet that
#: earns weight x return, as if trimmed back to weight for free -- right for a
#: cross-sectional book retraded every decision, and how every earlier result
#: was scored. ``drift`` holds what was bought: a position moves with its own
#: return between decisions (h -> h (1 + r)), so a trade held across several
#: decisions earns exactly its price change, and an order goes out only when
#: the target CHANGES (or an earlier order is still unfilled) -- an unchanged
#: target is held, not rebalanced back. ``drift`` needs the realized returns
#: to run from each decision to the NEXT one; horizons that overlap the next
#: decision would count the same move twice.
HOLDING_RULES = ("rebalanced", "drift")

#: ``mean_variance`` is the cost-aware, path-dependent allocator; the rest are
#: the path-independent rules that size a selected book directly.
ALLOCATORS = ("mean_variance", *WEIGHTING_RULES)

#: What each axis option needs of the market, collected from the modules that
#: define the options. ``mean_variance`` is charged the configured cost, so its
#: own requirement is empty and the cost model's carries the spread.
AXIS_REQUIRES: dict[str, dict[str, tuple[str, ...]]] = {
    "universe": UNIVERSE_REQUIRES,
    "selection": SELECTION_REQUIRES,
    "weighting": {"mean_variance": (), **WEIGHTING_REQUIRES},
    "risk_model": RISK_MODEL_REQUIRES,
    "cost_model": COST_MODEL_REQUIRES,
    "neutralize": SIGNAL_REQUIRES,
    "factor_exposure": SIGNAL_REQUIRES,
}


#: Settings that are ``None`` unless a model needs them, and that labels and
#: result rows leave out while unset.
OPTIONAL_SETTINGS = ("equity", "execution_days", "impact", "alpha", "alpha_ic",
                     "neutralize", "mv_solver", "factor_exposure", "factor_limit", *FEE_SETTINGS)


@dataclass(frozen=True)
class BacktestConfig:
    """One point in the harness configuration space.

    Defaults reproduce the library's standard pipeline: hold the whole
    cross-section long-short, forecast risk with a shrunk sample covariance,
    size with the cost-aware mean-variance allocator, rebalance every
    decision, and charge the observable quoted half-spread. The frictionless
    ``mid`` result is always reported alongside, so the default never hides
    what the spread cost.
    """

    universe: str = "all"
    selection: str = "all"
    weighting: str = "mean_variance"
    risk_model: str = "shrunk_sample"
    cost_model: str = "cross"
    #: Execution quality: the multiple of the quoted half-spread actually
    #: paid. Charged BOTH to the cost-aware allocator and to the scoring, so
    #: a book is always optimised against the cost it is then billed for.
    efq: float = 1.0
    rebalance: str = "every_decision"
    annualization: str = "daily"
    gross_exposure: float = 2.0
    shrinkage: float = 0.3
    #: The portfolio's value in the book's currency. Only ``cost_model='book'``
    #: reads it, to turn weight into shares, and it is required there: depth
    #: is in shares, so the same weights walk a book further at larger size.
    equity: float | None = None
    #: How long ``cost_model='impact'`` works each order, in trading days,
    #: measured in decisions from the ``days`` labels. ``None`` is two days,
    #: about the paper's trades (a median horizon of 1.7 days, a mean of 2.7);
    #: ``0`` fills at the decision, the paper's own accounting.
    execution_days: float | None = None
    #: Coefficients for ``cost_model='impact'`` (three, ``(a, b, c)``) or
    #: ``'impact_stock'`` (eleven, an
    #: :class:`~stable_finance.evaluation.execution.ImpactStockCoefficients`);
    #: ``None`` is the paper's U.S. calibration.
    impact: tuple | None = None
    #: Iteration budget for the cost-aware allocator's proximal solver. A
    #: NUMERICAL setting, not a modelling one: it is not swept, and a value too
    #: low changes the answer rather than approximating it. 400 reproduces the
    #: 1000-iteration solution to floating point on panels of a few hundred
    #: names while costing a fifth of the time.
    max_iter: int = 400
    #: How positions are carried between decisions (:data:`HOLDING_RULES`).
    #: Accounting rather than a modelling axis, so it is not swept.
    holding: str = "rebalanced"
    #: Convert the forecast to return units before construction:
    #: ``"grinold"`` is ``alpha = alpha_ic * sigma * z`` (Grinold, 1994), with
    #: ``sigma`` from the risk model and ``z`` the cross-sectional z-score;
    #: ``"grinold_rank"`` takes ``z`` as the Gaussian rank score, which an
    #: outlying forecast cannot dominate. ``None`` uses the forecast as given.
    alpha: str | None = None
    #: The information coefficient the ``alpha`` rule scales by.
    alpha_ic: float | None = None
    #: Residualise the forecast on exposures before construction, so the book
    #: expresses no market or industry bet: ``"beta"`` (beta estimated from
    #: the risk history), ``"industry"`` (needs an ``industry`` market
    #: quantity) or ``"beta_industry"``. ``None`` leaves it as forecast.
    neutralize: str | None = None
    #: How the cost-aware allocator's rebalance is solved, a NUMERICAL setting
    #: like ``max_iter`` and read only by ``weighting='mean_variance'``.
    #: ``None`` is the library's default,
    #: :data:`~stable_finance.evaluation.trades.rebalance.DEFAULT_SOLVER`:
    #: accelerated proximal gradient with adaptive restart, run until the step
    #: falls below the solver's tolerance under a cap of at least
    #: :data:`~stable_finance.evaluation.trades.rebalance.RESTART_MAX_ITER`
    #: iterations -- the converged book, so that a result about the allocator
    #: is not a result about where its solver was stopped. ``"fista"`` stops
    #: at the ``max_iter`` budget, for measuring what that budget costs.
    mv_solver: str | None = None
    #: Bound the frictionless mean-variance target's exposure to the library's
    #: beta, industry or PCA estimates: the same choices as neutralize, from
    #: the same estimators, independently of risk_model (the loadings fitted
    #: inside the factor risk model are not among them). A name whose loading
    #: is unknown -- a beta with fewer than two finite history decisions, an
    #: unknown industry code -- is left out of the target, where neutralization
    #: keeps it with its raw forecast. Set together with factor_limit. The held
    #: book may drift outside the bound; run_backtest(with_exposure=...)
    #: reports where it ended up.
    factor_exposure: str | None = None
    #: Symmetric bound on each exposure of factor_exposure, in signed weight
    #: units; zero imposes neutrality.
    factor_limit: float | None = None
    #: FEES, charged per unit of weight traded ON TOP of the cost model, and,
    #: like ``efq``, to the cost-aware allocator and in the scoring alike, so
    #: a book is planned against the fees it then pays. All three are refused
    #: with ``cost_model='mid'``, the frictionless benchmark (``cross`` at
    #: ``efq=0`` is a midpoint fill that still pays them). ``fee_bps`` is a
    #: commission in basis points of the notional traded, on either side.
    fee_bps: float | None = None
    #: Dollars per share traded, on either side, such as an exchange's take
    #: fee. It needs each name's share price (a ``mid``, or the bid and ask),
    #: and :func:`check_config` refuses it on a market without one.
    fee_per_share: float | None = None
    #: A fraction of the notional SOLD, the SEC's Section 31 fee. The
    #: allocator's cost is one rate for either side of a trade, so it is
    #: charged half of this on every unit, which is exact per round trip (see
    #: :func:`~stable_finance.evaluation.execution.fee_per_unit`); the scoring
    #: bills each sale in full. For a sample from 2020::
    #:
    #:     BacktestConfig(fee_per_share=0.003,    # Reg NMS Rule 610 access-fee cap
    #:                    sec_fee_rate=22.10e-6)  # SEC Section 31, fiscal 2020
    #:
    #: which is about 1.5 bp per side on a $20 stock, plus 0.22 bp on each sale.
    sec_fee_rate: float | None = None

    def __post_init__(self) -> None:
        for field, allowed in (
            ("universe", UNIVERSE_SCREENS), ("selection", SELECTION_RULES),
            ("weighting", ALLOCATORS), ("risk_model", RISK_MODELS),
            ("cost_model", COST_MODELS), ("rebalance", REBALANCE_RULES),
            ("annualization", ANNUALIZATIONS), ("holding", HOLDING_RULES),
        ):
            value = getattr(self, field)
            if value not in allowed:
                raise ValueError(
                    f"{field}={value!r} is not one of {allowed}"
                )
        # Type-checked before the numeric test: EFQ is the one axis whose
        # options are numbers, so a caller passing a name here (as they would
        # for every other axis) must get the same ValueError and not a
        # TypeError out of numpy.
        if (isinstance(self.efq, bool)
                or not isinstance(self.efq, (int, float))
                or not np.isfinite(self.efq) or self.efq < 0):
            raise ValueError("efq must be a finite, non-negative number")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.equity is not None and (isinstance(self.equity, bool)
                                        or not np.isfinite(self.equity) or self.equity <= 0):
            raise ValueError("equity must be a positive number")
        if self.cost_model in ("book", "impact", "impact_stock") and self.equity is None:
            raise ValueError(f"cost_model={self.cost_model!r} needs equity, the portfolio's "
                             "value in the market's currency, to size its orders")
        if self.execution_days is not None and (
                isinstance(self.execution_days, bool)
                or not np.isfinite(self.execution_days) or self.execution_days < 0):
            raise ValueError("execution_days must be a non-negative number")
        if self.impact is not None:
            values = tuple(self.impact)
            wanted = 11 if self.cost_model == "impact_stock" else 3
            if len(values) != wanted or not all(isinstance(v, (int, float)) and np.isfinite(v)
                                                for v in values):
                raise ValueError(f"impact must be {wanted} finite coefficients for "
                                 f"cost_model={self.cost_model!r}")
            object.__setattr__(self, "impact", tuple(float(v) for v in values))
        if self.alpha is not None and self.alpha not in ALPHA_RULES:
            raise ValueError(f"alpha={self.alpha!r} is not one of {ALPHA_RULES}")
        if self.alpha is not None and (
                self.alpha_ic is None or isinstance(self.alpha_ic, bool)
                or not np.isfinite(self.alpha_ic)):
            raise ValueError(f"alpha={self.alpha!r} needs alpha_ic, a finite information "
                             "coefficient to scale the forecast by")
        if self.neutralize is not None and self.neutralize not in NEUTRALIZATIONS:
            raise ValueError(f"neutralize={self.neutralize!r} is not one of {NEUTRALIZATIONS}")
        if (self.factor_exposure is None) != (self.factor_limit is None):
            raise ValueError("factor_exposure and factor_limit are set together; to report "
                             "held exposures without a bound, pass with_exposure to "
                             "run_backtest")
        if self.factor_exposure is not None:
            if self.factor_exposure not in NEUTRALIZATIONS:
                raise ValueError(f"factor_exposure={self.factor_exposure!r} is not one of "
                                 f"{NEUTRALIZATIONS}")
            if (isinstance(self.factor_limit, bool) or not np.isfinite(self.factor_limit)
                    or self.factor_limit < 0):
                raise ValueError("factor_limit must be finite and non-negative")
            if self.weighting != "mean_variance":
                raise ValueError("factor_limit constrains the mean_variance target")
        if self.mv_solver is not None and self.mv_solver not in REBALANCE_SOLVERS:
            raise ValueError(f"mv_solver={self.mv_solver!r} is not None or one of "
                             f"{REBALANCE_SOLVERS}")
        for name in FEE_SETTINGS:
            object.__setattr__(self, name, _fee_setting(name, getattr(self, name)))
        if self.cost_model == "mid" and self.fees:
            raise ValueError("cost_model='mid' is the frictionless benchmark and charges no fees; "
                             "charge them on top of 'cross' (efq=0 for midpoint fills that still "
                             "pay them), 'book' or an impact model")
        if self.rebalance == "horizon" and self.holding == "drift":
            raise ValueError("rebalance='horizon' books each held book's forward return once; "
                             "holding='drift' already carries positions step by step on "
                             "one-step returns, so use one or the other")

    @property
    def label(self) -> str:
        """A stable, filename-safe identifier for this configuration.

        The optional settings appear only when set, so every label written
        before they existed still names the same configuration.
        """
        return "|".join(
            f"{field.name}={getattr(self, field.name)}" for field in fields(self)
            if not (field.name in OPTIONAL_SETTINGS and getattr(self, field.name) is None)
            and not (field.name == "holding" and self.holding == "rebalanced")
        )

    @property
    def fees(self) -> dict[str, float]:
        """The fee settings this configuration charges, by name; empty when it charges none."""
        return {name: getattr(self, name) for name in FEE_SETTINGS
                if getattr(self, name) is not None}

    @property
    def requires(self) -> dict[str, tuple[str, ...]]:
        """Market quantities this configuration needs, by the axis needing them.

        A fee that needs the market is listed under its own name, as an axis
        option is: ``fee_per_share`` needs the share price.
        """
        out = {}
        for axis, table in AXIS_REQUIRES.items():
            needed = table.get(getattr(self, axis), ())
            if needed:
                out[axis] = tuple(needed)
        for name in self.fees:
            if FEE_REQUIRES[name]:
                out[name] = tuple(FEE_REQUIRES[name])
        return out

    def supported_by(self, market) -> bool:
        return all(market.has(*needed) for needed in self.requires.values())

    def replace(self, **changes) -> "BacktestConfig":
        """A copy with some axes changed, validated on construction."""
        return replace(self, **changes)


#: The library's opinion when the caller does not have one.
DEFAULT_BACKTEST = BacktestConfig()

#: The axes swept by :func:`enumerate_configs`, and their option lists. Scalar
#: settings (``gross_exposure``, ``shrinkage``, ``max_iter``) are deliberately
#: absent: the first two are continuous, so a study should choose its own grid
#: rather than inherit one, and the third is numerical, not a modelling choice.
#: ``annualization`` is deliberately ABSENT. It is pure accounting -- the same
#: trades folded into different periods -- and holding it at ``daily`` makes
#: every Sharpe here one non-overlapping observation per day annualized at
#: 252, which is what the execution-quality figure reports. ``cost_model`` is
#: absent because ``efq`` is now the cost axis.
#: ``rebalance`` sweeps the two schedules that are modelling choices;
#: ``horizon`` is the accounting for a horizon longer than the decision
#: spacing and is asked for by name. ``risk_model`` is every model in
#: :data:`RISK_MODELS`, the one list of them; :func:`enumerate_configs` keeps
#: the ones a given market supports.
CONFIG_SPACE = {
    "universe": UNIVERSE_SCREENS,
    "selection": SELECTION_RULES,
    "weighting": ALLOCATORS,
    "risk_model": RISK_MODELS,
    "efq": EFQ_GRID,
    "rebalance": ("every_decision", "daily"),
}


def _is_degenerate(config: BacktestConfig) -> bool:
    """Drop configurations that are duplicates of another point in the space.

    Two collapses are real rather than cosmetic. A book whose weights ignore
    the covariance entirely is the same book under every risk model, so only
    one risk model is kept for it. And under a diagonal risk model,
    inverse-volatility and equal-risk-contribution both reduce to ``w ~
    1/sigma``, so keeping all three would triple-count one portfolio and
    silently weight the summary toward it. Hierarchical risk parity and the
    long-only minimum-variance solution land on ``w ~ 1/sigma^2`` under the
    same model, which is ``min_var``'s book.

    A Grinold alpha reads each name's volatility from the risk model, so with
    it set even the covariance-free rules change with the risk model and none
    of the first collapse applies.
    """
    if (config.weighting in ("equal", "rank", "signal") and config.alpha is None
            and config.risk_model != "diagonal"):
        return True
    if config.risk_model == "diagonal" and config.weighting in ("erc", "hrp", "min_var_qp"):
        return True
    return False


def check_config(config: BacktestConfig, market) -> None:
    """Raise :class:`MissingRolesError` if ``market`` cannot support ``config``.

    The message names the axis and option, so a refused run says which
    choice to change: ``cost_model='cross'`` on a market without quotes, for
    instance, rather than a NaN Sharpe or a free trade.
    """
    hints = {"cost_model": "cost_model='mid' runs on any market data, frictionless at "
                           "the price the returns are marked at; 'cross' needs a "
                           "spread, quoted or estimated from bars; 'book' needs depth; "
                           "'impact' needs adv, the average daily dollar volume; "
                           "'impact_stock' also needs market_cap, idio_vol, market_vol "
                           "and time_trend",
             "universe": "universe='all' runs without quotes",
             "neutralize": "neutralize='beta' runs on any market data, estimating beta "
                           "from the risk history; 'industry' needs an industry code "
                           "for each name",
             "factor_exposure": "factor_exposure='beta' or 'pca3' estimates loadings "
                                "from the risk history; 'industry' needs industry codes",
             "fee_per_share": "fee_per_share is dollars per share, so it needs each name's "
                              "share price (mid, or bid_price and ask_price); fee_bps and "
                              "sec_fee_rate are charged on notional and need none"}
    for axis, needed in config.requires.items():
        try:
            market.require(*needed, purpose=f"{axis}={getattr(config, axis)!r}")
        except MissingRolesError as error:
            hint = hints.get(axis)
            raise MissingRolesError(f"{error}; {hint}" if hint else str(error)) from None


def enumerate_configs(*, market=None, **overrides) -> tuple[BacktestConfig, ...]:
    """The configuration space, or a named slice of it.

    Each keyword names a :class:`BacktestConfig` field. A list or tuple of
    options sweeps it -- any field, so ``enumerate_configs(neutralize=(None,
    "beta"))`` adds an axis the default space does not have -- and a single
    value pins it, so ``enumerate_configs(cost_model="mid")`` sweeps
    everything else at zero friction. (``impact`` is itself a tuple of
    coefficients: one tuple pins it, a list of tuples sweeps it.)
    Configurations that duplicate another point are removed, so every element
    is a distinct portfolio.

    ``market`` restricts the space to what that market data can support.
    Without a half-spread there is nothing to charge, so unless the caller
    pinned them the cost model becomes ``mid`` and the EFQ axis collapses to
    zero, and the liquidity screens drop out: what remains is the space of
    frictionless portfolios, each once.
    """
    axes = dict(CONFIG_SPACE)
    scalars = {}
    if market is not None and not market.has("half_spread"):
        if "cost_model" not in overrides:
            scalars["cost_model"] = "mid"
        if "efq" not in overrides:
            axes["efq"] = (0.0,)
    known = {f.name for f in fields(BacktestConfig)}
    for name, value in overrides.items():
        if name not in known:
            raise ValueError(f"{name!r} is not a configuration axis")
        sweep = isinstance(value, (list, tuple, np.ndarray)) and not (
            name == "impact" and all(isinstance(v, (int, float)) for v in value))
        if sweep:
            axes[name] = tuple(value)
        else:
            axes.pop(name, None)
            scalars[name] = value

    names = list(axes)
    out, seen = [], set()
    for combination in itertools.product(*(axes[name] for name in names)):
        options = {**dict(zip(names, combination)), **scalars}
        if options.get("factor_limit") is None:
            # No bound is the plain target whichever estimator is named.
            options.pop("factor_exposure", None)
        elif options.get("factor_exposure") is None:
            raise ValueError("factor_limit needs factor_exposure")
        elif options.get("weighting", DEFAULT_BACKTEST.weighting) != "mean_variance":
            continue
        config = BacktestConfig(**options)
        if _is_degenerate(config) or config in seen:
            continue
        seen.add(config)
        if market is not None and not config.supported_by(market):
            continue
        out.append(config)
    return tuple(out)
