"""Run configurations end to end, from whichever level you start at.

You may bring the pipeline's input at ANY level, and the harness runs only
the stages below it:

=====================  =====================================================
you bring              the harness runs
=====================  =====================================================
``Embeddings``         a ridge probe fit on ``train``, then everything below
``ForwardReturns``     universe, selection, risk and sizing, then trading
``PortfolioWeights``   the optimal portfolio: trading toward it under the
                       execution model, which decides the deltas under cost
``Orders``             your deltas, filled under the execution model
=====================  =====================================================

Every level is scored the same way, so a Sharpe from embeddings and one from
orders are the same statistic. The IC exists only where there is a forecast.
Every level runs against every level of market data that its configuration
allows: ``cost_model='mid'`` needs none, ``'cross'`` a spread (quoted, or
estimated from bars), and ``'book'`` depth.

This is the wiring, and it is deliberately thin. Every decision it makes is
named in the :class:`~stable_finance.evaluation.config.BacktestConfig` it was handed, and
every stage it calls is a public function that can be used on its own. The
point is that two runs differing only in their config differ only in the
harness -- so a change in Sharpe between them cannot be a change in what was
forecast, in which names were held, or in how the metric was computed.

THE IC AND THE SHARPE COME FROM THE SAME FORECAST ARRAY. That is the whole
reason this is one function rather than two: a study of how tightly IC and
Sharpe move together is worthless if the two numbers were produced by
separately fitted predictors.

Use :func:`run_backtest` for one configuration and :func:`sweep_configs` for
many. They agree exactly; the sweep is only faster, because it notices that
most of the space shares work.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.config import (
    DEFAULT_BACKTEST,
    OPTIONAL_SETTINGS,
    BacktestConfig,
)
from stable_finance.evaluation.embeddings.contract import Embeddings
from stable_finance.evaluation.execution import (
    FEE_SETTINGS,
    Orders,
    _efq_array,
    execute,
    fee_cost,
    fee_per_unit,
    fill_path,
    impact_cost,
    impact_stock_cost,
    orders_from_weights,
    trading_cost,
)
from stable_finance.dataset.market import MarketData, require_market_aligned
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.construction import (
    mean_variance_target, select_book, weight_book,
)
from stable_finance.evaluation.portfolio.risk import apply_screen, estimate_risk
from stable_finance.evaluation.portfolio.signals import (
    NEUTRALIZATIONS, SIGNAL_REQUIRES, alpha_from_rule, apply_signal, estimate_beta,
    estimate_loadings, industry_exposures,
)
from stable_finance.evaluation.preparation import (
    check_axes,
    decisions_spanning,
    impact_schedule,
    prepare,
    share_price,
)
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.returns.probes import RidgeProbe
from stable_finance.evaluation.scoring.ic import grouped_rank_ic
from stable_finance.evaluation.scoring.exposure import PortfolioExposure, portfolio_exposure
from stable_finance.evaluation.scoring.pnl import positions_from_fills
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.scoring.tc import transfer_coefficient
from stable_finance.evaluation.trades.rebalance import DEFAULT_SOLVER, rebalance

__all__ = ["LEVELS", "BacktestReport", "run_backtest", "sweep_configs"]

#: Axes that determine the weights. Two configurations agreeing on all of them
#: hold the identical book, so the expensive part is computed once and the
#: remaining axes -- which act only on an existing book -- are evaluated on top.
_BOOK_AXES = ("universe", "selection", "weighting", "risk_model", "cost_model",
              "efq", "gross_exposure", "shrinkage", "max_iter", "equity", "impact",
              "alpha", "alpha_ic", "neutralize", "mv_solver",
              "factor_exposure", "factor_limit", *FEE_SETTINGS)


#: The book axes only the cost-aware allocator reads. A path-independent rule
#: sizes a book from the forecast and the risk model alone, so configurations
#: differing only in what trading costs hold the same book and share it.
_COST_AXES = ("cost_model", "efq", "equity", "impact", "max_iter", "mv_solver",
              *FEE_SETTINGS)


def _book_key(config: BacktestConfig) -> tuple:
    """The settings that determine a configuration's book, and nothing else."""
    axes = (_BOOK_AXES if config.weighting == "mean_variance"
            else tuple(a for a in _BOOK_AXES if a not in _COST_AXES))
    return tuple((axis, getattr(config, axis)) for axis in axes)


@dataclass(frozen=True)
class PnLSeries:
    """The per-period profit path behind a report's scalar scores.

    A Sharpe ratio is a mean over a standard deviation, and two very different
    profit paths can share one. Keeping the path lets a caller plot what the
    configuration actually did -- and lets a test assert that the scalar and
    the series are the same number, which is the only way to know a chart and
    a table were not computed from different things.

    ``periods`` are the labels the series is indexed by: one per decision, or
    one per day when the configuration annualizes daily.

    ``fee`` is the part of ``cost`` that is fees, when the configuration
    charges any (else None), so ``cost - fee`` is what the cost model itself
    charged -- the part that scales with execution quality, which a
    break-even EFQ is read from.
    """

    periods: NDArray[np.str_]
    gross: NDArray[np.float64]
    net: NDArray[np.float64]
    cost: NDArray[np.float64]
    fee: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class BacktestReport:
    """One configuration's scores on one horizon.

    ``sharpe_mid`` marks every trade at the midpoint and ``sharpe_net``
    charges the configured cost model; reporting both is what keeps a
    frictionless number from being read as an executable one.
    ``information_coefficient`` is measured on the same forecast over the same
    screened universe, so the pair is comparable by construction.

    ``sharpe_mid`` is the gross of the book this configuration HOLDS, which is
    not always the book a cost-free configuration would hold. The
    mean-variance allocator plans against the cost it is billed (the cost
    model with its ``efq``, ``equity`` and ``impact``, and the fees), so a
    dearer configuration holds a book that trades less, and its gross Sharpe
    moves with EFQ BY DESIGN: that is the allocator doing its job, not cost
    leaking into the gross. Compare a configuration's gross with its own net;
    a gross at one EFQ against a net at another mixes two books. Every other
    weighting holds the same book at any cost, and its gross differs across
    EFQ only where a name without a usable quote fills at the mid (EFQ 0) but
    not across the spread.

    ``transfer_coefficient`` is how much of the forecast the book expresses
    (:func:`~stable_finance.evaluation.scoring.tc.transfer_coefficient`): the
    correlation, over the screened names, of the positions actually HELD --
    after the solver, the fills and ``rebalance`` -- with the forecast
    the book was built from, AFTER the signal transforms, both in the risk
    model's units. It is measured against the transformed forecast rather than
    the raw one because the transforms are what the book was asked to express;
    the IC is what scores the raw forecast. Below the forecast level there is
    no forecast to transfer, and it is NaN. It is averaged over the decisions
    at which the book earns -- the same time base as the profit path: every
    decision, except under ``rebalance="horizon"``, where a held book earns
    its return once, at the formation that set it, and so is scored there,
    against the forecast it was formed on. A carried decision holds a book
    formed on an earlier forecast and is not scored against a fresh one, nor
    is a formation whose forward return is unknown, which sets no book.

    UNITS. ``mean_gross_return_bps`` and ``mean_cost_bps`` are means over the
    periods of the profit path, in basis points of equity: per decision, per
    held book under ``rebalance="horizon"``, or per day under
    ``annualization="daily"``. ``mean_turnover`` is the two-way traded weight
    -- the sum over names of the absolute change in weight, in units of
    equity -- per decision, or per held book under ``rebalance="horizon"``,
    whatever the annualization. Under ``annualization="decisions"`` (which
    every experiment spec sets; the configuration's default is ``"daily"``)
    the two share a basis, so ``mean_cost_bps / mean_turnover`` is the cost
    per unit traded; under ``"daily"`` the cost is per day and the turnover
    is not. ``mean_gross_exposure`` and ``mean_names_held`` are means over
    decisions.
    """

    config: BacktestConfig
    horizon: int
    information_coefficient: float
    information_coefficient_se: float
    #: NaN where there is no forecast.
    transfer_coefficient: float
    sharpe_mid: float
    sharpe_net: float
    mean_gross_return_bps: float
    mean_cost_bps: float
    mean_turnover: float
    mean_gross_exposure: float
    periods_per_year: float
    n_periods: int
    mean_names_held: float
    #: The profit path, only when the caller asked for it. It is deliberately
    #: absent from :meth:`as_dict`: a sweep writes one row per configuration,
    #: and four thousand embedded arrays would turn a 3 MB result file into a
    #: gigabyte one.
    series: PnLSeries | None = None
    #: The mean fee per period, in basis points of equity, when the
    #: configuration charges fees; it is part of ``mean_cost_bps``, not on top
    #: of it. None otherwise, and then absent from :meth:`as_dict`.
    mean_fee_bps: float | None = None
    #: Factor exposure and concentration of the positions actually held,
    #: reconstructed from fills, on the estimator ``with_exposure`` named to
    #: :func:`run_backtest`; None otherwise, and then absent from
    #: :meth:`as_dict`. A configured bound applies to targets, not fills.
    exposure: PortfolioExposure | None = None

    def as_dict(self) -> dict:
        """A flat, JSON-serializable row: config axes beside the scores.

        ``equity`` is left out when unset, as in the config's label, so rows
        written before it existed keep the same columns; so is
        ``mean_fee_bps`` for a configuration that charges no fees. With
        ``exposure``, ``mean_concentration`` and ``max_abs_factor_exposure``
        summarize it over the decisions (and factors) where it is known.
        """
        row = {k: v for k, v in asdict(self).items()
               if k not in ("config", "series", "exposure")
               and not (k == "mean_fee_bps" and v is None)}
        if self.exposure is not None:
            concentration = self.exposure.concentration
            exposure = np.abs(self.exposure.factor_exposure)
            row["mean_concentration"] = float(
                concentration[np.isfinite(concentration)].mean()
                if np.isfinite(concentration).any() else np.nan)
            row["max_abs_factor_exposure"] = float(
                exposure[np.isfinite(exposure)].max() if np.isfinite(exposure).any() else np.nan)
        config = {k: v for k, v in asdict(self.config).items()
                  if not (k in OPTIONAL_SETTINGS and v is None)}
        return {**config, **row}


# ── shared setup ────────────────────────────────────────────────────────────


def _market(like, market, half_spread) -> MarketData:
    """The market data a run is checked against.

    ``half_spread`` is the historical argument and is wrapped as market data
    holding only that quantity. Neither given is a market with no quotes,
    which supports exactly the frictionless configurations.
    """
    if market is not None and half_spread is not None:
        raise ValueError("pass market or half_spread, not both")
    if market is None:
        if half_spread is None:
            return MarketData.empty(like.decisions, like.assets)
        spread = np.asarray(half_spread, dtype=np.float64)
        expected = like.values.shape[:2]
        if spread.shape != expected:
            raise ValueError(
                f"half_spread must have shape {expected}, got {spread.shape}")
        return MarketData({"half_spread": spread}, like.decisions, like.assets)
    require_market_aligned(market, like)
    return market


def _history_panel(context, history, realized) -> ForwardReturns:
    """The risk history at the horizon being scored, checked against the input.

    The history is matched to the input BY LABEL, not by position: its assets
    must be the input's assets in the input's order, and its horizon is found
    by value. A history whose columns were merely the same length used to be
    read as though they lined up, which silently paired one name's risk with
    another's forecast.
    """
    source = history if history is not None else realized
    if not np.array_equal(source.assets, context["assets"]):
        raise ValueError("assets of the history and the input are not aligned")
    matches = np.flatnonzero(np.asarray(source.horizons) == context["horizon"])
    if len(matches) == 0:
        raise ValueError(f"the history has no horizon {context['horizon']}; "
                         f"it has {list(source.horizons)}")
    return ForwardReturns(source.values[:, :, matches[0]][:, :, None],
                          source.decisions, source.assets, context["axis"])


def _risk(context, config, history, realized, embeddings, horizon_index):
    """The covariance for one risk model, on the horizon being scored.

    The ``factor`` model adds industry factors when the market carries an
    ``industry`` code, taking each name's FIRST known code: classifications
    are static data, and the earliest code in the period is the one in force
    when the model is fitted, so a reclassification later in the period
    cannot leak into it.
    """
    panel = _history_panel(context, history, realized)
    if embeddings is not None and not np.array_equal(embeddings.assets, context["assets"]):
        raise ValueError("assets of the embeddings and the input are not aligned")
    industry = None
    if config.risk_model == "factor" and context["market"].has("industry"):
        codes = context["market"].get("industry")
        known = np.isfinite(codes)
        first = np.argmax(known, axis=0)
        industry = np.where(known.any(axis=0),
                            codes[first, np.arange(codes.shape[1])], np.nan)
    return estimate_risk(config.risk_model, panel, embeddings=embeddings,
                         shrinkage=config.shrinkage, min_samples=2,
                         industry=industry)[0]


def _beta(context, history, realized) -> NDArray[np.float64]:
    """Each name's beta at the scored horizon, estimated once per run."""
    if "beta" not in context:
        context["beta"] = estimate_beta(_history_panel(context, history, realized))[0]
    return context["beta"]


def _loadings(context, history, realized) -> NDArray[np.float64]:
    """Each name's principal-component loadings at the scored horizon, estimated once per run."""
    if "loadings" not in context:
        context["loadings"] = estimate_loadings(_history_panel(context, history, realized))[0]
    return context["loadings"]


def _exposures(context, kind, history, realized) -> None:
    """Cache the library's factor estimates for bounds and held-book scoring.

    ``kind`` is one of :data:`NEUTRALIZATIONS`, or None for nothing. Like the
    risk models, estimates use history when supplied. Pass a prior period to
    keep them out of sample. Industry membership is read at each decision, so
    a reclassification is not carried back to earlier targets.
    """
    if kind is None:
        return
    cache = context.setdefault("factors", {})
    if kind in cache:
        return
    parts, names = [], []
    shape = (context["n_decisions"], context["n_assets"])
    if kind in ("beta", "beta_industry"):
        parts.append(np.broadcast_to(_beta(context, history, realized), shape)[:, :, None])
        names.append("beta")
    if kind in ("industry", "beta_industry"):
        codes = context["market"].get("industry")
        industries = np.unique(codes[np.isfinite(codes)])
        if len(industries):
            parts.append(industry_exposures(codes))
            names.extend(f"industry_{code:g}" for code in industries)
        else:
            parts.append(np.full((*shape, 1), np.nan))
            names.append("industry_unknown")
    if kind == "pca3":
        loadings = _loadings(context, history, realized).copy()
        panel = _history_panel(context, history, realized)
        loadings[np.isfinite(panel.values[:, :, 0]).sum(axis=0) < 2] = np.nan
        parts.append(np.broadcast_to(loadings, (*shape, loadings.shape[1])))
        names.extend(f"pc_{i + 1}" for i in range(loadings.shape[1]))
    cache[kind] = (np.concatenate(parts, axis=2), tuple(names))


def _signal(context: dict, config: BacktestConfig, sigma, screened) -> NDArray[np.float64]:
    """The forecast the book is built from, after the configured transforms.

    Neutralization runs first, so the alpha is scaled from the part of the
    forecast the book is meant to express; both read only the screened names.
    The IC is still measured on the raw forecast: these transforms change what
    the portfolio expresses, not what the model predicted.
    """
    mu = context["mu"]
    if config.neutralize is not None:
        market = context["market"]
        mu = apply_signal(
            config.neutralize, mu, eligible=screened,
            beta=context.get("beta"),
            industry=market.get("industry") if market.has("industry") else None,
            loadings=context.get("loadings"),
        )
    if config.alpha is not None:
        mu = alpha_from_rule(config.alpha, mu, np.sqrt(np.maximum(np.diag(sigma), 0.0)),
                             ic=config.alpha_ic, eligible=screened)
    return mu


# ── the book ────────────────────────────────────────────────────────────────

def _weights(context: dict, config: BacktestConfig,
             sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Target weights under one configuration's book-determining axes."""
    return _book(context, config, sigma)[0]


def _book(context: dict, config: BacktestConfig, sigma: NDArray[np.float64]
          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Target weights, and the transformed forecast they were built from.

    The forecast is returned beside the weights, rather than stashed in
    ``context``, because the context is shared by every configuration of a
    sweep and the transformed forecast is not: two configurations differing
    only in ``neutralize`` would read each other's. Returned together, the pair
    is cached together, so the transfer coefficient a sweep reports is the one
    :func:`run_backtest` would.

    A HELD BOOK HAS NO VIEW WHERE ITS RETURN IS NEVER KNOWN. A held book earns
    its forward return once, at its formation, and at the last formations of
    a day that return runs past the close and is NaN for every name: a book
    set there earns nothing. Under ``rebalance="horizon"`` the forecast at
    such a decision is therefore dropped before the book is built, exactly as
    though it were an oracle's, which is NaN wherever the return is. A
    path-independent rule then sets a flat book, closing the one before it;
    the cost-aware allocator unwinds as risk and cost allow. No trade into a
    book that cannot earn is made or charged, and the exit from the book
    before it is charged as an oracle's always was. (Setting the HELD book
    flat after the fact instead would force a mean-variance book, which
    barely trades, through a full exit and re-entry.)
    """
    spread = context["spread"]
    if _drops_forecast(context, config):
        context = {**context, "mu": np.where(_no_outcome(context)[:, None], np.nan,
                                             context["mu"])}
    mu = context["mu"]

    # THE SCREEN AND THE BOOK ARE DIFFERENT SETS. ``screened`` is what the
    # portfolio may trade at all; ``signs`` is what this configuration chose to
    # hold out of it. Both allocator families consume the second, which is what
    # keeps `selection` a real axis for mean-variance rather than a label on
    # four identical portfolios.
    screened = apply_screen(config.universe, spread, shape=mu.shape)
    mu = _signal(context, config, sigma, screened)
    signs = select_book(config.selection, mu, screened)
    if config.weighting != "mean_variance":
        return weight_book(config.weighting, signs, mu, sigma,
                           gross_exposure=config.gross_exposure), mu

    # A name outside the BOOK gets a zero expected return, so the optimizer
    # unwinds it as risk and cost allow. A name outside the UNIVERSE gets an
    # infinite cost, which the proximal step reads as "hold whatever you have".
    # The distinction matters: zeroing mu lets a position be exited, while an
    # infinite cost freezes it, and using the latter for both would trap the
    # portfolio in names the selection had already dropped.
    masked = np.where(signs != 0, mu, 0.0)
    forecast = ForwardReturns(masked[:, :, None], context["decisions"],
                              context["assets"], context["axis"])
    # lambda is set so the COST-FREE target averages the requested gross
    # exposure. mu is in return units and Sigma in squared ones, so their ratio
    # has no natural scale; fixing the exposure is what makes one forecast's
    # spread bill comparable with another's. The average runs over the
    # decisions UP TO each one, never past it: a lambda fit on the whole window
    # let a forecast made next month resize the book held today.
    risk_aversion = causal_risk_aversion(masked, sigma, config.gross_exposure)
    # The portfolio stage says what to hold; the trades stage how far to move
    # there, given what trading costs.
    limits = {}
    if config.factor_limit is not None:
        limits = dict(factor_loadings=context["factors"][config.factor_exposure][0],
                      factor_lower=-config.factor_limit, factor_upper=config.factor_limit)
    target = mean_variance_target(forecast, sigma[None], risk_aversion=risk_aversion,
                                  **limits)
    cost = _cost(config, spread, mu.shape, context=context, target=target.values[:, :, 0])
    return rebalance(
        target, sigma[None], risk_aversion=risk_aversion,
        cost=np.where(screened, np.nan_to_num(cost, nan=0.0), np.inf),
        max_iter=config.max_iter, solver=config.mv_solver or DEFAULT_SOLVER,
    ).values[:, :, 0], mu


def _cost(config: BacktestConfig, spread, shape, *, context=None,
          target=None) -> NDArray[np.float64]:
    """Per-unit trading cost the allocator is charged: the cost model's, plus fees.

    Fees are added on top of whatever the model charges, at
    :func:`~stable_finance.evaluation.execution.fee_per_unit`, so the book is
    planned against the same fees :func:`_score_fills` bills. A name the model
    cannot trade stays at infinite cost, and so does one without the price a
    per-share fee needs.
    """
    cost = _model_cost(config, spread, shape, context=context, target=target)
    if not config.fees:
        return cost
    return cost + fee_per_unit(share_price(context, config), **config.fees)


def _model_cost(config: BacktestConfig, spread, shape, *, context=None,
                target=None) -> NDArray[np.float64]:
    """Per-unit trading cost; zero for ``mid`` whether or not there are quotes.

    The impact model's cost depends on the order, so the allocator plans
    against the cost of the move the cost-free target would make: the one
    linearization the solver can take without knowing its own answer.
    """
    if config.cost_model in ("impact", "impact_stock"):
        market = context["market"]
        adv = market.get("adv")
        moves = np.abs(target - np.vstack([np.zeros((1, shape[1])), target[:-1]]))
        size = 100.0 * np.nan_to_num(moves) * config.equity / np.where(adv > 0, adv, np.nan)
        if config.cost_model == "impact_stock":
            traits = [market.get(n) for n in ("market_cap", "idio_vol", "market_vol", "time_trend")]
            known = np.isfinite(size) & np.all([np.isfinite(x) for x in traits], axis=0)
            planned = impact_stock_cost(np.nan_to_num(size), *[np.nan_to_num(x) for x in traits],
                                        coefficients=config.impact)
            return np.where(known, planned, np.inf)
        return np.where(np.isfinite(size), impact_cost(np.nan_to_num(size), coefficients=config.impact), np.inf)
    if spread is None:
        # check_config has already refused every other model without a spread.
        return np.zeros(shape, dtype=np.float64)
    efq = config.efq if context is None else context.get("efq", config.efq)
    return trading_cost(config.cost_model, spread, multiple=efq)


def _hold_within_day(weights, day):
    """Keep each day's first decision's weights and hold them all day.

    Applied AFTER weighting rather than by subsampling the decisions, so the
    book is still marked and still earns a return at every decision -- it just
    does not retrade. Turnover falls; exposure does not.
    """
    starts = np.r_[True, day[1:] != day[:-1]]
    anchor = np.maximum.accumulate(np.where(starts, np.arange(len(day)), 0))
    return weights[anchor]


# ── scoring ─────────────────────────────────────────────────────────────────

def causal_risk_aversion(mu: ArrayLike, sigma: ArrayLike,
                         gross_exposure: float) -> NDArray[np.float64]:
    """One lambda per decision, from the decisions up to and including it.

    ``lambda_t = mean_{s <= t} |Sigma^-1 mu_s|_1 / gross_exposure``: the
    cost-free mean-variance book averages the requested gross exposure over
    the forecasts seen so far. Only the past and the present enter, so a run
    cut short gives the same weights on the decisions it kept.
    """
    raw = np.linalg.solve(np.asarray(sigma, dtype=np.float64),
                          np.nan_to_num(np.asarray(mu, dtype=np.float64), nan=0.0).T).T
    gross = np.abs(raw).sum(axis=1)
    expanding = np.cumsum(gross) / np.arange(1, len(gross) + 1)
    return np.maximum(expanding / gross_exposure, 1e-12)





def _formations(context: dict) -> NDArray[np.bool_]:
    """The decisions where a horizon-held book is set: every k-th of each day.

    k is the forecast horizon in decisions. Blocks restart at each day, since
    a forward return does not run across the close; without ``days`` the
    decisions are one run. A formation whose forward return is unknown for
    every name still starts a period, but the forecast has no view there (see
    :func:`_book`), so the period carries the unwinding of the book before it.
    """
    if "formed" not in context:
        if context["horizon"] <= 0:
            raise ValueError("rebalance='horizon' needs a horizon of fixed length; a book "
                             "held to the close is already set once a day (rebalance='daily')")
        every = decisions_spanning(context, seconds=context["horizon"])
        day = context["day"]
        index = np.arange(len(day))
        if context["days_given"]:
            starts = np.r_[True, day[1:] != day[:-1]]
            index = index - np.maximum.accumulate(np.where(starts, index, 0))
        context["formed"] = index % every == 0
    return context["formed"]


def _no_outcome(context: dict) -> NDArray[np.bool_]:
    """The decisions at which no name's forward return is known."""
    if "dead" not in context:
        context["dead"] = ~np.isfinite(context["outcome_values"]).any(axis=1)
    return context["dead"]


def _drops_forecast(context: dict, config: BacktestConfig) -> bool:
    """Whether this configuration's book is built with part of the forecast dropped.

    True for a held book whose forecast has a view at a decision where no
    forward return is known (see :func:`_book`); a forecast already unknown
    at every such decision, as an oracle's is, has nothing to drop.
    """
    if config.rebalance != "horizon" or context["mu"] is None:
        return False
    return bool(np.isfinite(context["mu"][_no_outcome(context)]).any())


def _hold_from(weights, formed):
    """Carry each formation decision's weights until the next formation."""
    anchor = np.maximum.accumulate(np.where(formed, np.arange(len(formed)), 0))
    return weights[anchor]


def _score(context: dict, values: NDArray[np.float64],
           config: BacktestConfig, *,
           with_series: bool = False, alpha: NDArray[np.float64] | None = None,
           sigma: NDArray[np.float64] | None = None) -> BacktestReport:
    """Mark a book, charge it, and annualize -- the axes that act on a book.

    ``alpha`` and ``sigma`` are the transformed forecast and the covariance the
    book was built from; given both, the report carries a transfer coefficient.
    """
    day = context["day"]
    if config.rebalance == "daily":
        values = _hold_within_day(values, day)
    if config.rebalance == "horizon":
        values = _hold_from(values, _formations(context))
    # Every model is scored from its fills, so a cross fill is billed the
    # touch the order met (the arrival quotes, when the market carries them)
    # while the allocator planned against the quoted spread at the decision.
    return _score_fills(context, _fills_toward(context, values, config), config,
                        with_series=with_series, alpha=alpha, sigma=sigma)


def _fills_toward(context: dict, values: NDArray[np.float64], config: BacktestConfig):
    """Trade toward a target path from the positions actually held."""
    target = PortfolioWeights(values[:, :, None], context["decisions"],
                              context["assets"], context["axis"])
    if config.cost_model in ("impact", "impact_stock"):
        # Every order completes, so there is nothing to chase: the orders are
        # the target's moves, each worked over its schedule.
        return execute(orders_from_weights(target), context["market"], model=config.cost_model,
                       equity=config.equity, schedule=impact_schedule(context, config),
                       impact=config.impact)
    # Otherwise chase the target from what is really held: a name that is
    # not priced at a decision is held, not moved for free.
    return fill_path(target, context["market"], model=config.cost_model,
                     efq=context.get("efq", config.efq),
                     equity=config.equity, drift=_drift(context, config))


def _drift(context: dict, config: BacktestConfig):
    """The decision-to-decision returns positions move with, under drift holding; else None."""
    return context["outcome"].values if config.holding == "drift" else None


def _score_fills(context: dict, fills, config: BacktestConfig, *,
                 with_series: bool = False, alpha: NDArray[np.float64] | None = None,
                 sigma: NDArray[np.float64] | None = None) -> BacktestReport:
    """Mark the positions fills produced, and charge each fill's own cost."""
    positions = positions_from_fills(fills, drift=_drift(context, config)).values[:, :, 0]
    outcome_values = context["outcome_values"]
    valid = np.isfinite(positions) & np.isfinite(outcome_values)
    if config.rebalance == "horizon":
        # A held book earns its forward return ONCE, at the decision it was
        # set: the decisions it is merely carried through are inside that same
        # return, and booking them again would count it once per decision.
        valid &= _formations(context)[:, None]
    gross = np.where(valid, positions * outcome_values, 0.0).sum(axis=1)
    quantity = fills.quantity[:, :, 0]
    traded = quantity != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        slippage = np.where(traded, (fills.price[:, :, 0] - fills.mid_price[:, :, 0])
                            / fills.mid_price[:, :, 0], 0.0)
    costs = (quantity * slippage).sum(axis=1)
    fees = None
    if config.fees:
        # Fees are billed on the quantity each fill actually traded, beside
        # its slippage rather than inside its price (see the execution
        # module), so cost is still exactly slippage plus fees.
        fees = fee_cost(quantity, share_price(context, config), **config.fees).sum(axis=1)
        costs = costs + fees
    return _report(context, config, gross, costs, np.abs(quantity), positions,
                   with_series=with_series, alpha=alpha, sigma=sigma, fees=fees)


def _report(context: dict, config: BacktestConfig, gross, costs, turnover_per_asset,
            values, *, with_series: bool, alpha: NDArray[np.float64] | None = None,
            sigma: NDArray[np.float64] | None = None, fees=None) -> BacktestReport:
    """Annualize a profit path and summarize it, with the IC where there is a forecast.

    ``fees`` is the per-decision part of ``costs`` that is fees, folded into
    periods the same way, or None when the configuration charges none.
    """
    mu, spread, day = context["mu"], context["spread"], context["day"]
    kind = context.get("with_exposure")
    net = gross - costs
    turnover = turnover_per_asset.sum(axis=1)
    if config.rebalance == "horizon":
        # Turnover per held book, the basis its cost is reported on: the
        # trades into a book and any while it is carried are one rebalance.
        turnover = np.add.reduceat(turnover, np.flatnonzero(_formations(context)))

    if config.annualization == "daily":
        order = np.argsort(day, kind="stable")
        edges = np.flatnonzero(np.r_[True, day[order][1:] != day[order][:-1], True])
        fold = lambda series: np.array(  # noqa: E731
            [series[order][a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
        )
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        fee_series = None if fees is None else fold(fees)
        # One label per folded period, in the same order as the folded series.
        period_labels = context["day_labels"][order][edges[:-1]]
        periods_per_year = 252.0
    elif config.rebalance == "horizon":
        # One period per held book: the decisions it is carried through are
        # folded into the one at which it was set, which labels the period.
        block = np.cumsum(_formations(context)) - 1
        edges = np.flatnonzero(np.r_[True, block[1:] != block[:-1], True])
        fold = lambda series: np.add.reduceat(series, edges[:-1])
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        fee_series = None if fees is None else fold(fees)
        period_labels = np.asarray(context["decisions"])[edges[:-1]]
        periods_per_year = (len(edges) - 1) / max(len(np.unique(day)), 1) * 252.0
    else:
        costs_series = costs
        fee_series = fees
        period_labels = np.asarray(context["decisions"])
        periods_per_year = (context["n_decisions"]
                            / max(len(np.unique(day)), 1)) * 252.0

    # IC IS MEASURED ON THE SCREENED UNIVERSE, not on the book: the portfolio
    # can only express a view on names it is allowed to trade, but restricting
    # the IC to the tails it actually held would score the forecast on the
    # subset it was most confident about and call that its ranking ability.
    if mu is None:
        ic_mean = ic_se = float("nan")
    else:
        screened = apply_screen(config.universe, spread, shape=mu.shape)
        ic = grouped_rank_ic(np.where(screened, mu, np.nan), context["outcome_values"],
                             min_assets=2)
        ic_mean, ic_se = float(ic.mean), float(ic.standard_error)
    # THE TC READS THE SAME SCREENED NAMES AS THE IC, so the two factors of the
    # fundamental law are measured over one cross-section. The positions are
    # the HELD ones, with roundoff residue zeroed under the same threshold as
    # ``mean_names_held``: a book that returned to flat is flat, not a vector of
    # ulps whose correlation with the forecast is noise. NaN is the module
    # constant, not a fresh float, so two NaN reports compare equal as rows.
    tc = np.nan
    if mu is not None and alpha is not None and sigma is not None:
        held = np.where(np.isfinite(values) & (np.abs(values) <= 1e-12), 0.0, values)
        # A held book is scored where it was set, against the forecast it was
        # set from: the decisions it is carried through hold an older book.
        rows = (_formations(context) & ~_no_outcome(context)
                if config.rebalance == "horizon" else slice(None))
        mean = transfer_coefficient(held[rows], alpha[rows], sigma,
                                    eligible=screened[rows]).mean
        tc = float(mean) if np.isfinite(mean) else np.nan
    return BacktestReport(
        config=config, horizon=context["horizon"],
        information_coefficient=ic_mean,
        information_coefficient_se=ic_se,
        sharpe_mid=sharpe_ratio(gross, periods_per_year=periods_per_year),
        sharpe_net=sharpe_ratio(net, periods_per_year=periods_per_year),
        mean_gross_return_bps=float(np.nanmean(gross) * 1e4),
        mean_cost_bps=float(np.nanmean(costs_series) * 1e4),
        mean_turnover=float(turnover.mean()),
        mean_gross_exposure=float(np.abs(values).sum(axis=1).mean()),
        periods_per_year=float(periods_per_year),
        n_periods=int(len(net)),
        # A position is a holding above roundoff: a path of fills that returns
        # to flat leaves a weight of a few ulps, not a name held.
        mean_names_held=float((np.abs(values) > 1e-12).sum(axis=1).mean()),
        transfer_coefficient=tc,
        series=PnLSeries(
            periods=period_labels.astype(str),
            gross=gross.copy(), net=net.copy(), cost=costs_series.copy(),
            fee=None if fee_series is None else fee_series.copy(),
        ) if with_series else None,
        mean_fee_bps=None if fee_series is None else float(np.nanmean(fee_series) * 1e4),
        exposure=None if kind is None else portfolio_exposure(
            values, context["factors"][kind][0], factor_names=context["factors"][kind][1]),
    )


# ── public API ──────────────────────────────────────────────────────────────

#: What the harness accepts, from the top of the pipeline to the bottom.
LEVELS = (Embeddings, ForwardReturns, PortfolioWeights, Orders)


def _forecast(source, train, probe) -> ForwardReturns:
    """Turn embeddings into a forecast with a probe fit on ``train``."""
    if train is None:
        raise ValueError("embeddings need train=(embeddings, returns) to fit the probe on")
    train_embeddings, train_returns = train
    return (probe or RidgeProbe()).fit(train_embeddings, train_returns).predict(source)


def _check_exposure(kind: str | None, market: MarketData) -> None:
    """Refuse a ``with_exposure`` the market cannot support, before anything runs."""
    if kind is None:
        return
    if kind not in NEUTRALIZATIONS:
        raise ValueError(f"with_exposure={kind!r} is not None or one of {NEUTRALIZATIONS}")
    market.require(*SIGNAL_REQUIRES[kind], purpose=f"with_exposure={kind!r}")


def run_backtest(
    forecast,
    realized: ForwardReturns,
    *,
    market: MarketData | None = None,
    half_spread: ArrayLike | None = None,
    history: ForwardReturns | None = None,
    embeddings: Embeddings | None = None,
    days: ArrayLike | None = None,
    config: BacktestConfig = DEFAULT_BACKTEST,
    horizon_index: int = 0,
    with_series: bool = False,
    train: tuple[Embeddings, ForwardReturns] | None = None,
    probe: RidgeProbe | None = None,
    efq: ArrayLike | None = None,
    unfilled: str = "retry",
    with_exposure: str | None = None,
) -> BacktestReport:
    """Score one input, at any level, under one harness configuration.

    ``forecast`` is the input at any level -- embeddings, a forecast, target
    weights, or orders (see the module docstring) -- and the harness runs the
    stages below it. Embeddings need
    ``train``, the embeddings and realized returns to fit ``probe`` on (a
    default :class:`RidgeProbe` if not given), and also serve the
    ``embedding`` risk model. Below the forecast only the trading axes apply:
    ``cost_model``, ``efq``, ``equity``, the fees, ``rebalance`` (weights
    only) and ``annualization``.

    A forecast must already be in the units the realized returns are in; a
    rank-unit forecast traded against a return-unit spread would make the cost
    term meaningless. ``history`` supplies the returns the risk model is
    estimated from and should cover a period strictly before ``realized``; it
    defaults to ``realized`` itself, which is in-sample and appropriate only
    for tests. ``days`` labels each decision's trading day and is required by
    the ``daily`` rebalance and annualization options.

    ``market`` is the observable market on the input's grid; ``half_spread``
    is shorthand for a market holding only that. Each configuration axis
    declares what it needs (:attr:`BacktestConfig.requires`), and a
    configuration the market cannot support is refused with
    :class:`~stable_finance.dataset.schema.MissingRolesError` before anything
    runs. With neither given only frictionless configurations run, such as
    ``config=DEFAULT_BACKTEST.replace(cost_model="mid")``.

    ``efq`` is a per-name (or per-decision) execution quality, an array that
    broadcasts to ``(decision, asset)``, used in place of ``config.efq`` for
    both the allocator's charge and the fills. The config keeps a scalar so a
    configuration stays one hashable point of the sweep; this is for a
    caller who has measured execution quality by name.

    ``unfilled`` says what becomes of an ORDER (input at the orders level)
    that cannot fill where it is sent, e.g. a name with no quote then.
    ``"retry"`` (default) keeps it working: the orders are read as the
    position path they imply, and that path is chased from what is really
    held (:func:`fill_path`), so a missed exit goes out again at the next
    decision the name quotes instead of leaving the position open for good.
    ``"cancel"`` drops it, as the orders were literally sent. When every
    order fills the two are identical. The impact models complete every
    order, so it does not apply to them.

    ``config.holding="drift"`` carries positions with their own returns
    between decisions (see :data:`~stable_finance.evaluation.config.HOLDING_RULES`);
    ``realized`` must then hold each name's return from each decision to the
    next. It is supported for the ``mid`` and ``cross`` cost models.

    ``with_exposure`` names one of the library's factor estimators
    (:data:`~stable_finance.evaluation.portfolio.signals.NEUTRALIZATIONS`)
    and adds the held book's exposure to it, and its concentration, to the
    report as :attr:`BacktestReport.exposure`. Like ``with_series`` it is a
    reporting choice, not a configuration: it changes no label and no book,
    and reads any input level, including weights or orders supplied by a
    caller. Beta and the principal components are estimated from
    ``history``, so pass a prior period to keep them out of sample; the
    industry options need an ``industry`` code on the market.
    ``config.factor_limit`` bounds the mean-variance target's exposure to
    ``config.factor_exposure``; where the held book ended up after costs,
    partial fills and drift is what this reports.
    """
    source = forecast
    if isinstance(source, Embeddings):
        embeddings = source if embeddings is None else embeddings
        source = _forecast(source, train, probe)
    if not isinstance(source, LEVELS):
        raise TypeError(f"run_backtest takes one of {[c.__name__ for c in LEVELS]}, "
                        f"not {type(source).__name__}")
    market = _market(source, market, half_spread)
    if unfilled not in ("retry", "cancel"):
        raise ValueError(f"unfilled must be 'retry' or 'cancel', not {unfilled!r}")
    if config.holding == "drift" and config.cost_model not in ("mid", "cross"):
        raise ValueError("holding='drift' is supported for the mid and cross cost models")
    check_axes(config, market, source)
    _check_exposure(with_exposure, market)
    context = prepare(source, realized, market, days, horizon_index)
    _exposures(context, config.factor_exposure, history, realized)
    _exposures(context, with_exposure, history, realized)
    context["with_exposure"] = with_exposure
    if efq is not None:
        context["efq"] = np.broadcast_to(_efq_array(efq), (context["n_decisions"], context["n_assets"]))
    if isinstance(source, ForwardReturns):
        sigma = _risk(context, config, history, realized, embeddings, horizon_index)
        if config.neutralize in ("beta", "beta_industry"):
            _beta(context, history, realized)
        if config.neutralize == "pca3":
            _loadings(context, history, realized)
        weights, alpha = _book(context, config, sigma)
        return _score(context, weights, config, with_series=with_series,
                      alpha=alpha, sigma=sigma)
    if isinstance(source, PortfolioWeights):
        return _score(context, source.values[:, :, horizon_index], config,
                      with_series=with_series)
    one = slice(horizon_index, horizon_index + 1)
    orders = Orders(source.values[:, :, one], source.decisions, source.assets,
                    source.horizons[one],
                    limit=None if source.limit is None else source.limit[:, :, one])
    if unfilled == "retry" and config.cost_model not in ("impact", "impact_stock"):
        # The path the orders imply, chased from what is really held.
        path = PortfolioWeights(np.cumsum(np.nan_to_num(orders.values, nan=0.0), axis=0),
                                orders.decisions, orders.assets, orders.horizons)
        fills = fill_path(path, market, model=config.cost_model, efq=context.get("efq", config.efq),
                          equity=config.equity, limit=orders.limit, drift=_drift(context, config))
    else:
        fills = execute(orders, market, model=config.cost_model, efq=context.get("efq", config.efq),
                        equity=config.equity, impact=config.impact,
                        schedule=(impact_schedule(context, config)
                                  if config.cost_model in ("impact", "impact_stock") else 0))
    return _score_fills(context, fills, config, with_series=with_series)


def sweep_configs(
    forecast,
    realized: ForwardReturns,
    configs,
    *,
    market: MarketData | None = None,
    half_spread: ArrayLike | None = None,
    history: ForwardReturns | None = None,
    embeddings: Embeddings | None = None,
    days: ArrayLike | None = None,
    horizon_index: int = 0,
    on_error: str = "raise",
    book_cache_size: int = 8,
    with_series: bool = False,
    train: tuple[Embeddings, ForwardReturns] | None = None,
    probe: RidgeProbe | None = None,
    with_exposure: str | None = None,
) -> tuple[BacktestReport, ...]:
    """Score one input under many configurations, sharing the work.

    Identical to calling :func:`run_backtest` in a loop, and much faster,
    because the space is enormously redundant: a covariance depends only on the
    risk model, and a book depends only on the axes that produce weights.
    ``rebalance`` and ``annualization`` act on a book that already exists, so
    every configuration sharing the other axes reuses one solve -- which for
    the cost-aware allocator is essentially the entire cost of the run.
    Embeddings are turned into a forecast once. Weights and orders have no
    book to share, so they are run one configuration at a time.

    ``on_error='skip'`` drops configurations that raise and keeps the rest,
    for sweeps wide enough that one degenerate corner should not lose the
    others. ``book_cache_size`` bounds how many weight panels are held at once;
    the default suffices whenever the book-reusing axes vary fastest.

    Every configuration is checked against ``market`` before any is run, so a
    sweep refuses up front rather than failing partway; with ``on_error='skip'``
    the unsupported ones are dropped instead. ``enumerate_configs(market=...)``
    builds a space that needs no dropping. ``with_exposure`` is
    :func:`run_backtest`'s, reported on every configuration.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    source = forecast
    if isinstance(source, Embeddings):
        embeddings = source if embeddings is None else embeddings
        source = _forecast(source, train, probe)
    market = _market(source, market, half_spread)
    configs = tuple(configs)
    if on_error == "raise":
        for config in configs:
            check_axes(config, market, source)
    else:
        configs = tuple(c for c in configs if c.supported_by(market))
    _check_exposure(with_exposure, market)
    if not isinstance(source, ForwardReturns):
        reports = []
        for config in configs:
            try:
                reports.append(run_backtest(source, realized, market=market, days=days,
                                            config=config, horizon_index=horizon_index,
                                            with_series=with_series, history=history,
                                            with_exposure=with_exposure))
            except Exception:
                if on_error == "raise":
                    raise
        return tuple(reports)
    context = prepare(source, realized, market, days, horizon_index)
    _exposures(context, with_exposure, history, realized)
    context["with_exposure"] = with_exposure
    # The risk cache is tiny (one matrix per risk model) and kept whole. The
    # book cache is NOT: a book is a full weight panel, and the space holds
    # over a thousand distinct ones, which would be gigabytes held for nothing.
    # It is bounded instead, which costs nothing when the axes that reuse a
    # book vary fastest -- as they do in enumerate_configs -- and merely
    # recomputes on an unlucky ordering rather than exhausting memory.
    risk_cache: dict[tuple, NDArray[np.float64]] = {}
    # Each entry is a book AND the transformed forecast it was built from, so
    # a reused book reports the transfer coefficient its first solve would.
    book_cache: "OrderedDict[tuple, tuple[NDArray[np.float64], NDArray[np.float64]]]" = (
        OrderedDict())
    reports = []
    for config in configs:
        try:
            risk_key = (config.risk_model, config.shrinkage)
            if risk_key not in risk_cache:
                risk_cache[risk_key] = _risk(
                    context, config, history, realized, embeddings, horizon_index
                )
            if config.neutralize in ("beta", "beta_industry"):
                _beta(context, history, realized)
            if config.neutralize == "pca3":
                _loadings(context, history, realized)
            _exposures(context, config.factor_exposure, history, realized)
            # A held book that drops part of its forecast is a different book.
            book_key = (*_book_key(config), ("drops_forecast", _drops_forecast(context, config)))
            if book_key in book_cache:
                book_cache.move_to_end(book_key)
            else:
                book_cache[book_key] = _book(
                    context, config, risk_cache[risk_key]
                )
                if len(book_cache) > book_cache_size:
                    book_cache.popitem(last=False)
            weights, alpha = book_cache[book_key]
            reports.append(_score(context, weights, config, with_series=with_series,
                                  alpha=alpha, sigma=risk_cache[risk_key]))
        except Exception:
            if on_error == "raise":
                raise
    return tuple(reports)
