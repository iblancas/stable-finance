"""What an execution policy may see at a decision: the market as it stood, and nothing after.

A learned policy is only as honest as its inputs. Market data holds two kinds
of quantity (see :mod:`stable_finance.dataset.market`): those observed AT the
decision, and the ``exec_*`` and ``window_*`` quantities, which describe the
seconds AFTER it and exist only so the execution model can price a fill. A
policy that saw the arrival quotes would know whether its order was about to
be filled badly; one that saw the window would know where the price went. So
the observation is built from a fixed list of decision-time quantities,
:data:`OBSERVED`, read only as supplied or from the decision's own quotes,
never through a derivation that runs through the arrival quotes.

Every feature at a decision uses that decision and the ones before it, never
a later one, so corrupting the market after a decision leaves what a policy
saw there unchanged; the tests check exactly that. The features follow the
state the execution literature conditions on (Nevmyvaka, Feng and Kearns,
2006, "Reinforcement learning for optimized trade execution", ICML; Ning, Lin
and Jaimungal, 2018, "Double deep Q-learning for optimal execution",
arXiv:1812.06600), restricted to what a top-of-book dataset such as
Market-1T carries:

``half_spread``   the quoted half-spread over mid, the cost of crossing now.
``imbalance``     ``(bid_size - ask_size) / (bid_size + ask_size)`` at the
                  touch, the short-term pressure on the price.
``adv_scale``     equity over average daily dollar volume: the fraction of a
                  day's volume a unit of weight is, the scale price impact
                  is measured on.
``last_return``   the log change in the decision price since the previous
                  decision.
``volatility``    the standard deviation of ``last_return`` over the last
                  ``lookback`` decisions, this one included.
``forecast``      the forecast the target was built from, when given.

A feature is present only when its source is: a market without sizes has no
``imbalance`` column rather than a column of zeros, so a policy is never fed
an invented number. Within a column, what is unobserved at a decision is 0.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import ADV, MarketData

__all__ = ["OBSERVED", "DecisionFeatures", "decision_features", "observable"]

#: The market quantities a policy may observe: each is as of the decision.
OBSERVED = ("mid", "close", "vwap", "half_spread", "bid_size", "ask_size", ADV)

#: The price a decision is marked at, in the order the execution model reads it.
_PRICES = ("mid", "close", "vwap")


class DecisionFeatures(NamedTuple):
    """The per-name features of every decision, before the policy's own state."""

    #: One name per column of ``values``.
    names: tuple[str, ...]
    #: ``(n_decisions, n_assets, len(names))``, finite, 0 where unobserved.
    values: NDArray[np.float64]


def _touch(market: MarketData, side: str) -> NDArray[np.float64] | None:
    """The decision's best price on one side: the quote, or the book's first level."""
    if f"{side}_price" in market.columns:
        return market.columns[f"{side}_price"]
    if f"book_{side}_price" in market.columns:
        return market.columns[f"book_{side}_price"][:, :, 0]
    return None


def observable(market: MarketData, quantity: str) -> NDArray[np.float64] | None:
    """One decision-time quantity, or None when the market cannot say it without lookahead.

    A quantity is read as supplied, or for ``mid`` and ``half_spread`` from
    the decision's own bid and ask, and for the sizes from the book's first
    level. :meth:`MarketData.get` would also derive a touch from the ARRIVAL
    quotes when the decision's are missing; that is the one route closed
    here, since those quotes are after the decision.
    """
    if quantity not in OBSERVED:
        raise ValueError(f"{quantity!r} is not observable at the decision; a policy may "
                         f"see {OBSERVED}")
    if quantity in market.columns:
        return market.columns[quantity]
    if quantity in ("mid", "half_spread"):
        bid, ask = _touch(market, "bid"), _touch(market, "ask")
        if bid is None or ask is None:
            return None
        quotes = MarketData({"bid_price": bid, "ask_price": ask}, market.decisions, market.assets)
        return quotes.get(quantity)
    if quantity in ("bid_size", "ask_size") and f"book_{quantity}" in market.columns:
        return market.columns[f"book_{quantity}"][:, :, 0]
    return None


def _price(market: MarketData) -> NDArray[np.float64] | None:
    for quantity in _PRICES:
        price = observable(market, quantity)
        if price is not None:
            return price
    return None


def _rolling_std(values: NDArray[np.float64], finite: NDArray[np.bool_],
                 lookback: int) -> NDArray[np.float64]:
    """Population standard deviation over the trailing ``lookback`` rows; 0 below two values."""
    x = np.where(finite, values, 0.0)
    total, squares, count = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for lag in range(min(lookback, len(x))):
        rows = slice(lag, None)
        total[rows] += x[:len(x) - lag]
        squares[rows] += x[:len(x) - lag] ** 2
        count[rows] += finite[:len(x) - lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = np.maximum(squares / count - mean ** 2, 0.0)
    return np.where(count >= 2, np.sqrt(variance), 0.0)


def decision_features(
    market: MarketData,
    *,
    forecast: ArrayLike | None = None,
    equity: float | None = None,
    lookback: int = 12,
) -> DecisionFeatures:
    """The features of the module docstring, for every decision and name.

    ``forecast`` is ``(n_decisions, n_assets)``, in the realized return's
    units. ``equity`` is the portfolio's value in the market's currency and
    is what ``adv_scale`` needs besides ``adv``. ``lookback`` is the number
    of decisions ``volatility`` is measured over: twelve is an hour at
    Market-1T's five-minute decisions.
    """
    if int(lookback) != lookback or lookback < 2:
        raise ValueError("lookback must be an integer of at least two decisions")
    shape = market.shape
    names, columns = [], []

    def add(name, values):
        values = np.asarray(values, dtype=np.float64)
        names.append(name)
        columns.append(np.where(np.isfinite(values), values, 0.0))

    spread = observable(market, "half_spread")
    if spread is not None:
        add("half_spread", spread)
    bid, ask = observable(market, "bid_size"), observable(market, "ask_size")
    if bid is not None and ask is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            add("imbalance", (bid - ask) / (bid + ask))
    adv = observable(market, ADV)
    if adv is not None and equity is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            add("adv_scale", np.where(adv > 0, float(equity) / adv, np.nan))
    price = _price(market)
    if price is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            change = np.full(shape, np.nan)
            change[1:] = np.log(price[1:] / price[:-1])
        finite = np.isfinite(change)
        add("last_return", change)
        add("volatility", _rolling_std(change, finite, int(lookback)))
    if forecast is not None:
        forecast = np.asarray(forecast, dtype=np.float64)
        if forecast.shape != shape:
            raise ValueError(f"forecast must have shape {shape}, got {forecast.shape}")
        add("forecast", forecast)
    values = np.stack(columns, axis=2) if columns else np.zeros(shape + (0,))
    return DecisionFeatures(tuple(names), values)
