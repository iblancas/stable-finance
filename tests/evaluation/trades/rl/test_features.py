"""What a policy may observe: decision-time quantities, causally summarized."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance import MarketData
from stable_finance.evaluation.trades.rl.features import decision_features, observable

DECISIONS, ASSETS = np.arange(6) * 300, ["A", "B"]


def _quotes():
    mid = np.array([[100.0, 50.0], [101.0, 50.5], [100.5, np.nan],
                    [102.0, 51.0], [101.0, 50.0], [103.0, 50.5]])
    return MarketData({"bid_price": mid - 0.05, "ask_price": mid + 0.05,
                       "bid_size": np.full(mid.shape, 300.0),
                       "ask_size": np.full(mid.shape, 100.0)}, DECISIONS, ASSETS)


def test_only_decision_time_quantities_can_be_observed():
    market = _quotes()
    for quantity in ("exec_bid_price", "window_best_buy", "book_ask_size"):
        with pytest.raises(ValueError, match="not observable"):
            observable(market, quantity)
    np.testing.assert_allclose(observable(market, "mid"), market.get("mid"))


def test_the_touch_is_never_derived_from_the_arrival_quotes():
    """MarketData would read a touch off the quotes five seconds later; a policy may not."""
    arrival = _quotes().get("bid_price")
    market = MarketData({"exec_bid_price": arrival, "exec_ask_price": arrival + 0.1,
                         "bid_size": np.ones(arrival.shape), "ask_size": np.ones(arrival.shape)},
                        DECISIONS, ASSETS)
    assert market.has("mid") and market.has("half_spread")
    assert observable(market, "mid") is None
    assert observable(market, "half_spread") is None
    assert decision_features(market).names == ("imbalance",)


def test_features_are_present_only_where_their_source_is():
    quotes = decision_features(_quotes(), forecast=np.zeros((6, 2)))
    assert quotes.names == ("half_spread", "imbalance", "last_return", "volatility", "forecast")
    mid_only = MarketData({"mid": np.full((6, 2), 100.0)}, DECISIONS, ASSETS)
    assert decision_features(mid_only).names == ("last_return", "volatility")
    with_adv = MarketData({"mid": np.full((6, 2), 100.0), "adv": np.full((6, 2), 5e7)},
                          DECISIONS, ASSETS)
    assert "adv_scale" not in decision_features(with_adv).names     # no equity to scale
    scaled = decision_features(with_adv, equity=1e6)
    np.testing.assert_allclose(scaled.values[:, :, scaled.names.index("adv_scale")], 0.02)
    empty = decision_features(MarketData.empty(DECISIONS, ASSETS))
    assert empty.names == () and empty.values.shape == (6, 2, 0)


def test_feature_values():
    market = _quotes()
    features = decision_features(market, lookback=3)
    column = {name: features.values[:, :, i] for i, name in enumerate(features.names)}
    np.testing.assert_allclose(column["imbalance"], 0.5)
    assert np.all(np.isfinite(features.values))
    mid = market.get("mid")
    change = np.log(mid[1:] / mid[:-1])
    np.testing.assert_allclose(column["last_return"][0], 0.0)
    np.testing.assert_allclose(column["last_return"][1:], np.nan_to_num(change))
    # Over the last three decisions, this one included, from the changes observed.
    full = np.vstack([np.full((1, 2), np.nan), change])
    for t in range(6):
        for i in range(2):
            window = full[max(0, t - 2):t + 1, i]
            window = window[np.isfinite(window)]
            expected = np.std(window) if len(window) >= 2 else 0.0
            assert column["volatility"][t, i] == pytest.approx(expected, abs=1e-15)
    # B's quote is missing at decision 2: no spread, and no change into or out of it.
    assert column["half_spread"][2, 1] == 0.0
    assert column["last_return"][2, 1] == 0.0 and column["last_return"][3, 1] == 0.0


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="lookback"):
        decision_features(_quotes(), lookback=1)
    with pytest.raises(ValueError, match="forecast must have shape"):
        decision_features(_quotes(), forecast=np.zeros((5, 2)))
