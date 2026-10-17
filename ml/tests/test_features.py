import numpy as np
import pandas as pd

from btc_decision_tree.features import RELATIVE_FEATURES, add_relative_features


def make_prices(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


def test_returns_expected_columns_and_index() -> None:
    prices = make_prices()
    features = add_relative_features(prices)
    assert list(features.columns) == RELATIVE_FEATURES
    assert features.index.equals(prices.index)


def test_all_features_available_after_warmup() -> None:
    # EMA400 está lista en la fila 399 y su pendiente de 20 días en la 419.
    features = add_relative_features(make_prices())
    assert features.iloc[418].isna().any()
    assert features.iloc[419:].notna().all().all()


def test_features_do_not_depend_on_price_scale() -> None:
    prices = make_prices()
    pd.testing.assert_frame_equal(
        add_relative_features(prices),
        add_relative_features(prices * 10),
        check_exact=False,
        rtol=1e-9,
    )


def test_known_values() -> None:
    prices = make_prices()
    features = add_relative_features(prices)
    close = prices["Close"]
    t = 500
    assert np.isclose(features["ret_7d"].iloc[t], close.iloc[t] / close.iloc[t - 7] - 1)
    daily = close.pct_change().iloc[t - 29 : t + 1]
    assert np.isclose(features["vol_30d"].iloc[t], daily.std())
