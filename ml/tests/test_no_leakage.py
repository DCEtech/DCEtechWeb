"""Test anti-fuga: una feature en t no puede depender de nada posterior a t.

Idea: calculamos las features, cambiamos al azar todos los precios
posteriores a t y recalculamos. Si alguna feature hasta t cambia, es que
estaba "mirando el futuro".
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.features import add_relative_features
from btc_decision_tree.indicators import log_return_volatility

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


def make_prices(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


def alter_after(prices: pd.DataFrame, t: int, seed: int = 1) -> pd.DataFrame:
    """Copia de `prices` con todas las filas posteriores a t cambiadas al azar."""
    rng = np.random.default_rng(seed)
    altered = prices.copy()
    factors = rng.uniform(0.5, 2.0, size=(len(prices) - t - 1, 1))
    altered.iloc[t + 1 :] = altered.iloc[t + 1 :].to_numpy() * factors
    return altered


def leaks(feature_fn: FeatureFn, prices: pd.DataFrame, t: int) -> bool:
    """True si alguna feature en las filas 0..t cambia al alterar el futuro."""
    before = feature_fn(prices).iloc[: t + 1]
    after = feature_fn(alter_after(prices, t)).iloc[: t + 1]
    return not before.equals(after)


# t antes del calentamiento (EMA400 aún NaN), justo después y bien avanzado.
@pytest.mark.parametrize("t", [100, 399, 419, 600, 798])
def test_relative_features_do_not_look_ahead(t: int) -> None:
    assert not leaks(add_relative_features, make_prices(), t)


def test_detector_catches_a_leaky_feature() -> None:
    # Control: una feature que usa el precio de mañana debe ser detectada.
    def leaky(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"tomorrow": prices["Close"].shift(-1)})

    assert leaks(leaky, make_prices(), t=500)


@pytest.mark.parametrize("t", [20, 30, 400, 798])
def test_log_return_volatility_does_not_look_ahead(t: int) -> None:
    def vol(prices: pd.DataFrame) -> pd.DataFrame:
        return log_return_volatility(prices["Close"], 30).to_frame()

    assert not leaks(vol, make_prices(), t)
