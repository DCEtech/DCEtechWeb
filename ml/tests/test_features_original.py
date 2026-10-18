import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas_ta")  # baseline de la fase 2: solo en local

from btc_decision_tree.features_original import add_original_features, ema


def make_prices(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Paseo aleatorio positivo con forma de velas OHLCV."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


def test_ema_uses_requested_length() -> None:
    # Protege contra el bug de cleanPipeline.ipynb (window= ignorado → EMA10).
    series = make_prices()["Close"]
    assert not ema(series, 400).equals(ema(series, 10))
    assert ema(series, 400).first_valid_index() == series.index[399]


def test_ema_raises_with_too_few_rows() -> None:
    with pytest.raises(ValueError, match="EMA400"):
        ema(make_prices(n=100)["Close"], 400)


def test_relative_features_do_not_depend_on_price_scale() -> None:
    # Si BTC valiera 10 veces más, RSI y dist_ema400 deberían ser iguales.
    # Esto es lo que significa que una feature sea "relativa".
    prices = make_prices()
    base = add_original_features(prices)
    scaled = add_original_features(prices * 10)

    for col in ["RSI", "dist_ema400"]:
        pd.testing.assert_series_equal(base[col], scaled[col], check_exact=False, rtol=1e-9)
    # En cambio EMA400 (precio en bruto) sí cambia: por eso no sirve como feature.
    assert not np.allclose(base["EMA400"].dropna(), scaled["EMA400"].dropna())
