import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from btc_decision_tree.indicators import ema, log_return_volatility, rsi_wilder


def random_walk(n: int = 6000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=n, freq="D"))


def ema_loop(values: list[float], length: int) -> list[float]:
    """EMA escrita a mano, tal como se define: semilla SMA y fórmula recursiva."""
    alpha = 2 / (length + 1)
    out = [float("nan")] * len(values)
    prev = sum(values[:length]) / length
    out[length - 1] = prev
    for i in range(length, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi_loop(values: list[float], length: int) -> list[float]:
    """RSI de Wilder escrito a mano."""
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    out = [float("nan")] * len(values)
    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length
    out[length] = 100 * avg_gain / (avg_gain + avg_loss)
    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i - 1]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i - 1]) / length
        out[i] = 100 * avg_gain / (avg_gain + avg_loss)
    return out


@pytest.mark.parametrize("length", [10, 55, 400])
def test_ema_matches_handwritten_loop(length: int) -> None:
    series = random_walk(n=1000)
    expected = pd.Series(ema_loop(series.tolist(), length), index=series.index)
    pd.testing.assert_series_equal(ema(series, length), expected, check_names=False, rtol=1e-12)


@pytest.mark.parametrize("length", [10, 55, 400])
def test_ema_matches_pandas_ta_after_warmup(length: int) -> None:
    series = random_walk()
    warmup = 10 * length
    ours = ema(series, length).iloc[warmup:]
    theirs = ta.ema(series, length=length).iloc[warmup:]
    pd.testing.assert_series_equal(ours, theirs, check_names=False, rtol=1e-9)


def test_rsi_matches_handwritten_loop() -> None:
    series = random_walk(n=500)
    expected = pd.Series(rsi_loop(series.tolist(), 14), index=series.index)
    pd.testing.assert_series_equal(rsi_wilder(series, 14), expected, check_names=False, rtol=1e-9)


def test_rsi_matches_pandas_ta_after_warmup() -> None:
    # pandas_ta no usa semilla SMA en el RSI: al principio difiere (hasta ~18
    # puntos) y converge. Tras 10 × length filas la diferencia es < 0.01 puntos.
    series = random_walk()
    warmup = 10 * 14
    ours = rsi_wilder(series, 14).iloc[warmup:]
    theirs = ta.rsi(series, length=14).iloc[warmup:]
    assert (ours - theirs).abs().max() < 0.01


def test_rsi_first_values_are_nan_and_range_is_valid() -> None:
    rsi = rsi_wilder(random_walk(n=300), 14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].between(0, 100).all()


def test_indicators_raise_with_too_few_rows() -> None:
    short = random_walk(n=10)
    with pytest.raises(ValueError, match="EMA400"):
        ema(short, 400)
    with pytest.raises(ValueError, match="RSI14"):
        rsi_wilder(short, 14)


def test_log_return_volatility_matches_manual_std() -> None:
    series = random_walk(n=200)
    vol = log_return_volatility(series, 30)
    t = 150
    log_returns = np.log(series / series.shift(1)).iloc[t - 29 : t + 1]
    assert np.isclose(vol.iloc[t], np.std(log_returns, ddof=1))
    assert vol.iloc[:30].isna().all() and vol.iloc[30:].notna().all()
