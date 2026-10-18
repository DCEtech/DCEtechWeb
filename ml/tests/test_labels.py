import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.labels import (
    BUY,
    HOLD,
    SELL,
    calibrate_k,
    first_touch_labels,
    forward_return_target,
    rule_signals,
    triple_barrier_target,
    vol_adjusted_target,
)


def make_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Filas (Close, EMA400, RSI) con fechas consecutivas."""
    index = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["Close", "EMA400", "RSI"], index=index)


def test_buy_when_below_ema_and_oversold() -> None:
    df = make_frame([(90, 100, 25)])
    assert rule_signals(df).tolist() == [BUY]


def test_hold_when_rsi_not_oversold() -> None:
    df = make_frame([(90, 100, 45)])
    assert rule_signals(df).tolist() == [HOLD]


def test_sell_only_after_first_buy() -> None:
    df = make_frame([
        (130, 100, 80),  # condición de venta, pero aún no ha habido compra
        (90, 100, 25),   # primera compra
        (130, 100, 80),  # venta válida
    ])
    assert rule_signals(df).tolist() == [HOLD, BUY, SELL]


def test_no_sells_without_any_buy() -> None:
    df = make_frame([(130, 100, 80), (140, 100, 85)])
    assert rule_signals(df).tolist() == [HOLD, HOLD]


def test_missing_indicators_give_hold() -> None:
    # Al principio de la serie EMA400 y RSI son NaN: no debe haber señal.
    df = make_frame([(90, float("nan"), float("nan"))])
    assert rule_signals(df).tolist() == [HOLD]


def make_close(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


def test_forward_return_target_values() -> None:
    # horizonte 2: compara cada precio con el de dos días después
    close = make_close([100, 105, 115, 120, 90])
    target = forward_return_target(close, horizon=2, threshold=0.10)
    # 115/100 = +15 % → 1 ; 120/105 = +14 % → 1 ; 90/115 = -22 % → 0
    assert target.tolist() == [1, 1, 0]


def test_forward_return_target_drops_rows_without_future() -> None:
    close = make_close([float(i) for i in range(1, 101)])
    target = forward_return_target(close, horizon=30)
    assert len(target) == 70
    assert target.index[-1] == close.index[69]


def test_forward_return_target_rejects_bad_horizon() -> None:
    close = make_close([100, 101, 102])
    with pytest.raises(ValueError):
        forward_return_target(close, horizon=0)
    with pytest.raises(ValueError):
        forward_return_target(close, horizon=3)


def random_close(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return make_close(list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))))


def test_vol_adjusted_target_matches_formula() -> None:
    close = random_close()
    k, horizon = 0.6, 30
    target = vol_adjusted_target(close, k=k, horizon=horizon)
    t = 200
    log_returns = np.log(close / close.shift(1)).iloc[t - 29 : t + 1]
    threshold = k * np.std(log_returns, ddof=1) * np.sqrt(horizon)
    expected = int(np.log(close.iloc[t + horizon] / close.iloc[t]) > threshold)
    assert target.loc[close.index[t]] == expected
    # Sin futuro al final y sin volatilidad al principio.
    assert target.index[0] == close.index[30]
    assert target.index[-1] == close.index[-31]


def test_vol_adjusted_target_with_k_zero_is_any_rise() -> None:
    close = random_close()
    target = vol_adjusted_target(close, k=0)
    rose = (close.shift(-30) > close).astype(int).loc[target.index]
    assert target.equals(rose.rename("target"))


def test_calibrate_k_hits_target_rate_and_rate_falls_with_k() -> None:
    close = random_close(n=2000)
    train_index = close.index[30:1500]
    k, rate = calibrate_k(lambda x: vol_adjusted_target(close, k=x), train_index, target_rate=0.375)
    assert abs(rate - 0.375) < 0.01
    rates = [vol_adjusted_target(close, k=x).loc[train_index].mean() for x in (0.2, 0.6, 1.0)]
    assert rates[0] >= rates[1] >= rates[2]


def test_calibrate_k_rejects_train_rows_without_target() -> None:
    close = random_close()
    with pytest.raises(ValueError, match="no tienen target"):
        # incluye las últimas 30 filas, sin futuro
        calibrate_k(lambda x: vol_adjusted_target(close, k=x), close.index)


# --- triple barrier -------------------------------------------------------
# paths[i, j] = log-retorno acumulado hasta el día j+1; barrera ±0.10.

def test_first_touch_upper_first() -> None:
    paths = np.array([[0.02, 0.06, 0.11, -0.20]])  # arriba en el día 3, abajo en el 4
    assert first_touch_labels(paths, np.array([0.10])).tolist() == [1]


def test_first_touch_lower_first() -> None:
    paths = np.array([[-0.03, -0.12, 0.15, 0.20]])  # abajo en el día 2
    assert first_touch_labels(paths, np.array([0.10])).tolist() == [0]


def test_first_touch_expires() -> None:
    paths = np.array([[0.05, -0.05, 0.09, -0.09]])  # nunca llega a ±0.10
    assert first_touch_labels(paths, np.array([0.10])).tolist() == [0]


def test_first_touch_exact_barrier_counts_as_touch() -> None:
    assert first_touch_labels(np.array([[0.0, 0.10]]), np.array([0.10])).tolist() == [1]
    assert first_touch_labels(np.array([[0.0, -0.10]]), np.array([0.10])).tolist() == [0]


def test_triple_barrier_matches_loop_and_drops_rows_without_future() -> None:
    close = random_close()
    k, horizon = 0.8, 30
    target = triple_barrier_target(close, k=k, horizon=horizon)
    assert target.index[0] == close.index[30]     # antes no hay volatilidad
    assert target.index[-1] == close.index[-31]   # las últimas 30 no tienen futuro

    vol = np.log(close / close.shift(1)).rolling(30).std()
    for t in (100, 250, 400):
        barrier = k * vol.iloc[t] * np.sqrt(horizon)
        expected = 0
        for j in range(1, horizon + 1):
            move = np.log(close.iloc[t + j] / close.iloc[t])
            if move >= barrier:
                expected = 1
                break
            if move <= -barrier:
                break
        assert target.loc[close.index[t]] == expected


def test_calibrate_k_works_with_triple_barrier() -> None:
    close = random_close(n=2000)
    train_index = close.index[30:1500]
    k, rate = calibrate_k(lambda x: triple_barrier_target(close, k=x), train_index, 0.375)
    assert abs(rate - 0.375) < 0.01
