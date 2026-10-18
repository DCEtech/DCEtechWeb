"""Etiquetas (targets) para los modelos.

- `rule_signals`: la estrategia de reglas original (EMA400 + RSI). OJO: se
  calcula con las mismas columnas que el modelo recibe como features (RSI,
  Close, EMA400), por eso el árbol solo reaprende las reglas y saca un 99 %
  de accuracy sin predecir nada. Solo se usa en el baseline de la fase 2.
- `forward_return_target`: target de la fase 3 (experimento 1). Mira el
  precio futuro, que es justo lo que el modelo no puede ver en sus features.
- `vol_adjusted_target`: experimento 2. Igual, pero el umbral se mide en
  unidades de volatilidad reciente, así significa lo mismo en cada régimen.
- `triple_barrier_target`: experimento 3. Barreras simétricas de
  take-profit y stop-loss, más un límite de tiempo: ¿qué se toca primero?
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from btc_decision_tree.indicators import log_return_volatility

BUY, HOLD, SELL = 1, 0, -1

K_GRID = np.round(np.arange(0, 2.001, 0.01), 2)


def forward_return_target(
    close: pd.Series, horizon: int = 30, threshold: float = 0.10
) -> pd.Series:
    """1 si dentro de `horizon` filas el precio sube más de `threshold`, si no 0.

    Las filas sin precio futuro conocido (las últimas `horizon`) se eliminan:
    no son 0, son desconocidas.
    """
    if horizon <= 0:
        raise ValueError(f"horizon debe ser positivo; es {horizon}")
    if len(close) <= horizon:
        raise ValueError(f"Hacen falta más de {horizon} filas; hay {len(close)}")

    future_return = close.shift(-horizon) / close - 1
    known = future_return.notna()
    return (future_return[known] > threshold).astype(int).rename("target")


def _future_log_return_and_unit_threshold(
    close: pd.Series, horizon: int, window: int
) -> tuple[pd.Series, pd.Series]:
    """log(C[t+h] / C[t]) y vol_30d · √h (el umbral para k = 1), en filas conocidas."""
    if horizon <= 0:
        raise ValueError(f"horizon debe ser positivo; es {horizon}")
    if len(close) <= horizon:
        raise ValueError(f"Hacen falta más de {horizon} filas; hay {len(close)}")
    prices = close.astype(float)
    future_log_return = np.log(prices.shift(-horizon) / prices)
    unit_threshold = log_return_volatility(prices, window) * np.sqrt(horizon)
    known = future_log_return.notna() & unit_threshold.notna()
    return future_log_return[known], unit_threshold[known]


def vol_adjusted_target(
    close: pd.Series, k: float, horizon: int = 30, window: int = 30
) -> pd.Series:
    """1 si log(C[t+horizon] / C[t]) > k · vol_window[t] · √horizon, si no 0.

    Se eliminan las filas sin futuro conocido (las últimas `horizon`) y las
    primeras, donde la volatilidad aún no está calculada.
    """
    if k < 0:
        raise ValueError(f"k no puede ser negativo; es {k}")
    future, unit = _future_log_return_and_unit_threshold(close, horizon, window)
    return (future > k * unit).astype(int).rename("target")


def first_touch_labels(paths: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    """Etiqueta *triple barrier* a partir de los log-retornos futuros.

    `paths[i, j]` es log(C[t+j+1] / C[t]) para la fila i, y `barrier[i]` la
    distancia de las dos barreras (simétricas). Devuelve 1 si la superior
    (≥ +b) se toca antes que la inferior (≤ −b), y 0 si se toca antes la
    inferior o si no se toca ninguna (expira).
    """
    horizon = paths.shape[1]
    hit_up = paths >= barrier[:, None]
    hit_down = paths <= -barrier[:, None]
    # argmax da el primer True; si no hay ninguno, usamos `horizon` (= nunca).
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), horizon)
    first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), horizon)
    return (first_up < first_down).astype(int)


def triple_barrier_target(
    close: pd.Series, k: float, horizon: int = 30, window: int = 30
) -> pd.Series:
    """*Triple barrier* con barreras ±k · vol_window[t] · √horizon y límite a
    `horizon` días, evaluado con el cierre diario.

    Se eliminan las últimas `horizon` filas (aunque ya hayan tocado una
    barrera) y las primeras, sin volatilidad calculada.
    """
    if k < 0:
        raise ValueError(f"k no puede ser negativo; es {k}")
    if horizon <= 0:
        raise ValueError(f"horizon debe ser positivo; es {horizon}")
    if len(close) <= horizon:
        raise ValueError(f"Hacen falta más de {horizon} filas; hay {len(close)}")

    prices = close.astype(float)
    log_prices = np.log(prices.to_numpy())
    n = len(log_prices)
    paths = np.full((n, horizon), np.nan)
    for j in range(1, horizon + 1):
        paths[: n - j, j - 1] = log_prices[j:] - log_prices[: n - j]
    barrier = (k * log_return_volatility(prices, window) * np.sqrt(horizon)).to_numpy()

    known = ~np.isnan(barrier) & (np.arange(n) < n - horizon)
    labels = first_touch_labels(paths[known], barrier[known])
    return pd.Series(labels, index=close.index[known], name="target")


def calibrate_k(
    label_fn: Callable[[float], pd.Series],
    train_index: pd.Index,
    target_rate: float = 0.375,
    grid: np.ndarray = K_GRID,
) -> tuple[float, float]:
    """k de `grid` cuya tasa base en `train_index` queda más cerca de `target_rate`.

    `label_fn(k)` devuelve las etiquetas para esa k (p. ej. una llamada a
    `vol_adjusted_target` o `triple_barrier_target`). Solo mira la proporción
    de clases, nunca el rendimiento del modelo. Devuelve (k, tasa obtenida).
    """
    rates = []
    for k in grid:
        labels = label_fn(float(k))
        missing = train_index.difference(labels.index)
        if len(missing):
            raise ValueError(f"{len(missing)} filas de train no tienen target conocido")
        rates.append(labels.loc[train_index].mean())
    rates_array = np.array(rates)
    best = int(np.argmin(np.abs(rates_array - target_rate)))  # en empate, el k más pequeño
    return float(grid[best]), float(rates_array[best])


def buy_condition(df: pd.DataFrame) -> pd.Series:
    """RSI <= 30 y precio en o bajo la EMA400, o más de un 15 % por encima.

    La rama "> EMA400 * 1.15" contradice el texto del post (comprar cerca de
    la EMA); se conserva tal cual para reproducir el modelo publicado.
    """
    near_or_below = df["Close"] <= df["EMA400"]
    far_above = df["Close"] > df["EMA400"] * 1.15
    return (near_or_below | far_above) & (df["RSI"] <= 30)


def sell_condition(df: pd.DataFrame) -> pd.Series:
    """Precio más de un 25 % sobre la EMA400 y RSI > 70."""
    return (df["Close"] > df["EMA400"] * 1.25) & (df["RSI"] > 70)


def only_after_first_buy(sell: pd.Series, buy: pd.Series) -> pd.Series:
    """Anula las ventas anteriores (o iguales) a la primera compra."""
    if not buy.any():
        return pd.Series(False, index=sell.index)
    first_buy = buy.idxmax()  # primera fecha con True
    return sell & (sell.index > first_buy)


def rule_signals(df: pd.DataFrame) -> pd.Series:
    """Señal de la estrategia: 1 = compra, -1 = venta, 0 = mantener."""
    buy = buy_condition(df)
    sell = only_after_first_buy(sell_condition(df), buy)
    signal = pd.Series(HOLD, index=df.index, name="signal")
    signal[buy] = BUY
    signal[sell] = SELL  # como en el original, venta pisa a compra
    return signal
