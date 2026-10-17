"""Indicadores técnicos en pandas puro (sin pandas_ta).

Cada valor en la fecha t se calcula solo con datos hasta t, así que se pueden
usar como features sin fuga de información.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_enough_rows(series: pd.Series, needed: int, name: str) -> None:
    if len(series) < needed:
        raise ValueError(f"{name} necesita al menos {needed} filas; hay {len(series)}")


def ema(series: pd.Series, length: int) -> pd.Series:
    """Media móvil exponencial con semilla SMA (como TA-Lib y pandas_ta).

    El primer valor (fila `length - 1`) es la media simple de las primeras
    `length` filas. Desde ahí: ema_t = a * x_t + (1 - a) * ema_{t-1}, con
    a = 2 / (length + 1). Eso es `ewm(span=length, adjust=False)`; con el
    `adjust=True` por defecto de pandas el resultado sería otro.
    """
    _check_enough_rows(series, length, f"EMA{length}")
    seeded = series.astype(float)  # astype devuelve una copia
    seeded.iloc[: length - 1] = np.nan
    seeded.iloc[length - 1] = series.iloc[:length].mean()
    return seeded.ewm(span=length, adjust=False).mean()


def _wilder_average(values: pd.Series, length: int) -> pd.Series:
    """Media de Wilder: semilla SMA de las `length` primeras variaciones y
    después ewm con a = 1 / length (no 2 / (length + 1) como la EMA)."""
    seeded = values.copy()
    seeded.iloc[:length] = np.nan  # fila 0 no tiene variación; 1..length-1 van a la semilla
    seeded.iloc[length] = values.iloc[1 : length + 1].mean()
    return seeded.ewm(alpha=1 / length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI de Wilder (0-100). Las primeras `length` filas son NaN."""
    _check_enough_rows(close, length + 1, f"RSI{length}")
    delta = close.astype(float).diff()
    avg_gain = _wilder_average(delta.clip(lower=0), length)
    avg_loss = _wilder_average((-delta).clip(lower=0), length)
    return 100 * avg_gain / (avg_gain + avg_loss)
