"""Features del modelo publicado (solo para baseline.py).

Fase 2: mismos cálculos que el modelo publicado (pandas_ta, EMA sobre HL2).
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

EMA_LENGTH = 400
RSI_LENGTH = 14

# Features del modelo publicado. `Close` y `EMA400` son precios en bruto
# (no estacionarios): se mantienen solo para reproducir el original.
ORIGINAL_FEATURES = ["RSI", "EMA400", "Close", "dist_ema400"]


def hl2(df: pd.DataFrame) -> pd.Series:
    """Punto medio de la vela: (High + Low) / 2."""
    return (df["High"] + df["Low"]) / 2


def ema(series: pd.Series, length: int) -> pd.Series:
    """EMA de pandas_ta (arranca con la SMA de las primeras `length` velas).

    Importante: el parámetro es `length=`. pandas_ta ignora en silencio
    `window=` y usa su valor por defecto (10), que fue el bug de
    `cleanPipeline.ipynb`.
    """
    result = ta.ema(series, length=length)
    if result is None:  # pandas_ta devuelve None si no hay suficientes filas
        raise ValueError(f"EMA{length} necesita al menos {length} filas; hay {len(series)}")
    return result


def rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI de pandas_ta (Wilder)."""
    result = ta.rsi(series, length=length)
    if result is None:
        raise ValueError(f"RSI{length} necesita al menos {length} filas; hay {len(series)}")
    return result


def add_original_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Añade HL2, EMA400, RSI y dist_ema400 a una copia de `prices`."""
    df = prices.copy()
    df["HL2"] = hl2(df)
    df["EMA400"] = ema(df["HL2"], EMA_LENGTH)
    df["RSI"] = rsi(df["Close"], RSI_LENGTH)
    df["dist_ema400"] = df["Close"] / df["EMA400"] - 1
    return df
