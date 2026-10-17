"""Features relativas para el modelo de la fase 3 (sin pandas_ta).

Todas son ratios, retornos u osciladores: no dependen del nivel de precio,
así que "+20 % sobre la EMA400" significa lo mismo en 2017 que en 2025. Cada
valor en t usa solo datos hasta t (lo comprueba `tests/test_no_leakage.py`).
"""

from __future__ import annotations

import pandas as pd

from btc_decision_tree.indicators import ema, rsi_wilder

EMA_LONG = 400
EMA_MID = 55
SLOPE_DAYS = 20
RSI_LENGTH = 14
RETURN_DAYS = (1, 7, 30)
VOL_WINDOW = 30

RELATIVE_FEATURES = [
    "dist_ema400",
    "dist_ema55",
    "ema400_slope_20d",
    "rsi14",
    "ret_1d",
    "ret_7d",
    "ret_30d",
    "vol_30d",
]


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Cambio porcentual respecto a `periods` filas antes."""
    return series / series.shift(periods) - 1


def add_relative_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame (mismo índice que `prices`) con RELATIVE_FEATURES.

    Las primeras filas tienen NaN mientras los indicadores se calientan:
    EMA400 necesita 400 filas y su pendiente 20 más.
    """
    close = prices["Close"]
    hl2 = (prices["High"] + prices["Low"]) / 2
    ema_long = ema(hl2, EMA_LONG)
    ema_mid = ema(hl2, EMA_MID)

    features = pd.DataFrame(index=prices.index)
    features["dist_ema400"] = close / ema_long - 1
    features["dist_ema55"] = close / ema_mid - 1
    features["ema400_slope_20d"] = pct_change(ema_long, SLOPE_DAYS)
    features["rsi14"] = rsi_wilder(close, RSI_LENGTH)
    for days in RETURN_DAYS:
        features[f"ret_{days}d"] = pct_change(close, days)
    features["vol_30d"] = pct_change(close, 1).rolling(VOL_WINDOW).std()
    return features[RELATIVE_FEATURES]
