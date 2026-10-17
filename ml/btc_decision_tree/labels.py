"""Etiquetas (targets) para los modelos.

- `rule_signals`: la estrategia de reglas original (EMA400 + RSI). OJO: se
  calcula con las mismas columnas que el modelo recibe como features (RSI,
  Close, EMA400), por eso el árbol solo reaprende las reglas y saca un 99 %
  de accuracy sin predecir nada. Solo se usa en el baseline de la fase 2.
- `forward_return_target`: target de la fase 3. Mira el precio futuro, que
  es justo lo que el modelo no puede ver en sus features.
"""

from __future__ import annotations

import pandas as pd

BUY, HOLD, SELL = 1, 0, -1


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
