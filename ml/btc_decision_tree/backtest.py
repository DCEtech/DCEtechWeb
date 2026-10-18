"""Fase 4: backtest de los baselines (buy & hold, reglas originales y volatilidad).

Dos pasos separados para poder testearlos a mano:
1. Planificar operaciones (`plan_*`): con las señales al cierre de t se decide
   entrar o salir en la apertura de t+1. Solo se usan datos hasta t.
2. Simular (`simulate`): retorno diario con la cartera valorada al cierre y
   una comisión de `fee` por lado.

Cada periodo se simula por separado y empieza fuera del mercado; lo que siga
abierto al final se liquida al cierre del último día (con comisión).

Uso (desde `ml/`):
    python -m btc_decision_tree.backtest
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from btc_decision_tree.data import DEFAULT_SNAPSHOT, load_snapshot
from btc_decision_tree.features_original import add_original_features
from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.labels import BUY, SELL, rule_signals
from btc_decision_tree.model import HORIZON, N_SPLITS, VOL_WINDOW, build_dataset, split_train_test

FEE = 0.001
HOLD_DAYS = 30
TRAIN = ("2015-10-21", "2024-08-04")
TEST = ("2024-08-05", "2026-10-16")
POST_PUBLICATION = ("2025-12-13", "2026-10-16")  # fecha del post original


@dataclass(frozen=True)
class Trade:
    """Operación en posiciones del periodo: entra en la apertura de `entry` y
    sale en la apertura de `exit`; `exit=None` = liquidar al cierre del último día."""

    entry: int
    exit: int | None


@dataclass(frozen=True)
class Simulation:
    returns: pd.Series   # retorno diario de la cartera
    invested: pd.Series  # True si al cierre del día hay posición abierta
    trades: list[Trade]


# --- 1. Planificación -----------------------------------------------------

def plan_buy_and_hold(n_days: int) -> list[Trade]:
    """Entra en la apertura del primer día y mantiene hasta el final."""
    return [Trade(0, None)] if n_days > 0 else []


def plan_rules(signal: Sequence[int]) -> list[Trade]:
    """Largo con la señal de compra si se está fuera; salida con la de venta."""
    trades: list[Trade] = []
    entry: int | None = None
    for i, value in enumerate(signal[:-1]):  # la señal del último día no tiene apertura siguiente
        if entry is None and value == BUY:
            entry = i + 1
        elif entry is not None and entry <= i and value == SELL:
            trades.append(Trade(entry, i + 1))
            entry = None
    if entry is not None:
        trades.append(Trade(entry, None))
    return trades


def plan_hold_after_signal(
    condition: Sequence[bool], hold_days: int = HOLD_DAYS, cooldown: int = HOLD_DAYS
) -> list[Trade]:
    """Señal en i → entrada en i+1 y salida en i+1+hold_days. No se acepta otra
    señal hasta i+cooldown (la misma regla de bloqueo que C4)."""
    n_days = len(condition)
    trades: list[Trade] = []
    next_allowed = 0
    for i in range(n_days - 1):
        if condition[i] and i >= next_allowed:
            exit_day = i + 1 + hold_days
            trades.append(Trade(i + 1, exit_day if exit_day <= n_days - 1 else None))
            next_allowed = i + cooldown
    return trades


# --- 2. Simulación --------------------------------------------------------

def _validate_trades(trades: Sequence[Trade], n_days: int) -> None:
    previous_exit = 0
    for trade in trades:
        if not 0 <= trade.entry < n_days:
            raise ValueError(f"Entrada fuera del periodo: {trade}")
        if trade.exit is not None and not trade.entry < trade.exit <= n_days - 1:
            raise ValueError(f"Salida no válida: {trade}")
        if trade.entry < previous_exit:
            raise ValueError(f"Operaciones solapadas: {trade}")
        previous_exit = n_days if trade.exit is None else trade.exit


def simulate(prices: pd.DataFrame, trades: Sequence[Trade], fee: float = FEE) -> Simulation:
    """Retornos diarios de una lista de operaciones sobre `prices` (un periodo)."""
    n_days = len(prices)
    _validate_trades(trades, n_days)
    open_ = prices["Open"].to_numpy(dtype=float)
    close = prices["Close"].to_numpy(dtype=float)
    growth = np.ones(n_days)
    invested = np.zeros(n_days, dtype=bool)

    for trade in trades:
        e = trade.entry
        growth[e] *= close[e] / open_[e] * (1 - fee)
        last_held = n_days - 1 if trade.exit is None else trade.exit - 1
        for d in range(e + 1, last_held + 1):
            growth[d] *= close[d] / close[d - 1]
        invested[e : last_held + 1] = True
        if trade.exit is None:
            growth[n_days - 1] *= 1 - fee  # liquidación al cierre del último día
        else:
            x = trade.exit
            growth[x] *= open_[x] / close[x - 1] * (1 - fee)

    return Simulation(
        returns=pd.Series(growth - 1, index=prices.index, name="return"),
        invested=pd.Series(invested, index=prices.index, name="invested"),
        trades=list(trades),
    )


# --- 3. Métricas ----------------------------------------------------------

def metrics(sim: Simulation) -> dict[str, float]:
    """CAGR, máximo drawdown, Sharpe (365), % de tiempo invertido y nº de operaciones."""
    returns = sim.returns
    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    std = returns.std(ddof=1)
    return {
        "CAGR": float(equity.iloc[-1] ** (365.25 / len(returns)) - 1),
        "max_drawdown": float((equity.to_numpy() / peak - 1).min()),
        "sharpe": float(returns.mean() / std * np.sqrt(365)) if std > 0 else np.nan,
        "pct_invertido": float(sim.invested.mean()),
        "operaciones": len(sim.trades),
    }


# --- 4. Estrategias sobre un periodo --------------------------------------

def rules_trades(signals: pd.Series, period: pd.Index) -> list[Trade]:
    """Reglas originales; `signals` se calcula sobre toda la historia."""
    return plan_rules(signals.reindex(period).fillna(0).astype(int).tolist())


def vol_trades(vol: pd.Series, median: float | pd.Series, period: pd.Index) -> list[Trade]:
    """Entrada si vol_30d < mediana (escalar o serie alineada); NaN = sin señal."""
    vol_period = vol.reindex(period)
    threshold = median.reindex(period) if isinstance(median, pd.Series) else median
    return plan_hold_after_signal((vol_period < threshold).fillna(False).tolist())


def expanding_median(vol: pd.Series) -> pd.Series:
    """Mediana de vol de todos los días anteriores a t (solo pasado)."""
    return vol.expanding().median().shift(1)


def evaluate_period(
    prices: pd.DataFrame, start: str, end: str, signals: pd.Series, vol: pd.Series,
    median: float | pd.Series, fee: float = FEE,
) -> dict[str, dict[str, float]]:
    """Métricas de las tres estrategias en [start, end]."""
    period_prices = prices.loc[start:end]
    period = period_prices.index
    plans = {
        "buy & hold": plan_buy_and_hold(len(period)),
        "reglas originales": rules_trades(signals, period),
        "baseline vol": vol_trades(vol, median, period),
    }
    return {name: metrics(simulate(period_prices, trades, fee)) for name, trades in plans.items()}


def fold_windows(prices: pd.DataFrame) -> list[tuple[pd.Index, pd.Timestamp, pd.Timestamp]]:
    """(índice de train del fold, inicio y fin de validación) del walk-forward."""
    X, y = build_dataset(prices)
    X_train = split_train_test(X, y).X_train
    folds = TimeSeriesSplit(n_splits=N_SPLITS, gap=HORIZON).split(X_train)
    return [(X_train.index[tr], X_train.index[va[0]], X_train.index[va[-1]]) for tr, va in folds]


def run(prices: pd.DataFrame, fee: float = FEE) -> pd.DataFrame:
    """Tabla con una fila por (periodo, estrategia)."""
    signals = rule_signals(add_original_features(prices))
    vol = log_return_volatility(prices["Close"], VOL_WINDOW)
    train_median = float(vol.loc[TRAIN[0] : TRAIN[1]].median())

    periods: list[tuple[str, str, str, float | pd.Series]] = [
        ("train (mediana expansiva)", *TRAIN, expanding_median(vol)),
    ]
    for k, (fold_train, start, end) in enumerate(fold_windows(prices), start=1):
        periods.append((f"fold {k}", str(start.date()), str(end.date()), float(vol.loc[fold_train].median())))
    periods.append(("test", *TEST, train_median))
    periods.append(("post-publicación", *POST_PUBLICATION, train_median))

    rows = []
    for name, start, end, median in periods:
        for strategy, values in evaluate_period(prices, start, end, signals, vol, median, fee).items():
            rows.append({"periodo": name, "desde": start, "hasta": end, "estrategia": strategy, **values})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest de los baselines.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    pd.set_option("display.width", 160)

    table = run(load_snapshot(args.data))
    formatters = {
        "CAGR": "{:.1%}".format,
        "max_drawdown": "{:.1%}".format,
        "sharpe": "{:.2f}".format,
        "pct_invertido": "{:.0%}".format,
    }
    for name, group in table.groupby("periodo", sort=False):
        start, end = group.iloc[0][["desde", "hasta"]]
        print(f"\n=== {name}: {start} → {end} ===")
        print(group.drop(columns=["periodo", "desde", "hasta"]).to_string(index=False, formatters=formatters))
    print("\nNota: las reglas originales son in-sample en train y test; solo el "
          "periodo post-publicación está fuera de muestra para ellas.")


if __name__ == "__main__":
    main()
