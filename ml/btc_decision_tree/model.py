"""Fase 3: target a 30 días, split temporal con purga y walk-forward en train.

El periodo de test se separa aquí pero NO se evalúa: se usará una única vez,
al final, con los hiperparámetros ya elegidos.

Uso (desde `ml/`):
    python -m btc_decision_tree.model
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from btc_decision_tree.data import DEFAULT_SNAPSHOT, load_snapshot
from btc_decision_tree.features import add_relative_features
from btc_decision_tree.labels import forward_return_target

HORIZON = 30
THRESHOLD = 0.10
TRAIN_END = "2024-08-04"
TEST_START = "2024-08-05"
N_SPLITS = 5
MAX_DEPTHS = range(2, 7)
MIN_SAMPLES_LEAFS = (20, 50, 100, 200)
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainTest:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def build_dataset(
    prices: pd.DataFrame, horizon: int = HORIZON, threshold: float = THRESHOLD
) -> tuple[pd.DataFrame, pd.Series]:
    """Filas con todas las features calculadas y con target conocido."""
    features = add_relative_features(prices).dropna()
    target = forward_return_target(prices["Close"], horizon, threshold)
    rows = features.index.intersection(target.index)
    return features.loc[rows], target.loc[rows]


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    purge: int = HORIZON,
) -> TrainTest:
    """Train hasta `train_end` sin sus últimas `purge` filas; test desde `test_start`.

    La purga evita que las etiquetas del final del train, que miran `purge`
    días hacia delante, contengan información del periodo de test.
    """
    in_train = X.index <= pd.Timestamp(train_end)
    in_test = X.index >= pd.Timestamp(test_start)
    keep = int(in_train.sum()) - purge
    if keep <= 0 or not in_test.any():
        raise ValueError("El train o el test quedan vacíos con estas fechas")
    return TrainTest(
        X_train=X[in_train].iloc[:keep],
        y_train=y[in_train].iloc[:keep],
        X_test=X[in_test],
        y_test=y[in_test],
    )


def evaluate_fold(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Tasa base, nº de predicciones de clase 1, precision y lift."""
    base_rate = float(y_true.mean())
    n_pred_1 = int((y_pred == 1).sum())
    precision = float(y_true[y_pred == 1].mean()) if n_pred_1 else np.nan
    return {
        "base_rate": base_rate,
        "n_pred_1": n_pred_1,
        "precision_1": precision,
        "lift": precision - base_rate,
    }


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    max_depths: Iterable[int] = MAX_DEPTHS,
    min_samples_leafs: Iterable[int] = MIN_SAMPLES_LEAFS,
    n_splits: int = N_SPLITS,
    gap: int = HORIZON,
) -> pd.DataFrame:
    """Una fila por (fold, max_depth, min_samples_leaf) con sus métricas."""
    rows = []
    folds = TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X)
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        for depth in max_depths:
            for leaf in min_samples_leafs:
                clf = DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE
                ).fit(X_tr, y_tr)
                rows.append({
                    "fold": fold,
                    "val_start": X_val.index[0].date(),
                    "val_end": X_val.index[-1].date(),
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    **evaluate_fold(y_val, clf.predict(X_val)),
                })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Media y peor fold de cada combinación, ordenadas por lift medio.

    Un fold sin ninguna predicción de clase 1 no tiene precision: se cuenta
    en `folds_sin_pred` en vez de puntuar como acierto o fallo.
    """
    grouped = results.groupby(["max_depth", "min_samples_leaf"])
    summary = pd.DataFrame({
        "lift_mean": grouped["lift"].mean(),
        "lift_min": grouped["lift"].min(),
        "precision_mean": grouped["precision_1"].mean(),
        "n_pred_1_mean": grouped["n_pred_1"].mean(),
        "folds_sin_pred": grouped["precision_1"].apply(lambda s: int(s.isna().sum())),
    })
    return summary.sort_values("lift_mean", ascending=False)


def describe(name: str, index: pd.Index) -> str:
    return f"{name}: {index[0].date()} → {index[-1].date()} ({len(index)} filas)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward del árbol con target a futuro.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()

    X, y = build_dataset(load_snapshot(args.data))
    split = split_train_test(X, y)
    print(describe("Train", split.X_train.index), f"| tasa base {split.y_train.mean():.1%}")
    print(describe("Test ", split.X_test.index), "| NO evaluado")

    results = walk_forward(split.X_train, split.y_train)
    summary = summarize(results)

    folds = results.groupby("fold")[["val_start", "val_end", "base_rate"]].first()
    print("\nFolds de validación (walk-forward, gap = 30):")
    print(folds.to_string(formatters={"base_rate": "{:.1%}".format}))

    pd.set_option("display.width", 120)
    print("\nResumen por combinación (ordenado por lift medio):")
    print(summary.round(3).to_string())

    best_depth, best_leaf = summary.index[0]
    best = results[(results["max_depth"] == best_depth) & (results["min_samples_leaf"] == best_leaf)]
    print(f"\nDetalle por fold de la mejor combinación (max_depth={best_depth}, min_samples_leaf={best_leaf}):")
    columns = ["fold", "val_start", "val_end", "base_rate", "n_pred_1", "precision_1", "lift"]
    print(best[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
