"""Fase 3: target a futuro, split temporal con purga y walk-forward en train.

Dos experimentos:
- `return`: retorno a 30 días > +10 % (experimento 1).
- `vol`: log-retorno a 30 días > k · vol_30d · √30, con k calibrado solo con
  la tasa base del train (experimento 2).

El periodo de test se separa aquí pero NO se evalúa: se usará una única vez,
al final, con el candidato ya elegido.

Uso (desde `ml/`):
    python -m btc_decision_tree.model                 # experimento 1
    python -m btc_decision_tree.model --target vol    # experimento 2
    python -m btc_decision_tree.model --target both   # los dos + veredictos
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
from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.labels import calibrate_k, forward_return_target, vol_adjusted_target

HORIZON = 30
THRESHOLD = 0.10
VOL_WINDOW = 30
K_TARGET_RATE = 0.375
K_RATE_RANGE = (0.37, 0.38)
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


@dataclass(frozen=True)
class Experiment:
    name: str
    split: TrainTest
    vol: pd.Series  # volatilidad de log-retornos, para el baseline
    note: str = ""


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


def prepare_experiment(prices: pd.DataFrame, target: str) -> Experiment:
    """Construye el split del experimento `target` ("return" o "vol")."""
    X, y_return = build_dataset(prices)
    split = split_train_test(X, y_return)
    vol = log_return_volatility(prices["Close"], VOL_WINDOW)
    if target == "return":
        return Experiment("return >+10 %", split, vol)
    if target != "vol":
        raise ValueError(f"target desconocido: {target}")

    # k se calibra UNA vez con la tasa base de todo el train (incluye las
    # ventanas de validación): solo fija la proporción de clases.
    k, rate = calibrate_k(prices["Close"], split.X_train.index, K_TARGET_RATE, HORIZON, VOL_WINDOW)
    low, high = K_RATE_RANGE
    if not low <= rate <= high:
        raise ValueError(f"Con k={k} la tasa base del train es {rate:.1%}, fuera de {K_RATE_RANGE}")
    y_vol = vol_adjusted_target(prices["Close"], k, HORIZON, VOL_WINDOW).reindex(X.index)
    if y_vol.isna().any():
        raise ValueError("Hay filas del dataset sin target ajustado por volatilidad")
    split = split_train_test(X, y_vol.astype(int))
    return Experiment(f"vol k={k:.2f}", split, vol, f"k={k:.2f} (tasa base train {rate:.1%})")


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


def vol_baseline_predict(vol_train: pd.Series, vol_val: pd.Series) -> np.ndarray:
    """Baseline: 1 si la volatilidad está por debajo de la mediana del train del fold."""
    return (vol_val < vol_train.median()).astype(int).to_numpy()


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    vol: pd.Series,
    max_depths: Iterable[int] = MAX_DEPTHS,
    min_samples_leafs: Iterable[int] = MIN_SAMPLES_LEAFS,
    n_splits: int = N_SPLITS,
    gap: int = HORIZON,
) -> pd.DataFrame:
    """Una fila por (fold, max_depth, min_samples_leaf) con las métricas del
    árbol y las del baseline de volatilidad en ese mismo fold."""
    rows = []
    folds = TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X)
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        baseline = evaluate_fold(
            y_val, vol_baseline_predict(vol.loc[X_tr.index], vol.loc[X_val.index])
        )
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
                    "vol_base_precision": baseline["precision_1"],
                    "vol_base_lift": baseline["lift"],
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


def success_criteria(fold_rows: pd.DataFrame) -> dict[str, float | bool]:
    """Los 3 criterios de éxito pre-registrados, para las filas (una por fold)
    de una combinación. Un fold sin predicciones (lift NaN) no cuenta como
    lift positivo; media y desviación se calculan sobre los folds con lift."""
    lifts = fold_rows["lift"]
    positive = int((lifts > 0).sum())
    lift_mean = float(lifts.mean())
    lift_std = float(lifts.std(ddof=1))
    base_mean = float(fold_rows["vol_base_lift"].mean())
    criteria = {
        "folds_lift_pos": positive,
        "c1_lift_pos_4de5": positive >= 4,
        "lift_mean": lift_mean,
        "lift_std": lift_std,
        "c2_media_gt_std": lift_mean > lift_std,
        "vol_base_lift_mean": base_mean,
        "c3_gana_baseline": lift_mean > base_mean,
    }
    criteria["cumple"] = bool(
        criteria["c1_lift_pos_4de5"] and criteria["c2_media_gt_std"] and criteria["c3_gana_baseline"]
    )
    return criteria


def combo_rows(results: pd.DataFrame, depth: int, leaf: int) -> pd.DataFrame:
    return results[(results["max_depth"] == depth) & (results["min_samples_leaf"] == leaf)]


def verdict(name: str, results: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    """Veredicto de la combinación elegida (mejor lift medio) y datos informativos."""
    depth, leaf = summary.index[0]
    passing = sum(
        success_criteria(combo_rows(results, d, l))["cumple"] for d, l in summary.index
    )
    return {
        "experimento": name,
        "elegida": f"{depth}/{leaf}",
        **success_criteria(combo_rows(results, depth, leaf)),
        "combos_que_cumplen": f"{passing}/{len(summary)}",
        "lift_mean_mediana": float(summary["lift_mean"].median()),
    }


def describe(name: str, index: pd.Index) -> str:
    return f"{name}: {index[0].date()} → {index[-1].date()} ({len(index)} filas)"


def report(exp: Experiment, results: pd.DataFrame, summary: pd.DataFrame) -> None:
    split = exp.split
    print(f"\n{'=' * 30} Experimento: {exp.name} {'=' * 30}")
    if exp.note:
        print(exp.note)
    print(describe("Train", split.X_train.index), f"| tasa base {split.y_train.mean():.1%}")
    print(describe("Test ", split.X_test.index), "| NO evaluado")

    folds = results.groupby("fold")[["val_start", "val_end", "base_rate"]].first()
    print("\nFolds de validación (walk-forward, gap = 30):")
    print(folds.to_string(formatters={"base_rate": "{:.1%}".format}))

    print("\nResumen por combinación (ordenado por lift medio):")
    print(summary.round(3).to_string())

    depth, leaf = summary.index[0]
    print(f"\nDetalle por fold de la elegida (max_depth={depth}, min_samples_leaf={leaf}):")
    columns = ["fold", "base_rate", "n_pred_1", "precision_1", "lift",
               "vol_base_precision", "vol_base_lift"]
    print(combo_rows(results, depth, leaf)[columns].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward del árbol con target a futuro.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--target", choices=["return", "vol", "both"], default="return")
    args = parser.parse_args()
    pd.set_option("display.width", 160)

    prices = load_snapshot(args.data)
    targets = ["return", "vol"] if args.target == "both" else [args.target]
    verdicts = []
    for target in targets:
        exp = prepare_experiment(prices, target)
        results = walk_forward(exp.split.X_train, exp.split.y_train, exp.vol)
        summary = summarize(results)
        report(exp, results, summary)
        verdicts.append(verdict(exp.name, results, summary))

    print(f"\n{'=' * 30} Veredictos (criterios pre-registrados) {'=' * 30}")
    print(pd.DataFrame(verdicts).set_index("experimento").T.to_string())


if __name__ == "__main__":
    main()
