"""Fase 2: reproduce el modelo publicado en el post, ahora de forma reproducible.

Es el punto de partida contra el que se medirá el modelo nuevo. Recuerda que
este modelo NO predice el mercado: imita la estrategia de reglas.

Uso (desde `ml/`):
    python -m btc_decision_tree.baseline

Regenerar el modelo de la gráfica del post original (fase 5):
    python -m btc_decision_tree.baseline --end 2025-12-13 --random-state 2 \\
        --save ../models/decision_tree_sklearn-1.9.1.joblib

- `--end 2025-12-13`: los datos que tenía el original, entrenado ese día.
- `--random-state 2`: elegida por FIDELIDAD, no por rendimiento. El original
  se entrenó sin `random_state` y resolvió al azar un empate entre dos cortes
  (RSI o Close, ambos con las dos ramas en −1). La semilla 2 es la primera
  que resuelve ese empate igual: `export_text` sale idéntico al de
  `models/decision_tree.plk`. Las predicciones son las mismas con cualquier
  semilla.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from btc_decision_tree.data import DEFAULT_SNAPSHOT, load_snapshot
from btc_decision_tree.features_original import ORIGINAL_FEATURES, add_original_features
from btc_decision_tree.labels import rule_signals

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def build_dataset(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features originales (sin filas con NaN) y señales de la estrategia."""
    df = add_original_features(prices)
    X = df[ORIGINAL_FEATURES].dropna()
    y = rule_signals(df).loc[X.index]
    return X, y


def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE) -> Split:
    """Primer 80 % para entrenar, último 20 % para test, sin barajar."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    return Split(X_train, X_test, y_train, y_test)


def train_tree(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """Mismos hiperparámetros que el post, más `random_state` fijo."""
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
    )
    return clf.fit(X, y)


def describe_period(index: pd.Index) -> str:
    return f"{index[0].date()} → {index[-1].date()} ({len(index)} días)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce el árbol publicado.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--end", help="Usar solo datos hasta esta fecha (incluida)")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--save", type=Path, help="Guardar el árbol entrenado (joblib)")
    args = parser.parse_args()

    prices = load_snapshot(args.data)
    if args.end:
        prices = prices.loc[: args.end]
    X, y = build_dataset(prices)
    split = temporal_split(X, y)
    clf = train_tree(split.X_train, split.y_train, random_state=args.random_state)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, args.save)
        print(f"Modelo guardado en {args.save}")
    y_pred = clf.predict(split.X_test)

    print(f"Train: {describe_period(split.X_train.index)}")
    print(f"Test:  {describe_period(split.X_test.index)}")
    print("\nClases en test:", split.y_test.value_counts().sort_index().to_dict())
    print("\n" + classification_report(split.y_test, y_pred, zero_division=0))
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Importancia de las features:")
    print(importances.round(3).to_string())


if __name__ == "__main__":
    main()
