import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.model import build_dataset, evaluate_fold, split_train_test, walk_forward


def make_prices(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


def test_dataset_ends_horizon_rows_before_data_end() -> None:
    prices = make_prices()
    X, y = build_dataset(prices, horizon=30)
    assert X.index[-1] == prices.index[-31]
    assert X.index.equals(y.index)
    assert X.notna().all().all()


def test_split_purges_train_and_does_not_overlap() -> None:
    X, y = build_dataset(make_prices(), horizon=30)
    split = split_train_test(X, y, train_end="2022-12-31", test_start="2023-01-01", purge=30)

    assert split.X_train.index[-1] < split.X_test.index[0]
    # Entre la última fila de train y la primera de test hay al menos 30 filas
    # (las purgadas), así ninguna etiqueta de train mira dentro del test.
    gap_rows = X.index.get_loc(split.X_test.index[0]) - X.index.get_loc(split.X_train.index[-1])
    assert gap_rows > 30


def test_split_raises_when_train_is_empty() -> None:
    X, y = build_dataset(make_prices(), horizon=30)
    with pytest.raises(ValueError):
        split_train_test(X, y, train_end="2000-01-01", test_start="2001-01-01")


def test_evaluate_fold_without_positive_predictions() -> None:
    metrics = evaluate_fold(pd.Series([1, 0, 0, 0]), np.array([0, 0, 0, 0]))
    assert metrics["base_rate"] == 0.25
    assert metrics["n_pred_1"] == 0
    assert np.isnan(metrics["precision_1"])


def test_evaluate_fold_lift() -> None:
    metrics = evaluate_fold(pd.Series([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert metrics["precision_1"] == 1.0
    assert metrics["lift"] == 0.5


def test_walk_forward_shape() -> None:
    X, y = build_dataset(make_prices(), horizon=30)
    results = walk_forward(X, y, max_depths=[2, 3], min_samples_leafs=[20], n_splits=3, gap=30)
    assert len(results) == 3 * 2
    assert results["base_rate"].between(0, 1).all()
