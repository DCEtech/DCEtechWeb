import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.model import (
    build_dataset,
    evaluate_fold,
    split_train_test,
    success_criteria,
    vol_baseline_predict,
    walk_forward,
)


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
    prices = make_prices()
    X, y = build_dataset(prices, horizon=30)
    vol = log_return_volatility(prices["Close"], 30)
    results = walk_forward(X, y, vol, max_depths=[2, 3], min_samples_leafs=[20], n_splits=3, gap=30)
    assert len(results) == 3 * 2
    assert results["base_rate"].between(0, 1).all()
    # El baseline no depende de los hiperparámetros: igual en todas las filas de un fold.
    assert results.groupby("fold")["vol_base_lift"].nunique().eq(1).all()


def test_vol_baseline_uses_train_median() -> None:
    vol_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # mediana 3
    vol_val = pd.Series([10.0, 2.9, 3.0, 0.5])         # su mediana sería 2.95
    assert vol_baseline_predict(vol_train, vol_val).tolist() == [0, 1, 0, 1]


def fold_rows(lifts: list[float], base_lifts: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"lift": lifts, "vol_base_lift": base_lifts})


def test_success_criteria_all_pass() -> None:
    result = success_criteria(fold_rows([0.05, 0.04, 0.03, 0.02, -0.01], [0.0] * 5))
    assert result["folds_lift_pos"] == 4
    assert result["c1_lift_pos_4de5"] and result["c2_media_gt_std"] and result["c3_gana_baseline"]
    assert result["cumple"]


def test_success_criteria_fails_with_few_positive_folds() -> None:
    result = success_criteria(fold_rows([0.10, -0.01, 0.10, -0.01, 0.10], [0.0] * 5))
    assert not result["c1_lift_pos_4de5"]
    assert not result["cumple"]


def test_success_criteria_fails_when_mean_below_std() -> None:
    # Todos positivos, pero muy dispersos: la media no supera la desviación.
    result = success_criteria(fold_rows([0.20, 0.001, 0.001, 0.001, 0.001], [0.0] * 5))
    assert result["c1_lift_pos_4de5"]
    assert not result["c2_media_gt_std"]
    assert not result["cumple"]


def test_success_criteria_fails_against_better_baseline() -> None:
    result = success_criteria(fold_rows([0.05, 0.04, 0.03, 0.02, 0.01], [0.06] * 5))
    assert result["c1_lift_pos_4de5"] and result["c2_media_gt_std"]
    assert not result["c3_gana_baseline"]
    assert not result["cumple"]


def test_success_criteria_fold_without_predictions_is_not_positive() -> None:
    result = success_criteria(fold_rows([0.05, float("nan"), 0.03, 0.02, 0.04], [0.0] * 5))
    assert result["folds_lift_pos"] == 4
