import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.model import (
    bet_entries,
    build_dataset,
    evaluate_bets,
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


def fold_rows(
    lifts: list[float], base_lifts: list[float], n_bets: list[int] | None = None
) -> pd.DataFrame:
    bets = n_bets if n_bets is not None else [5] * len(lifts)
    return pd.DataFrame({"lift": lifts, "vol_base_lift": base_lifts, "n_bets": bets})


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


# --- apuestas (C4) --------------------------------------------------------

def test_bet_entries_without_predictions() -> None:
    assert bet_entries(np.zeros(50, dtype=int)) == []


def test_bet_entries_long_streak_opens_a_new_bet_after_horizon() -> None:
    pred = np.ones(31, dtype=int)  # 31 días seguidos en 1
    assert bet_entries(pred, cooldown=30) == [0, 30]
    assert bet_entries(np.ones(30, dtype=int), cooldown=30) == [0]


def test_bet_entries_short_gap_does_not_open_a_new_bet() -> None:
    pred = np.array([1] * 5 + [0] * 5 + [1] * 5)  # dos rachas separadas por 5 días
    assert bet_entries(pred, cooldown=30) == [0]


def test_bet_entries_fold_edges() -> None:
    # Cada fold se evalúa por separado: empieza sin apuesta abierta, así que
    # un 1 en la primera fila abre apuesta aunque el fold anterior acabara en 1.
    assert bet_entries(np.array([1, 1, 0, 0]), cooldown=30) == [0]
    # Un 1 en la última fila también abre apuesta.
    assert bet_entries(np.array([0, 0, 0, 1]), cooldown=30) == [3]


def test_evaluate_bets_uses_entry_day_label() -> None:
    y_true = pd.Series([0, 1, 1, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    result = evaluate_bets(y_true, y_pred)  # cooldown 30 → una sola apuesta, el día 0
    assert result == {"n_bets": 1, "bet_precision": 0.0}


def test_c4_fails_with_a_fold_below_three_bets_only_when_required() -> None:
    rows = fold_rows([0.05, 0.04, 0.03, 0.02, -0.01], [0.0] * 5, n_bets=[5, 2, 4, 3, 6])
    informative = success_criteria(rows, require_c4=False)
    assert not informative["c4_3_apuestas"] and informative["cumple"]
    required = success_criteria(rows, require_c4=True)
    assert required["min_apuestas_fold"] == 2 and not required["cumple"]


def test_c4_passes_with_three_bets_in_every_fold() -> None:
    rows = fold_rows([0.05, 0.04, 0.03, 0.02, -0.01], [0.0] * 5, n_bets=[3, 3, 4, 3, 6])
    assert success_criteria(rows, require_c4=True)["cumple"]
