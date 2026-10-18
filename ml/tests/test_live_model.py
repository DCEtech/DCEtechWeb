"""El modelo de la gráfica del post original: regenerado, debe ser el mismo árbol."""

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, export_text

from btc_decision_tree.data import load_snapshot
from btc_decision_tree.features import RULE_FEATURES, add_rule_features
from btc_decision_tree.labels import BUY, HOLD, rule_signals
from btc_decision_tree.paths import LIVE_MODEL_PATH, ORIGINAL_MODEL_PATH

PLOT_START = "2022-12-01"


def load_original() -> DecisionTreeClassifier:
    # El original es de scikit-learn 1.6.1: avisa al cargarlo con otra versión.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(ORIGINAL_MODEL_PATH)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return add_rule_features(load_snapshot())


def test_regenerated_tree_text_is_identical() -> None:
    original, live = load_original(), joblib.load(LIVE_MODEL_PATH)
    names = list(original.feature_names_in_)
    assert export_text(live, feature_names=names, decimals=10) == export_text(
        original, feature_names=names, decimals=10
    )


def test_regenerated_tree_predicts_like_original_on_full_history(features: pd.DataFrame) -> None:
    # (a) Mismas X para los dos modelos, calculadas sin pandas_ta.
    X = features[RULE_FEATURES].dropna()
    original, live = load_original(), joblib.load(LIVE_MODEL_PATH)
    np.testing.assert_array_equal(live.predict(X), original.predict(X))


def test_disagreements_with_rules_are_the_two_known_zones(features: pd.DataFrame) -> None:
    # (b) El árbol no copia las reglas al 100 %. Cada diferencia debe caer en
    # una de las dos zonas explicadas en el plan (fase 5, hallazgos).
    live = joblib.load(LIVE_MODEL_PATH)
    rsi_cut = float(live.tree_.threshold[0])  # raíz: RSI <= 29.94
    assert live.feature_names_in_[live.tree_.feature[0]] == "RSI"

    X = features[RULE_FEATURES].dropna()
    model = pd.Series(live.predict(X), index=X.index)
    rules = rule_signals(features).loc[X.index]
    differ = model != rules
    rsi, dist = X["RSI"], X["dist_ema400"]

    # Zona 1: el árbol compra con RSI bajo aunque el precio esté entre la EMA400
    # y +15 % (franja que las reglas excluyen).
    zone_1 = (model == BUY) & (rules == HOLD) & (rsi <= rsi_cut) & (dist > 0) & (dist <= 0.15)
    # Zona 2: las reglas compran con RSI entre el corte del árbol y 30.
    zone_2 = (model == HOLD) & (rules == BUY) & (rsi > rsi_cut) & (rsi <= 30)

    unexplained = differ & ~(zone_1 | zone_2)
    assert not unexplained.any(), X[unexplained]
    assert differ.loc[PLOT_START:].sum() == 5  # lo medido al planificar la fase 5


def test_rule_features_match_pandas_ta_in_plotted_window(features: pd.DataFrame) -> None:
    pytest.importorskip("pandas_ta")
    from btc_decision_tree.features_original import add_original_features

    original = add_original_features(load_snapshot())
    pd.testing.assert_frame_equal(
        features.loc[PLOT_START:, RULE_FEATURES],
        original.loc[PLOT_START:, RULE_FEATURES],
        rtol=1e-9,
    )
