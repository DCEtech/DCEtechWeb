from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from btc_decision_tree import live
from btc_decision_tree.backtest import POST_PUBLICATION, evaluate_period, expanding_median, simulate
from btc_decision_tree.data import DataError, load_snapshot
from btc_decision_tree.features import add_rule_features
from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.labels import rule_signals
from btc_decision_tree.paths import LIVE_MODEL_PATH


def random_prices(n: int = 900, seed: int = 5, sigma: float = 0.04) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.005, n))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * 1.01,
         "Low": np.minimum(open_, close) * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


# --- tracker --------------------------------------------------------------

def test_tracker_final_return_is_the_product_of_daily_returns() -> None:
    prices = random_prices()
    start = str(prices.index[500].date())
    tracker = live.build_tracker(prices, start=start)

    period = prices.loc[start:]
    hold = simulate(period, live.plan_buy_and_hold(len(period)))
    assert np.isclose(tracker.final["buy_and_hold"], (1 + hold.returns).prod() - 1)

    signals = rule_signals(add_rule_features(prices))
    rules = simulate(period, live.rules_trades(signals, period.index))
    assert np.isclose(tracker.final["rules"], (1 + rules.returns).prod() - 1)
    assert tracker.n_trades == len(rules.trades)


def test_tracker_matches_phase_4_post_publication_on_snapshot() -> None:
    # Mismo backtest que la fase 4: el retorno acumulado del tracker equivale
    # al CAGR del periodo post-publicación de la tabla de la fase 4.
    prices = load_snapshot()
    tracker = live.build_tracker(prices.loc[: POST_PUBLICATION[1]])
    vol = log_return_volatility(prices["Close"], 30)
    table = evaluate_period(prices, *POST_PUBLICATION, rule_signals(add_rule_features(prices)),
                            vol, expanding_median(vol))
    years = len(tracker.curves) / 365.25
    for column, strategy in (("rules", "reglas originales"), ("buy_and_hold", "buy & hold")):
        cagr = (1 + tracker.final[column]) ** (1 / years) - 1
        assert np.isclose(cagr, table[strategy]["CAGR"])
    assert tracker.n_trades == table["reglas originales"]["operaciones"]


# --- gráficas -------------------------------------------------------------

def test_plots_are_written(tmp_path: Path) -> None:
    prices = random_prices()
    model = joblib.load(LIVE_MODEL_PATH)
    start = str(prices.index[500].date())

    live.plot_signals(live.predict_signals(prices, model), tmp_path / "signals.png", start=start)
    live.plot_tracker(live.build_tracker(prices, start=start), tmp_path / "tracker.png", start=start)

    for name in ("signals.png", "tracker.png"):
        content = (tmp_path / name).read_bytes()
        assert content.startswith(b"\x89PNG") and len(content) > 10_000


# --- descarga y código de salida para el CI -------------------------------

def test_check_fresh_rejects_stale_data() -> None:
    prices = random_prices(n=10)  # termina el 2020-01-10
    live.check_fresh(prices, today=date(2020, 1, 12))
    with pytest.raises(DataError, match="antigüedad"):
        live.check_fresh(prices, today=date(2020, 1, 20))


def test_main_exits_with_download_failed_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing_download(*args: object) -> pd.DataFrame:
        raise DataError("sin conexión")

    monkeypatch.setattr(live, "download_ohlcv", failing_download)
    with pytest.raises(SystemExit) as exc_info:
        live.main(["--images-dir", str(tmp_path)])
    assert exc_info.value.code == live.DOWNLOAD_FAILED
    assert not list(tmp_path.iterdir())  # no se escribe ninguna imagen


# --- marcadores y ejes ----------------------------------------------------

def test_tracker_entries_and_shading_come_from_the_trades() -> None:
    prices = random_prices()
    start = str(prices.index[500].date())
    tracker = live.build_tracker(prices, start=start)

    period = prices.loc[start:]
    signals = rule_signals(add_rule_features(prices))
    sim = simulate(period, live.rules_trades(signals, period.index))
    assert tracker.entries == [period.index[t.entry] for t in sim.trades]
    assert tracker.entries  # con estos datos las reglas operan
    pd.testing.assert_series_equal(tracker.invested, sim.invested)


def test_out_of_market_spans() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="D")
    invested = pd.Series([False, False, True, True, False, True, False], index=index)
    assert live.out_of_market_spans(invested) == [
        (index[0], index[1]), (index[4], index[4]), (index[6], index[6]),
    ]
    assert live.out_of_market_spans(pd.Series(True, index=index)) == []


def test_axis_formats() -> None:
    assert live.format_usd_thousands(20_000) == "$20k"
    assert live.format_usd_thousands(125_000) == "$125k"
    ten_months = pd.date_range("2025-12-13", "2026-10-17", freq="D")
    assert live.month_interval(ten_months) == 2      # 11 meses → una fecha cada 2
    assert live.month_interval(ten_months[:60]) == 1


def test_simulated_data_failure_exits_with_download_failed_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def must_not_download(*args: object) -> pd.DataFrame:
        raise AssertionError("con la simulación activa no debe llegar a descargar")

    monkeypatch.setattr(live, "download_ohlcv", must_not_download)
    monkeypatch.setenv(live.SIMULATE_ENV, "true")
    with pytest.raises(SystemExit) as exc_info:
        live.main(["--images-dir", str(tmp_path)])
    assert exc_info.value.code == live.DOWNLOAD_FAILED


@pytest.mark.parametrize("value", ["", "false"])
def test_simulation_is_off_unless_true(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(live.SIMULATE_ENV, value)
    monkeypatch.setattr(live, "download_ohlcv", lambda *args: random_prices(n=10))
    with pytest.raises(DataError, match="antigüedad"):  # descarga de verdad (datos de 2020)
        live.download_history(date(2026, 10, 18))
