import numpy as np
import pandas as pd
import pytest

from btc_decision_tree.backtest import (
    Simulation,
    Trade,
    expanding_median,
    metrics,
    plan_buy_and_hold,
    plan_hold_after_signal,
    plan_rules,
    rules_trades,
    simulate,
    vol_trades,
)
from btc_decision_tree.features import add_rule_features
from btc_decision_tree.indicators import log_return_volatility
from btc_decision_tree.labels import BUY, HOLD, SELL, rule_signals

FEE = 0.001


def make_prices(opens: list[float], closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(opens), freq="D")
    return pd.DataFrame({"Open": opens, "Close": closes}, index=index)


def random_prices(n: int = 900, seed: int = 0, sigma: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.005, n))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * 1.01,
         "Low": np.minimum(open_, close) * 0.99, "Close": close, "Volume": 1.0},
        index=index,
    )


# --- simulación a mano ----------------------------------------------------

def test_buy_and_hold_by_hand() -> None:
    prices = make_prices(opens=[100, 108, 115], closes=[110, 99, 121])
    sim = simulate(prices, plan_buy_and_hold(3), fee=FEE)
    expected = [110 / 100 * (1 - FEE) - 1, 99 / 110 - 1, 121 / 99 * (1 - FEE) - 1]
    assert np.allclose(sim.returns.to_numpy(), expected)
    total = (1 + sim.returns).prod()
    assert np.isclose(total, 121 / 100 * (1 - FEE) ** 2)
    assert sim.invested.all()


def test_single_trade_by_hand() -> None:
    prices = make_prices(opens=[10, 11, 12, 13, 14, 15], closes=[11, 12, 13, 14, 15, 16])
    sim = simulate(prices, [Trade(entry=2, exit=4)], fee=FEE)
    expected = [0, 0, 13 / 12 * (1 - FEE) - 1, 14 / 13 - 1, 14 / 14 * (1 - FEE) - 1, 0]
    assert np.allclose(sim.returns.to_numpy(), expected)
    # Entra en la apertura de 2 (12) y sale en la apertura de 4 (14).
    assert np.isclose((1 + sim.returns).prod(), 14 / 12 * (1 - FEE) ** 2)
    assert sim.invested.tolist() == [False, False, True, True, False, False]


def test_commissions_charged_on_entry_and_exit() -> None:
    prices = make_prices(opens=[100, 104, 103, 107], closes=[102, 101, 106, 110])
    no_fee = (1 + simulate(prices, plan_buy_and_hold(4), fee=0).returns).prod()
    with_fee = (1 + simulate(prices, plan_buy_and_hold(4), fee=FEE).returns).prod()
    assert np.isclose(no_fee, 110 / 100)  # sin comisión = retorno del precio
    assert np.isclose(with_fee, no_fee * (1 - FEE) ** 2)


def test_exit_and_reentry_on_same_open_pay_two_commissions() -> None:
    prices = make_prices(opens=[10, 11, 12, 13], closes=[11, 12, 13, 14])
    sim = simulate(prices, [Trade(0, 2), Trade(2, None)], fee=FEE)
    day2 = 12 / 12 * (1 - FEE) * (13 / 12) * (1 - FEE) - 1
    assert np.isclose(sim.returns.iloc[2], day2)
    assert np.isclose((1 + sim.returns).prod(), 12 / 10 * (1 - FEE) ** 2 * 14 / 12 * (1 - FEE) ** 2)


def test_simulate_rejects_overlapping_trades() -> None:
    prices = make_prices(opens=[1, 1, 1, 1], closes=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="solapadas"):
        simulate(prices, [Trade(0, 3), Trade(2, None)])


# --- planificación --------------------------------------------------------

def test_plan_rules() -> None:
    signal = [BUY, BUY, HOLD, SELL, SELL, BUY, HOLD, BUY]
    # compra en 0 → entra en 1; la compra de 1 se ignora (ya dentro);
    # venta en 3 → sale en 4; la venta de 4 se ignora (fuera);
    # compra en 5 → entra en 6 y se liquida al final; la señal del último día se ignora.
    assert plan_rules(signal) == [Trade(1, 4), Trade(6, None)]


def test_plan_rules_sell_on_entry_day_exits_next_open() -> None:
    # Compra al cierre de 0 → dentro desde la apertura de 1; venta al cierre de 1 → sale en 2.
    assert plan_rules([BUY, SELL, HOLD]) == [Trade(1, 2)]


def test_plan_hold_after_signal_with_cooldown() -> None:
    trades = plan_hold_after_signal([True] * 70, hold_days=30, cooldown=30)
    assert trades == [Trade(1, 31), Trade(31, 61), Trade(61, None)]


def test_plan_hold_after_signal_last_day_signal_is_ignored() -> None:
    assert plan_hold_after_signal([False, False, True]) == []


# --- sin información de t en t --------------------------------------------

def executed_up_to(trades: list[Trade], t: int) -> list[tuple[int, int | None]]:
    """Entradas y salidas ejecutadas en una apertura ≤ t (salida None = aún abierta en t)."""
    return [
        (tr.entry, tr.exit if tr.exit is not None and tr.exit <= t else None)
        for tr in trades
        if tr.entry <= t
    ]


def shock_from(prices: pd.DataFrame, t: int, factor: float) -> pd.DataFrame:
    """Multiplica Open/High/Low/Close por `factor` desde el día t (incluido)."""
    shocked = prices.copy()
    cols = shocked.columns.get_indexer(["Open", "High", "Low", "Close"])
    shocked.iloc[t:, cols] = shocked.iloc[t:, cols].to_numpy() * factor
    return shocked


RULES_PERIOD_START = 420


def rules_trades_for(prices: pd.DataFrame, lookahead: int = 0) -> list[Trade]:
    """`lookahead=1` simula una estrategia tramposa que usa la señal de mañana."""
    signals = rule_signals(add_rule_features(prices)).shift(-lookahead)
    return rules_trades(signals, prices.index[RULES_PERIOD_START:])


# Con estos datos las reglas operan 5 veces. Un cambio aleatorio en t casi
# nunca altera la señal de las reglas, así que forzamos una: un desplome
# (×0,3) estando fuera crea una compra en t; un pico (×3) estando dentro crea
# una venta en t. Una estrategia honesta no puede reaccionar antes de t+1.
SHOCKS = [(660, 0.3), (1000, 0.3), (900, 3.0), (1300, 3.0)]


@pytest.mark.parametrize(("t", "factor"), SHOCKS)
def test_rules_strategy_does_not_use_day_t(t: int, factor: float) -> None:
    prices = random_prices(n=1500, seed=5, sigma=0.04)
    pos_t = t - RULES_PERIOD_START
    before = executed_up_to(rules_trades_for(prices), pos_t)
    assert before  # el test no puede pasar en vacío
    assert before == executed_up_to(rules_trades_for(shock_from(prices, t, factor)), pos_t)


@pytest.mark.parametrize(("t", "factor"), SHOCKS)
def test_detector_catches_rules_using_tomorrow(t: int, factor: float) -> None:
    # Control: la misma comprobación debe fallar con una estrategia tramposa.
    prices = random_prices(n=1500, seed=5, sigma=0.04)
    pos_t = t - RULES_PERIOD_START
    original = executed_up_to(rules_trades_for(prices, lookahead=1), pos_t)
    shocked = executed_up_to(rules_trades_for(shock_from(prices, t, factor), lookahead=1), pos_t)
    assert original != shocked


VOL_PERIOD_START = 40


def vol_trades_for(prices: pd.DataFrame, lookahead: int = 0) -> list[Trade]:
    vol = log_return_volatility(prices["Close"], 30)
    median = expanding_median(vol)
    return vol_trades(vol.shift(-lookahead), median.shift(-lookahead), prices.index[VOL_PERIOD_START:])


# Días en los que la estrategia honesta da señal (entra al día siguiente). Un
# pico ×3 en t dispara la volatilidad de t: una estrategia que mirase t en t
# dejaría de entrar en la apertura de t; la honesta no puede verlo hasta t+1.
VOL_SIGNAL_DAYS = [102, 198, 370, 508]


@pytest.mark.parametrize("t", VOL_SIGNAL_DAYS)
def test_vol_strategy_does_not_use_day_t(t: int) -> None:
    prices = random_prices()
    pos_t = t - VOL_PERIOD_START
    before = executed_up_to(vol_trades_for(prices), pos_t)
    assert before  # el test no puede pasar en vacío
    assert before == executed_up_to(vol_trades_for(shock_from(prices, t, 3.0)), pos_t)


@pytest.mark.parametrize("t", VOL_SIGNAL_DAYS)
def test_detector_catches_vol_using_tomorrow(t: int) -> None:
    prices = random_prices()
    pos_t = t - VOL_PERIOD_START
    original = executed_up_to(vol_trades_for(prices, lookahead=1), pos_t)
    shocked = executed_up_to(vol_trades_for(shock_from(prices, t, 3.0), lookahead=1), pos_t)
    assert original != shocked


def test_expanding_median_uses_only_previous_days() -> None:
    vol = pd.Series([1.0, 3.0, 2.0, 10.0])
    assert expanding_median(vol).tolist()[1:] == [1.0, 2.0, 2.0]
    assert np.isnan(expanding_median(vol).iloc[0])


# --- métricas -------------------------------------------------------------

def make_sim(returns: list[float], invested: list[bool] | None = None, n_trades: int = 1) -> Simulation:
    index = pd.date_range("2024-01-01", periods=len(returns), freq="D")
    flags = invested if invested is not None else [True] * len(returns)
    return Simulation(pd.Series(returns, index=index), pd.Series(flags, index=index), [Trade(0, None)] * n_trades)


def test_metrics_cagr_and_drawdown() -> None:
    returns = [0.0] * 731
    returns[10], returns[20] = 0.10, 0.10  # crece un 21 % en 731 días
    result = metrics(make_sim(returns))
    assert np.isclose(result["CAGR"], 1.21 ** (365.25 / 731) - 1)
    assert result["max_drawdown"] == 0.0

    dd = metrics(make_sim([0.10, -0.50, 1.00]))
    assert np.isclose(dd["max_drawdown"], -0.5)


def test_metrics_sharpe_invested_and_trades() -> None:
    returns = [0.02, -0.01, 0.03, 0.0]
    result = metrics(make_sim(returns, invested=[True, True, False, False], n_trades=2))
    expected_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365)
    assert np.isclose(result["sharpe"], expected_sharpe)
    assert result["pct_invertido"] == 0.5
    assert result["operaciones"] == 2


def test_metrics_drawdown_from_initial_capital() -> None:
    # Si el primer día ya cae, el drawdown se mide desde el capital inicial (1).
    assert np.isclose(metrics(make_sim([-0.2, 0.1]))["max_drawdown"], -0.2)
