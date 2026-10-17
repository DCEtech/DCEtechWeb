import pandas as pd

from btc_decision_tree.labels import BUY, HOLD, SELL, rule_signals


def make_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Filas (Close, EMA400, RSI) con fechas consecutivas."""
    index = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["Close", "EMA400", "RSI"], index=index)


def test_buy_when_below_ema_and_oversold() -> None:
    df = make_frame([(90, 100, 25)])
    assert rule_signals(df).tolist() == [BUY]


def test_hold_when_rsi_not_oversold() -> None:
    df = make_frame([(90, 100, 45)])
    assert rule_signals(df).tolist() == [HOLD]


def test_sell_only_after_first_buy() -> None:
    df = make_frame([
        (130, 100, 80),  # condición de venta, pero aún no ha habido compra
        (90, 100, 25),   # primera compra
        (130, 100, 80),  # venta válida
    ])
    assert rule_signals(df).tolist() == [HOLD, BUY, SELL]


def test_no_sells_without_any_buy() -> None:
    df = make_frame([(130, 100, 80), (140, 100, 85)])
    assert rule_signals(df).tolist() == [HOLD, HOLD]


def test_missing_indicators_give_hold() -> None:
    # Al principio de la serie EMA400 y RSI son NaN: no debe haber señal.
    df = make_frame([(90, float("nan"), float("nan"))])
    assert rule_signals(df).tolist() == [HOLD]
