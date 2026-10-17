from pathlib import Path

import pandas as pd
import pytest

from btc_decision_tree.data import DataError, load_snapshot, save_snapshot, snapshot_path, validate_ohlcv


def make_prices(n: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="D", name="Date")
    values = [float(i + 1) for i in range(n)]
    return pd.DataFrame(
        {"Open": values, "High": values, "Low": values, "Close": values, "Volume": values},
        index=index,
    )


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(DataError, match="Close"):
        validate_ohlcv(make_prices().drop(columns="Close"))


def test_validate_rejects_unsorted_dates() -> None:
    with pytest.raises(DataError, match="ordenadas"):
        validate_ohlcv(make_prices().iloc[::-1])


def test_validate_rejects_duplicated_dates() -> None:
    prices = make_prices()
    with pytest.raises(DataError, match="duplicadas"):
        validate_ohlcv(pd.concat([prices, prices.iloc[[-1]]]))


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    prices = make_prices()
    path = snapshot_path("BTC-USD", prices, data_dir=tmp_path)
    assert path.name == "btc-usd_2024-01-01_2024-01-05.csv"

    save_snapshot(prices, path)
    pd.testing.assert_frame_equal(load_snapshot(path), prices, check_freq=False)


def test_load_missing_snapshot_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="No existe"):
        load_snapshot(tmp_path / "missing.csv")
