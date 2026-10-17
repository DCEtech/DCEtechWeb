"""Descarga de precios diarios (OHLCV) y snapshots locales en CSV.

yfinance devuelve datos distintos según el día en que se descargan (hay días
nuevos y a veces corrige los antiguos). Para que los resultados sean
reproducibles, el modelo se entrena siempre desde un snapshot guardado en
`ml/data/`, nunca directamente desde internet.

Crear un snapshot nuevo (desde `ml/`):
    python -m btc_decision_tree.data --end 2026-10-17
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Snapshot usado por defecto en los experimentos. Se cambia a mano y a
# propósito, nunca de forma automática.
DEFAULT_SNAPSHOT = DATA_DIR / "btc-usd_2014-09-17_2026-10-16.csv"


class DataError(RuntimeError):
    """Los datos descargados o leídos del snapshot no son válidos."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba columnas e índice y devuelve una copia con valores float."""
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise DataError(f"Faltan columnas OHLCV: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("El índice debe ser de fechas (DatetimeIndex)")
    if df.index.has_duplicates:
        raise DataError("Hay fechas duplicadas en el índice")
    if not df.index.is_monotonic_increasing:
        raise DataError("Las fechas no están ordenadas de forma ascendente")
    return df[OHLCV_COLUMNS].astype(float)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga velas diarias de yfinance. `end` es exclusivo."""
    try:
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
    except Exception as exc:  # yfinance puede lanzar errores de red de muchos tipos
        raise DataError(f"No se pudo descargar {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(f"yfinance no devolvió datos para {ticker} ({start} → {end})")

    # yfinance devuelve columnas con dos niveles (campo, ticker); nos quedamos con el campo.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"
    return validate_ohlcv(raw)


def snapshot_path(ticker: str, df: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    """Nombre del snapshot a partir del ticker y de la primera y última fecha."""
    first, last = df.index[0].date(), df.index[-1].date()
    return data_dir / f"{ticker.lower()}_{first}_{last}.csv"


def save_snapshot(df: pd.DataFrame, path: Path) -> None:
    """Guarda el DataFrame en CSV, creando la carpeta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


def load_snapshot(path: Path = DEFAULT_SNAPSHOT) -> pd.DataFrame:
    """Lee un snapshot CSV y lo valida."""
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        raise DataError(
            f"No existe el snapshot {path}. "
            "Créalo con: python -m btc_decision_tree.data"
        ) from None
    return validate_ohlcv(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y guarda un snapshot OHLCV.")
    parser.add_argument("--ticker", default="BTC-USD")
    # yfinance no tiene datos de BTC-USD anteriores al 2014-09-17.
    parser.add_argument("--start", default="2014-09-17")
    parser.add_argument("--end", default=date.today().isoformat(), help="Exclusivo")
    parser.add_argument("--force", action="store_true", help="Sobrescribir si existe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = download_ohlcv(args.ticker, args.start, args.end)
    path = snapshot_path(args.ticker, df)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} ya existe. Usa --force para sobrescribirlo.")
    save_snapshot(df, path)
    logger.info("Guardadas %d velas en %s", len(df), path)


if __name__ == "__main__":
    main()
