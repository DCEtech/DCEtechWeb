"""Fase 5: gráficas diarias de la web.

- `btc-latest.png` (post original): señales del árbol publicado, ahora con el
  histórico completo desde 2014 (corrige la EMA400) e indicadores propios.
- `btc-tracker.png` (Part 2): reglas originales frente a buy & hold desde la
  publicación del post, el único tramo fuera de muestra para las reglas.

Uso (desde `ml/`; es lo que ejecuta el CI cada día):
    python -m btc_decision_tree.live

Si la descarga de yfinance falla o los datos están desactualizados, sale con
el código DOWNLOAD_FAILED para que el CI reutilice las imágenes anteriores.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # sin pantalla (CI)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from btc_decision_tree.backtest import (  # noqa: E402
    FEE,
    POST_PUBLICATION,
    plan_buy_and_hold,
    rules_trades,
    simulate,
)
from btc_decision_tree.data import DataError, download_ohlcv  # noqa: E402
from btc_decision_tree.features import RULE_FEATURES, add_rule_features  # noqa: E402
from btc_decision_tree.labels import BUY, SELL, rule_signals  # noqa: E402
from btc_decision_tree.paths import IMAGES_DIR, LIVE_MODEL_PATH  # noqa: E402

logger = logging.getLogger(__name__)

HISTORY_START = "2014-09-17"  # primer día de BTC-USD en yfinance
SIGNALS_START = "2022-12-01"  # ventana que pinta el post original
TRACKER_START = POST_PUBLICATION[0]
MAX_STALENESS_DAYS = 3
DOWNLOAD_FAILED = 3  # código de salida que el CI trata como "reutilizar imágenes"
# Con SIMULATE_DATA_FAILURE=true la descarga falla a propósito (prueba del
# fallback del CI; solo se activa al lanzar el workflow a mano).
SIMULATE_ENV = "SIMULATE_DATA_FAILURE"

# Las imágenes se ven a ~700 px de ancho en el blog: letra ~1,6× la de
# matplotlib por defecto (10 pt de base, 12 pt el título).
FONT_RC = {
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}
NOTE_FONTSIZE = 14
FIGSIZE = (14, 7)  # pulgadas; a 150 dpi, 2100 × 1050 px
RULES_COLOR = "#4fc3f7"


@dataclass(frozen=True)
class Tracker:
    curves: pd.DataFrame        # retorno acumulado: columnas "rules" y "buy_and_hold"
    entries: list[pd.Timestamp]  # días de entrada de las reglas (de la lista de operaciones)
    invested: pd.Series          # True si las reglas tienen posición al cierre

    @property
    def final(self) -> pd.Series:
        return self.curves.iloc[-1]

    @property
    def n_trades(self) -> int:
        return len(self.entries)


# --- datos ----------------------------------------------------------------

def check_fresh(prices: pd.DataFrame, today: date, max_days: int = MAX_STALENESS_DAYS) -> None:
    """DataError si el último dato tiene más de `max_days` días."""
    age = (pd.Timestamp(today) - prices.index[-1]).days
    if age > max_days:
        raise DataError(f"Último dato del {prices.index[-1].date()}: {age} días de antigüedad")


def download_history(today: date) -> pd.DataFrame:
    """Histórico completo hasta ayer (el día en curso aún no ha cerrado)."""
    if os.environ.get(SIMULATE_ENV, "").lower() == "true":
        raise DataError(f"Fallo de descarga simulado ({SIMULATE_ENV}=true)")
    prices = download_ohlcv("BTC-USD", HISTORY_START, today.isoformat())
    check_fresh(prices, today)
    return prices


# --- estilo común ---------------------------------------------------------

@contextmanager
def chart_style() -> Iterator[None]:
    """Fondo oscuro y letra grande, solo dentro del bloque (no cambia el estado global)."""
    with plt.style.context("dark_background"), plt.rc_context(FONT_RC):
        yield


def format_usd_thousands(value: float, _pos: int | None = None) -> str:
    """20000 → "$20k"."""
    return f"${value / 1000:,.0f}k"


def format_signed_percent(value: float, _pos: int | None = None) -> str:
    """Valor ya en %: 10 → "+10%", -30 → "−30%" (signo menos tipográfico), 0 → "0%"."""
    if round(value) == 0:
        return "0%"
    sign = "+" if value > 0 else "\u2212"
    return f"{sign}{abs(value):.0f}%"


def out_of_market_spans(invested: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Tramos consecutivos (primer día, último día) sin posición."""
    spans = []
    start = None
    for day, held in invested.items():
        if not held and start is None:
            start = day
        elif held and start is not None:
            spans.append((start, previous))
            start = None
        previous = day
    if start is not None:
        spans.append((start, invested.index[-1]))
    return spans


def month_interval(index: pd.Index, max_ticks: int = 8) -> int:
    """Cada cuántos meses poner una fecha para no pasar de `max_ticks`."""
    months = (index[-1].year - index[0].year) * 12 + index[-1].month - index[0].month + 1
    return max(1, math.ceil(months / max_ticks))


# --- gráfica 1: señales del modelo publicado ------------------------------

def predict_signals(prices: pd.DataFrame, model: DecisionTreeClassifier) -> pd.DataFrame:
    """Features de las reglas y predicción del árbol donde están todas calculadas."""
    features = add_rule_features(prices)
    X = features[RULE_FEATURES].dropna()
    features.loc[X.index, "prediction"] = model.predict(X)
    return features


def plot_signals(features: pd.DataFrame, path: Path, start: str = SIGNALS_START) -> None:
    """Mismo aspecto que la gráfica original del post."""
    data = features.loc[start:].dropna(subset=["prediction"])
    buys = data[data["prediction"] == BUY]
    sells = data[data["prediction"] == SELL]

    with chart_style():
        # 14 × 7 pulgadas, como el tracker: a 700 px la letra se lee igual en las dos.
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(data.index, data["Close"], label="BTC Price", alpha=0.8, color="white")
        ax.plot(data.index, data["EMA400"], label="EMA400", linestyle="--", color="orange")
        ax.scatter(buys.index, buys["Close"], marker="^", label="Buy", s=90, color="green")
        ax.scatter(sells.index, sells["Close"], marker="v", label="Sell", s=60, color="red")
        ax.legend(facecolor="black")
        ax.set_title(
            "BTC Decision Tree Signals (original model, kept for reference)\n"
            f"Last update: {data.index[-1].date()}",
            color="white",
        )
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Price (USD)", color="white")
        ax.yaxis.set_major_formatter(FuncFormatter(format_usd_thousands))
        fig.tight_layout()
        _save(fig, path)


# --- gráfica 2: seguimiento fuera de muestra ------------------------------

def build_tracker(prices: pd.DataFrame, start: str = TRACKER_START, fee: float = FEE) -> Tracker:
    """Retorno acumulado de las reglas y de buy & hold desde `start`.

    Usa el mismo backtest que la fase 4: señal al cierre, ejecución en la
    apertura siguiente, comisión por lado y la posición abierta valorada como
    si se vendiera al cierre del último día.
    """
    signals = rule_signals(add_rule_features(prices))
    period_prices = prices.loc[start:]
    period = period_prices.index
    rules = simulate(period_prices, rules_trades(signals, period), fee)
    hold = simulate(period_prices, plan_buy_and_hold(len(period)), fee)
    curves = pd.DataFrame({
        "rules": (1 + rules.returns).cumprod() - 1,
        "buy_and_hold": (1 + hold.returns).cumprod() - 1,
    })
    entries = [period[trade.entry] for trade in rules.trades]
    return Tracker(curves, entries=entries, invested=rules.invested)


def plot_tracker(tracker: Tracker, path: Path, start: str = TRACKER_START) -> None:
    curves = tracker.curves * 100
    final = tracker.final
    trades = "trade" if tracker.n_trades == 1 else "trades"
    one_day = pd.Timedelta(days=1)

    with chart_style():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        # Sombreado de los tramos en que las reglas están fuera del mercado.
        for i, (first, last) in enumerate(out_of_market_spans(tracker.invested)):
            ax.axvspan(first, last + one_day, color="grey", alpha=0.15, linewidth=0,
                       label="Rules out of the market" if i == 0 else None)
        ax.plot(curves.index, curves["rules"], color=RULES_COLOR, linewidth=2.5,
                label=f"Original rules: {final['rules']:+.1%}")
        ax.plot(curves.index, curves["buy_and_hold"], color="white", alpha=0.8,
                label=f"Buy & hold: {final['buy_and_hold']:+.1%}")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

        # Un marcador por cada entrada de las reglas, sacado de las operaciones.
        for entry in tracker.entries:
            y = curves.loc[entry, "rules"]
            ax.scatter([entry], [y], marker="^", s=160, color=RULES_COLOR,
                       edgecolors="white", zorder=5)
            ax.annotate(f"Entry (rules): {entry.date()}", xy=(entry, y), xytext=(14, 48),
                        textcoords="offset points", fontsize=NOTE_FONTSIZE, color=RULES_COLOR,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black",
                              "edgecolor": "none", "alpha": 0.8},
                        arrowprops={"arrowstyle": "-", "color": RULES_COLOR, "linewidth": 1})

        ax.set_title(
            f"Out-of-sample since publication ({start})\n"
            f"Last update: {curves.index[-1].date()}",
            color="white",
        )
        ax.set_ylabel("Cumulative return", color="white")
        ax.yaxis.set_major_formatter(FuncFormatter(format_signed_percent))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval(curves.index)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        # Leyenda fuera del gráfico, en una fila: no tapa ninguna línea sean cuales sean los datos.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
        fig.text(
            0.99, 0.01,
            f"Rule-based signals, not a forecast. {tracker.n_trades} {trades} so far. "
            "0.1% fee per side.",
            ha="right", va="bottom", fontsize=NOTE_FONTSIZE, color="lightgrey",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera las gráficas diarias de BTC.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        prices = download_history(date.today())
    except DataError as exc:
        logger.error("No se pudieron actualizar los datos: %s", exc)
        sys.exit(DOWNLOAD_FAILED)

    model = joblib.load(LIVE_MODEL_PATH)
    plot_signals(predict_signals(prices, model), args.images_dir / "btc-latest.png")
    tracker = build_tracker(prices)
    plot_tracker(tracker, args.images_dir / "btc-tracker.png")
    logger.info(
        "Gráficas actualizadas hasta %s | reglas %+.1f%% | buy & hold %+.1f%% | %d operaciones",
        prices.index[-1].date(), tracker.final["rules"] * 100,
        tracker.final["buy_and_hold"] * 100, tracker.n_trades,
    )


if __name__ == "__main__":
    main()
