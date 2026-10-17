# Plan: rehacer el Decision Tree de BTC para que aprenda de verdad

Post actual: https://www.danielcejudo.com/ai-meets-investing-decision-tree-for-btc-and-stock-strategies.html
Repo: github.com/DCEtech/DCEtechWeb (Pelican, tema propio, sin plugins).

## Reglas innegociables
- Nunca usar como target algo calculado a partir de las propias features
  (el modelo original solo reproducía las reglas: accuracy 99 % sin valor).
- Features solo relativas/estacionarias (ratios, RSI, retornos), nunca
  precios en bruto.
- Validación temporal (walk-forward, sin shuffle, con embargo = horizonte).
- Comparar siempre contra baselines (buy & hold, estrategia de reglas).
- Reportar resultados con honestidad aunque no batan al buy & hold.
- Python: type hints, manejo de excepciones explícito, funciones pequeñas y
  testeables. Nada de lógica importante solo en notebooks.
- Daniel está aprendiendo: explicar la lógica de cada cambio, no solo el código.

## 0. Estado actual del repo (fase 1 completada, 2026-10-16)

**Dónde está cada cosa**
- Post: `content/AI Meets Investing Decision Trees for BTC and Stock Strategies.md`.
  Es el único sitio donde existe el código de entrenamiento (en bloques de código
  del post). **No hay script ni notebook de entrenamiento en el repo.**
- Modelo: `models/decision_tree.plk` (pickle joblib, sklearn 1.6.1; la extensión
  "plk" es una errata de "pkl").
- Inferencia + gráfica: `scripts/update_btc.py`. Descarga BTC-USD de yfinance desde
  2022-12-01, calcula `EMA400` (sobre HL2), `RSI` (14) y `dist_ema400` con pandas_ta,
  y llama a `model.predict` con las features `["RSI", "EMA400", "Close", "dist_ema400"]`.
  Pinta el precio, la EMA400 y las señales buy (1) y sell (-1), y guarda la imagen en
  `content/images/btc-latest.png`.
- No hay `requirements.txt` ni tests.

**Cómo se actualiza la gráfica**
- `.github/workflows/deploy.yml` se ejecuta en cada push a `main` y por cron
  diario a las 06:00 UTC. Pasos:
  1. Instala `pelican markdown yfinance pandas_ta matplotlib scikit-learn joblib`
     sin versiones fijadas, con Python 3.12.
  2. Ejecuta `update_btc.py`.
  3. Ejecuta `pelican content`.
  4. Publica `output/` en `gh-pages` con peaceiris/actions-gh-pages@v3.
- La PNG nueva solo llega a `gh-pages`; no se commitea a `main`.
- Estado verificado con `gh`: el workflow está activo, con 313 de 318 ejecuciones
  correctas. Hubo dos fallos aislados en 2026: el 29-06, un segfault (exit code 139)
  en `update_btc.py`, probablemente de numba/pandas_ta; y el 30-09, un fallo de git
  en el paso de deploy.
- Avisos en cada ejecución: `checkout@v2`, `setup-python@v2` y `peaceiris@v3` usan
  Node 20 (obsoleto), y **`ubuntu-latest` pasa a Ubuntu 26 el 2026-10-19**.

**Bugs encontrados en la inferencia actual**
- La EMA400 se calcula solo con datos desde 2022-12-01. Tras `dropna()`, la gráfica
  empieza hacia enero de 2024, aunque el post dice diciembre de 2022. Además, la EMA
  no coincide con la del entrenamiento, que se calculó con datos desde 2014. Como
  `EMA400` y `Close` entran en bruto como features, las predicciones cambian.
- Sklearn no está fijado en CI y el pickle es de la 1.6.1, así que puede romperse.

## 1. Diagnóstico del modelo actual

| Problema | Consecuencia |
|---|---|
| Etiquetas `buy`/`sell` = reglas fijas sobre RSI, Close y EMA400, y esas mismas columnas son features | El árbol redescubre los umbrales. Accuracy del 99 % sin capacidad predictiva |
| `Close` y `EMA400` en bruto como features | No estacionarias: el test (2023-25) tiene precios nunca vistos y un árbol no extrapola |
| Clase `buy` con 11 casos en test | Métricas estadísticamente irrelevantes |
| Sin backtest ni baseline | No se sabe si la estrategia gana dinero frente a mantener BTC |
| Sin `random_state`, código duplicado, EMA10/EMA55 sin usar, EMA sobre HL2 vs reglas sobre Close | Poca reproducibilidad y limpieza |
| `buy` incluye `Close > EMA400 * 1.15` | Contradice el texto (comprar cerca de la EMA). Confirmar si es intencionado |

## 2. Nuevo planteamiento

**Pregunta:** dadas las condiciones de hoy, ¿subirá BTC más de un X % en los
próximos N días?

### Target (sin fuga de información)
```python
horizon = 30
data["future_return"] = data["Close"].shift(-horizon) / data["Close"] - 1
data["target"] = (data["future_return"] > 0.10).astype(int)
data = data.iloc[:-horizon]  # las últimas N filas no tienen futuro conocido
```
Versión avanzada (fase posterior): *triple barrier* (take-profit, stop-loss y
límite temporal).

### Features (todas relativas, calculadas solo con datos hasta t)
- `dist_ema400`, `dist_ema55`, pendiente de la EMA400 (cambio % en k días)
- `RSI(14)`
- Retornos pasados: 1, 7 y 30 días
- Volatilidad: desviación típica móvil de retornos (30 días)
- Opcional: señal de la estrategia original como una feature más

### Validación
- `TimeSeriesSplit` walk-forward, nunca `shuffle`
- Hueco (embargo) de `horizon` días entre train y test, porque el target del
  final del train mira dentro del test
- Hiperparámetros (`max_depth`, `min_samples_leaf`) ajustados solo con CV
  sobre train. Test final usado una única vez
- `random_state` fijo y snapshot de datos en disco (yfinance cambia)

### Evaluación
- Métricas de clasificación: precision de la clase 1 (cuando dice "compra",
  ¿acierta?), matriz de confusión
- **Backtest:** señal al cierre de t, ejecución en apertura de t+1 (evita
  *look-ahead*), comisión ~0.1 % por operación
- Comparar contra: buy & hold, estrategia de reglas original, y siempre-clase-mayoritaria
- Métricas financieras: retorno total/CAGR, max drawdown, Sharpe, nº de operaciones
- Bonus: `feature_importances_` y diagrama del árbol para explicar qué aprendió

## 3. Estructura de código propuesta
```
ml/btc_decision_tree/
  data.py        # descarga + caché local
  features.py    # indicadores y features (type hints)
  labels.py      # construcción del target
  model.py       # entrenamiento y validación walk-forward
  backtest.py    # simulación con comisiones
  plot.py        # genera images/btc-latest.png
tests/
  test_no_leakage.py  # alterar datos futuros no debe cambiar features en t
requirements.txt      # versiones fijadas
```
Notebook solo para exploración. La gráfica diaria pasará a generarse desde
`plot.py`, que sustituirá a `scripts/update_btc.py`. El paso del workflow
tendrá que apuntar al nuevo script y seguir escribiendo en
`content/images/btc-latest.png`, que es la ruta que usa el post.

## 4. Fases
1. ~~Explorar el repo y completar los TODO de `CLAUDE.md`.~~ Hecho el 2026-10-16
   (ver sección 0).
   1b. ~~Estabilizar el CI.~~ Hecho el 2026-10-17 (commits 2448b69 y 48ecb1d):
   - runner `ubuntu-24.04`;
   - todas las dependencias fijadas en `requirements-ci.txt`, con las versiones
     del último run correcto (scikit-learn 1.9.1);
   - `workflow_dispatch` para lanzarlo a mano;
   - acciones con Node 24: checkout@v5, setup-python@v6, actions-gh-pages@v4.
   Queda pendiente que el `.plk` se creó con sklearn 1.6.1 y el CI lo carga con
   la 1.9.1 (`InconsistentVersionWarning`); se resuelve en la fase 5.
2. Reconstruir en `ml/` el entrenamiento del modelo actual a partir del código del
   post, porque no existe en el repo. Será la baseline de partida y servirá para
   comprobar que reproduce el classification report publicado.
3. Implementar target, features y validación nuevos con test anti-fuga.
4. Backtest y comparación con baselines.
5. Adaptar el workflow existente, que ya funciona a diario.
   - Sustituir `update_btc.py` por `plot.py`.
   - Calcular los indicadores con el histórico completo, para corregir el bug de la EMA400.
   - Commitear o versionar el modelo nuevo.
6. Escribir el post.

## 5. Decisiones abiertas
- ¿Post nuevo ("Part 2: making the Decision Tree actually learn") enlazado
  desde el original, o reescribir el original? Recomendado: post nuevo. Contar
  el error y su corrección demuestra más criterio que un post perfecto.
- Horizonte y umbral del target (30 días / 10 % es un punto de partida).
- ¿Incluir una acción (p. ej. SPY) para cumplir el título "BTC and Stock"?
- ¿Sustituir pandas_ta, que está poco mantenido y es la causa probable del segfault,
  por indicadores propios en `features.py`? EMA y RSI son unas pocas líneas de pandas
  y además se pueden testear.
- ¿Qué hacer con la gráfica del post original cuando exista el modelo nuevo?
  Opciones: congelarla o seguir mostrando las señales del modelo viejo.
