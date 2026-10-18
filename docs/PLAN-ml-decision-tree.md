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
**Decidido (2026-10-18):** el target es un retorno a 30 días mayor del +10 %.
- Lo calcula una función nueva, `forward_return_target`, en `labels.py`.
- La regla original se queda en el mismo archivo, solo para el baseline.
- Se descartan las últimas 30 filas, que no tienen futuro conocido.
- Si el modelo no mejora la tasa base, se probará primero un umbral ajustado
  por volatilidad y después *triple barrier* (take-profit, stop-loss y límite
  temporal).

Tasa base en el snapshot: 37,6 % en train y 23,3 % en test. Varía mucho por
año (13 % en 2018, 65 % en 2017), así que la precision siempre se compara con
la tasa base del mismo periodo.

### Indicadores (decidido)
- El código nuevo usa indicadores propios en pandas, en `indicators.py`, sin
  pandas_ta. `baseline.py` sigue con pandas_ta para reproducir el original.
- Dos trampas a evitar:
  - La EMA lleva semilla SMA (la media de las primeras `length` filas) y
    `ewm(span=length, adjust=False)`. El `ewm(span)` por defecto da otro resultado.
  - El RSI de Wilder usa `ewm(alpha=1/length)`, no `span=length`. pandas_ta no
    usa semilla SMA en el RSI, así que los dos solo coinciden tras un calentamiento.
- Tests:
  - comparar con pandas_ta descartando las primeras 10×length filas, con tolerancia;
  - comparar la EMA con un bucle recursivo escrito a mano.

### Features (decidido: todas relativas, calculadas solo con datos hasta t)
- `dist_ema400` y `dist_ema55`: Close / EMA − 1. Las EMAs se calculan sobre HL2.
- `ema400_slope_20d`: cambio % de la EMA400 en 20 días.
- `rsi14`.
- `ret_1d`, `ret_7d`, `ret_30d`: retornos pasados.
- `vol_30d`: desviación típica móvil de los retornos diarios en 30 días.
- Test anti-fuga: alterar los precios posteriores a t no cambia las features en t.

### Validación (decidido)
- Split temporal:
  - Train hasta el 2024-08-04, sin sus últimas 30 filas (purga): termina el
    2024-07-05. Así ninguna etiqueta del train mira dentro del test.
  - Test desde el 2024-08-05 hasta 30 días antes del final del snapshot.
  - **El test no se evalúa todavía.**
- Walk-forward solo en train: `TimeSeriesSplit(n_splits=5, gap=30)`, con
  `max_depth` de 2 a 6 y `min_samples_leaf` en {20, 50, 100, 200}.
- Sin `class_weight="balanced"`: lo que importa es acertar cuando dice clase 1.
- Por fold se reportan la precision de la clase 1, la tasa base, el lift
  (precision − tasa base) y el número de predicciones de clase 1. Se elige la
  combinación con mejor lift medio, mirando también el peor fold.
- `random_state` fijo y snapshot de datos en disco (yfinance cambia).

### Experimento 2: target ajustado por volatilidad (decidido el 2026-10-18)
- `vol_30d` del target: desviación típica de los log-retornos diarios de los
  últimos 30 días, hasta t incluido (solo pasado).
- Etiqueta 1 si `log(C[t+30] / C[t]) > k · vol_30d · √30`.
- `k` se calibra una sola vez para que la tasa base del train quede entre el 37 %
  y el 38 %. La estimación previa era k ≈ 0,60, que da un 37,5 %. **Prohibido
  ajustar k mirando el lift.**
  - Se calibra con la tasa base de todo el train tras la purga, que incluye las
    ventanas de validación del walk-forward. Es aceptable porque solo fija la
    proporción de clases y no ajusta el modelo.
- Mismas features, split, purga y rejilla de hiperparámetros que el experimento 1.
  La feature `vol_30d` (retornos simples) no cambia.
- Nuevo baseline: predice 1 si `vol_30d` (la de log-retornos) está por debajo de
  su mediana en la parte de train de cada fold. Su precision y su lift se
  reportan por fold junto a los del árbol. Es necesario porque, con vol alta, el
  umbral sube y la etiqueta 1 es más difícil: un árbol podría "acertar" solo
  aprendiendo esa relación mecánica.

**Criterio de éxito (escrito antes de ver resultados).** Se aplica a la
combinación elegida con la regla del experimento 1 (la de mejor lift medio):
1. Lift > 0 en al menos 4 de los 5 folds.
2. Lift medio > desviación típica muestral (`ddof=1`) de los 5 lifts.
3. Lift medio del árbol > lift medio del baseline de volatilidad, en los
   mismos 5 folds.

Elegir la mejor de 20 combinaciones la favorece por azar, así que un
cumplimiento justo en el límite se señalará.

Se aplica también al experimento 1 (`--target return`), y los dos se presentan
en la misma tabla de veredictos. Además se informa, **solo como dato y sin que
decida nada**, de:
- cuántas de las 20 combinaciones cumplen los 3 criterios;
- el lift medio de la combinación mediana.

El test sigue sin evaluarse.

### Experimento 3: *triple barrier* (decidido el 2026-10-18, último experimento)
- Barreras simétricas `±b`, con `b = k · vol_30d[t] · √30`. Se usa la misma
  `log_return_volatility` que en el experimento 2, y la barrera vertical está a
  30 días.
- Toques con el cierre diario:
  - superior si `log(C[t+j] / C[t]) ≥ +b` para algún j entre 1 y 30;
  - inferior si `≤ −b`.

  El "mayor o igual" es porque llegar justo al nivel cuenta como tocar. Un mismo
  día no puede tocar las dos.
- Etiqueta 1 si toca primero la superior. Etiqueta 0 si toca primero la
  inferior o si expira sin tocar ninguna.
- Se descartan las últimas 30 filas aunque ya hayan tocado una barrera, para
  que los tres experimentos usen las mismas filas y el mismo split.
- `k` se calibra una sola vez con `calibrate_k` (generalizado), apuntando a una
  tasa base de train del 37,5 %.
  - La tasa no es monótona en k: sube de 53 % con k = 0 a 57 % con k = 0,2 y
    luego baja. El 37,5 % solo se alcanza en un punto, hacia k ≈ 0,84, con un
    ~35 % de días que expiran.
  - Si la tasa obtenida queda fuera del 37–38 %, la ejecución se detiene y se
    avisa.
- Mismas features, split, purga (30), rejilla y baseline de volatilidad.

**Apuesta = entrada.**
- El primer día con predicción 1 abre una apuesta. No se cuenta otra hasta 30
  días después (el horizonte), aunque el modelo siga prediciendo 1: la siguiente
  entrada es posible en el día de entrada + 30.
- Precision por apuesta = etiqueta del día de entrada.
- Cada fold empieza sin apuesta abierta; el bloqueo no cruza folds.

**Criterio de éxito (escrito antes de ver resultados):**
- C1–C3 iguales que en el experimento 2.
- **C4 (nuevo):** al menos 3 apuestas en cada uno de los 5 folds. Un fold con
  menos hace fallar C4.

Qué significa "cumple" en cada experimento:
- Experimento 3: C1, C2, C3 y C4.
- Experimentos 1 y 2: sigue siendo C1–C3, tal como se registró. Su C4 se informa
  solo como dato.

Se presenta una tabla conjunta de los 3 experimentos, con el número de
combinaciones que cumplen y el lift medio de la combinación mediana, que son
solo datos. El test sigue sin evaluarse.

### Evaluación
- Métricas de clasificación: precision de la clase 1 (cuando dice "compra",
  ¿acierta?), matriz de confusión
- **Backtest:** señal al cierre de t, ejecución en apertura de t+1 (evita
  *look-ahead*), comisión ~0.1 % por operación
- Comparar contra: buy & hold, estrategia de reglas original, y siempre-clase-mayoritaria
- Métricas financieras: retorno total/CAGR, max drawdown, Sharpe, nº de operaciones
- Bonus: `feature_importances_` y diagrama del árbol para explicar qué aprendió

### Fase 4: backtest de los baselines (decidido el 2026-10-18, sin candidato ML)
- **Periodos:** todo empieza el 2015-10-21, la primera EMA400 válida, también
  buy & hold. El reporte separa train (con desglose por folds) y test.
- **El periodo de test se usa para los baselines y queda consumido.** Ya se miró
  en la fase 2 y se conoce su tasa base. Ver la sección 5.
- **Estrategias:**
  - Buy & hold.
  - Reglas originales: `rule_signals` tal cual, incluida la rama
    `> EMA400 × 1,15`. Se entra en largo con la señal de compra si se está
    fuera y se sale con la de venta. Sin cortos.
  - Baseline de volatilidad: entrada si `vol_30d` está por debajo de la mediana
    del train, y salida a los 30 días. Usa la misma regla de entrada y de
    bloqueo que C4.
- **Ejecución:** apertura de t+1 para una señal al cierre de t. Comisión del
  0,1 % por lado.
- **Métricas, fijadas antes de ejecutar:**
  - CAGR;
  - máximo drawdown;
  - Sharpe con 365 días por año;
  - % de tiempo invertido;
  - número de operaciones.
- **Tests:**
  - buy & hold y una estrategia de una sola operación, calculados a mano;
  - comisiones aplicadas en la entrada y en la salida;
  - ninguna decisión ejecutada en t usa información del propio día t.

Precisiones (aprobadas el 2026-10-18):
1. **Contabilidad diaria**, con el valor de la cartera marcado al cierre:
   - día de entrada (apertura de t+1): `Close/Open × (1 − 0,001)`;
   - días con posición: `Close/Close anterior`;
   - día de salida (apertura): `Open/Close anterior × (1 − 0,001)`;
   - días fuera del mercado: 0.
2. **Bordes de cada periodo.** Cada periodo (train, cada fold, test y
   post-publicación) se simula por separado y empieza fuera del mercado.
   - La primera entrada posible es en la apertura del segundo día. Buy & hold
     entra en la apertura del primer día.
   - Lo que siga abierto al final se liquida al cierre del último día, con
     comisión.
   - Una señal en el último día se ignora.
3. **Periodos:**
   - Train: 2015-10-21 → 2024-08-04, sin purga, porque no se entrena nada.
   - Folds: las 5 ventanas de validación del walk-forward.
   - Test: 2024-08-05 → 2026-10-16.
   - **Post-publicación:** 2025-12-13, la fecha del post original según sus
     metadatos en `content/`, → 2026-10-16.
4. **Reglas originales:** `rule_signals` sobre la serie completa desde 2014, con
   `features_original` (pandas_ta). Una compra con la posición abierta o una
   venta estando fuera se ignoran.
5. **Baseline de volatilidad:**
   - Señal en i, entrada en la apertura de i+1 y salida en la apertura de i+31.
   - Siguiente señal posible desde i+30.
   - Si una salida y una nueva entrada coinciden, se cobran las dos comisiones y
     cuentan como dos operaciones.
6. **Métricas:**
   - CAGR = `(final/inicial)^(365,25/días) − 1`;
   - máximo drawdown sobre el valor diario de la cartera;
   - Sharpe = `media/desv. × √365` de los retornos diarios, con tipo libre de
     riesgo 0 y contando los días fuera;
   - % de tiempo invertido = días con la posición abierta al cierre;
   - número de operaciones = entradas.

**Mediana del baseline de volatilidad, según el periodo:**
- Train completo: **mediana expansiva**, la de `vol_30d` de todos los días
  anteriores a t (solo pasado). `vol_30d` existe desde 2014-10, así que el
  2015-10-21 ya hay un año de historia.
- Folds: la mediana de la parte de train de cada fold.
- Test y post-publicación: la mediana de todo el train (2015-10-21 → 2024-08-04).

**Las reglas originales son in-sample en train y test.** Sus umbrales (RSI 30
y 70, EMA400 × 1,15 y × 1,25) se eligieron mirando el histórico de BTC. El
único tramo fuera de muestra para ellas es el **post-publicación**.

### Fase 5: gráficas en vivo y CI (decidido el 2026-10-18)
**A) `btc-latest.png` (post original): se mantiene el modelo, de forma mantenible.**
- Regenerar el árbol con `baseline.py` y la versión de scikit-learn del CI, y
  guardarlo con otro nombre. Se conserva `models/decision_tree.plk`, que usan
  los tests de la fase 2.
- Verificar que `export_text` es idéntico al original y que las predicciones
  son iguales en todo el histórico. Documentar en CLAUDE.md cómo regenerarlo.
- Inferencia con el histórico completo desde 2014, lo que corrige la EMA400, y
  con indicadores propios en lugar de pandas_ta. Comprobar que las features
  coinciden con las del pipeline original en la ventana que se pinta.
- Test: en esa ventana, las predicciones del modelo coinciden con `rule_signals`.

**B) `btc-tracker.png` (Part 2): seguimiento fuera de muestra.**
- En inglés y con dos líneas: reglas originales y buy & hold desde el
  2025-12-13, con `simulate` y `rule_signals` de `backtest.py` y un 0,1 % de
  comisión por lado.
- La imagen incluye:
  - el título "Out-of-sample since publication (2025-12-13)";
  - la fecha de la última actualización;
  - el retorno acumulado de cada línea en la leyenda;
  - la nota "Rule-based signals, not a forecast. N trades so far".

**C) CI**
- Descargar el histórico completo en cada ejecución.
- Ejecutar `pytest ml` antes del build: si falla, no hay deploy.
- Quitar pandas_ta, y lo que ya no se use, de `requirements-ci.txt`.
- Retirar `scripts/update_btc.py` cuando lo sustituya el código nuevo.

**D) Aviso "Update" en el post original**, en el mismo push que A–C. Primero
un borrador en inglés para revisarlo antes de hacer el commit.

**Hallazgos al planificar (2026-10-18), que afectan a A:**
- **`export_text` idéntico.** Solo se consigue entrenando con los datos
  cortados en el 2025-12-13 (el día del entrenamiento original) y
  `random_state=2`: coincide hasta 10 decimales, y también con las semillas
  10, 14, 16, 23, 34, 40 y 46.
  - Con el snapshot completo, o con `random_state=42`, cambia un nodo: un corte
    en `Close` en lugar de `RSI`, cuyas dos ramas dan −1.
  - El original se entrenó sin `random_state`, así que ese empate se resolvió
    al azar. En todos los casos, las predicciones son iguales en todo el
    histórico.
- **Modelo == `rule_signals` no se cumple.** El árbol publicado difiere de las
  reglas en 5 días de la ventana pintada (desde el 2022-12-01) y en 21 de todo
  el histórico:
  - El árbol compra con RSI ≤ 29,94 aunque el precio esté entre la EMA400 y
    +15 %. Las reglas excluyen esa franja, pero las dos ramas de ese nodo dan
    clase 1 y el árbol la "olvidó". Pasa en 3 días de la ventana y en 19 del
    histórico.
  - Las reglas compran con RSI entre 29,94 y 30 y el árbol no, porque su corte
    es 29,94. Pasa en 2 días.
- Las features con indicadores propios coinciden **exactamente** (diferencia 0)
  con las de pandas_ta en la ventana pintada, y las predicciones también.

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
2. ~~Reconstruir en `ml/` el entrenamiento del modelo actual.~~ Hecho el 2026-10-17.
   - Código: `ml/btc_decision_tree/` con `data.py`, `features.py`, `labels.py` y
     `baseline.py`, más 13 tests en `ml/tests/`.
   - Datos: snapshot fijo `ml/data/btc-usd_2014-09-17_2026-10-16.csv` y
     `random_state=42`. Dos ejecuciones seguidas dan un resultado idéntico.
   - Fidelidad: sobre el mismo snapshot, X e y son idénticos a los del código de
     `Untitled.ipynb`, y las predicciones en test coinciden al 100 % con
     `models/decision_tree.plk`.
   - Resultado (test del 2024-08-05 al 2026-10-16, 803 días): accuracy del 100 %,
     y en la clase compra precision y recall de 0,92. La importancia de las
     features es RSI 0,95, dist_ema400 0,04, EMA400 0,02 y Close 0,00: el árbol
     solo reaprende los umbrales de la regla.
3. Implementar target, features y validación nuevos con test anti-fuga.
   Código hecho el 2026-10-18:
   - `indicators.py`, `features.py`, `forward_return_target` y `model.py`.
   - 42 tests, entre ellos el anti-fuga y la comparación de los indicadores
     con pandas_ta y con bucles escritos a mano.

   Resultado del walk-forward (5 folds en train, test NO evaluado):
   - **Sin ventaja clara.** La mejor combinación (`max_depth=6`,
     `min_samples_leaf=100`) tiene un lift medio de +1,8 puntos, con un rango
     por fold de −6,9 a +10,4. La segunda (3/50) da +1,7 de media, pero −12,5
     en su peor fold.
   - 11 de las 20 combinaciones están por debajo de la tasa base.
   - La tasa base cambia entre folds del 21 % al 54 %.
   - Siguiente paso, según lo decidido: probar un umbral ajustado por
     volatilidad y, si sigue sin haber señal, *triple barrier*.

   Veredictos con los criterios pre-registrados (2026-10-18, commit 9130b3b,
   `python -m btc_decision_tree.model --target both`; test NO evaluado):

   | | Exp. 1: return > +10 % | Exp. 2: vol, k = 0,60 |
   |---|---|---|
   | Combinación elegida | 6/100 | 2/200 |
   | C1: lift > 0 en ≥ 4/5 folds | ✅ 4/5 | ✅ 4/5 |
   | C2: lift medio > desv. típica | ❌ 0,018 < 0,062 | ❌ 0,298 < 0,311 |
   | C3: lift medio > baseline de vol | ❌ 0,018 < 0,056 | ✅ 0,298 > 0,094 |
   | **Cumple** | **No** | **No** |
   | Combinaciones que cumplen (dato) | 0/20 | 0/20 |
   | Lift medio de la combinación mediana (dato) | −0,082 | +0,018 |

   Lectura:
   - **Experimento 1:** no hay señal, y el baseline de volatilidad, una regla de
     una línea, tiene más lift que el árbol.
   - **Experimento 2:** el lift medio es alto, pero no supera a la dispersión
     entre folds (C2 falla por poco). Además, las predicciones de clase 1 están
     muy agrupadas en el tiempo:
     - fold 1: 16 días en **1 sola racha** (abril y mayo de 2017), con precision 1,0;
     - fold 4: 67 días en 2 rachas;
     - folds 2, 3 y 5: 8, 6 y 12 rachas.

     Son muy pocas apuestas independientes, así que el lift alto puede deberse
     al azar.
   - La mediana de las 20 combinaciones está en +1,8 puntos: elegir la mejor
     infla el resultado.
   - Siguiente paso, según la regla de parada: *triple barrier* como último
     experimento y después la fase 4.

   Experimento 3 y tabla conjunta (2026-10-18, commit dfdc1cb,
   `python -m btc_decision_tree.model --target all`; test NO evaluado).
   Calibración: k = 0,83, con una tasa base de train del 37,5 %. Al cambiar el
   código, los experimentos 1 y 2 dan los mismos números que antes.

   | | Exp. 1: return | Exp. 2: vol k=0,60 | Exp. 3: barrier k=0,83 |
   |---|---|---|---|
   | Elegida | 6/100 | 2/200 | 2/200 |
   | C1: lift > 0 en ≥ 4/5 | ✅ 4/5 | ✅ 4/5 | ✅ 4/5 |
   | C2: media > desv. típica | ❌ 0,018 < 0,062 | ❌ 0,298 < 0,311 | ❌ 0,319 < 0,325 |
   | C3: gana al baseline de vol | ❌ 0,018 < 0,056 | ✅ 0,298 > 0,094 | ✅ 0,319 > 0,108 |
   | C4: ≥ 3 apuestas por fold | (dato) ✅ mín. 5 | (dato) ❌ mín. 1 | ❌ mín. 1 |
   | **Cumple** | **No** | **No** | **No** |
   | Precision media por apuesta (dato) | 0,36 | 0,57 | 0,62 |
   | Combinaciones que cumplen (dato) | 0/20 | 0/20 | 0/20 |
   | Lift medio de la combinación mediana (dato) | −0,082 | +0,018 | +0,141 |

   Detalle del experimento 3 (2/200), por fold:

   | Fold | Apuestas | Precision por apuesta | Lift diario |
   |---|---|---|---|
   | 1 | 1 | 1,00 | +0,625 |
   | 2 | 6 | 0,50 | −0,098 |
   | 3 | 3 | 0,67 | +0,538 |
   | 4 | 3 | 0,67 | +0,490 |
   | 5 | 8 | 0,25 | +0,041 |

   **Conclusión de la fase 3: ningún candidato cumple los criterios.**
   - El mejor fold de los experimentos 2 y 3 es una única apuesta (abril y mayo
     de 2017).
   - En el experimento 3, la combinación mediana (+0,14) es mejor que en los
     anteriores. Es solo un dato: no cambia el veredicto ni justifica seguir
     probando.
   - Según la regla de parada, se pasa a la fase 4 sin candidato. El test sigue
     sin evaluarse.
4. Backtest y comparación con baselines. Hecho el 2026-10-18, sin candidato ML.
   - Código del commit 3803873, guardado antes de ejecutar, en `backtest.py`.
   - 99 tests. Los anti-fuga fuerzan una señal en t con un choque de precio e
     incluyen casos de control que detectan una estrategia que usa la señal de
     mañana.
   - Comando: `python -m btc_decision_tree.backtest`.

   | Periodo | Estrategia | CAGR | Máx. DD | Sharpe | % invertido | Operaciones |
   |---|---|---|---|---|---|---|
   | Train 2015-10-21 → 2024-08-04 | Buy & hold | **84,2 %** | −83,4 % | **1,22** | 100 % | 1 |
   | | Reglas originales (in-sample) | 28,2 % | −72,8 % | 0,78 | 45 % | 10 |
   | | Baseline vol (mediana expansiva) | 36,4 % | −73,4 % | 0,87 | 68 % | 73 |
   | Test 2024-08-05 → 2026-10-16 | Buy & hold | 16,5 % | −53,1 % | 0,56 | 100 % | 1 |
   | | Reglas originales (in-sample) | −0,4 % | −39,6 % | 0,15 | 50 % | 2 |
   | | Baseline vol | 21,2 % | −54,0 % | 0,65 | 99 % | 27 |
   | Post-publicación 2025-12-13 → 2026-10-16 | Buy & hold | −11,6 % | −39,6 % | −0,06 | 100 % | 1 |
   | | Reglas originales (**fuera de muestra**) | 4,1 % | −28,7 % | 0,30 | 84 % | 1 |
   | | Baseline vol | −11,9 % | −39,1 % | −0,09 | 92 % | 10 |

   Sharpe por fold (buy & hold / reglas / vol):
   - fold 1: 1,71 / 2,09 / 2,70
   - fold 2: 0,52 / 0,99 / 0,77
   - fold 3: 1,91 / 0,98 / 2,10
   - fold 4: −0,41 / −0,49 / −0,63
   - fold 5: 1,54 / 1,24 / 1,52

   Lectura:
   - **En el periodo largo (train) ninguna estrategia bate a buy & hold**, ni en
     CAGR ni en Sharpe. Las reglas originales ganan un 28 % anual frente a un
     84 %, incluso siendo in-sample. Reducen el tiempo invertido, pero el
     drawdown sigue en torno al −73 %.
   - Por folds no hay un ganador estable: cada estrategia gana en algún tramo.
     En el bajista de 2022 (fold 4) pierden todas.
   - En el test, el baseline de vol queda algo por encima de buy & hold, pero
     está invertido el 99 % del tiempo: en la práctica es buy & hold, y la
     diferencia es ruido.
   - En post-publicación, el único tramo fuera de muestra para las reglas, estas
     ganan un 4,1 % frente a −11,6 %. Pero es **una sola operación** en 10
     meses: es una anécdota, no una prueba.
   - Esto contradice la frase del post original ("works surprisingly well"),
     y conviene contarlo en el post nuevo.
5. ~~Adaptar el workflow existente.~~ Hecho el 2026-10-18 (decisiones y detalles
   en la sección 2, "Fase 5"):
   - `live.py` sustituye a `update_btc.py` y genera `btc-latest.png` y `btc-tracker.png`.
   - El árbol está regenerado con sklearn 1.9.1, idéntico al original.
   - Indicadores propios con el histórico completo, lo que corrige el bug de la EMA400.
   - El CI ejecuta los tests antes del deploy y reutiliza las imágenes de `gh-pages`
     si falla yfinance.
   - Aviso "Update" añadido al post original.
6. Escribir el post.

**Orden de trabajo acordado (2026-10-18), tras la fase 4:**
1. ~~Fase 5: gráfica en vivo (Claude Code).~~ Hecho el 2026-10-18.
2. ~~Aviso "Update" en el post original.~~ Hecho el 2026-10-18. Al publicar la
   Part 2, cambiar "Part 2 (coming soon)" por el enlace.
3. Part 2: la escribe Daniel en inglés y Claude la revisa. **← siguiente**
4. Part 3: volatilidad, cuando la Part 2 esté publicada (ver sección 5).

**Índice de la Part 2:**
1. *The 99% that meant nothing*: por qué el modelo era circular.
2. *Doing it properly*: snapshot fijo, sin fugas, criterios fijados antes de
   ver los resultados.
3. *Three honest attempts*: la tabla de los 3 experimentos.
4. *Did the rules at least work?*: el backtest contra buy & hold.
5. *What I learned*: el método importa más que el resultado.
6. *Live tracker + next steps*: enlace a la gráfica y la idea de la Part 3.

## 5. Decisiones abiertas
- ¿Post nuevo ("Part 2: making the Decision Tree actually learn") enlazado
  desde el original, o reescribir el original? Recomendado: post nuevo. Contar
  el error y su corrección demuestra más criterio que un post perfecto.
- ~~Horizonte y umbral del target.~~ Decidido el 2026-10-18: 30 días y +10 %
  (ver sección 2).
- ¿Incluir una acción (p. ej. SPY) para cumplir el título "BTC and Stock"?
- ~~¿Sustituir pandas_ta?~~ Decidido el 2026-10-18: el código nuevo usa
  indicadores propios y el baseline sigue con pandas_ta (ver sección 2).
- ¿Qué hacer con la gráfica del post original cuando exista el modelo nuevo?
  Opciones: congelarla o seguir mostrando las señales del modelo viejo.
- Limpiar `ml/environment.yml`: ahora es el export completo de `ml_env`
  (~154 líneas, con Jupyter y dependencias indirectas). Decidir qué dejar,
  por ejemplo solo las dependencias directas (`--from-history` más las de pip).
- **Regla de parada (decidida el 2026-10-18):** después del experimento 2, como
  mucho se prueba *triple barrier*. Luego se pasa a la fase 4 con el mejor
  candidato, o sin ninguno si ninguno cumple el criterio.
- **Para la fase 4 (backtest):** en *triple barrier*, los días que expiran
  tienen etiqueta 0, pero sí tuvieron un retorno real, que puede ser positivo.
  El backtest debe usar el retorno de la operación, no la etiqueta.
- **El periodo de test (2024-08-05 → 2026-10-16) está consumido (decidido el
  2026-10-18).**
  - Ya se miró en la fase 2, se conoce su tasa base y la fase 4 lo usa para los
    baselines.
  - Cualquier modelo futuro se valida **con datos posteriores al 2026-10-16**,
    es decir, con un snapshot nuevo que empiece después de esa fecha.

### Ideas para Part 3 (no empezar hasta publicar Part 2)
- **Hipótesis de Daniel:** la volatilidad baja indica acumulación y precede a
  un movimiento fuerte. La estrategia sería:
  - comprar en compresión (volatilidad baja);
  - si rompe al alza, mantener hasta la siguiente compresión y vender;
  - si rompe a la baja, comprar más en la siguiente compresión.
- **Indicio exploratorio (in-sample):**
  - en el cuartil de volatilidad más baja, el 45 % de los días acaban con una
    subida > 10 % a 30 días;
  - en los cuartiles centrales, entre el 29 % y el 32 %.

  Pendiente medir también las caídas fuertes desde compresión.
- **Por definir antes de ejecutar:**
  - umbral de compresión (percentil expansivo);
  - definición de ruptura;
  - tamaño por compra;
  - posición máxima.
- **Baselines:** buy & hold, DCA mensual y el baseline de vol de la fase 4.
- **Validación:** con datos posteriores al 2026-10-16, porque el test está
  consumido.
