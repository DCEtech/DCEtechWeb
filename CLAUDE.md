# DCEtechWeb — danielcejudo.com

Web personal tipo blog/portfolio de Daniel. Objetivo: mostrar capacidades como
software developer a reclutadores técnicos y responsables de pymes. Escaparate
de la futura consultora Odoo, pero NO es el punto de entrada comercial.

## Stack
- Pelican 4.12.0 (fijado en `requirements-ci.txt`). Local: no instalado en el
  pyenv 3.10.12. Tema propio en `theme/mytheme/`.
- Plugins: ninguno (no hay `PLUGINS` en `pelicanconf.py`).
- Contenido: posts en `content/*.md`, páginas en `content/pages/`, imágenes en
  `content/images/`, CSS en `content/css/custom.css`. `images/` en la raíz es
  solo para el README.
- Hosting: GitHub Pages (rama `gh-pages`), dominio www.danielcejudo.com (CNAME
  creado en CI).
- **CI** (`.github/workflows/deploy.yml`), en push a `main`, por cron diario
  (06:00 UTC) y a mano. Pasos:
  1. instala `requirements-ci.txt`, todo fijado y sin pandas_ta;
  2. `pytest ml`: si falla, no hay deploy;
  3. `scripts/update_charts.sh`;
  4. build de Pelican y deploy a `gh-pages`.
  - Runner `ubuntu-24.04` y acciones con Node 24.
  - Las PNG solo se publican en `gh-pages`; no se commitean a `main`.
- **Gráficas diarias**: `scripts/update_charts.sh` ejecuta `ml/btc_decision_tree/live.py`.
  - Descarga el histórico completo desde 2014 y genera dos imágenes:
    - `btc-latest.png`: señales del árbol original, en el post original;
    - `btc-tracker.png`: reglas frente a buy & hold desde el 2025-12-13, para la
      Part 2.
  - Si `live.py` sale con **código 3** (yfinance falla o los datos tienen más de
    3 días), reutiliza las PNG de `gh-pages` que existan, avisa con
    `::warning::` y publica. Con **otro código**, el job falla sin deploy.
  - Para probar el fallback en GitHub:
    `gh workflow run deploy.yml -f simulate_data_failure=true`. Solo existe en el
    lanzamiento manual.
- Revisar ejecuciones con `gh run list --workflow deploy.yml`. Los logs se
  obtienen con `gh api repos/DCEtech/DCEtechWeb/actions/jobs/<job_id>/logs`,
  porque `gh run view --log` sale vacío.
- **Modelos**:
  - `models/decision_tree.plk`: el árbol original, de sklearn 1.6.1. Solo lo
    usan los tests de fidelidad.
  - `models/decision_tree_sklearn-1.9.1.joblib`: el mismo árbol regenerado. Es
    el que usa la gráfica.
  - Para regenerarlo, desde `ml/`:
    `python -m btc_decision_tree.baseline --end 2025-12-13 --random-state 2 --save ../models/decision_tree_sklearn-1.9.1.joblib`.
    - `--end` corta los datos el día en que se entrenó el original.
    - **La semilla 2 se eligió por fidelidad al desempate del original, no por
      rendimiento**: con ella, `export_text` sale idéntico. Las predicciones son
      las mismas con cualquier semilla.
    - Si cambia la versión de sklearn del CI, se regenera con el nuevo nombre y
      se actualiza `paths.py`.
  - El árbol **no** copia las reglas al 100 %: difiere en 5 días desde el
    2022-12-01, en dos zonas conocidas que comprueba `test_live_model.py`.
- El código de entrenamiento del modelo NO está en el repo. Original en
  `../ML/BTCpriceOracle/` (relativo a la raíz del repo, fuera de git):
  `Untitled.ipynb` = mismo código que el post; `cleanPipeline.ipynb` = experimento
  posterior "good_sell" (no es el modelo publicado). Ningún notebook guarda el `.plk`.

## Entorno Python (ML)
- Conda env `ml_env` (Python 3.12.13; kernel de Jupyter "Python (ml_env)").
  Versiones exportadas en `ml/environment.yml` (versionado en git).
- Ejecutar: `conda run -n ml_env python <script>`.
- Recrear: `conda env create -f ml/environment.yml`.
- `ml_env` y el CI usan scikit-learn 1.9.1. `ml_env` además tiene pandas_ta,
  solo para el baseline de la fase 2 y los tests que comparan con él; en CI
  esos 6 tests se saltan.

## Comandos
- Build: `make html` (= `pelican content -o output -s pelicanconf.py`)
- Servidor local con recarga: `make devserver` (http://localhost:8000)
- Build producción: `make publish` (usa `publishconf.py`)
- Regenerar las gráficas: `bash scripts/update_charts.sh` desde la raíz del
  repo, o `python -m btc_decision_tree.live --images-dir <carpeta>` desde
  `ml/` para no tocar `content/images/`.
- Publicar: push a `main` (GitHub Actions); `make github` existe pero no se usa.
- ML (desde `ml/`, con `conda run -n ml_env`):
  - Tests: `python -m pytest` (o `python -m pytest ml` desde la raíz)
  - Modelo base (fase 2): `python -m btc_decision_tree.baseline`
  - Walk-forward del modelo nuevo (fase 3): `python -m btc_decision_tree.model
    --target {return,vol,barrier,all}` (por defecto `return`; `all` imprime la
    tabla de veredictos de los 3 experimentos). No evalúa el test.
  - Backtest de baselines (fase 4): `python -m btc_decision_tree.backtest`
    (train con mediana expansiva, folds, test y post-publicación). El test ya está
    consumido: un modelo futuro se valida con datos posteriores al 2026-10-16.
  - Gráficas diarias (fase 5): `python -m btc_decision_tree.live`.
  - `features.py`, `indicators.py`, `backtest.py` y `live.py` no usan pandas_ta;
    solo `features_original.py` (para el baseline) lo usa.
  - Snapshot nuevo de datos: `python -m btc_decision_tree.data --end YYYY-MM-DD`
    (`end` exclusivo). Los experimentos leen `DEFAULT_SNAPSHOT` de `data.py`,
    nunca yfinance directamente.

## Reglas de trabajo
- Antes de cambiar algo de Pelican, explica el porqué y respeta la estructura
  existente del proyecto (tema propio, plugins, rutas).
- Daniel está aprendiendo: explica la lógica de cada cambio, no solo el código.
- Python: type hints, manejo de excepciones explícito, funciones pequeñas y
  testeables. Nada de lógica importante solo en notebooks.

## Posts (en inglés)
- Daniel escribe en inglés para mejorar. Al revisar: corrige y explica
  brevemente cada error (gramática, naturalidad, vocabulario técnico).
- Errores recurrentes conocidos: `´` en vez de apóstrofo `'`; ordinales `1º`;
  letras dobles (recommendation, professional, useful); tiempos pasados.
- Encabezados con `##`, bloques de código con lenguaje (```python), enlace al
  código fuente cuando el post incluya código.

## Proyecto ML: Decision Tree BTC
Ver `docs/PLAN-ml-decision-tree.md`. Reglas innegociables:
- Nunca usar como target algo calculado a partir de las propias features
  (el modelo original solo reproducía las reglas: accuracy 99 % sin valor).
- Features solo relativas/estacionarias (ratios, RSI, retornos), nunca
  precios en bruto.
- Validación temporal (walk-forward, sin shuffle, con embargo = horizonte).
- Comparar siempre contra baselines (buy & hold, estrategia de reglas).
- Reportar resultados con honestidad aunque no batan al buy & hold.
