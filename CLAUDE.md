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
- Gráfica BTC: `.github/workflows/deploy.yml` corre en push a `main` y por cron
  diario (06:00 UTC): `scripts/update_btc.py` carga `models/decision_tree.plk`,
  predice y guarda `content/images/btc-latest.png`; luego build y deploy. La PNG
  actualizada solo se publica en `gh-pages`, no se commitea a `main`.
  CI estabilizado el 2026-10-17: runner `ubuntu-24.04`, todas las dependencias
  fijadas en `requirements-ci.txt`, acciones con Node 24 (checkout@v5,
  setup-python@v6, actions-gh-pages@v4) y lanzamiento manual con
  `gh workflow run deploy.yml`. Pendiente: el CI carga el `.plk` (sklearn 1.6.1)
  con sklearn 1.9.1 (`InconsistentVersionWarning`); se resuelve al reentrenar.
  Revisar con `gh run list --workflow deploy.yml`. Los logs se obtienen con
  `gh api repos/DCEtech/DCEtechWeb/actions/jobs/<job_id>/logs` (`gh run view --log` sale vacío).
- El código de entrenamiento del modelo NO está en el repo. Original en
  `../ML/BTCpriceOracle/` (relativo a la raíz del repo, fuera de git):
  `Untitled.ipynb` = mismo código que el post; `cleanPipeline.ipynb` = experimento
  posterior "good_sell" (no es el modelo publicado). Ningún notebook guarda el `.plk`.

## Entorno Python (ML)
- Conda env `ml_env` (Python 3.12.13; kernel de Jupyter "Python (ml_env)").
  Versiones exportadas en `ml/environment.yml` (versionado en git).
- Ejecutar: `conda run -n ml_env python <script>`.
- Recrear: `conda env create -f ml/environment.yml`.
- Ojo: `ml_env` tiene scikit-learn 1.8.0, pero `models/decision_tree.plk` es de
  la 1.6.1, y el CI instala versiones sin fijar.

## Comandos
- Build: `make html` (= `pelican content -o output -s pelicanconf.py`)
- Servidor local con recarga: `make devserver` (http://localhost:8000)
- Build producción: `make publish` (usa `publishconf.py`)
- Regenerar gráfica: `python scripts/update_btc.py` (desde la raíz del repo)
- Publicar: push a `main` (GitHub Actions); `make github` existe pero no se usa.
- ML (desde `ml/`, con `conda run -n ml_env`):
  - Tests: `python -m pytest` (o `python -m pytest ml` desde la raíz)
  - Modelo base (fase 2): `python -m btc_decision_tree.baseline`
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
