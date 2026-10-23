# ml/ — Decision Tree BTC

Plan completo: `../docs/PLAN-ml-decision-tree.md`.

## Reglas innegociables
- Nunca usar como target algo calculado a partir de las propias features
  (el modelo original solo reproducía las reglas: accuracy 99 % sin valor).
- Features solo relativas/estacionarias (ratios, RSI, retornos), nunca precios en bruto.
- Validación temporal (walk-forward, sin shuffle, con embargo = horizonte).
- Comparar siempre contra baselines (buy & hold, estrategia de reglas).
- Reportar resultados con honestidad aunque no batan al buy & hold.
- El test ya está consumido: un modelo futuro se valida con datos posteriores al 2026-10-16.

## Entorno
- Conda env `ml_env` (Python 3.12.13; kernel "Python (ml_env)"), versiones en `environment.yml`.
- Ejecutar: `conda run -n ml_env python <script>` · Recrear: `conda env create -f environment.yml`.
- `ml_env` y el CI usan scikit-learn 1.9.1. Solo `ml_env` tiene pandas_ta (para el
  baseline de la fase 2 y los 6 tests que lo comparan; en CI se saltan).
  Únicamente `features_original.py` usa pandas_ta.

## Comandos (desde `ml/`, con `conda run -n ml_env`)
- Tests: `python -m pytest`
- Fase 2, modelo base: `python -m btc_decision_tree.baseline`
- Fase 3, walk-forward: `python -m btc_decision_tree.model --target {return,vol,barrier,all}`
  (por defecto `return`; `all` imprime la tabla de veredictos). No evalúa el test.
- Fase 4, backtest de baselines: `python -m btc_decision_tree.backtest`
- Fase 5, gráficas diarias: `python -m btc_decision_tree.live --images-dir <carpeta>`
  (sin `--images-dir` escribe en `content/images/`).
- Snapshot nuevo: `python -m btc_decision_tree.data --end YYYY-MM-DD` (`end` exclusivo).
  Los experimentos leen `DEFAULT_SNAPSHOT` de `data.py`, nunca yfinance directamente.

## Modelos (`../models/`)
- `decision_tree.plk`: árbol original (sklearn 1.6.1). Solo para los tests de fidelidad.
- `decision_tree_sklearn-1.9.1.joblib`: el mismo árbol regenerado; lo usa la gráfica.
- Regenerar: `python -m btc_decision_tree.baseline --end 2025-12-13 --random-state 2 --save ../models/decision_tree_sklearn-1.9.1.joblib`
  - `--end` corta en el día en que se entrenó el original.
  - La semilla 2 se eligió por fidelidad al desempate del original (`export_text`
    idéntico), no por rendimiento; las predicciones son iguales con cualquier semilla.
  - Si cambia la versión de sklearn del CI: regenerar con el nuevo nombre y actualizar `paths.py`.
- No copia las reglas al 100 %: difiere en 5 días desde 2022-12-01, en dos zonas
  conocidas que comprueba `tests/test_live_model.py`.
- El código de entrenamiento original NO está en el repo: `../../ML/BTCpriceOracle/`
  (fuera de git). `Untitled.ipynb` = código del post; `cleanPipeline.ipynb` = experimento
  "good_sell" (no publicado). Ningún notebook guarda el `.plk`.
