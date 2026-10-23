# CI y despliegue

Workflow: `.github/workflows/deploy.yml`. Se lanza en push a `main`, por cron diario
(06:00 UTC) y a mano. Runner `ubuntu-24.04`, acciones con Node 24.

## Pasos
1. Instala `requirements-ci.txt` (todo fijado, sin pandas_ta).
2. `pytest ml`: si falla, no hay deploy.
3. `scripts/update_charts.sh` (gráficas diarias).
4. Build de Pelican y deploy a `gh-pages` (el CNAME se crea en CI).

Las PNG solo se publican en `gh-pages`; no se commitean a `main`.
`make github` existe pero no se usa.

## Gráficas diarias
`scripts/update_charts.sh` ejecuta `ml/btc_decision_tree/live.py`, que descarga el
histórico desde 2014 y genera:
- `btc-latest.png`: señales del árbol original, para el post original.
- `btc-tracker.png`: reglas frente a buy & hold desde 2025-12-13, para la Part 2.

Fallback: si `live.py` sale con **código 3** (yfinance falla o datos con más de 3 días),
reutiliza las PNG de `gh-pages`, avisa con `::warning::` y publica. Con **otro código**,
el job falla sin deploy.

Probar el fallback (solo en lanzamiento manual):
`gh workflow run deploy.yml -f simulate_data_failure=true`

Regenerar en local: `bash scripts/update_charts.sh` desde la raíz.

## Diagnóstico
- Ejecuciones: `gh run list --workflow deploy.yml`
- Logs: `gh api repos/DCEtech/DCEtechWeb/actions/jobs/<job_id>/logs`
  (`gh run view --log` sale vacío).
