#!/usr/bin/env bash
# Genera las gráficas diarias de BTC (ml/btc_decision_tree/live.py).
#
# Según el código de salida de live.py:
#   0  -> gráficas nuevas, se sigue normalmente.
#   3  -> yfinance falló o los datos están desactualizados: se reutilizan las
#         imágenes publicadas en la rama gh-pages (solo las que existan), se
#         avisa en el log de GitHub Actions y se sigue con el deploy.
#   otro -> error real (bug): el job falla y no hay deploy.
#
# Variables para probarlo en local:
#   LIVE_CMD    comando que genera las gráficas (por defecto, live.py)
#   IMAGES_DIR  carpeta de las imágenes (por defecto, content/images)
#   PAGES_REMOTE / PAGES_BRANCH  rama publicada (por defecto, origin gh-pages)
set -uo pipefail

LIVE_CMD="${LIVE_CMD:-cd ml && python -m btc_decision_tree.live}"
IMAGES_DIR="${IMAGES_DIR:-content/images}"
PAGES_REMOTE="${PAGES_REMOTE:-origin}"
PAGES_BRANCH="${PAGES_BRANCH:-gh-pages}"
DOWNLOAD_FAILED=3
IMAGES=(btc-latest.png btc-tracker.png)

bash -c "$LIVE_CMD"
status=$?

if [ "$status" -eq 0 ]; then
  echo "Gráficas actualizadas."
  exit 0
fi

if [ "$status" -ne "$DOWNLOAD_FAILED" ]; then
  echo "::error::live.py terminó con el código $status; no se publica."
  exit "$status"
fi

echo "::warning::No se pudieron actualizar los datos de BTC; se reutilizan las imágenes publicadas en $PAGES_BRANCH."
if ! git fetch --quiet --depth=1 "$PAGES_REMOTE" "$PAGES_BRANCH"; then
  echo "::warning::No se pudo leer $PAGES_BRANCH; se publican las imágenes que ya hay en el repo."
  exit 0
fi

mkdir -p "$IMAGES_DIR"
for name in "${IMAGES[@]}"; do
  if git cat-file -e "FETCH_HEAD:images/$name" 2>/dev/null; then
    git show "FETCH_HEAD:images/$name" > "$IMAGES_DIR/$name"
    echo "Reutilizada $name de $PAGES_BRANCH."
  else
    echo "::warning::$name no existe en $PAGES_BRANCH; se omite."
  fi
done
exit 0
