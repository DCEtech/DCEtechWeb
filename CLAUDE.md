# DCEtechWeb — danielcejudo.com

Blog/portfolio de Daniel para reclutadores técnicos y responsables de pymes.
Escaparate de la futura consultora Odoo, pero NO es el punto de entrada comercial.

## Stack
- Pelican 4.12.0 (fijado en `requirements-ci.txt`), sin plugins. Tema propio en `theme/mytheme/`.
- Contenido: posts en `content/*.md`, páginas en `content/pages/`, imágenes en
  `content/images/`, CSS en `content/css/custom.css`. `images/` (raíz) es solo para el README.
- Hosting: GitHub Pages (rama `gh-pages`), dominio www.danielcejudo.com.
- Publicar = push a `main`; lo despliega GitHub Actions. Detalle del CI y del
  fallback de gráficas: `docs/ci-deploy.md` (léelo antes de tocar el workflow o `scripts/`).
- `ml/` es un subproyecto (Decision Tree BTC) con su propio `ml/CLAUDE.md`.

## Comandos
- Build: `make html` · Servidor local con recarga: `make devserver` (http://localhost:8000)
- Build producción: `make publish` (usa `publishconf.py`)

## Reglas de trabajo
- Antes de cambiar algo de Pelican o del tema, explica el porqué y respeta la
  estructura existente (tema propio, rutas, configuración).
- Daniel está aprendiendo: explica la lógica de cada cambio, no solo el código.
- Python: type hints, excepciones explícitas, funciones pequeñas y testeables.
- Posts en inglés: para escribir o revisar uno, usa la skill `english-post-review`.
