"""Rutas del repo usadas por el código de ML."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
IMAGES_DIR = REPO_ROOT / "content" / "images"

# Árbol publicado en diciembre de 2025 (scikit-learn 1.6.1). Se conserva para
# los tests de fidelidad; la gráfica usa el regenerado.
ORIGINAL_MODEL_PATH = MODELS_DIR / "decision_tree.plk"
# Mismo árbol regenerado con la versión del CI (ver baseline.py y CLAUDE.md).
LIVE_MODEL_PATH = MODELS_DIR / "decision_tree_sklearn-1.9.1.joblib"
