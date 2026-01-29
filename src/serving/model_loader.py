from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


def _pick_best_model_path(models_dir: Path) -> Path:
    """
    Preference order:
    1) strong_xgboost
    2) strong_random_forest
    3) baseline_logreg

    This makes the API serve the best available model by default.
    """
    candidates = [
        models_dir / "strong_xgboost.joblib",
        models_dir / "strong_random_forest.joblib",
        models_dir / "baseline_logreg.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No model artifacts found in: {models_dir}")


def _load_threshold(reports_dir: Path, model_prefix: str) -> float:
    """
    Load tuned threshold from metrics json if present.
    Fallback to 0.5.
    """
    # We wrote threshold into *_test_metrics.json during compare phase.
    # Baseline: baseline_logreg_test_metrics.json
    # Strong: strong_xgboost_test_metrics.json or strong_random_forest_test_metrics.json
    metrics_path = reports_dir / f"{model_prefix}_test_metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            t = data.get("threshold", None)
            if t is not None:
                return float(t)
        except Exception:
            pass
    return 0.5


def load_model_bundle(
    models_dir: str | Path = "artifacts/models",
    reports_dir: str | Path = "artifacts/reports",
) -> Tuple[Any, Dict[str, Any], float, str]:
    """
    Returns:
      pipeline, config, threshold, model_name
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)

    model_path = _pick_best_model_path(models_dir)
    bundle = joblib.load(model_path)

    # Determine model prefix for threshold loading
    # File names: baseline_logreg.joblib, strong_xgboost.joblib, strong_random_forest.joblib
    model_prefix = model_path.stem  # e.g., "strong_xgboost"
    threshold = _load_threshold(reports_dir, model_prefix=model_prefix)

    pipeline = bundle["pipeline"]
    config = bundle.get("config", {})
    model_name = bundle.get("model_name", model_prefix)

    return pipeline, config, threshold, str(model_name)

