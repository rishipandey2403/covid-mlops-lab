from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def setup_mlflow(experiment_name: str, tracking_uri: str = "file:./mlruns") -> None:
    """
    Use local file-based tracking so the lab runs offline.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_dict_params(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Flatten and log params as strings (MLflow params are key/value strings).
    """
    for k, v in params.items():
        key = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            log_dict_params(v, prefix=f"{key}.")
        else:
            mlflow.log_param(key, v)


def log_artifact_if_exists(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p))


def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run

