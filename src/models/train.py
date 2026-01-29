from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import mlflow

from src.data.load_data import load_dataset_from_config
from src.features.preprocess import build_preprocessor, clean_features, default_cleaning_spec_from_config
from src.models.mlflow_utils import log_artifact_if_exists, log_dict_params, setup_mlflow, start_run
from src.utils.io import ensure_dir, load_yaml


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float,
    val_size: float,
    stratify: bool = True,
) -> SplitData:
    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    strat2 = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=strat2
    )
    return SplitData(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)


def build_baseline_pipeline(cfg: dict) -> Pipeline:
    num_feats = cfg["data"]["features"]["numeric"]
    cat_feats = cfg["data"]["features"]["categorical"]

    pre = build_preprocessor(numeric_features=num_feats, categorical_features=cat_feats)

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=int(cfg["project"]["seed"]),
    )
    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def quick_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def main() -> None:
    cfg = load_yaml("configs/config.yaml")
    seed = int(cfg["project"]["seed"])

    # MLflow setup
    exp_name = cfg["project"]["name"]
    setup_mlflow(exp_name)

    # Load + clean
    data = load_dataset_from_config("configs/config.yaml", sample_n=None)
    spec = default_cleaning_spec_from_config(cfg)
    Xc = clean_features(data.X, spec=spec)

    # Splits
    split_cfg = cfg["split"]
    sdata = make_splits(
        Xc,
        data.y,
        seed=seed,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        stratify=bool(split_cfg["stratify"]),
    )

    # Train
    pipe = build_baseline_pipeline(cfg)

    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])
    model_path = Path(models_dir) / "baseline_logreg.joblib"
    metrics_path = Path(reports_dir) / "baseline_val_metrics.json"

    with start_run(
        run_name="baseline_logreg_train",
        tags={"stage": "train", "model_family": "logreg", "model_name": "baseline_logreg"},
    ):
        # log config params (selected)
        log_dict_params(
            {
                "seed": seed,
                "split": split_cfg,
                "target": cfg["data"]["target_col"],
                "features_numeric": cfg["data"]["features"]["numeric"],
                "features_categorical": cfg["data"]["features"]["categorical"],
            }
        )

        # log model hyperparams
        log_dict_params(pipe.get_params(), prefix="sklearn.")

        pipe.fit(sdata.X_train, sdata.y_train)

        # val metrics
        val_proba = pipe.predict_proba(sdata.X_val)[:, 1]
        val_metrics = quick_metrics(sdata.y_val, val_proba)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Save artifacts to disk (then log to MLflow)
        joblib.dump(
            {
                "pipeline": pipe,
                "config": cfg,
                "feature_spec": {
                    "numeric": cfg["data"]["features"]["numeric"],
                    "categorical": cfg["data"]["features"]["categorical"],
                },
            },
            model_path,
        )

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"split": "val", **val_metrics}, f, indent=2)

        log_artifact_if_exists(model_path)
        log_artifact_if_exists(metrics_path)

        # Also log model via mlflow.sklearn
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(pipe, artifact_path="model")
        except Exception:
            pass

    print("✅ Baseline training logged to MLflow")
    print(f"Saved model: {model_path}")
    print(f"Saved val metrics: {metrics_path}")


if __name__ == "__main__":
    main()

