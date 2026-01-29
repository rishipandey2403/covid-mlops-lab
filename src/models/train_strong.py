from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
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


def make_splits(X, y, seed, test_size, val_size, stratify=True) -> SplitData:
    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    strat2 = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=strat2
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def quick_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def build_strong_model(seed: int):
    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        return model, "xgboost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
        return model, "random_forest"


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

    # Build pipeline
    num_feats = cfg["data"]["features"]["numeric"]
    cat_feats = cfg["data"]["features"]["categorical"]
    pre = build_preprocessor(numeric_features=num_feats, categorical_features=cat_feats)

    model, model_name = build_strong_model(seed=seed)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])

    model_path = Path(models_dir) / f"strong_{model_name}.joblib"
    metrics_path = Path(reports_dir) / f"strong_{model_name}_val_metrics.json"

    with start_run(
        run_name=f"strong_{model_name}_train",
        tags={"stage": "train", "model_family": model_name, "model_name": f"strong_{model_name}"},
    ):
        log_dict_params(
            {
                "seed": seed,
                "split": split_cfg,
                "target": cfg["data"]["target_col"],
                "features_numeric": num_feats,
                "features_categorical": cat_feats,
                "model_name": model_name,
            }
        )

        # hyperparams
        try:
            log_dict_params(model.get_params(), prefix="model.")
        except Exception:
            pass

        pipe.fit(sdata.X_train, sdata.y_train)

        # val metrics
        val_proba = pipe.predict_proba(sdata.X_val)[:, 1]
        val_metrics = quick_metrics(sdata.y_val, val_proba)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Save artifacts
        joblib.dump(
            {"pipeline": pipe, "config": cfg, "model_name": model_name, "feature_spec": {"numeric": num_feats, "categorical": cat_feats}},
            model_path,
        )

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"split": "val", "model": f"strong_{model_name}", **val_metrics}, f, indent=2)

        log_artifact_if_exists(model_path)
        log_artifact_if_exists(metrics_path)

        # log model via mlflow (sklearn wrapper works for pipeline too)
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(pipe, artifact_path="model")
        except Exception:
            pass

    print("✅ Strong model training logged to MLflow")
    print(f"Model used: {model_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved val metrics: {metrics_path}")


if __name__ == "__main__":
    main()

