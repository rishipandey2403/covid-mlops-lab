from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import mlflow

from src.data.load_data import load_dataset_from_config
from src.features.preprocess import clean_features, default_cleaning_spec_from_config
from src.models.mlflow_utils import log_artifact_if_exists, log_dict_params, setup_mlflow, start_run
from src.utils.io import ensure_dir, load_yaml


def make_splits(X, y, seed, test_size, val_size, stratify=True):
    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    strat2 = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=strat2
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def pick_threshold_max_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def eval_model(pipe, X_val, y_val, X_test, y_test) -> Dict:
    val_proba = pipe.predict_proba(X_val)[:, 1]
    t = pick_threshold_max_f1(y_val, val_proba)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= t).astype(int)

    metrics = {
        "threshold": float(t),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "positive_rate": float(y_test.mean()),
    }

    cm = confusion_matrix(y_test, test_pred)
    report = classification_report(y_test, test_pred, digits=4, zero_division=0)
    return {"metrics": metrics, "confusion_matrix": cm, "classification_report": report}


def load_pipeline(path: Path):
    bundle = joblib.load(path)
    return bundle["pipeline"]


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

    split_cfg = cfg["split"]
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        Xc,
        data.y,
        seed=seed,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        stratify=bool(split_cfg["stratify"]),
    )

    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])

    baseline_path = models_dir / "baseline_logreg.joblib"
    strong_xgb = models_dir / "strong_xgboost.joblib"
    strong_rf = models_dir / "strong_random_forest.joblib"
    if strong_xgb.exists():
        strong_path = strong_xgb
        strong_name = "strong_xgboost"
    elif strong_rf.exists():
        strong_path = strong_rf
        strong_name = "strong_random_forest"
    else:
        raise FileNotFoundError("No strong model artifact found. Run: python -m src.models.train_strong")

    baseline_pipe = load_pipeline(baseline_path)
    strong_pipe = load_pipeline(strong_path)

    baseline_eval = eval_model(baseline_pipe, X_val, y_val, X_test, y_test)
    strong_eval = eval_model(strong_pipe, X_val, y_val, X_test, y_test)

    # Save per-model reports locally
    def save_reports(prefix: str, out: Dict):
        metrics_path = Path(reports_dir) / f"{prefix}_test_metrics.json"
        cm_path = Path(reports_dir) / f"{prefix}_confusion_matrix.csv"
        report_path = Path(reports_dir) / f"{prefix}_classification_report.txt"

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"model": prefix, **out["metrics"]}, f, indent=2)

        cm = out["confusion_matrix"]
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(cm_path, index=True)

        with report_path.open("w", encoding="utf-8") as f:
            f.write(out["classification_report"])

        return metrics_path, cm_path, report_path

    b_metrics, b_cm, b_rep = save_reports("baseline_logreg", baseline_eval)
    s_metrics, s_cm, s_rep = save_reports(strong_name, strong_eval)

    comp = {
        "baseline": baseline_eval["metrics"],
        "strong": strong_eval["metrics"],
        "delta": {k: float(strong_eval["metrics"][k] - baseline_eval["metrics"][k]) for k in ["roc_auc", "pr_auc", "f1"]},
        "notes": {"thresholds": {"baseline": baseline_eval["metrics"]["threshold"], "strong": strong_eval["metrics"]["threshold"]}},
    }

    comp_path = Path(reports_dir) / "comparison_baseline_vs_strong.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)

    # Log to MLflow
    with start_run(
        run_name="compare_baseline_vs_strong",
        tags={"stage": "compare", "baseline": "baseline_logreg", "strong": strong_name},
    ):
        log_dict_params({"seed": seed, "split": split_cfg, "strong_model": strong_name})
        mlflow.log_metrics(
            {
                "baseline_test_roc_auc": comp["baseline"]["roc_auc"],
                "baseline_test_pr_auc": comp["baseline"]["pr_auc"],
                "baseline_test_f1": comp["baseline"]["f1"],
                "strong_test_roc_auc": comp["strong"]["roc_auc"],
                "strong_test_pr_auc": comp["strong"]["pr_auc"],
                "strong_test_f1": comp["strong"]["f1"],
                "delta_test_roc_auc": comp["delta"]["roc_auc"],
                "delta_test_pr_auc": comp["delta"]["pr_auc"],
                "delta_test_f1": comp["delta"]["f1"],
            }
        )

        for p in [b_metrics, b_cm, b_rep, s_metrics, s_cm, s_rep, comp_path]:
            log_artifact_if_exists(p)

    print("✅ Comparison logged to MLflow and saved locally")
    print(f"Saved comparison: {comp_path}")
    print("Baseline metrics:", baseline_eval["metrics"])
    print("Strong metrics:", strong_eval["metrics"])
    print("Deltas:", comp["delta"])


if __name__ == "__main__":
    main()

