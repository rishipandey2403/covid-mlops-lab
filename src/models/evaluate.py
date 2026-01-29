from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset_from_config
from src.features.preprocess import clean_features, default_cleaning_spec_from_config
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
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def pick_threshold_max_f1(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Choose threshold that maximizes F1 on validation set.
    Returns (best_threshold, best_f1).
    """
    thresholds = np.linspace(0.05, 0.95, 19)  # coarse grid for clarity
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
    }


def main() -> None:
    cfg = load_yaml("configs/config.yaml")
    seed = int(cfg["project"]["seed"])

    # Load + clean
    data = load_dataset_from_config("configs/config.yaml", sample_n=None)
    spec = default_cleaning_spec_from_config(cfg)
    Xc = clean_features(data.X, spec=spec)

    # Same split logic as training
    split_cfg = cfg["split"]
    sdata = make_splits(
        Xc,
        data.y,
        seed=seed,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        stratify=bool(split_cfg["stratify"]),
    )

    # Load trained model
    model_path = Path(cfg["paths"]["models_dir"]) / "baseline_logreg.joblib"
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]

    # Validation: threshold tuning
    val_proba = pipe.predict_proba(sdata.X_val)[:, 1]
    best_t, best_f1 = pick_threshold_max_f1(sdata.y_val, val_proba)

    # Test evaluation using tuned threshold
    test_proba = pipe.predict_proba(sdata.X_test)[:, 1]
    metrics = compute_metrics(sdata.y_test, test_proba, threshold=best_t)

    y_test_pred = (test_proba >= best_t).astype(int)
    cm = confusion_matrix(sdata.y_test, y_test_pred)

    # Save reports
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])
    metrics_path = Path(reports_dir) / "baseline_test_metrics.json"
    cm_path = Path(reports_dir) / "confusion_matrix.csv"
    report_path = Path(reports_dir) / "classification_report.txt"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "baseline_logreg",
                "threshold_tuned_on": "val",
                "best_val_f1": best_f1,
                **metrics,
            },
            f,
            indent=2,
        )

    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(cm_path, index=True)

    report_str = classification_report(sdata.y_test, y_test_pred, digits=4, zero_division=0)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_str)

    print("✅ Evaluation complete")
    print(f"Loaded model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved classification report: {report_path}")
    print("\nTest metrics:", metrics)
    print("\nClassification report:\n", report_str)


if __name__ == "__main__":
    main()

