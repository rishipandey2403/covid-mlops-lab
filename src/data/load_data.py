from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io import load_yaml


@dataclass(frozen=True)
class LoadResult:
    X: pd.DataFrame
    y: pd.Series


def _infer_dtypes(columns: List[str]) -> Dict[str, str]:
    """
    Memory-aware dtype mapping for this dataset.
    - DATE_DIED must be read as string for label creation.
    - AGE fits in int16 (0..120).
    - Most coded fields fit in int8.
    """
    dtypes: Dict[str, str] = {}
    for c in columns:
        if c == "DATE_DIED":
            dtypes[c] = "string"
        elif c == "AGE":
            dtypes[c] = "int16"
        else:
            # Many columns are small coded integers (1/2/97/99 etc.)
            dtypes[c] = "int16"  # safe default; we can downcast later
    return dtypes


def load_raw_csv(
    csv_path: str | Path,
    sample_n: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load raw COVID CSV with optional row sampling for faster iteration.
    Sampling strategy:
      - If sample_n is provided, read full file index and sample rows with pandas.
      - For simplicity + reliability in teaching: load the file, then sample.
        (For very large files, later extension can do chunk sampling.)
    """
    csv_path = Path(csv_path)
    # Read header first
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    dtypes = _infer_dtypes(cols)

    df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
    # Downcast numeric columns where possible (saves RAM)
    for c in df.columns:
        if c != "DATE_DIED":
            df[c] = pd.to_numeric(df[c], downcast="integer")

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    return df


def make_label_from_date_died(
    df: pd.DataFrame,
    date_died_col: str,
    alive_code: str = "9999-99-99",
    target_col: str = "DIED",
) -> pd.DataFrame:
    """
    Create binary label:
      - DIED = 1 if DATE_DIED != alive_code
      - DIED = 0 if DATE_DIED == alive_code
    """
    if date_died_col not in df.columns:
        raise KeyError(f"Missing required column: {date_died_col}")

    date_series = df[date_died_col].astype("string")
    died = (date_series != alive_code).astype("int8")
    df = df.copy()
    df[target_col] = died
    return df


def select_features_and_target(
    df: pd.DataFrame,
    feature_numeric: List[str],
    feature_categorical: List[str],
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> LoadResult:
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    feature_cols = feature_numeric + feature_categorical
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return LoadResult(X=X, y=y)


def load_dataset_from_config(
    config_path: str | Path = "configs/config.yaml",
    sample_n: Optional[int] = None,
) -> LoadResult:
    cfg = load_yaml(config_path)

    seed = int(cfg["project"]["seed"])
    raw_csv = cfg["paths"]["raw_csv"]

    date_died_col = cfg["data"]["date_died_col"]
    alive_code = cfg["data"]["alive_code"]
    target_col = cfg["data"]["target_col"]

    feat_num = cfg["data"]["features"]["numeric"]
    feat_cat = cfg["data"]["features"]["categorical"]
    drop_cols = cfg["data"].get("drop_cols", [])

    df = load_raw_csv(raw_csv, sample_n=sample_n, seed=seed)
    df = make_label_from_date_died(df, date_died_col=date_died_col, alive_code=alive_code, target_col=target_col)
    return select_features_and_target(df, feat_num, feat_cat, target_col, drop_cols=drop_cols)


def main() -> None:
    """
    Smoke test:
    - loads a sample
    - prints shape + label distribution
    """
    res = load_dataset_from_config(sample_n=50_000)
    y = res.y
    print("X shape:", res.X.shape)
    print("y distribution:")
    print(y.value_counts(dropna=False).to_string())
    print("y positive rate:", float(y.mean()))


if __name__ == "__main__":
    main()

