from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def replace_unknown_codes_with_nan(
    df: pd.DataFrame,
    cols: Sequence[str],
    unknown_codes: Sequence[int] = (97, 98, 99),
) -> pd.DataFrame:
    """
    Replace dataset-specific unknown/not-applicable codes with NaN.
    Works for integer-coded columns (e.g., 1/2/97/99).
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(list(unknown_codes), np.nan)
    return df


def map_binary_1_2_to_0_1(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    """
    Many fields use 1 = Yes, 2 = No.
    Convert to 1/0 (keeping NaN as NaN).
    """
    df = df.copy()
    mapping = {1: 1, 2: 0}
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(mapping).astype("float32")  # float to allow NaN
    return df


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Sklearn preprocessor:
    - numeric: median impute
    - categorical: most_frequent impute + one-hot
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class CleaningSpec:
    numeric_features: List[str]
    categorical_features: List[str]
    unknown_codes: List[int]

    # Columns that are truly binary 1/2 (yes/no) in this dataset
    # We keep some coded columns as categorical (e.g., MEDICAL_UNIT, CLASIFFICATION_FINAL)
    binary_1_2_cols: List[str]


def default_cleaning_spec_from_config(cfg: dict) -> CleaningSpec:
    feat_num = cfg["data"]["features"]["numeric"]
    feat_cat = cfg["data"]["features"]["categorical"]
    unknown = cfg["data"]["unknown_codes"]

    # Binary yes/no columns among the selected features
    binary = [
        "PNEUMONIA",
        "PREGNANT",
        "DIABETES",
        "COPD",
        "ASTHMA",
        "INMSUPR",
        "HIPERTENSION",
        "OTHER_DISEASE",
        "CARDIOVASCULAR",
        "OBESITY",
        "RENAL_CHRONIC",
        "TOBACCO",
        "USMER",
        "PATIENT_TYPE",
        "SEX",
    ]
    # Keep only those present in categorical features to avoid surprises
    binary = [c for c in binary if c in feat_cat]

    return CleaningSpec(
        numeric_features=list(feat_num),
        categorical_features=list(feat_cat),
        unknown_codes=list(unknown),
        binary_1_2_cols=binary,
    )


def clean_features(
    X: pd.DataFrame,
    spec: CleaningSpec,
) -> pd.DataFrame:
    """
    Clean X for modeling:
    1) replace unknown codes with NaN on all modeled columns
    2) convert selected 1/2 columns to 1/0 (keep NaN)
    """
    cols = list(spec.numeric_features) + list(spec.categorical_features)
    Xc = replace_unknown_codes_with_nan(X, cols=cols, unknown_codes=spec.unknown_codes)
    Xc = map_binary_1_2_to_0_1(Xc, cols=spec.binary_1_2_cols)

    # Ensure AGE is numeric (can become object if upstream changed)
    if "AGE" in Xc.columns:
        Xc["AGE"] = pd.to_numeric(Xc["AGE"], errors="coerce")

    return Xc

