"""Data loading and validation module for Employee Attrition.

Handles programmatic schema discovery, target encoding, data validation,
and seamless fallback to synthetic demo data when IBM data is absent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

from src.config import (
    RAW_IBM_DATA_PATH,
    SYNTHETIC_DATA_PATH,
    TARGET_COLUMN,
    POSITIVE_TARGET_VALUE,
    NEGATIVE_TARGET_VALUE,
    TARGET_MAPPING,
    DEFAULT_DROP_COLUMNS,
    ALLOWED_CATEGORIES,
    NUMERIC_LIMITS,
)
from src.data_generator import generate_synthetic_attrition_data


@dataclass
class DatasetValidationReport:
    """Detailed summary of dataset integrity checks."""
    is_valid: bool
    row_count: int
    column_count: int
    missing_values_count: int
    duplicate_rows_count: int
    target_present: bool
    target_distribution: Dict[str, int]
    missing_columns: List[str]
    unexpected_columns: List[str]
    warnings: List[str]
    errors: List[str]
    is_synthetic: bool
    data_source_path: str


def discover_schema(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
    """Programmatically discovers schema types, unique values, and column roles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str, default=TARGET_COLUMN
        Name of target column.

    Returns
    -------
    Dict[str, Any]
        Discovered numerical columns, categorical columns, drop candidates,
        and value cardinalities.
    """
    candidate_drop = [c for c in DEFAULT_DROP_COLUMNS if c in df.columns]

    feature_cols = [c for c in df.columns if c != target_col and c not in candidate_drop]

    num_cols = []
    cat_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return {
        "target_col": target_col if target_col in df.columns else None,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "drop_cols": candidate_drop,
        "all_feature_cols": feature_cols,
    }


def validate_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    is_synthetic: bool = False,
    source_path: str = "In-memory",
) -> DatasetValidationReport:
    """Validates schema conformance, data types, and value boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to evaluate.
    target_col : str, default=TARGET_COLUMN
        Expected target column name.
    is_synthetic : bool, default=False
        Flag indicating if data is synthetic.
    source_path : str, default='In-memory'
        Path or label for origin.

    Returns
    -------
    DatasetValidationReport
        Structured report of validation findings.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if df.empty:
        return DatasetValidationReport(
            is_valid=False,
            row_count=0,
            column_count=0,
            missing_values_count=0,
            duplicate_rows_count=0,
            target_present=False,
            target_distribution={},
            missing_columns=[],
            unexpected_columns=[],
            warnings=[],
            errors=["Uploaded dataset is empty (0 rows)."],
            is_synthetic=is_synthetic,
            data_source_path=source_path,
        )

    row_count = len(df)
    column_count = len(df.columns)
    missing_count = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    target_present = target_col in df.columns
    target_dist: Dict[str, int] = {}

    if target_present:
        target_counts = df[target_col].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in target_counts.items()}
        # Check that target contains expected positive/negative classes
        str_keys = {str(k).strip() for k in target_counts.keys()}
        if not (POSITIVE_TARGET_VALUE in str_keys or 1 in target_counts):
            warnings.append(
                f"Target column '{target_col}' missing expected positive class '{POSITIVE_TARGET_VALUE}' (1)."
            )
    else:
        warnings.append(
            f"Target column '{target_col}' not found. Dataset can only be used for prediction inference, not model training."
        )

    # Check for expected core categorical columns
    missing_cols = []
    for cat_col in ALLOWED_CATEGORIES.keys():
        if cat_col not in df.columns:
            missing_cols.append(cat_col)

    for num_col in NUMERIC_LIMITS.keys():
        if num_col not in df.columns and num_col not in missing_cols:
            missing_cols.append(num_col)

    if missing_cols:
        warnings.append(
            f"Missing {len(missing_cols)} standard IBM schema column(s): {', '.join(missing_cols[:5])}..."
        )

    # Check category domain adherence
    for col, allowed_vals in ALLOWED_CATEGORIES.items():
        if col in df.columns:
            observed = set(df[col].dropna().astype(str).unique())
            unseen = observed - set(allowed_vals)
            if unseen:
                warnings.append(
                    f"Column '{col}' contains unexpected categories: {unseen}. These will be handled gracefully via one-hot unknown encoding."
                )

    is_valid = len(errors) == 0

    return DatasetValidationReport(
        is_valid=is_valid,
        row_count=row_count,
        column_count=column_count,
        missing_values_count=missing_count,
        duplicate_rows_count=dup_count,
        target_present=target_present,
        target_distribution=target_dist,
        missing_columns=missing_cols,
        unexpected_columns=[c for c in df.columns if c not in ALLOWED_CATEGORIES and c not in NUMERIC_LIMITS and c != target_col and c not in DEFAULT_DROP_COLUMNS],
        warnings=warnings,
        errors=errors,
        is_synthetic=is_synthetic,
        data_source_path=source_path,
    )


def load_dataset(
    file_path: Optional[Path] = None,
    ensure_fallback: bool = True,
) -> Tuple[pd.DataFrame, DatasetValidationReport]:
    """Loads dataset from path or seamlessly falls back to synthetic data.

    Parameters
    ----------
    file_path : Optional[Path], default=None
        Custom file path to load. If None, checks RAW_IBM_DATA_PATH then SYNTHETIC_DATA_PATH.
    ensure_fallback : bool, default=True
        Whether to generate synthetic dataset if no file exists on disk.

    Returns
    -------
    Tuple[pd.DataFrame, DatasetValidationReport]
        Loaded DataFrame and its associated validation report.
    """
    is_synthetic = False
    chosen_path: Optional[Path] = None

    if file_path is not None and Path(file_path).exists():
        chosen_path = Path(file_path)
        is_synthetic = "synthetic" in chosen_path.name.lower()
    elif RAW_IBM_DATA_PATH.exists():
        chosen_path = RAW_IBM_DATA_PATH
        is_synthetic = False
    elif SYNTHETIC_DATA_PATH.exists():
        chosen_path = SYNTHETIC_DATA_PATH
        is_synthetic = True
    elif ensure_fallback:
        # Generate synthetic data cleanly
        generate_synthetic_attrition_data(output_path=SYNTHETIC_DATA_PATH)
        chosen_path = SYNTHETIC_DATA_PATH
        is_synthetic = True
    else:
        raise FileNotFoundError(
            f"No dataset found at {RAW_IBM_DATA_PATH} or {SYNTHETIC_DATA_PATH}. "
            f"Please place the IBM HR Analytics CSV at '{RAW_IBM_DATA_PATH}' "
            "or set ensure_fallback=True."
        )

    df = pd.read_csv(chosen_path)
    report = validate_dataset(
        df=df,
        target_col=TARGET_COLUMN,
        is_synthetic=is_synthetic,
        source_path=str(chosen_path),
    )

    return df, report


def encode_target(
    y: pd.Series,
    target_mapping: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Safely encodes the attrition target into binary integer format (0/1).

    Ensures the positive class (1) reliably represents employees who left.

    Parameters
    ----------
    y : pd.Series
        Raw target column values.
    target_mapping : Optional[Dict[str, int]], default=None
        Custom mapping dictionary. If None, uses default TARGET_MAPPING.

    Returns
    -------
    np.ndarray
        1D binary numpy array of target values.
    """
    mapping = target_mapping or TARGET_MAPPING

    # If already integer binary (0 and 1)
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = set(y.dropna().unique())
        if unique_vals.issubset({0, 1}):
            return y.astype(int).to_numpy()

    # Map string representations
    y_str = y.astype(str).str.strip()
    encoded = y_str.map(mapping)

    if encoded.isna().any():
        unmapped = y_str[encoded.isna()].unique()
        raise ValueError(
            f"Unable to encode target values: {unmapped}. Expected {list(mapping.keys())}"
        )

    return encoded.astype(int).to_numpy()
