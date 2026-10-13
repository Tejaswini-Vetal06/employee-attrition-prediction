"""Data preprocessing and feature transformation pipeline module.

Constructs reusable Scikit-Learn ColumnTransformer and Pipeline objects with
stratified data splitting, missing value imputation, and feature scaling.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COLUMN,
    DEFAULT_DROP_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import discover_schema, encode_target


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Constructs a Scikit-Learn ColumnTransformer for numerical and categorical features.

    Parameters
    ----------
    numerical_cols : List[str]
        List of numerical feature names.
    categorical_cols : List[str]
        List of categorical feature names.
    scale_numeric : bool, default=True
        Whether to apply StandardScaler to numerical features.

    Returns
    -------
    ColumnTransformer
        Configured, un-fitted ColumnTransformer.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Retrieves post-transformation feature names including one-hot encoded categories.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer.
    numerical_cols : List[str]
        Original numerical column names.
    categorical_cols : List[str]
        Original categorical column names.

    Returns
    -------
    List[str]
        Complete ordered list of output feature names.
    """
    feature_names: List[str] = list(numerical_cols)

    if "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        if "onehot" in cat_pipeline.named_steps:
            onehot = cat_pipeline.named_steps["onehot"]
            try:
                cat_names = onehot.get_feature_names_out(categorical_cols).tolist()
                feature_names.extend(cat_names)
            except Exception:
                # Fallback if get_feature_names_out fails
                for i, col in enumerate(categorical_cols):
                    feature_names.append(f"{col}_cat_{i}")

    return feature_names


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """Splits raw DataFrame into stratified training and testing partitions.

    Prepares X and y, dropping non-predictive ID / constant columns. Ensures
    no data leakage by postponing transformer fitting to training routines.

    Parameters
    ----------
    df : pd.DataFrame
        Complete raw dataset.
    target_col : str, default=TARGET_COLUMN
        Target column name.
    test_size : float, default=0.20
        Proportion of dataset to reserve for held-out testing.
    random_state : int, default=42
        Deterministic random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]
        X_train, X_test, y_train, y_test, numerical_cols, categorical_cols.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Supervised training requires a labeled target."
        )

    # Encode target cleanly
    y = encode_target(df[target_col])

    # Discover columns
    schema_info = discover_schema(df, target_col=target_col)
    drop_cols = schema_info["drop_cols"]
    cols_to_drop = [c for c in drop_cols if c in df.columns] + [target_col]

    X = df.drop(columns=cols_to_drop)

    numerical_cols = schema_info["numerical_cols"]
    categorical_cols = schema_info["categorical_cols"]

    # Stratified split to maintain positive class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols

