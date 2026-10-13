"""Unit tests for src/preprocessing.py."""

import pytest
import pandas as pd
import numpy as np

from src.data_generator import generate_synthetic_attrition_data
from src.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
    prepare_train_test_data,
)
from src.config import TARGET_COLUMN


def test_prepare_train_test_data_no_leakage():
    """Verify stratified splitting retains class distribution and excludes target from X."""
    df = generate_synthetic_attrition_data(n_samples=200, random_state=42, output_path=None)
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_data(
        df, target_col=TARGET_COLUMN, test_size=0.25, random_state=42
    )

    assert len(X_train) == 150
    assert len(X_test) == 50
    assert len(y_train) == 150
    assert len(y_test) == 50
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns

    # Verify stratified balance is roughly equal
    train_pos_rate = np.mean(y_train)
    test_pos_rate = np.mean(y_test)
    assert abs(train_pos_rate - test_pos_rate) < 0.05


def test_build_preprocessor_transformation_and_imputation():
    """Verify ColumnTransformer handles missing values and categorical encoding."""
    num_cols = ["Age", "MonthlyIncome"]
    cat_cols = ["Department", "OverTime"]

    preprocessor = build_preprocessor(numerical_cols=num_cols, categorical_cols=cat_cols)

    # Data with a missing value in each type
    data = pd.DataFrame(
        {
            "Age": [25, np.nan, 45],
            "MonthlyIncome": [3000, 5000, np.nan],
            "Department": ["Sales", None, "Research & Development"],
            "OverTime": [None, "Yes", "No"],
        }
    )

    transformed = preprocessor.fit_transform(data)
    assert transformed.shape[0] == 3
    # No NaNs should remain in the output
    assert not np.isnan(transformed).any()

    # Feature names
    names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)
    assert len(names) == transformed.shape[1]
    assert names[0] == "Age"
    assert names[1] == "MonthlyIncome"

