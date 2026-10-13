"""Unit tests for src/data_loader.py and src/data_generator.py."""

import pytest
import pandas as pd
import numpy as np

from src.data_generator import generate_synthetic_attrition_data
from src.data_loader import (
    discover_schema,
    validate_dataset,
    encode_target,
    load_dataset,
)
from src.config import TARGET_COLUMN


def test_synthetic_data_generation():
    """Verify synthetic data generator produces correct shape and schema."""
    df = generate_synthetic_attrition_data(n_samples=100, random_state=42, output_path=None)
    assert len(df) == 100
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({"Yes", "No"})
    assert "Department" in df.columns
    assert "JobRole" in df.columns
    assert "MonthlyIncome" in df.columns
    assert "OverTime" in df.columns


def test_discover_schema():
    """Verify programmatic schema discovery extracts correct column types."""
    df = generate_synthetic_attrition_data(n_samples=50, random_state=42, output_path=None)
    schema = discover_schema(df, target_col=TARGET_COLUMN)

    assert schema["target_col"] == TARGET_COLUMN
    assert "Age" in schema["numerical_cols"]
    assert "Department" in schema["categorical_cols"]
    assert "EmployeeCount" in schema["drop_cols"]
    assert TARGET_COLUMN not in schema["all_feature_cols"]


def test_encode_target_valid():
    """Verify consistent binary target encoding (Yes=1, No=0)."""
    series = pd.Series(["No", "Yes", "No", "Yes"])
    encoded = encode_target(series)
    np.testing.assert_array_equal(encoded, np.array([0, 1, 0, 1]))


def test_encode_target_invalid():
    """Verify invalid target labels raise ValueError."""
    series = pd.Series(["No", "Maybe", "Yes"])
    with pytest.raises(ValueError):
        encode_target(series)


def test_validate_empty_dataset():
    """Verify empty dataframe fails validation gracefully with error."""
    empty_df = pd.DataFrame()
    report = validate_dataset(empty_df)
    assert not report.is_valid
    assert len(report.errors) > 0
    assert report.row_count == 0


def test_validate_missing_target():
    """Verify dataset missing target is marked as invalid for training."""
    df = generate_synthetic_attrition_data(n_samples=50, random_state=42, output_path=None)
    df_no_target = df.drop(columns=[TARGET_COLUMN])
    report = validate_dataset(df_no_target)
    assert not report.target_present
    assert any("not found" in w for w in report.warnings)


def test_load_dataset_fallback():
    """Verify load_dataset runs without raising FileNotFoundError."""
    df, report = load_dataset(ensure_fallback=True)
    assert not df.empty
    assert report.is_valid

