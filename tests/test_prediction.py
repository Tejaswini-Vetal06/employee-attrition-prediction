"""Unit tests for src/predict.py and inference pipelines."""

import pytest
import numpy as np

from src.predict import (
    load_trained_model,
    predict_employee_attrition,
    validate_inference_input,
)
from src.config import BEST_MODEL_PATH, MODEL_METADATA_PATH


@pytest.fixture(scope="module")
def model_artifacts():
    """Loads saved model pipeline and metadata fixture."""
    pipeline, metadata = load_trained_model(BEST_MODEL_PATH, MODEL_METADATA_PATH)
    return pipeline, metadata


def test_predict_single_employee_valid(model_artifacts):
    """Verify inference pipeline returns valid probability and risk classification."""
    pipeline, metadata = model_artifacts

    valid_input = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Medical",
        "EnvironmentSatisfaction": 3,
        "Gender": "Female",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5200,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": 1,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }

    result = predict_employee_attrition(
        input_dict=valid_input,
        pipeline=pipeline,
        metadata=metadata,
        decision_threshold=0.45,
    )

    assert 0.0 <= result["probability_attrition"] <= 1.0
    assert 0.0 <= result["probability_retention"] <= 1.0
    assert abs(result["probability_attrition"] + result["probability_retention"] - 1.0) < 1e-4
    assert result["risk_tier"] in [
        "Low Retention Risk",
        "Moderate Retention Risk",
        "High Retention Risk",
    ]
    assert isinstance(result["is_flagged_for_review"], bool)
    assert "interpretation" in result
    assert "ethical_disclaimer" in result


def test_predict_missing_feature_raises_error(model_artifacts):
    """Verify input missing critical features raises descriptive error."""
    pipeline, metadata = model_artifacts

    incomplete_input = {"Age": 30, "MonthlyIncome": 4000}  # Missing other features

    with pytest.raises(ValueError) as excinfo:
        predict_employee_attrition(
            input_dict=incomplete_input,
            pipeline=pipeline,
            metadata=metadata,
        )
    assert "Missing required input features" in str(excinfo.value)


def test_validate_inference_input_warning_for_unknown_category(model_artifacts):
    """Verify unknown categories generate informative warnings rather than silent failure."""
    _, metadata = model_artifacts

    input_with_novel_cat = {
        col: 1 if col in metadata["feature_schema"]["numerical_cols"] else "UnknownCategory"
        for col in metadata["feature_schema"]["all_features"]
    }

    is_valid, warnings, errors = validate_inference_input(input_with_novel_cat, metadata)
    assert any("not in training domain" in w for w in warnings)

