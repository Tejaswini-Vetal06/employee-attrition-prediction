"""Inference and single/batch prediction module for Employee Attrition.

Provides schema-validated inference pipelines, configurable risk thresholds,
probabilistic risk tier classification, and contextual interpretations.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
    NUMERIC_LIMITS,
    ALLOWED_CATEGORIES,
    ETHICAL_DISCLAIMER,
)


def load_trained_model(
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> Tuple[Any, Dict[str, Any]]:
    """Safely loads persisted Scikit-Learn pipeline and training metadata.

    Parameters
    ----------
    model_path : Path
        Path to best pipeline .joblib artifact.
    metadata_path : Path
        Path to model metadata .joblib artifact.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (pipeline, metadata)

    Raises
    ------
    FileNotFoundError
        If model artifacts do not exist on disk.
    """
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Trained model artifacts not found at {model_path} or {metadata_path}. "
            "Please run the training pipeline first using 'python -m src.train'."
        )

    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return pipeline, metadata


def validate_inference_input(
    input_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Tuple[bool, List[str], List[str]]:
    """Validates single employee input against schema boundaries and allowed categories.

    Parameters
    ----------
    input_data : Dict[str, Any]
        Dictionary of input feature values.
    metadata : Dict[str, Any]
        Model metadata containing expected schema.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    required_features = metadata["feature_schema"]["all_features"]

    # Check for missing required features
    missing = [f for f in required_features if f not in input_data]
    if missing:
        errors.append(f"Missing required input features: {', '.join(missing[:5])}...")

    # Check categorical domains
    for cat_col, allowed in ALLOWED_CATEGORIES.items():
        if cat_col in input_data:
            val = str(input_data[cat_col])
            if val not in allowed:
                warnings.append(
                    f"Category '{val}' for '{cat_col}' was not in training domain. "
                    "Will be handled gracefully via unknown category encoding."
                )

    # Check numeric ranges
    for num_col, limits in NUMERIC_LIMITS.items():
        if num_col in input_data:
            val = input_data[num_col]
            try:
                numeric_val = float(val)
                if numeric_val < limits["min"] or numeric_val > limits["max"]:
                    warnings.append(
                        f"Value {numeric_val} for '{num_col}' lies outside standard training range "
                        f"[{limits['min']}, {limits['max']}]."
                    )
            except (ValueError, TypeError):
                errors.append(f"Value for '{num_col}' must be numeric, received: {val}")

    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def predict_employee_attrition(
    input_dict: Dict[str, Any],
    pipeline: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    decision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes attrition probability, risk classification, and guidance for a single employee record.

    Parameters
    ----------
    input_dict : Dict[str, Any]
        Dictionary containing employee attributes.
    pipeline : Optional[Any], default=None
        Pre-loaded fitted pipeline. If None, loaded from disk.
    metadata : Optional[Dict[str, Any]], default=None
        Pre-loaded metadata. If None, loaded from disk.
    decision_threshold : Optional[float], default=None
        Threshold for positive risk classification. If None, uses metadata optimal threshold.

    Returns
    -------
    Dict[str, Any]
        Comprehensive prediction payload with probability, risk tier, interpretation, and safeguards.
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = load_trained_model()

    if decision_threshold is None:
        decision_threshold = float(metadata.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD))

    # Validate input
    is_valid, warnings, errors = validate_inference_input(input_dict, metadata)
    if not is_valid:
        raise ValueError(f"Inference input validation failed: {'; '.join(errors)}")

    # Format into DataFrame with correct columns
    feature_cols = metadata["feature_schema"]["all_features"]
    row_df = pd.DataFrame([{col: input_dict.get(col, np.nan) for col in feature_cols}])

    # Compute probability
    probs = pipeline.predict_proba(row_df)[0]
    prob_stay = float(probs[0])
    prob_leave = float(probs[1])

    is_flagged = prob_leave >= decision_threshold

    # Define Risk Tier
    if prob_leave < 0.25:
        risk_tier = "Low Retention Risk"
        risk_color = "#10B981"  # Emerald
        tier_description = "Employee exhibits strong historical retention alignment."
    elif prob_leave < 0.50:
        risk_tier = "Moderate Retention Risk"
        risk_color = "#F59E0B"  # Amber
        tier_description = "Employee shows minor risk signals; periodic check-ins recommended."
    else:
        risk_tier = "High Retention Risk"
        risk_color = "#EF4444"  # Red
        tier_description = "Significant historical attrition indicators detected; leadership engagement advised."

    # Cautious result interpretation
    interpretation = (
        f"The model estimates a **{prob_leave * 100:.1f}% probability of departure** based on historical patterns. "
        f"Using the calibrated operating threshold of **{decision_threshold * 100:.1f}%**, this profile is "
        f"**{'flagged for proactive retention review' if is_flagged else 'not flagged as high-risk'}**. "
        "Remember that statistical probability is NOT an outcome certainty. "
        "Individual decisions are shaped by nuanced, personal, and situational factors."
    )

    return {
        "probability_attrition": prob_leave,
        "probability_retention": prob_stay,
        "probability_percentage": round(prob_leave * 100, 1),
        "threshold_used": decision_threshold,
        "is_flagged_for_review": is_flagged,
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "tier_description": tier_description,
        "interpretation": interpretation,
        "warnings": warnings,
        "ethical_disclaimer": ETHICAL_DISCLAIMER,
        "model_used": metadata.get("selected_model_name", "Scikit-Learn Classifier"),
        "model_version": metadata.get("model_version", "1.0.0"),
    }

