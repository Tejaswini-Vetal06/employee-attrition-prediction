"""Unit tests for src/train.py and src/evaluate.py."""

import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_attrition_data
from src.train import get_candidate_models, train_and_evaluate_models
from src.evaluate import compute_metrics, find_optimal_threshold, get_imbalance_explanation


def test_candidate_models_instantiation():
    """Verify all 3 candidate models instantiate with random states."""
    models = get_candidate_models()
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "Gradient Boosting" in models


def test_compute_metrics_validity():
    """Verify metric calculations for binary predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["confusion_matrix"]["tp"] == 3
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_find_optimal_threshold():
    """Verify optimal threshold computation returns sensible bound."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
    opt_t = find_optimal_threshold(y_true, y_prob)
    assert 0.15 <= opt_t <= 0.75


def test_train_pipeline_execution():
    """Verify training executes cleanly and returns fitted pipeline with complete metadata."""
    df_small = generate_synthetic_attrition_data(n_samples=150, random_state=42, output_path=None)
    pipeline, metadata = train_and_evaluate_models(df=df_small, save_artifacts=False)

    assert pipeline is not None
    assert "selected_model_name" in metadata
    assert metadata["selected_model_name"] in ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    assert "best_cv_score" in metadata
    assert metadata["best_cv_score"] > 0.50
    assert "test_metrics" in metadata
    assert "feature_schema" in metadata
