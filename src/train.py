"""Model training, cross-validation, selection, and persistence module.

Trains and evaluates multiple candidate classifiers (Logistic Regression,
Random Forest, HistGradientBoosting), applies stratified cross-validation,
prevents data leakage, and persists the optimal production pipeline.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_MODELS_PATH,
    BEST_MODEL_PATH,
    CV_FOLDS,
    MODEL_METADATA_PATH,
    PRIMARY_SELECTION_METRIC,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_dataset
from src.evaluate import compute_metrics, get_imbalance_explanation
from src.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
    prepare_train_test_data,
)


def get_candidate_models() -> Dict[str, Any]:
    """Returns initialized candidate classifier algorithms with class-weight handling."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate_models(
    df: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Executes full training lifecycle: CV evaluation, test scoring, model selection, and serialization.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        Input dataset. If None, loaded via load_dataset().
    save_artifacts : bool, default=True
        Whether to serialize trained pipeline and metadata to disk.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        The best fitted Pipeline and complete evaluation metadata dictionary.
    """
    is_synthetic = False
    source_path = "In-memory"

    if df is None:
        df, report = load_dataset()
        is_synthetic = report.is_synthetic
        source_path = report.data_source_path

    # Prepare data without leakage
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_data(
        df=df,
        target_col=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = get_candidate_models()
    cv_splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    candidate_results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    best_model_name: Optional[str] = None
    best_cv_score: float = -1.0

    scoring_metrics = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    for name, clf in models.items():
        # Preprocessor: scale numerical features for Logistic Regression, optional for trees
        preprocessor = build_preprocessor(
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            scale_numeric=True,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        # 5-Fold Stratified Cross-Validation strictly on training data
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring_metrics,
            return_train_score=False,
            n_jobs=-1,
        )

        mean_roc_auc = float(np.mean(cv_res["test_roc_auc"]))
        std_roc_auc = float(np.std(cv_res["test_roc_auc"]))

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test set
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_prob, threshold=0.50)

        candidate_results[name] = {
            "cv_metrics": {
                "roc_auc_mean": mean_roc_auc,
                "roc_auc_std": std_roc_auc,
                "accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
                "precision_mean": float(np.mean(cv_res["test_precision"])),
                "recall_mean": float(np.mean(cv_res["test_recall"])),
                "f1_mean": float(np.mean(cv_res["test_f1"])),
            },
            "test_metrics": test_metrics,
        }

        # Track best model by primary selection metric (Cross-Validated ROC-AUC)
        if mean_roc_auc > best_cv_score:
            best_cv_score = mean_roc_auc
            best_model_name = name

    assert best_model_name is not None
    best_pipeline = fitted_pipelines[best_model_name]
    best_test_metrics = candidate_results[best_model_name]["test_metrics"]

    # Extract transformed feature names
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_transformed_feature_names(
        fitted_preprocessor, num_cols, cat_cols
    )

    attrition_rate = float(np.mean(y_train))

    metadata: Dict[str, Any] = {
        "model_version": "1.0.0",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_model_name": best_model_name,
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "selection_rationale": (
            f"Selected '{best_model_name}' based on the highest {CV_FOLDS}-fold stratified "
            f"cross-validation {PRIMARY_SELECTION_METRIC.upper()} score ({best_cv_score:.4f}). "
            "ROC-AUC evaluates discrimination capacity across all decision thresholds without "
            "being artificially inflated by majority class dominance."
        ),
        "best_cv_score": best_cv_score,
        "candidate_results": candidate_results,
        "test_metrics": best_test_metrics,
        "optimal_threshold": best_test_metrics["optimal_threshold"],
        "attrition_rate_train": attrition_rate,
        "imbalance_explanation": get_imbalance_explanation(attrition_rate),
        "feature_schema": {
            "numerical_cols": num_cols,
            "categorical_cols": cat_cols,
            "all_features": list(X_train.columns),
            "transformed_feature_names": transformed_feature_names,
        },
        "dataset_provenance": {
            "is_synthetic": is_synthetic,
            "source_path": source_path,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    }

    if save_artifacts:
        ALL_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, BEST_MODEL_PATH)
        joblib.dump(metadata, MODEL_METADATA_PATH)
        joblib.dump(
            {
                "pipelines": fitted_pipelines,
                "candidate_results": candidate_results,
                "test_data": {"X_test": X_test, "y_test": y_test},
            },
            ALL_MODELS_PATH,
        )

    return best_pipeline, metadata


if __name__ == "__main__":
    print("Initiating model training and cross-validation...")
    pipeline, meta = train_and_evaluate_models()
    print(f"\nTraining Complete!")
    print(f"Selected Model: {meta['selected_model_name']}")
    print(f"Validation ROC-AUC: {meta['best_cv_score']:.4f}")
    print(f"Test ROC-AUC: {meta['test_metrics']['roc_auc']:.4f}")
    print(f"Optimal Threshold: {meta['optimal_threshold']:.3f}")
    print(f"Artifacts saved to {BEST_MODEL_PATH.parent}")
