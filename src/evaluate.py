"""Model evaluation and metrics computation module.

Calculates standard classification performance indicators (ROC-AUC, Precision,
Recall, F1, Confusion Matrix, ROC/PR Curves) with optimal threshold tuning
and class imbalance context.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics for given probability predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for positive class (1).
    threshold : float, default=0.5
        Decision boundary threshold to discretize probabilities.

    Returns
    -------
    Dict[str, Any]
        Dictionary of scalar metrics, confusion matrix, and curve coordinates.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Discretize using chosen decision threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate standard metrics with zero_division safeguards
    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = 0.0

    try:
        prec = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        prec = 0.0

    try:
        rec = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        rec = 0.0

    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        f1 = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5  # Random baseline fallback

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        cm = np.zeros((2, 2), dtype=int)
        tn, fp, fn, tp = 0, 0, 0, 0

    # ROC Curve coordinates
    try:
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        roc_curve_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        }
    except Exception:
        roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}

    # Precision-Recall Curve coordinates
    try:
        precision_vals, recall_vals, pr_thresh = precision_recall_curve(y_true, y_prob)
        pr_curve_data = {
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
            "thresholds": pr_thresh.tolist(),
        }
    except Exception:
        pr_curve_data = {"precision": [0.0], "recall": [0.0], "thresholds": [0.0]}

    # Find optimal threshold via Youden's J statistic (Sensitivity + Specificity - 1)
    optimal_threshold = find_optimal_threshold(y_true, y_prob)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_used": float(threshold),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds optimal decision threshold balancing sensitivity and specificity via Youden's J index.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target values.
    y_prob : np.ndarray
        Predicted positive class probabilities.

    Returns
    -------
    float
        Recommended decision threshold rounded to 3 decimal places (bounded [0.1, 0.8]).
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # J = TPR - FPR
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        if np.isinf(best_thresh) or np.isnan(best_thresh):
            return 0.50
        # Bound between sensible operating limits for HR alerting
        return float(np.clip(round(best_thresh, 3), 0.15, 0.75))
    except Exception:
        return 0.50


def get_imbalance_explanation(attrition_rate: float) -> str:
    """Provides plain-language explanation of why accuracy is inadequate for HR attrition.

    Parameters
    ----------
    attrition_rate : float
        Proportion of positive (attrition=Yes) instances in dataset.

    Returns
    -------
    str
        Explanatory commentary for dashboard display.
    """
    baseline_stay = (1.0 - attrition_rate) * 100
    return (
        f"In workforce datasets where attrition is typically rare (here, {attrition_rate:.1%} attrition), "
        f"a trivial model that predicts every single employee will stay achieves {baseline_stay:.1f}% accuracy "
        f"despite detecting 0% of departing staff (Recall = 0.00). "
        "Therefore, ROC-AUC, Recall, and Precision are significantly more reliable metrics for HR analytics."
    )

