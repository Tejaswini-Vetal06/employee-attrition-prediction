"""Model explainability and feature importance analysis module.

Extracts model-specific coefficients/feature importances, permutation importance,
and optional SHAP explanations with plain-English interpretations and ethical boundaries.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.config import RANDOM_STATE


def extract_global_feature_importance(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extracts intrinsic feature importance or model coefficients.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-Learn pipeline.
    metadata : Dict[str, Any]
        Model metadata containing feature names.
    top_n : int, default=15
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Ranked feature importance DataFrame with directionality and descriptions.
    """
    feature_names = metadata["feature_schema"]["transformed_feature_names"]
    clf = pipeline.named_steps["classifier"]

    importances = []
    directions = []
    raw_weights = []

    if isinstance(clf, LogisticRegression):
        coefs = clf.coef_[0]
        # Align lengths in case of mismatch
        min_len = min(len(feature_names), len(coefs))
        f_names = feature_names[:min_len]
        coefs = coefs[:min_len]

        for val in coefs:
            raw_weights.append(float(val))
            importances.append(float(abs(val)))
            directions.append("Increases Attrition Risk" if val > 0 else "Reduces Attrition Risk (Protective)")

    elif hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        min_len = min(len(feature_names), len(fi))
        f_names = feature_names[:min_len]
        fi = fi[:min_len]

        for val in fi:
            raw_weights.append(float(val))
            importances.append(float(val))
            directions.append("Significant Driver")
    else:
        # Fallback uniform
        f_names = feature_names
        importances = [1.0 / len(feature_names)] * len(feature_names)
        raw_weights = importances
        directions = ["Factor"] * len(feature_names)

    df_importance = pd.DataFrame(
        {
            "Feature": f_names,
            "Importance": importances,
            "Raw_Weight": raw_weights,
            "Impact_Direction": directions,
        }
    )

    df_importance = df_importance.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_importance.head(top_n)


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Computes model-agnostic permutation feature importance on held-out test data.

    Measures mean drop in ROC-AUC when each feature is randomly shuffled.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-Learn pipeline.
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : np.ndarray
        True target array.
    n_repeats : int, default=5
        Number of permutation iterations.
    random_state : int, default=42
        Deterministic random state.

    Returns
    -------
    pd.DataFrame
        Sorted dataframe of permutation importance scores.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    features = list(X_test.columns)
    df_perm = pd.DataFrame(
        {
            "Feature": features,
            "Mean_ROC_AUC_Drop": result.importances_mean,
            "Std_Dev": result.importances_std,
        }
    ).sort_values(by="Mean_ROC_AUC_Drop", ascending=False).reset_index(drop=True)

    return df_perm


def compute_shap_explanations(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Generates SHAP values for a sample dataset if compatible.

    Gracefully falls back if SHAP fails or model architecture is incompatible.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-Learn pipeline.
    X_sample : pd.DataFrame
        Representative sample of employee features.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing mean absolute SHAP values and transformed feature names,
        or None if computation is unavailable.
    """
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        # Transform features
        X_trans = preprocessor.transform(X_sample)

        # Pick appropriate explainer
        if isinstance(clf, (RandomForestClassifier, HistGradientBoostingClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans)
            # In binary classification, choose positive class values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                values = shap_values[:, :, 1]
            else:
                values = shap_values
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_trans)
            values = explainer.shap_values(X_trans)
        else:
            explainer = shap.Explainer(clf, X_trans)
            values = explainer(X_trans).values

        mean_abs_shap = np.mean(np.abs(values), axis=0)
        return {
            "mean_abs_shap": mean_abs_shap.tolist(),
            "status": "success",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


def explain_feature_in_plain_english(feature_name: str, direction: str) -> str:
    """Translates statistical feature patterns into constructive HR workforce narratives.

    Parameters
    ----------
    feature_name : str
        Name of the feature.
    direction : str
        Direction of impact (Increases Risk or Protective).

    Returns
    -------
    str
        Contextual narrative.
    """
    narratives = {
        "OverTime": (
            "Consistent overtime demands correlate with burnout and departure. "
            "Leadership should examine team bandwidth and rebalance workloads."
        ),
        "MonthlyIncome": (
            "Compensation competitiveness is a primary retention pillar. "
            "Employees below market benchmarks exhibit higher turnover propensity."
        ),
        "YearsWithCurrManager": (
            "Managerial rapport significantly stabilizes tenure. "
            "Frequent supervisory changes often disrupt employee engagement."
        ),
        "JobSatisfaction": (
            "Day-to-day role contentment strongly shields against outside solicitations."
        ),
        "EnvironmentSatisfaction": (
            "Physical workplace culture and operational climate directly impact employee well-being."
        ),
        "DistanceFromHome": (
            "Extended daily commute times compound fatigue and work-life imbalance."
        ),
        "BusinessTravel": (
            "Frequent business travel elevates personal strain and turnover likelihood."
        ),
        "YearsSinceLastPromotion": (
            "Stagnation in career progression often motivates employees to explore external opportunities."
        ),
        "StockOptionLevel": (
            "Financial equity alignment creates mutual long-term commitment and retention incentives."
        ),
    }

    clean_name = feature_name.split("_")[0]
    base_explanation = narratives.get(
        clean_name,
        f"This factor demonstrates a statistical association with {direction.lower()} across historical records."
    )

    return base_explanation
