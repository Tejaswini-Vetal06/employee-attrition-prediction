"""Page 4: Model Performance & Comparative Benchmarking.

Evaluates and compares candidate classification architectures (Logistic Regression,
Random Forest, Gradient Boosting) using cross-validation and held-out test data.
Features interactive ROC curves, Precision-Recall curves, confusion matrices, and
in-depth class imbalance diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import (
    inject_custom_css,
    render_page_header,
    render_metric_card,
    get_plotly_layout,
    get_cached_pipeline_and_metadata,
)
from src.config import THEME_COLORS, ALL_MODELS_PATH
from src.evaluate import compute_metrics
import joblib

st.set_page_config(
    page_title="Model Performance | RetainIQ",
    page_icon="📈",
    layout="wide",
)

inject_custom_css()

render_page_header(
    title="Model Benchmarking & Evaluation",
    subtitle="Comparative performance analysis across Scikit-Learn classification pipelines on held-out test data.",
    badge_label="Model Selection",
    badge_type="info",
)

# Load metadata and artifacts
try:
    pipeline, metadata = get_cached_pipeline_and_metadata()
    candidate_results = metadata.get("candidate_results", {})
    selected_model_name = metadata.get("selected_model_name", "Logistic Regression")
except Exception as e:
    st.error(f"Unable to load model evaluation metadata: {e}. Please run `python -m src.train`.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Selection Strategy & Class Imbalance Rationale
# ---------------------------------------------------------------------------
st.markdown("### 🏆 Production Model Selection")

sel_col1, sel_col2 = st.columns([2, 1])

with sel_col1:
    st.markdown(
        f"""
        <div class="saas-card" style="border-left: 5px solid {THEME_COLORS['primary']};">
            <h4 style="margin-top: 0; color: #0F172A;">Winner: {selected_model_name}</h4>
            <p style="color: #334155; font-size: 0.92rem; line-height: 1.6;">
                {metadata.get('selection_rationale', '')}
            </p>
            <div style="font-size: 0.85rem; color: #64748B;">
                <strong>Primary Metric:</strong> Cross-Validated ROC-AUC | 
                <strong>Validation Folds:</strong> 5-Fold Stratified K-Fold
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sel_col2:
    st.markdown(
        f"""
        <div class="saas-card" style="border-left: 5px solid {THEME_COLORS['warning']};">
            <h4 style="margin-top: 0; color: #0F172A;">⚠️ Why Accuracy is Misleading</h4>
            <p style="color: #475569; font-size: 0.85rem; line-height: 1.5;">
                {metadata.get('imbalance_explanation', 'Raw accuracy fails in imbalanced domains.')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 2: Comparative Metric Tables
# ---------------------------------------------------------------------------
st.markdown("### 📊 Architecture Comparison Matrix")

# Build comparison DataFrame
comp_rows = []
for model_name, res in candidate_results.items():
    cv = res.get("cv_metrics", {})
    test = res.get("test_metrics", {})

    comp_rows.append(
        {
            "Candidate Architecture": model_name,
            "CV ROC-AUC (Mean ± Std)": f"{cv.get('roc_auc_mean', 0):.3f} ± {cv.get('roc_auc_std', 0):.3f}",
            "Test ROC-AUC": f"{test.get('roc_auc', 0):.3f}",
            "Test PR-AUC": f"{test.get('pr_auc', 0):.3f}",
            "Test Recall": f"{test.get('recall', 0):.3f}",
            "Test Precision": f"{test.get('precision', 0):.3f}",
            "Test F1-Score": f"{test.get('f1', 0):.3f}",
            "Test Accuracy": f"{test.get('accuracy', 0):.3f}",
        }
    )

df_comp = pd.DataFrame(comp_rows)
st.dataframe(df_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Interactive ROC Curves & Precision-Recall Curves
# ---------------------------------------------------------------------------
st.markdown("### 📈 Discrimination Curves")

curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    fig_roc = go.Figure()
    # Random baseline
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8"),
            name="Random Baseline (AUC = 0.50)",
        )
    )

    colors = [THEME_COLORS["primary"], THEME_COLORS["secondary"], "#EC4899"]
    for i, (m_name, res) in enumerate(candidate_results.items()):
        test_m = res.get("test_metrics", {})
        roc_data = test_m.get("roc_curve", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc_val = test_m.get("roc_auc", 0.0)

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{m_name} (AUC = {auc_val:.3f})",
                line=dict(color=colors[i % len(colors)], width=2.5),
            )
        )

    fig_roc.update_layout(
        get_plotly_layout("Receiver Operating Characteristic (ROC) Curves", height=420),
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall)",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with curve_col2:
    fig_pr = go.Figure()

    for i, (m_name, res) in enumerate(candidate_results.items()):
        test_m = res.get("test_metrics", {})
        pr_data = test_m.get("pr_curve", {})
        prec = pr_data.get("precision", [])
        rec = pr_data.get("recall", [])
        pr_auc = test_m.get("pr_auc", 0.0)

        fig_pr.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"{m_name} (PR-AUC = {pr_auc:.3f})",
                line=dict(color=colors[i % len(colors)], width=2.5),
            )
        )

    fig_pr.update_layout(
        get_plotly_layout("Precision-Recall (PR) Curves", height=420),
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Confusion Matrix with Dynamic Threshold Calibration
# ---------------------------------------------------------------------------
st.markdown("### 🎯 Decision Boundary & Confusion Matrix")

st.markdown(
    "Test set evaluation using dynamic decision threshold calibration for the winning model:"
)

# Load test predictions if available
if ALL_MODELS_PATH.exists():
    try:
        all_artifacts = joblib.load(ALL_MODELS_PATH)
        test_data = all_artifacts.get("test_data", {})
        X_test = test_data.get("X_test")
        y_test = test_data.get("y_test")

        if X_test is not None and y_test is not None:
            winning_pipeline = all_artifacts["pipelines"][selected_model_name]
            y_test_probs = winning_pipeline.predict_proba(X_test)[:, 1]

            optimal_t = float(metadata.get("optimal_threshold", 0.50))
            thresh_slider = st.slider(
                "Adjust Classification Decision Threshold:",
                min_value=0.10,
                max_value=0.90,
                value=optimal_t,
                step=0.01,
                help=f"Validation Youden's J optimal threshold is {optimal_t:.3f}.",
            )

            # Recompute metrics dynamically
            dyn_metrics = compute_metrics(y_test, y_test_probs, threshold=thresh_slider)
            cm = dyn_metrics["confusion_matrix"]

            cm_col1, cm_col2 = st.columns([1, 1])

            with cm_col1:
                # Plotly Confusion Matrix Heatmap
                z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
                x_labels = ["Predicted Stayed (0)", "Predicted Left (1)"]
                y_labels = ["Actual Stayed (0)", "Actual Left (1)"]

                fig_cm = px.imshow(
                    z,
                    x=x_labels,
                    y=y_labels,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(color="Count"),
                )
                fig_cm.update_layout(
                    get_plotly_layout(f"Confusion Matrix @ Threshold = {thresh_slider:.2f}", height=340),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with cm_col2:
                st.markdown("#### Performance at Selected Threshold")
                m1, m2 = st.columns(2)
                with m1:
                    render_metric_card("Precision", f"{dyn_metrics['precision']:.1%}", "True positives / All flagged", THEME_COLORS["primary"])
                    st.write("")
                    render_metric_card("Accuracy", f"{dyn_metrics['accuracy']:.1%}", "Overall correct rate", THEME_COLORS["navy"])
                with m2:
                    render_metric_card("Recall (Sensitivity)", f"{dyn_metrics['recall']:.1%}", "Captured true leavers", THEME_COLORS["danger"])
                    st.write("")
                    render_metric_card("F1-Score", f"{dyn_metrics['f1']:.3f}", "Harmonic mean", THEME_COLORS["secondary"])
    except Exception as e:
        st.warning(f"Interactive confusion matrix evaluation error: {e}")
