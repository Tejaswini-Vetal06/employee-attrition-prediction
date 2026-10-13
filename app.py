"""Employee Attrition Prediction & HR Analytics Platform - Main Application.

Entry point for the Streamlit dashboard providing executive navigation,
system telemetry, architecture overview, and responsible AI governance.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure project root is in sys.path for clean module imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    RAW_IBM_DATA_PATH,
    SYNTHETIC_DATA_PATH,
    THEME_COLORS,
)
from src.utils import (
    inject_custom_css,
    render_page_header,
    render_provenance_badge,
    render_ethical_banner,
    render_metric_card,
    get_cached_dataset,
    get_cached_pipeline_and_metadata,
)

# Page configuration
st.set_page_config(
    page_title="Workforce Intelligence | Attrition AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified SaaS styling
inject_custom_css()

# Sidebar Branding & Global Telemetry
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">💼</span>
                <div>
                    <h2 style="font-size: 1.15rem; margin: 0; color: #F8FAFC; font-weight: 700;">RetainIQ Analytics</h2>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Workforce Intelligence Suite</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System Status")

    # Data Status
    try:
        df, report = get_cached_dataset()
        if report.is_synthetic:
            st.markdown("📊 **Data:** `Synthetic Demo` (seed 42)")
        else:
            st.markdown("📊 **Data:** `Enterprise IBM Data`")
        st.markdown(f"👥 **Records:** `{report.row_count:,}` employees")
    except Exception as e:
        st.error(f"Data loading error: {e}")

    # Model Status
    try:
        pipeline, metadata = get_cached_pipeline_and_metadata()
        st.markdown(f"🤖 **Model:** `{metadata.get('selected_model_name', 'Active')}`")
        st.markdown(f"🎯 **Validation ROC-AUC:** `{metadata.get('best_cv_score', 0):.3f}`")
        st.markdown(f"⚡ **Threshold:** `{metadata.get('optimal_threshold', 0.5):.2f}`")
    except Exception:
        st.warning("⚠️ Model artifacts missing. Run `python -m src.train` or train via Data Management.")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748B;">
            <strong>RetainIQ Platform v1.0.0</strong><br/>
            Compliant with Responsible AI & Human Review Standards.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Page Header
render_page_header(
    title="Employee Attrition Prediction & Workforce Analytics",
    subtitle="AI-powered predictive modeling and exploratory telemetry for proactive talent retention.",
    badge_label="Enterprise Edition",
    badge_type="info",
)

# Provenance alert
try:
    _, rep = get_cached_dataset()
    render_provenance_badge(rep.is_synthetic)
except Exception:
    pass

# Ethical Banner
render_ethical_banner()

# Quick System Metrics
col1, col2, col3, col4 = st.columns(4)

try:
    df, report = get_cached_dataset()
    pipeline, metadata = get_cached_pipeline_and_metadata()

    with col1:
        render_metric_card(
            title="Total Workforce",
            value=f"{report.row_count:,}",
            subtext=f"{report.column_count} attributes tracked",
            color=THEME_COLORS["primary"],
        )
    with col2:
        attrition_count = report.target_distribution.get("Yes", 0)
        attrition_rate = (attrition_count / report.row_count) if report.row_count > 0 else 0
        render_metric_card(
            title="Historical Attrition",
            value=f"{attrition_rate:.1%}",
            subtext=f"{attrition_count} historical departures",
            color=THEME_COLORS["danger"],
        )
    with col3:
        best_auc = metadata.get("best_cv_score", 0.0)
        render_metric_card(
            title="Model ROC-AUC",
            value=f"{best_auc:.3f}",
            subtext="5-Fold Stratified CV",
            color=THEME_COLORS["secondary"],
        )
    with col4:
        threshold = metadata.get("optimal_threshold", 0.5)
        render_metric_card(
            title="Alert Boundary",
            value=f"{threshold * 100:.1f}%",
            subtext="Optimal Youden's J Index",
            color=THEME_COLORS["warning"],
        )
except Exception as e:
    st.info("System initializing. Please navigate to the desired module in the sidebar.")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Platform Architecture & Modules
st.markdown("### Platform Modules")

mod_col1, mod_col2, mod_col3 = st.columns(3)

with mod_col1:
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #4F46E5; margin-top: 0;">📊 Executive Dashboard</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Strategic workforce telemetry, attrition velocity, department comparisons, and interactive demographic filters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #4F46E5; margin-top: 0;">🎯 Individual Prediction</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Real-time attrition risk estimation with configurable thresholds, risk tiers, and confidential human-review recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mod_col2:
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #0D9488; margin-top: 0;">🔍 Exploratory Analysis</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Multi-dimensional pattern discovery, compensation correlation matrices, and satisfaction factor distributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #0D9488; margin-top: 0;">📈 Model Performance</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Rigorous benchmarking of Logistic Regression, Random Forest, and Gradient Boosting with ROC, PR, and Confusion Matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mod_col3:
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #2563EB; margin-top: 0;">💡 Explainability</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Global feature importance, permutation impact, plain-English workforce drivers, and SHAP explainability insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="saas-card">
            <h3 style="color: #2563EB; margin-top: 0;">📁 Data Management</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Secure CSV upload, schema verification, missing value auditing, and on-demand model retraining pipelines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

