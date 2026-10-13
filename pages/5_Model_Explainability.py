"""Page 5: Model Explainability & Interpretability.

Surfaces global feature importance, model coefficients, permutation importance,
and SHAP values alongside plain-English organizational translations and ethical boundaries.
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
    render_ethical_banner,
    get_plotly_layout,
    get_cached_pipeline_and_metadata,
)
from src.config import THEME_COLORS, ALL_MODELS_PATH
from src.explainability import (
    extract_global_feature_importance,
    compute_permutation_importance,
    compute_shap_explanations,
    explain_feature_in_plain_english,
)
import joblib

st.set_page_config(
    page_title="Model Explainability | RetainIQ",
    page_icon="💡",
    layout="wide",
)

inject_custom_css()

render_page_header(
    title="Model Explainability & Factor Analysis",
    subtitle="Unpack predictive drivers, global feature weights, permutation sensitivity, and ethical guidelines.",
    badge_label="Interpretability",
    badge_type="info",
)

render_ethical_banner()

# Load model artifacts
try:
    pipeline, metadata = get_cached_pipeline_and_metadata()
except Exception as e:
    st.error(f"Model artifacts could not be loaded: {e}. Please train models first.")
    st.stop()

# ---------------------------------------------------------------------------
# Governance & Non-Discrimination Notice
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 8px; padding: 14px 18px; margin-bottom: 24px;">
        <h4 style="color: #991B1B; margin: 0 0 6px 0; font-size: 0.95rem;">⚖️ Strict Regulatory & Non-Discrimination Governance</h4>
        <p style="color: #7F1D1D; font-size: 0.85rem; margin: 0; line-height: 1.5;">
            Federal and state labor laws strictly prohibit adverse employment decisions based on protected demographic attributes 
            such as <strong>Age, Gender, Race, or Marital Status</strong>. While machine learning algorithms may observe statistical 
            correlations involving demographic fields in historical data, these attributes <strong>must never</strong> serve as criteria 
            for compensation adjustments, promotion gating, or retention interventions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "📊 Global Feature Importance",
    "🔄 Permutation Sensitivity",
    "🧠 SHAP Summary Insights",
])

# ---------------------------------------------------------------------------
# Tab 1: Global Feature Importance
# ---------------------------------------------------------------------------
with tab1:
    top_n = st.slider("Select Number of Top Features to Display:", 5, 25, 15)
    df_importance = extract_global_feature_importance(pipeline, metadata, top_n=top_n)

    fig_imp = px.bar(
        df_importance.sort_values(by="Importance", ascending=True),
        y="Feature",
        x="Importance",
        color="Impact_Direction",
        orientation="h",
        color_discrete_map={
            "Increases Attrition Risk": THEME_COLORS["danger"],
            "Reduces Attrition Risk (Protective)": THEME_COLORS["secondary"],
            "Significant Driver": THEME_COLORS["primary"],
            "Factor": THEME_COLORS["navy"],
        },
        labels={"Importance": "Magnitude (|Weight| or Impurity)", "Feature": "Feature Name"},
    )
    fig_imp.update_layout(
        get_plotly_layout(f"Top {top_n} Global Feature Drivers ({metadata.get('selected_model_name', 'Model')})", height=450),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🗣️ Plain-English Operational Translations")
    st.markdown("Contextual interpretation of the highest-impact factors observed by the model:")

    for _, row in df_importance.head(6).iterrows():
        f_name = row["Feature"]
        direction = row["Impact_Direction"]
        explanation = explain_feature_in_plain_english(f_name, direction)

        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong style="color: #0F172A; font-size: 0.95rem;">{f_name}</strong>
                    <span style="font-size: 0.78rem; font-weight: 600; color: {'#EF4444' if 'Increases' in direction else '#0D9488'};">
                        {direction}
                    </span>
                </div>
                <div style="color: #475569; font-size: 0.85rem; line-height: 1.4;">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab 2: Permutation Feature Importance
# ---------------------------------------------------------------------------
with tab2:
    st.markdown(
        "Model-agnostic permutation importance measures the degradation in **held-out test ROC-AUC** "
        "when each feature's values are randomly shuffled. Higher drops indicate critical decision dependencies."
    )

    if ALL_MODELS_PATH.exists():
        try:
            all_art = joblib.load(ALL_MODELS_PATH)
            X_test = all_art["test_data"]["X_test"]
            y_test = all_art["test_data"]["y_test"]

            with st.spinner("Computing permutation importance across held-out test data..."):
                df_perm = compute_permutation_importance(pipeline, X_test, y_test, n_repeats=5)

            fig_perm = px.bar(
                df_perm.head(15).sort_values(by="Mean_ROC_AUC_Drop", ascending=True),
                y="Feature",
                x="Mean_ROC_AUC_Drop",
                error_x="Std_Dev",
                orientation="h",
                color="Mean_ROC_AUC_Drop",
                color_continuous_scale="Blues",
                labels={"Mean_ROC_AUC_Drop": "Mean Drop in ROC-AUC", "Feature": "Feature Name"},
            )
            fig_perm.update_layout(
                get_plotly_layout("Top 15 Features by Permutation ROC-AUC Impact", height=420, show_legend=False),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_perm, use_container_width=True)
            st.dataframe(df_perm.head(15), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute permutation importance: {e}")
    else:
        st.info("Test set artifacts not found. Please retrain models to inspect permutation importance.")

# ---------------------------------------------------------------------------
# Tab 3: SHAP Explainability
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### SHAP (SHapley Additive exPlanations) Global Insights")
    st.markdown(
        "SHAP values attribute model prediction outcomes to cooperative game-theoretic feature contributions."
    )

    if ALL_MODELS_PATH.exists():
        try:
            all_art = joblib.load(ALL_MODELS_PATH)
            X_test_sample = all_art["test_data"]["X_test"].head(80)

            with st.spinner("Calculating SHAP sample values..."):
                shap_result = compute_shap_explanations(pipeline, X_test_sample)

            if shap_result and shap_result.get("status") == "success":
                f_names = metadata["feature_schema"]["transformed_feature_names"]
                mean_shap = shap_result["mean_abs_shap"]

                min_l = min(len(f_names), len(mean_shap))
                df_shap = pd.DataFrame(
                    {"Feature": f_names[:min_l], "Mean_Abs_SHAP": mean_shap[:min_l]}
                ).sort_values(by="Mean_Abs_SHAP", ascending=False).head(15)

                fig_shap = px.bar(
                    df_shap.sort_values(by="Mean_Abs_SHAP", ascending=True),
                    y="Feature",
                    x="Mean_Abs_SHAP",
                    orientation="h",
                    color="Mean_Abs_SHAP",
                    color_continuous_scale="Teal",
                    labels={"Mean_Abs_SHAP": "Mean |SHAP Value|", "Feature": "Feature Name"},
                )
                fig_shap.update_layout(
                    get_plotly_layout("Top 15 Features by Mean Absolute SHAP Value", height=420, show_legend=False),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning(f"SHAP explainer notice: {shap_result.get('error_message', 'Computation unavailable')}")
        except Exception as e:
            st.warning(f"SHAP explanation computation notice: {e}")
    else:
        st.info("Test data artifacts needed for SHAP evaluation.")
