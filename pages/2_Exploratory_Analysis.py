"""Page 2: Exploratory Data Analysis.

Deep-dive exploratory telemetry covering data types, nulls, duplicates,
multivariate correlation matrices, and demographic feature distributions.
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
    render_provenance_badge,
    get_plotly_layout,
    get_cached_dataset,
)
from src.config import THEME_COLORS, TARGET_COLUMN

st.set_page_config(
    page_title="Exploratory Analysis | RetainIQ",
    page_icon="🔍",
    layout="wide",
)

inject_custom_css()

render_page_header(
    title="Exploratory Data Analysis",
    subtitle="Inspect dataset dimensions, feature distributions, cross-tabulations, and correlation topology.",
    badge_label="Workforce Patterns",
    badge_type="info",
)

df, report = get_cached_dataset()
render_provenance_badge(report.is_synthetic)

# Cautionary guidance
st.markdown(
    """
    <div style="background-color: #F8FAFC; border-left: 4px solid #0D9488; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 20px; font-size: 0.88rem; color: #334155;">
        <strong>Statistical Interpretation Caution:</strong>
        Observed statistical associations reflect historical workforce patterns and do NOT establish direct causation.
        Confounding variables (e.g. career stage, life events, market demand) frequently drive observed correlations.
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Hygiene & Overview",
    "🎯 Attrition Distribution",
    "📊 Categorical Patterns",
    "📈 Numeric Distributions & Correlation",
])

# ---------------------------------------------------------------------------
# Tab 1: Dataset Overview & Hygiene
# ---------------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Rows", f"{len(df):,}", "Employee entries", THEME_COLORS["primary"])
    with col2:
        render_metric_card("Total Features", f"{len(df.columns)}", "Captured attributes", THEME_COLORS["secondary"])
    with col3:
        render_metric_card("Missing Values", f"{report.missing_values_count}", "Complete records", THEME_COLORS["success"])
    with col4:
        render_metric_card("Duplicate Rows", f"{report.duplicate_rows_count}", "Zero redundancy", THEME_COLORS["navy"])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Attribute Schema & Data Types")
    dtype_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique().values,
            "Sample Value": [df[c].dropna().iloc[0] if not df[c].dropna().empty else None for c in df.columns],
        }
    )
    st.dataframe(dtype_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Attrition Target Distribution
# ---------------------------------------------------------------------------
with tab2:
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        stayed = int(counts.get("No", 0))
        left = int(counts.get("Yes", 0))
        total = len(df)
        rate = left / total if total > 0 else 0

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Class Balance Summary")
            st.markdown(
                f"""
                - **Retained (No):** `{stayed:,}` employees ({(stayed / total):.1%})
                - **Departed (Yes):** `{left:,}` employees ({(left / total):.1%})
                - **Class Imbalance Ratio:** ~`{stayed / max(1, left):.1f} : 1`
                """
            )
            st.info(
                "💡 **Machine Learning Impact:** The severe class imbalance (~1:4 ratio) "
                "means standard raw accuracy is deceptive. Cost-sensitive weighting and ROC-AUC optimization "
                "are mandatory during model development."
            )

        with c2:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Retained (Stayed)", "Departed (Left)"],
                        values=[stayed, left],
                        hole=0.55,
                        marker=dict(colors=[THEME_COLORS["secondary"], THEME_COLORS["danger"]]),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie.update_layout(get_plotly_layout("Workforce Attrition Breakdown", height=320))
            st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Categorical Workforce Patterns
# ---------------------------------------------------------------------------
with tab3:
    cat_columns = [
        "Department",
        "BusinessTravel",
        "EducationField",
        "Gender",
        "MaritalStatus",
        "JobRole",
        "OverTime",
    ]
    available_cats = [c for c in cat_columns if c in df.columns]

    selected_cat = st.selectbox(
        "Select Feature to Cross-Tabulate against Attrition:",
        options=available_cats,
        index=0,
    )

    if selected_cat and TARGET_COLUMN in df.columns:
        cross_tab = (
            pd.crosstab(df[selected_cat], df[TARGET_COLUMN], normalize="index") * 100
        ).reset_index()

        col_left = cross_tab["Yes"] if "Yes" in cross_tab.columns else pd.Series([0] * len(cross_tab))
        cross_tab["Attrition_Rate"] = col_left

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            fig_pct = px.bar(
                cross_tab.sort_values(by="Attrition_Rate", ascending=False),
                x=selected_cat,
                y="Attrition_Rate",
                color="Attrition_Rate",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                labels={"Attrition_Rate": "Attrition Rate (%)", selected_cat: selected_cat},
            )
            fig_pct.update_layout(
                get_plotly_layout(f"Attrition Proportion across {selected_cat} (%)", show_legend=False),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        with chart_c2:
            raw_tab = df.groupby([selected_cat, TARGET_COLUMN]).size().reset_index(name="Count")
            fig_vol = px.bar(
                raw_tab,
                x=selected_cat,
                y="Count",
                color=TARGET_COLUMN,
                barmode="group",
                color_discrete_map={"No": THEME_COLORS["secondary"], "Yes": THEME_COLORS["danger"]},
            )
            fig_vol.update_layout(get_plotly_layout(f"Absolute Headcount across {selected_cat}"))
            st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Numeric Distributions & Correlation
# ---------------------------------------------------------------------------
with tab4:
    numeric_candidates = [
        "Age",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "DistanceFromHome",
        "PercentSalaryHike",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
    ]
    present_numerics = [c for c in numeric_candidates if c in df.columns]

    st.markdown("### Interactive Feature Distribution")
    sel_num = st.selectbox("Select Numeric Feature for Detailed Distribution:", options=present_numerics, index=1)

    if sel_num and TARGET_COLUMN in df.columns:
        fig_dist = px.histogram(
            df,
            x=sel_num,
            color=TARGET_COLUMN,
            marginal="box",
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={"No": THEME_COLORS["secondary"], "Yes": THEME_COLORS["danger"]},
            labels={sel_num: sel_num, TARGET_COLUMN: "Attrition"},
        )
        fig_dist.update_layout(get_plotly_layout(f"Distribution of {sel_num} by Attrition Status", height=420))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Multivariate Correlation Matrix")
    corr_cols = [c for c in present_numerics if c in df.columns]
    if corr_cols:
        corr_matrix = df[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Pearson Corr"),
        )
        fig_corr.update_layout(get_plotly_layout("Pearson Correlation Matrix among Numeric Workforce Features", height=520))
        st.plotly_chart(fig_corr, use_container_width=True)

