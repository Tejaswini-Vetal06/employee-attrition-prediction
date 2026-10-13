"""Page 1: Executive Dashboard.

Provides executive workforce telemetry, attrition velocity by department and job role,
interactive demographic slicing, and clear distinction of historical metrics from model predictions.
"""

import streamlit as st
import pandas as pd
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
    render_ethical_banner,
    get_plotly_layout,
    get_cached_dataset,
)
from src.config import THEME_COLORS, TARGET_COLUMN

st.set_page_config(
    page_title="Executive Dashboard | RetainIQ",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()

# Header
render_page_header(
    title="Executive Workforce Telemetry",
    subtitle="Macro workforce patterns, historical attrition distribution, and departmental comparisons.",
    badge_label="Historical Analytics",
    badge_type="info",
)

# Load data
df, report = get_cached_dataset()
render_provenance_badge(report.is_synthetic)

# Filter Sidebar
st.sidebar.markdown("### 🔍 Dashboard Filters")

# Departments filter
available_depts = sorted(df["Department"].dropna().unique().tolist()) if "Department" in df.columns else []
selected_depts = st.sidebar.multiselect(
    "Department",
    options=available_depts,
    default=available_depts,
    help="Filter workforce metrics by operational department.",
)

# Job Role filter
available_roles = (
    sorted(df[df["Department"].isin(selected_depts)]["JobRole"].dropna().unique().tolist())
    if "JobRole" in df.columns and "Department" in df.columns
    else sorted(df["JobRole"].dropna().unique().tolist()) if "JobRole" in df.columns else []
)
selected_roles = st.sidebar.multiselect(
    "Job Role",
    options=available_roles,
    default=available_roles,
    help="Filter by specific employee positions.",
)

# Overtime filter
available_ot = sorted(df["OverTime"].dropna().unique().tolist()) if "OverTime" in df.columns else []
selected_ot = st.sidebar.multiselect(
    "Overtime Status",
    options=available_ot,
    default=available_ot,
)

# Apply filters
filtered_df = df.copy()
if selected_depts and "Department" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Department"].isin(selected_depts)]
if selected_roles and "JobRole" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["JobRole"].isin(selected_roles)]
if selected_ot and "OverTime" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["OverTime"].isin(selected_ot)]

if filtered_df.empty:
    st.warning("No employee records match the active filter criteria. Please broaden your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------------------------
total_filtered = len(filtered_df)
has_attrition = TARGET_COLUMN in filtered_df.columns

left_count = int((filtered_df[TARGET_COLUMN] == "Yes").sum()) if has_attrition else 0
stayed_count = int((filtered_df[TARGET_COLUMN] == "No").sum()) if has_attrition else 0
attrition_rate = (left_count / total_filtered) if total_filtered > 0 and has_attrition else 0.0

avg_tenure = float(filtered_df["YearsAtCompany"].mean()) if "YearsAtCompany" in filtered_df.columns else 0.0
avg_monthly_income = float(filtered_df["MonthlyIncome"].mean()) if "MonthlyIncome" in filtered_df.columns else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    render_metric_card(
        title="Active Cohort",
        value=f"{total_filtered:,}",
        subtext=f"{(total_filtered / len(df)):.1%} of total dataset",
        color=THEME_COLORS["primary"],
    )
with kpi2:
    render_metric_card(
        title="Historical Attrition",
        value=f"{attrition_rate:.1%}",
        subtext=f"{left_count} departed employees",
        color=THEME_COLORS["danger"] if attrition_rate > 0.20 else THEME_COLORS["warning"],
    )
with kpi3:
    render_metric_card(
        title="Retained Employees",
        value=f"{stayed_count:,}",
        subtext=f"{(stayed_count / total_filtered):.1%} retention rate" if total_filtered > 0 else "",
        color=THEME_COLORS["success"],
    )
with kpi4:
    render_metric_card(
        title="Average Tenure",
        value=f"{avg_tenure:.1f} yrs",
        subtext="Mean company service",
        color=THEME_COLORS["secondary"],
    )
with kpi5:
    render_metric_card(
        title="Average Income",
        value=f"${avg_monthly_income:,.0f}",
        subtext="Monthly compensation",
        color=THEME_COLORS["navy"],
    )

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Methodology Note
st.info(
    "ℹ️ **Executive Clarification:** All metrics and charts on this dashboard reflect "
    "**retrospective historical observations** recorded in the workforce registry. "
    "They represent realized past outcomes rather than model-forecasted risks."
)

# ---------------------------------------------------------------------------
# Visualizations Row 1: Department Breakdown & Job Role Velocity
# ---------------------------------------------------------------------------
chart_row1_col1, chart_row1_col2 = st.columns(2)

with chart_row1_col1:
    if "Department" in filtered_df.columns and has_attrition:
        dept_summary = (
            filtered_df.groupby(["Department", TARGET_COLUMN])
            .size()
            .reset_index(name="Count")
        )
        fig_dept = px.bar(
            dept_summary,
            x="Department",
            y="Count",
            color=TARGET_COLUMN,
            barmode="group",
            color_discrete_map={"No": THEME_COLORS["secondary"], "Yes": THEME_COLORS["danger"]},
            labels={"Count": "Employees", "Department": "Department", TARGET_COLUMN: "Attrition Status"},
        )
        fig_dept.update_layout(get_plotly_layout("Workforce by Department & Attrition Status"))
        st.plotly_chart(fig_dept, use_container_width=True)

with chart_row1_col2:
    if "JobRole" in filtered_df.columns and has_attrition:
        # Calculate attrition rate per role
        role_stats = (
            filtered_df.groupby("JobRole")[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Attrition_Rate_Pct")
            .sort_values(by="Attrition_Rate_Pct", ascending=True)
        )
        fig_role = px.bar(
            role_stats,
            y="JobRole",
            x="Attrition_Rate_Pct",
            orientation="h",
            color="Attrition_Rate_Pct",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            labels={"Attrition_Rate_Pct": "Attrition Rate (%)", "JobRole": "Job Role"},
        )
        fig_role.update_layout(
            get_plotly_layout("Attrition Rate by Job Role (%)", show_legend=False),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_role, use_container_width=True)

# ---------------------------------------------------------------------------
# Visualizations Row 2: OverTime Impact & Compensation Boxplots
# ---------------------------------------------------------------------------
chart_row2_col1, chart_row2_col2 = st.columns(2)

with chart_row2_col1:
    if "OverTime" in filtered_df.columns and has_attrition:
        ot_summary = (
            filtered_df.groupby(["OverTime", TARGET_COLUMN])
            .size()
            .reset_index(name="Count")
        )
        fig_ot = px.bar(
            ot_summary,
            x="OverTime",
            y="Count",
            color=TARGET_COLUMN,
            barmode="stack",
            color_discrete_map={"No": "#3B82F6", "Yes": "#EF4444"},
            labels={"Count": "Employees", "OverTime": "Overtime Status", TARGET_COLUMN: "Attrition"},
        )
        fig_ot.update_layout(get_plotly_layout("Overtime Exposure vs Attrition Volume"))
        st.plotly_chart(fig_ot, use_container_width=True)

with chart_row2_col2:
    if "MonthlyIncome" in filtered_df.columns and has_attrition:
        fig_income = px.box(
            filtered_df,
            x=TARGET_COLUMN,
            y="MonthlyIncome",
            color=TARGET_COLUMN,
            color_discrete_map={"No": THEME_COLORS["secondary"], "Yes": THEME_COLORS["danger"]},
            labels={"MonthlyIncome": "Monthly Income ($)", TARGET_COLUMN: "Attrition Status"},
            points="outliers",
        )
        fig_income.update_layout(get_plotly_layout("Monthly Income Distribution by Attrition"))
        st.plotly_chart(fig_income, use_container_width=True)

# ---------------------------------------------------------------------------
# Visualizations Row 3: Age vs Tenure Scatter
# ---------------------------------------------------------------------------
if "Age" in filtered_df.columns and "YearsAtCompany" in filtered_df.columns and has_attrition:
    st.markdown("### Age vs Tenure Correlation")
    fig_scatter = px.scatter(
        filtered_df,
        x="Age",
        y="YearsAtCompany",
        color=TARGET_COLUMN,
        color_discrete_map={"No": "#4F46E5", "Yes": "#EF4444"},
        hover_data=["JobRole", "Department", "MonthlyIncome"],
        opacity=0.75,
        labels={"YearsAtCompany": "Years at Company", "Age": "Age (years)"},
    )
    fig_scatter.update_layout(get_plotly_layout("Employee Age vs Company Tenure", height=420))
    st.plotly_chart(fig_scatter, use_container_width=True)

