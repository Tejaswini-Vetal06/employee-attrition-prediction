"""Page 3: Employee Attrition Prediction.

Interactive prediction form leveraging the persisted production Scikit-Learn pipeline.
Offers configurable decision boundaries, probability estimation, risk tier classification,
and downloadable audit reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
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
    get_cached_pipeline_and_metadata,
)
from src.predict import predict_employee_attrition
from src.config import (
    ALLOWED_CATEGORIES,
    NUMERIC_LIMITS,
    THEME_COLORS,
    DEFAULT_DECISION_THRESHOLD,
)

st.set_page_config(
    page_title="Attrition Prediction | RetainIQ",
    page_icon="🎯",
    layout="wide",
)

inject_custom_css()

render_page_header(
    title="Employee Attrition Risk Prediction",
    subtitle="Evaluate individual employee retention probability based on calibrated historical patterns.",
    badge_label="Machine Learning Inference",
    badge_type="info",
)

render_ethical_banner()

# Check for model artifacts
try:
    pipeline, metadata = get_cached_pipeline_and_metadata()
except Exception as e:
    st.error(
        f"⚠️ Model pipeline is not loaded: {e}. "
        "Please visit **Page 4 (Model Performance)** or run `python -m src.train` to train and save the model."
    )
    st.stop()

# Optimal threshold from model training
optimal_thresh = float(metadata.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD))

# Preset Profiles for immediate testing
st.markdown("### 📋 Quick Demo Profiles")
profile_col1, profile_col2 = st.columns([2, 1])

with profile_col1:
    archetype = st.selectbox(
        "Load Pre-Configured Benchmark Archetype (Optional):",
        options=[
            "Custom Profile (Manual Input)",
            "High Retention Benchmark (Experienced, Satisfied, No Overtime)",
            "Moderate Risk Benchmark (Mid-Career, Occasional Travel, Neutral Satisfaction)",
            "Elevated Risk Benchmark (Overtime, Low Income, Frequent Travel, Low Satisfaction)",
        ],
        index=0,
    )

# Archetype defaults
defaults = {}
for k, v in NUMERIC_LIMITS.items():
    defaults[k] = v["default"]

for k, v in ALLOWED_CATEGORIES.items():
    defaults[k] = v[0]

if archetype == "High Retention Benchmark (Experienced, Satisfied, No Overtime)":
    defaults.update({
        "Age": 45,
        "Department": "Research & Development",
        "JobRole": "Manager",
        "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 3,
        "MonthlyIncome": 14500,
        "JobLevel": 4,
        "YearsAtCompany": 12,
        "YearsInCurrentRole": 8,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 7,
        "TotalWorkingYears": 22,
        "OverTime": "No",
        "JobSatisfaction": 4,
        "EnvironmentSatisfaction": 4,
        "WorkLifeBalance": 4,
        "JobInvolvement": 4,
        "StockOptionLevel": 2,
    })
elif archetype == "Moderate Risk Benchmark (Mid-Career, Occasional Travel, Neutral Satisfaction)":
    defaults.update({
        "Age": 32,
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 12,
        "MonthlyIncome": 5400,
        "JobLevel": 2,
        "YearsAtCompany": 4,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2,
        "TotalWorkingYears": 8,
        "OverTime": "No",
        "JobSatisfaction": 2,
        "EnvironmentSatisfaction": 2,
        "WorkLifeBalance": 3,
        "JobInvolvement": 3,
        "StockOptionLevel": 1,
    })
elif archetype == "Elevated Risk Benchmark (Overtime, Low Income, Frequent Travel, Low Satisfaction)":
    defaults.update({
        "Age": 24,
        "Department": "Sales",
        "JobRole": "Sales Representative",
        "BusinessTravel": "Travel_Frequently",
        "DistanceFromHome": 24,
        "MonthlyIncome": 2400,
        "JobLevel": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
        "TotalWorkingYears": 2,
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "EnvironmentSatisfaction": 1,
        "WorkLifeBalance": 1,
        "JobInvolvement": 2,
        "StockOptionLevel": 0,
        "MaritalStatus": "Single",
    })

# Form Layout
st.markdown("### 📝 Employee Attributes")

with st.form("attrition_prediction_form"):
    sec1, sec2, sec3 = st.columns(3)

    with sec1:
        st.markdown("#### 1. Personal & Role")
        age = st.slider("Age", 18, 65, value=int(defaults["Age"]))
        department = st.selectbox(
            "Department",
            options=ALLOWED_CATEGORIES["Department"],
            index=ALLOWED_CATEGORIES["Department"].index(defaults["Department"]) if defaults["Department"] in ALLOWED_CATEGORIES["Department"] else 0,
        )
        job_role = st.selectbox(
            "Job Role",
            options=ALLOWED_CATEGORIES["JobRole"],
            index=ALLOWED_CATEGORIES["JobRole"].index(defaults["JobRole"]) if defaults["JobRole"] in ALLOWED_CATEGORIES["JobRole"] else 0,
        )
        business_travel = st.selectbox(
            "Business Travel",
            options=ALLOWED_CATEGORIES["BusinessTravel"],
            index=ALLOWED_CATEGORIES["BusinessTravel"].index(defaults["BusinessTravel"]) if defaults["BusinessTravel"] in ALLOWED_CATEGORIES["BusinessTravel"] else 0,
        )
        distance = st.slider("Distance From Home (miles)", 1, 35, value=int(defaults["DistanceFromHome"]))
        marital_status = st.selectbox(
            "Marital Status",
            options=ALLOWED_CATEGORIES["MaritalStatus"],
            index=ALLOWED_CATEGORIES["MaritalStatus"].index(defaults["MaritalStatus"]) if defaults["MaritalStatus"] in ALLOWED_CATEGORIES["MaritalStatus"] else 0,
        )
        gender = st.selectbox(
            "Gender",
            options=ALLOWED_CATEGORIES["Gender"],
            index=ALLOWED_CATEGORIES["Gender"].index(defaults["Gender"]) if defaults["Gender"] in ALLOWED_CATEGORIES["Gender"] else 0,
        )
        education_field = st.selectbox(
            "Education Field",
            options=ALLOWED_CATEGORIES["EducationField"],
            index=ALLOWED_CATEGORIES["EducationField"].index(defaults["EducationField"]) if defaults["EducationField"] in ALLOWED_CATEGORIES["EducationField"] else 0,
        )

    with sec2:
        st.markdown("#### 2. Work & Compensation")
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=1000,
            max_value=25000,
            value=int(defaults["MonthlyIncome"]),
            step=250,
        )
        job_level = st.selectbox("Job Level (1=Entry, 5=Executive)", options=[1, 2, 3, 4, 5], index=int(defaults["JobLevel"]) - 1)
        over_time = st.selectbox(
            "Overtime Requirement",
            options=ALLOWED_CATEGORIES["OverTime"],
            index=ALLOWED_CATEGORIES["OverTime"].index(defaults["OverTime"]) if defaults["OverTime"] in ALLOWED_CATEGORIES["OverTime"] else 0,
        )
        stock_options = st.selectbox("Stock Option Level (0-3)", options=[0, 1, 2, 3], index=int(defaults["StockOptionLevel"]))
        total_working_years = st.slider("Total Working Years", 0, 45, value=int(defaults["TotalWorkingYears"]))
        years_at_company = st.slider("Years at Company", 0, 40, value=int(defaults["YearsAtCompany"]))
        years_in_role = st.slider("Years in Current Role", 0, 20, value=min(int(defaults["YearsInCurrentRole"]), int(defaults["YearsAtCompany"])))
        years_since_promo = st.slider("Years Since Last Promotion", 0, 20, value=min(int(defaults["YearsSinceLastPromotion"]), int(defaults["YearsAtCompany"])))
        years_with_mgr = st.slider("Years with Current Manager", 0, 20, value=min(int(defaults["YearsWithCurrManager"]), int(defaults["YearsAtCompany"])))

    with sec3:
        st.markdown("#### 3. Engagement & Well-being")
        job_satisfaction = st.select_slider(
            "Job Satisfaction",
            options=[1, 2, 3, 4],
            value=int(defaults["JobSatisfaction"]),
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        )
        env_satisfaction = st.select_slider(
            "Environment Satisfaction",
            options=[1, 2, 3, 4],
            value=int(defaults["EnvironmentSatisfaction"]),
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        )
        work_life_balance = st.select_slider(
            "Work-Life Balance",
            options=[1, 2, 3, 4],
            value=int(defaults["WorkLifeBalance"]),
            format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}[x],
        )
        job_involvement = st.select_slider(
            "Job Involvement",
            options=[1, 2, 3, 4],
            value=int(defaults["JobInvolvement"]),
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        )
        rel_satisfaction = st.select_slider(
            "Relationship Satisfaction",
            options=[1, 2, 3, 4],
            value=int(defaults.get("RelationshipSatisfaction", 3)),
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        )
        num_companies = st.slider("Prior Companies Worked", 0, 10, value=int(defaults.get("NumCompaniesWorked", 2)))
        hike = st.slider("Salary Hike (%)", 10, 30, value=int(defaults.get("PercentSalaryHike", 15)))

    st.markdown("---")
    thresh_col1, thresh_col2 = st.columns([3, 1])
    with thresh_col1:
        custom_threshold = st.slider(
            "Configurable Risk Alert Threshold (%):",
            min_value=15,
            max_value=85,
            value=int(optimal_thresh * 100),
            step=1,
            help=f"Calibrated default based on validation Youden's J statistic is {optimal_thresh * 100:.1f}%.",
        ) / 100.0

    with thresh_col2:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("⚡ Predict Attrition Risk", use_container_width=True)

if submitted:
    # Build complete input payload
    input_payload = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": defaults.get("DailyRate", 800),
        "Department": department,
        "DistanceFromHome": distance,
        "Education": defaults.get("Education", 3),
        "EducationField": education_field,
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender,
        "HourlyRate": defaults.get("HourlyRate", 65),
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": defaults.get("MonthlyRate", 14000),
        "NumCompaniesWorked": num_companies,
        "OverTime": over_time,
        "PercentSalaryHike": hike,
        "PerformanceRating": 4 if hike >= 20 else 3,
        "RelationshipSatisfaction": rel_satisfaction,
        "StockOptionLevel": stock_options,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": defaults.get("TrainingTimesLastYear", 2),
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_mgr,
    }

    try:
        res = predict_employee_attrition(
            input_dict=input_payload,
            pipeline=pipeline,
            metadata=metadata,
            decision_threshold=custom_threshold,
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Result & Assessment")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"""
                <div class="saas-card" style="border-top: 5px solid {res['risk_color']}; text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase;">
                        Estimated Attrition Probability
                    </div>
                    <div style="font-size: 3.0rem; font-weight: 800; color: {res['risk_color']}; margin: 8px 0;">
                        {res['probability_percentage']}%
                    </div>
                    <div style="background-color: {res['risk_color']}20; color: {res['risk_color']}; font-weight: 700; padding: 6px 14px; border-radius: 9999px; display: inline-block; font-size: 0.85rem;">
                        {res['risk_tier']}
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 12px;">
                        Alert Threshold: <strong>{res['threshold_used'] * 100:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f"""
                <div class="saas-card">
                    <h4 style="margin-top: 0; color: #0F172A;">Operating Guidance & Interpretation</h4>
                    <p style="color: #334155; font-size: 0.95rem; line-height: 1.6;">
                        {res['interpretation']}
                    </p>
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 12px; margin-top: 12px;">
                        <span style="font-size: 0.82rem; color: #64748B;">
                            <strong>Model Pipeline:</strong> {res['model_used']} (v{res['model_version']})
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Profile contribution highlights
        st.markdown("#### Key Profile Context Factors")
        fac1, fac2, fac3 = st.columns(3)
        with fac1:
            st.metric("Overtime Status", over_time, delta="Elevates Risk" if over_time == "Yes" else "Standard Baseline", delta_color="inverse")
        with fac2:
            st.metric("Monthly Income", f"${monthly_income:,.0f}", delta="Below Median" if monthly_income < 5000 else "Competitive")
        with fac3:
            st.metric("Job Satisfaction", f"{job_satisfaction} / 4", delta="Low" if job_satisfaction <= 2 else "Favorable")

        # Downloadable Prediction Report
        report_data = {
            "prediction_summary": {
                "probability_percentage": res["probability_percentage"],
                "risk_tier": res["risk_tier"],
                "is_flagged_for_review": res["is_flagged_for_review"],
                "threshold_applied": res["threshold_used"],
                "model_version": res["model_version"],
            },
            "employee_inputs": input_payload,
            "governance": {
                "statement": res["ethical_disclaimer"],
                "notice": "This document is confidential HR decision-support documentation."
            }
        }

        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download Formal Prediction Audit Report (JSON)",
            data=report_json,
            file_name=f"attrition_prediction_audit_{age}_{job_role.replace(' ', '_')}.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"Prediction execution failed: {e}")
