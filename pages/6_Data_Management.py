"""Page 6: Data Management & Schema Governance.

Handles CSV file uploads, strict schema validation, missing column reporting,
downloadable cleaned datasets, demo data controls, and on-demand model retraining.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import (
    inject_custom_css,
    render_page_header,
    render_provenance_badge,
    render_metric_card,
    get_cached_dataset,
)
from src.config import (
    RAW_IBM_DATA_PATH,
    SYNTHETIC_DATA_PATH,
    TARGET_COLUMN,
    THEME_COLORS,
)
from src.data_loader import validate_dataset, load_dataset
from src.data_generator import generate_synthetic_attrition_data
from src.train import train_and_evaluate_models

st.set_page_config(
    page_title="Data Management | RetainIQ",
    page_icon="📁",
    layout="wide",
)

inject_custom_css()

render_page_header(
    title="Data Management & Pipeline Governance",
    subtitle="Ingest custom workforce datasets, validate schema fidelity, and trigger reproducible model retraining.",
    badge_label="Data Ops",
    badge_type="info",
)

# Current dataset status
current_df, current_report = get_cached_dataset()
render_provenance_badge(current_report.is_synthetic)

tab_upload, tab_synthetic, tab_schema = st.tabs([
    "📤 Ingest Custom Dataset",
    "🧪 Demo Synthetic Data Generator",
    "📋 Schema Specifications",
])

# ---------------------------------------------------------------------------
# Tab 1: Ingest Custom CSV
# ---------------------------------------------------------------------------
with tab_upload:
    st.markdown("### Upload Workforce CSV")
    st.markdown(
        "Upload an enterprise CSV dataset adhering to the 35-attribute IBM HR Analytics schema. "
        "Files are validated in-memory and will never automatically overwrite system data."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV safely
            upload_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully read `{uploaded_file.name}` ({len(upload_df):,} rows, {len(upload_df.columns)} columns).")

            # Validate in memory
            validation_report = validate_dataset(upload_df, source_path=uploaded_file.name)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Row Count", f"{validation_report.row_count:,}", "Uploaded rows", THEME_COLORS["primary"])
            with col2:
                render_metric_card("Columns", f"{validation_report.column_count}", "Uploaded attributes", THEME_COLORS["secondary"])
            with col3:
                render_metric_card("Missing Values", f"{validation_report.missing_values_count}", "Total null cells", THEME_COLORS["warning"] if validation_report.missing_values_count > 0 else THEME_COLORS["success"])
            with col4:
                status_val = "Present" if validation_report.target_present else "Missing"
                render_metric_card("Target ('Attrition')", status_val, "Supervised training eligibility", THEME_COLORS["success"] if validation_report.target_present else THEME_COLORS["danger"])

            # Warnings & Errors display
            if validation_report.errors:
                st.error("### Validation Errors Found:")
                for err in validation_report.errors:
                    st.write(f"❌ {err}")
            elif validation_report.warnings:
                with st.expander("⚠️ Schema Warnings & Notes", expanded=True):
                    for w in validation_report.warnings:
                        st.write(f"⚠️ {w}")

            st.markdown("### Data Preview")
            st.dataframe(upload_df.head(10), use_container_width=True)

            # Cleaned copy export
            st.markdown("### Export Cleaned Copy")
            cleaned_csv = upload_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Standardized CSV Copy",
                data=cleaned_csv,
                file_name=f"cleaned_{uploaded_file.name}",
                mime="text/csv",
            )

            # Retraining Options
            st.markdown("---")
            st.markdown("### 🔄 Model Retraining")

            if validation_report.target_present:
                st.info(
                    "This dataset contains the required **'Attrition'** target column and is eligible for "
                    "training and updating the production machine learning pipeline."
                )
                if st.button("🚀 Train Production Pipeline with Uploaded Data"):
                    with st.spinner("Executing 5-Fold Stratified Cross-Validation & Model Training..."):
                        try:
                            # Train with uploaded data
                            best_p, meta = train_and_evaluate_models(df=upload_df, save_artifacts=True)
                            st.success(
                                f"🎉 Training Complete! Selected Model: **{meta['selected_model_name']}** "
                                f"(Validation ROC-AUC: **{meta['best_cv_score']:.4f}**, Test ROC-AUC: **{meta['test_metrics']['roc_auc']:.4f}**)."
                            )
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.info("Cache refreshed. Visit Executive Dashboard or Model Performance to view updated analytics.")
                        except Exception as train_err:
                            st.error(f"Training failed: {train_err}")
            else:
                st.warning(
                    "⚠️ **Target Column Missing:** Because this dataset does not have the **'Attrition'** target column, "
                    "it cannot be used to train a supervised model. However, you can generate batch predictions "
                    "using our active production pipeline."
                )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Synthetic Generator Controls
# ---------------------------------------------------------------------------
with tab_synthetic:
    st.markdown("### 🧪 Synthetic Benchmark Generator")
    st.markdown(
        "If you do not have access to proprietary workforce data, this generator creates a realistic, "
        "deterministic benchmark dataset using seed **42** matching the IBM HR Analytics schema."
    )

    sample_size = st.number_input("Synthetic Records Count", min_value=200, max_value=5000, value=1470, step=100)
    seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)

    if st.button("⚙️ Generate & Save Synthetic Demo Dataset"):
        with st.spinner("Generating synthetic workforce cohort..."):
            gen_df = generate_synthetic_attrition_data(n_samples=sample_size, random_state=seed, output_path=SYNTHETIC_DATA_PATH)
            st.success(f"Generated {len(gen_df)} synthetic records saved to `{SYNTHETIC_DATA_PATH}`.")

            # Trigger retrain on new demo data
            train_and_evaluate_models(df=gen_df, save_artifacts=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.info("Models successfully retrained on the fresh synthetic benchmark!")

# ---------------------------------------------------------------------------
# Tab 3: Schema Specifications
# ---------------------------------------------------------------------------
with tab_schema:
    st.markdown("### 📋 Required IBM HR Analytics Schema")
    st.markdown(
        """
        To guarantee full compatibility, custom CSV datasets must contain standard HR attributes:
        
        - **Target Column:** `Attrition` ('Yes' / 'No')
        - **Core Numerical Fields:** `Age`, `DailyRate`, `DistanceFromHome`, `Education`, `HourlyRate`, `JobInvolvement`, 
          `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, 
          `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, 
          `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`
        - **Categorical Fields:** `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`
        - **Ignored / ID Constants:** `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
        """
    )

