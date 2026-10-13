"""Configuration module for the Employee Attrition Prediction platform.

Centralizes file paths, feature schemas, training hyperparameters,
and domain validation rules.
"""

from pathlib import Path
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Standard dataset paths
RAW_IBM_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_employee_attrition.csv"
DEMO_DATA_PATH = SYNTHETIC_DATA_PATH  # Default fallback

# Artifact paths
BEST_MODEL_PATH = MODELS_DIR / "best_attrition_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
ALL_MODELS_PATH = MODELS_DIR / "all_trained_models.joblib"

# ---------------------------------------------------------------------------
# Target & ID Column Definitions
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Attrition"
POSITIVE_TARGET_VALUE = "Yes"
NEGATIVE_TARGET_VALUE = "No"

# Target mapping for modeling
TARGET_MAPPING = {
    NEGATIVE_TARGET_VALUE: 0,
    POSITIVE_TARGET_VALUE: 1
}
REVERSE_TARGET_MAPPING = {0: "Stayed (No)", 1: "Left (Yes)"}

# Columns with zero variance or arbitrary employee identifiers
DEFAULT_DROP_COLUMNS = [
    "EmployeeCount",     # Constant value 1
    "Over18",             # Constant value 'Y'
    "StandardHours",      # Constant value 80
    "EmployeeNumber",     # Identifier
]

# ---------------------------------------------------------------------------
# Feature Schemas
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Allowed categorical domains for strict validation
ALLOWED_CATEGORIES: Dict[str, List[str]] = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Human Resources",
        "Other"
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["No", "Yes"],
}

# Numeric boundaries and standard defaults for UI sliders/inputs
NUMERIC_LIMITS: Dict[str, Dict[str, Any]] = {
    "Age": {"min": 18, "max": 65, "default": 35, "step": 1, "label": "Age (years)"},
    "DailyRate": {"min": 100, "max": 1500, "default": 800, "step": 10, "label": "Daily Rate ($)"},
    "DistanceFromHome": {"min": 1, "max": 35, "default": 7, "step": 1, "label": "Distance from Home (miles)"},
    "Education": {"min": 1, "max": 5, "default": 3, "step": 1, "label": "Education Level (1=Below College, 5=Doctorate)"},
    "EnvironmentSatisfaction": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Environment Satisfaction (1-4)"},
    "HourlyRate": {"min": 30, "max": 100, "default": 65, "step": 1, "label": "Hourly Rate ($)"},
    "JobInvolvement": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Job Involvement (1-4)"},
    "JobLevel": {"min": 1, "max": 5, "default": 2, "step": 1, "label": "Job Level (1-5)"},
    "JobSatisfaction": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Job Satisfaction (1-4)"},
    "MonthlyIncome": {"min": 1000, "max": 25000, "default": 5000, "step": 100, "label": "Monthly Income ($)"},
    "MonthlyRate": {"min": 2000, "max": 30000, "default": 14000, "step": 100, "label": "Monthly Rate ($)"},
    "NumCompaniesWorked": {"min": 0, "max": 10, "default": 2, "step": 1, "label": "Prior Companies Worked"},
    "PercentSalaryHike": {"min": 10, "max": 30, "default": 15, "step": 1, "label": "Percent Salary Hike (%)"},
    "PerformanceRating": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Performance Rating (1-4)"},
    "RelationshipSatisfaction": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Relationship Satisfaction (1-4)"},
    "StockOptionLevel": {"min": 0, "max": 3, "default": 1, "step": 1, "label": "Stock Option Level (0-3)"},
    "TotalWorkingYears": {"min": 0, "max": 45, "default": 10, "step": 1, "label": "Total Working Years"},
    "TrainingTimesLastYear": {"min": 0, "max": 6, "default": 2, "step": 1, "label": "Training Times Last Year"},
    "WorkLifeBalance": {"min": 1, "max": 4, "default": 3, "step": 1, "label": "Work-Life Balance (1-4)"},
    "YearsAtCompany": {"min": 0, "max": 40, "default": 5, "step": 1, "label": "Years at Company"},
    "YearsInCurrentRole": {"min": 0, "max": 20, "default": 3, "step": 1, "label": "Years in Current Role"},
    "YearsSinceLastPromotion": {"min": 0, "max": 20, "default": 1, "step": 1, "label": "Years Since Last Promotion"},
    "YearsWithCurrManager": {"min": 0, "max": 20, "default": 3, "step": 1, "label": "Years with Current Manager"},
}

# ---------------------------------------------------------------------------
# Model Training Settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_SELECTION_METRIC = "roc_auc"
DEFAULT_DECISION_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# UI Theme Constants
# ---------------------------------------------------------------------------
THEME_COLORS = {
    "primary": "#4F46E5",        # Indigo
    "primary_dark": "#3730A3",   # Deep Indigo
    "secondary": "#0D9488",      # Teal
    "secondary_light": "#14B8A6",
    "navy": "#0F172A",           # Slate Navy
    "card_bg": "#FFFFFF",
    "background": "#F8FAFC",     # Light Slate
    "text_main": "#1E293B",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "danger": "#EF4444",         # Red/High Risk
    "warning": "#F59E0B",        # Amber/Medium Risk
    "success": "#10B981",        # Emerald/Low Risk
}

# ---------------------------------------------------------------------------
# Responsible AI Disclaimers & Ethics
# ---------------------------------------------------------------------------
ETHICAL_DISCLAIMER = (
    "Attrition risk predictions are probabilistic estimates derived from historical patterns, "
    "not deterministic certainty regarding individual employees. These models must NOT be used "
    "as the sole basis for hiring, compensation, promotion, discipline, or termination. "
    "Protected demographic characteristics (such as age, gender, and marital status) must "
    "never be utilized as adverse factors in HR decision-making. All retention interventions "
    "require proactive, supportive, and confidential human leadership review."
)
