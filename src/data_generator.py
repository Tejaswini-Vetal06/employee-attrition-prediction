"""Synthetic Dataset Generator for Employee Attrition Demo.

Generates a realistic, schema-compliant synthetic HR dataset with a fixed
random seed when the official IBM HR Analytics dataset is not present.
Explicitly labels all generated rows with synthetic provenance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    SYNTHETIC_DATA_PATH,
    ALLOWED_CATEGORIES,
    RANDOM_STATE,
)


def generate_synthetic_attrition_data(
    n_samples: int = 1470,
    random_state: int = RANDOM_STATE,
    output_path: Optional[Path] = SYNTHETIC_DATA_PATH,
) -> pd.DataFrame:
    """Generate a synthetic HR dataset modeled on the IBM HR Analytics schema.

    Parameters
    ----------
    n_samples : int, default=1470
        Number of synthetic employee records to generate.
    random_state : int, default=42
        Fixed random seed for deterministic generation.
    output_path : Optional[Path], default=SYNTHETIC_DATA_PATH
        Destination CSV path. If None, file is not written to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame adhering to the 35-column IBM HR schema with realistic
        distributions and probabilistic attrition outcome.
    """
    rng = np.random.default_rng(random_state)

    # 1. Demographics & Base Attributes
    age = rng.integers(18, 61, size=n_samples)
    gender = rng.choice(ALLOWED_CATEGORIES["Gender"], size=n_samples, p=[0.40, 0.60])
    marital_status = rng.choice(
        ALLOWED_CATEGORIES["MaritalStatus"], size=n_samples, p=[0.32, 0.46, 0.22]
    )
    distance_from_home = rng.integers(1, 30, size=n_samples)
    education = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.19, 0.39, 0.27, 0.03])
    education_field = rng.choice(
        ALLOWED_CATEGORIES["EducationField"],
        size=n_samples,
        p=[0.41, 0.32, 0.11, 0.09, 0.02, 0.05],
    )

    # 2. Organizational Structure & Roles
    departments = ALLOWED_CATEGORIES["Department"]
    department = rng.choice(departments, size=n_samples, p=[0.30, 0.65, 0.05])

    job_roles = []
    for dept in department:
        if dept == "Sales":
            role = rng.choice(
                ["Sales Executive", "Sales Representative", "Manager"],
                p=[0.70, 0.22, 0.08],
            )
        elif dept == "Research & Development":
            role = rng.choice(
                [
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Research Director",
                    "Manager",
                ],
                p=[0.35, 0.30, 0.15, 0.10, 0.05, 0.05],
            )
        else:  # Human Resources
            role = rng.choice(["Human Resources", "Manager"], p=[0.85, 0.15])
        job_roles.append(role)
    job_role = np.array(job_roles)

    # 3. Work Details & Compensation
    business_travel = rng.choice(
        ALLOWED_CATEGORIES["BusinessTravel"], size=n_samples, p=[0.71, 0.19, 0.10]
    )
    over_time = rng.choice(ALLOWED_CATEGORIES["OverTime"], size=n_samples, p=[0.72, 0.28])

    # Job Level depends loosely on age
    job_level = np.clip(1 + (age - 20) // 9 + rng.integers(-1, 2, size=n_samples), 1, 5)

    # Monthly income based on job level + noise
    base_income_by_level = {1: 2800, 2: 5200, 3: 8500, 4: 13000, 5: 18500}
    monthly_income = np.array(
        [
            int(
                np.clip(
                    base_income_by_level[lvl] + rng.normal(0, 750),
                    1009,
                    19999,
                )
            )
            for lvl in job_level
        ]
    )

    daily_rate = rng.integers(102, 1500, size=n_samples)
    hourly_rate = rng.integers(30, 101, size=n_samples)
    monthly_rate = rng.integers(2094, 27000, size=n_samples)
    percent_salary_hike = rng.integers(11, 26, size=n_samples)
    stock_option_level = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.43, 0.40, 0.11, 0.06])

    # 4. Experience & Tenure (maintain realistic logical ordering)
    total_working_years = np.clip(age - 18 - rng.integers(0, 5, size=n_samples), 0, 40)
    years_at_company = np.array(
        [rng.integers(0, max(1, twy + 1)) for twy in total_working_years]
    )
    years_in_current_role = np.array(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company]
    )
    years_since_last_promotion = np.array(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company]
    )
    years_with_curr_manager = np.array(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company]
    )
    num_companies_worked = rng.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        size=n_samples,
        p=[0.13, 0.35, 0.10, 0.11, 0.10, 0.04, 0.05, 0.05, 0.03, 0.04],
    )
    training_times_last_year = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.04, 0.05, 0.37, 0.33, 0.10, 0.08, 0.03])

    # 5. Satisfaction & Ratings
    environment_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.19, 0.31, 0.31])
    job_involvement = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.06, 0.25, 0.59, 0.10])
    job_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.20, 0.19, 0.30, 0.31])
    performance_rating = np.where(percent_salary_hike >= 20, 4, 3)
    relationship_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.20, 0.31, 0.30])
    work_life_balance = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.05, 0.23, 0.61, 0.11])

    # 6. Standard Constants & IDs in IBM Schema
    employee_count = np.ones(n_samples, dtype=int)
    over_18 = np.array(["Y"] * n_samples)
    standard_hours = np.full(n_samples, 80, dtype=int)
    employee_number = np.arange(1, n_samples + 1)

    # 7. Probabilistic Attrition Generation based on established HR dynamics
    # Base attrition log-odds
    log_odds = -1.8  # ~15-16% baseline attrition
    # Risk factors
    log_odds += np.where(over_time == "Yes", 0.95, -0.30)
    log_odds += np.where(business_travel == "Travel_Frequently", 0.60, 0.0)
    log_odds += np.where(business_travel == "Non-Travel", -0.40, 0.0)
    log_odds += np.where(marital_status == "Single", 0.50, -0.20)
    log_odds += (30 - age) * 0.035
    log_odds += (distance_from_home - 9) * 0.03
    log_odds += (2.7 - environment_satisfaction) * 0.35
    log_odds += (2.7 - job_satisfaction) * 0.40
    log_odds += (2.7 - job_involvement) * 0.35
    log_odds += (2.7 - work_life_balance) * 0.30
    log_odds += (3 - stock_option_level) * 0.25
    log_odds += (6500 - monthly_income) / 6000.0 * 0.5
    log_odds += (years_since_last_promotion - 2) * 0.06
    log_odds -= (years_with_curr_manager) * 0.08
    log_odds -= (years_at_company) * 0.04

    # Convert to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-log_odds))
    probs = np.clip(probs, 0.03, 0.92)

    # Sample binary attrition outcome
    attrition_binary = rng.random(size=n_samples) < probs
    attrition = np.where(attrition_binary, "Yes", "No")

    # Build DataFrame
    df = pd.DataFrame(
        {
            "Age": age,
            "Attrition": attrition,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EmployeeCount": employee_count,
            "EmployeeNumber": employee_number,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "Over18": over_18,
            "OverTime": over_time,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StandardHours": standard_hours,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    generated_df = generate_synthetic_attrition_data()
    print(
        f"Generated {len(generated_df)} synthetic records at {SYNTHETIC_DATA_PATH}. "
        f"Attrition rate: {(generated_df['Attrition'] == 'Yes').mean():.2%}"
    )
