"""Smoke tests for Streamlit application pages and utilities."""

import pytest
import importlib.util
from pathlib import Path

from src.utils import get_cached_pipeline_and_metadata, get_cached_dataset
from src.explainability import (
    extract_global_feature_importance,
    explain_feature_in_plain_english,
)


def test_app_and_pages_syntax_and_compilation():
    """Verify app.py and all multi-page scripts compile without syntax errors."""
    project_root = Path(__file__).resolve().parent.parent

    files_to_check = [
        project_root / "app.py",
        project_root / "pages" / "1_Executive_Dashboard.py",
        project_root / "pages" / "2_Exploratory_Analysis.py",
        project_root / "pages" / "3_Attrition_Prediction.py",
        project_root / "pages" / "4_Model_Performance.py",
        project_root / "pages" / "5_Model_Explainability.py",
        project_root / "pages" / "6_Data_Management.py",
    ]

    for py_file in files_to_check:
        assert py_file.exists(), f"File {py_file} is missing"
        # Compile file to verify Python AST syntax validity
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(py_file), "exec")
        assert compiled is not None


def test_explainability_feature_importance_smoke():
    """Verify feature importance extraction returns valid DataFrame."""
    pipeline, metadata = get_cached_pipeline_and_metadata()
    df_imp = extract_global_feature_importance(pipeline, metadata, top_n=10)

    assert not df_imp.empty
    assert "Feature" in df_imp.columns
    assert "Importance" in df_imp.columns
    assert "Impact_Direction" in df_imp.columns
    assert len(df_imp) == 10

    sample_expl = explain_feature_in_plain_english("OverTime_Yes", "Increases Attrition Risk")
    assert isinstance(sample_expl, str)
    assert len(sample_expl) > 10

