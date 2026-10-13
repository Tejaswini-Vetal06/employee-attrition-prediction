"""UI utilities, custom CSS injection, caching, and Plotly styling helpers.

Provides a unified design system with Navy, Indigo, and Teal accents,
consistent typography, responsive cards, and responsible AI badges.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.config import (
    THEME_COLORS,
    ETHICAL_DISCLAIMER,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    ALL_MODELS_PATH,
)
from src.data_loader import load_dataset, DatasetValidationReport
from src.predict import load_trained_model


def inject_custom_css():
    """Injects modern SaaS CSS design system into the Streamlit application."""
    custom_css = f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Global background */
    .stApp {{
        background-color: {THEME_COLORS['background']};
        color: {THEME_COLORS['text_main']};
    }}

    /* Main Container Padding */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #0F172A;
        color: #F8FAFC;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
        color: #94A3B8;
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: #F8FAFC;
    }}

    /* Cards */
    .saas-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .saas-card:hover {{
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    }}

    /* KPI Metric Cards */
    .metric-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }}
    .metric-card-top-accent {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }}
    .metric-label {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }}
    .metric-subtext {{
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 6px;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }}
    .badge-synthetic {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }}
    .badge-real {{
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }}
    .badge-info {{
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }}

    /* Ethical Alert Box */
    .ethical-banner {{
        background: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 24px;
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.5;
    }}

    /* Custom Buttons */
    div.stButton > button:first-child {{
        background-color: {THEME_COLORS['primary']};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {THEME_COLORS['primary_dark']};
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
    }}

    /* Plotly Chart Containers */
    .js-plotly-plot {{
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def get_plotly_layout(
    title: str = "",
    height: int = 380,
    show_legend: bool = True,
) -> Dict[str, Any]:
    """Provides consistent, clean styling parameters for Plotly figures."""
    return {
        "title": {
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": THEME_COLORS["navy"], "family": "Inter"},
            "x": 0.01,
            "xanchor": "left",
        },
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 40, "r": 30, "t": 50 if title else 20, "b": 40},
        "font": {"family": "Inter", "color": THEME_COLORS["text_main"], "size": 12},
        "xaxis": {
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "showline": True,
            "linecolor": "#CBD5E1",
        },
        "yaxis": {
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "showline": True,
            "linecolor": "#CBD5E1",
        },
        "legend": {
            "visible": show_legend,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
    }


def render_metric_card(
    title: str,
    value: str,
    subtext: Optional[str] = None,
    color: str = THEME_COLORS["primary"],
):
    """Renders an attractive KPI card with subtle SaaS top accent border."""
    subtext_html = f'<div class="metric-subtext">{subtext}</div>' if subtext else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-card-top-accent" style="background-color: {color};"></div>
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: str,
    badge_label: Optional[str] = None,
    badge_type: str = "info",
):
    """Renders a prominent, SaaS-style header with optional badge."""
    badge_html = ""
    if badge_label:
        badge_class = f"badge-{badge_type}"
        badge_html = f'<span class="badge {badge_class}" style="margin-left: 10px; vertical-align: middle;">{badge_label}</span>'

    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.0rem; font-weight: 700; color: #0F172A; margin-bottom: 4px; display: inline-block;">
                {title}
            </h1>
            {badge_html}
            <p style="font-size: 0.95rem; color: #64748B; margin-top: 4px;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ethical_banner():
    """Renders responsible AI usage disclaimer."""
    st.markdown(
        f"""
        <div class="ethical-banner">
            <strong style="color: #4F46E5;">Responsible AI & Governance Guidance:</strong><br/>
            {ETHICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_provenance_badge(is_synthetic: bool):
    """Renders dataset provenance banner."""
    if is_synthetic:
        st.markdown(
            """
            <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center;">
                <span style="font-size: 1.1rem; margin-right: 10px;">⚠️</span>
                <div>
                    <strong style="color: #92400E; font-size: 0.88rem;">Demo Synthetic Dataset Active</strong>
                    <div style="color: #78350F; font-size: 0.82rem;">
                        This environment is running on a synthetically generated benchmark dataset (seed 42). Metrics reflect synthetic data dynamics.
                        To utilize real enterprise data, upload or place the official IBM dataset at <code>data/WA_Fn-UseC_-HR-Employee-Attrition.csv</code>.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #D1FAE5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; align-items: center;">
                <span style="font-size: 1.1rem; margin-right: 10px;">🛡️</span>
                <span style="color: #065F46; font-size: 0.85rem; font-weight: 500;">
                    Official Enterprise Dataset Active (IBM HR Analytics Schema).
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def get_cached_dataset(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, Any]:
    """Caches dataset loading to prevent repetitive disk I/O on UI interactions."""
    p = Path(file_path) if file_path else None
    return load_dataset(file_path=p)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline_and_metadata():
    """Caches model artifact loading in memory."""
    return load_trained_model()
