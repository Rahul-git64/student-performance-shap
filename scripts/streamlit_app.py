"""Streamlit interactive student performance predictor and explainability console."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Predictor & Explainability Console")
st.markdown(
    "Predict student **math exam scores**, inspect **local feature attributions (SHAP-style)**, "
    "view **actionable intervention recommendations**, and monitor **demographic subgroup fairness**."
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race_ethnicity = st.selectbox(
        "Race / Ethnicity Group",
        ["group A", "group B", "group C", "group D", "group E"],
        index=2,
    )
    education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
        index=2,
    )

with col2:
    lunch = st.selectbox("Lunch Program", ["standard", "free/reduced"])
    prep = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading = st.slider("Reading Score (0-100)", 0.0, 100.0, 72.0, 1.0)
    writing = st.slider("Writing Score (0-100)", 0.0, 100.0, 74.0, 1.0)

if st.button("Predict Student Performance", type="primary"):
    payload = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": education,
        "lunch": lunch,
        "test_preparation_course": prep,
        "reading_score": reading,
        "writing_score": writing,
    }
    try:
        resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        resp.raise_for_status()
        res = resp.json()

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Math Score", f"{res['predicted_math_score']:.1f} / 100")
        m2.metric(
            "90% Prediction Interval",
            f"[{res['prediction_interval']['lower']:.1f}, {res['prediction_interval']['upper']:.1f}]",
        )
        m3.metric("Model", res["model_name"])

        st.subheader("Feature Attributions (Contribution to Score)")
        st.json(res["feature_attributions"])

        st.subheader("Actionable Academic Interventions")
        for tip in res["interventions"]:
            st.info(f"💡 {tip}")

    except Exception as exc:
        st.error(f"Valuation request failed: {exc}")

st.divider()
if st.button("Inspect Demographic & Score Drift (/monitor/drift)"):
    try:
        drift_resp = httpx.get(f"{API_URL}/monitor/drift", timeout=10.0)
        st.json(drift_resp.json())
    except Exception as exc:
        st.error(f"Drift telemetry request failed: {exc}")
