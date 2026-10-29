"""Pydantic v2 schemas for student performance inference and drift monitoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StudentPredictionRequest(BaseModel):
    """Input payload for single student performance prediction."""

    gender: str = Field(..., description="Student gender (female, male)")
    race_ethnicity: str = Field(..., description="Demographic group: group A through group E")
    parental_level_of_education: str = Field(
        ...,
        description="Highest parental education: some high school, high school, some college, associate's degree, bachelor's degree, master's degree",
    )
    lunch: str = Field(..., description="Lunch category: standard, free/reduced")
    test_preparation_course: str = Field(..., description="Test prep status: none, completed")
    reading_score: float = Field(..., ge=0.0, le=100.0, description="Reading test score (0-100)")
    writing_score: float = Field(..., ge=0.0, le=100.0, description="Writing test score (0-100)")


class StudentPredictionResponse(BaseModel):
    """Output score prediction, confidence interval, and local feature attributions."""

    predicted_math_score: float
    prediction_interval: dict[str, float]
    feature_attributions: dict[str, float]
    interventions: list[str]
    model_name: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health and holdout model metrics."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_model: str
    model_metrics: dict[str, float]
    subgroup_fairness: dict[str, Any]
    buffered_requests: int


class NumericalDriftMetric(BaseModel):
    """KS-test and PSI drift telemetry for a numerical test score."""

    ks_statistic: float
    ks_p_value: float
    psi: float
    mean_reference: float
    mean_current: float
    drift_detected: bool


class CategoricalDriftMetric(BaseModel):
    """L1 distribution shift telemetry for a demographic category."""

    l1_divergence: float
    drift_detected: bool


class DriftReportResponse(BaseModel):
    """Complete drift telemetry across scores and demographic attributes."""

    status: str
    overall_drift_detected: bool
    sample_size: int
    reference_size: int
    numerical_features: dict[str, NumericalDriftMetric]
    categorical_features: dict[str, CategoricalDriftMetric]
