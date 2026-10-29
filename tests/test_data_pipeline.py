"""Unit tests for student data ingestion, schema validation, and feature preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from student_performance.core.config import load_config
from student_performance.core.exceptions import DataValidationError
from student_performance.data.ingestion import (
    generate_synthetic_students,
    ingest_data,
)
from student_performance.data.preprocessing import (
    StudentFeatureEngineer,
    build_preprocessing_pipeline,
)
from student_performance.data.validation import (
    validate_single_student,
    validate_student_dataframe,
)


def test_synthetic_generation_and_ingestion(tmp_path: Path) -> None:
    cfg = load_config()
    df = generate_synthetic_students(n_rows=120, random_state=42)
    assert len(df) == 120
    assert (df["math_score"] >= 0).all() and (df["math_score"] <= 100).all()
    assert set(df["gender"].unique()).issubset({"female", "male"})

    csv_path = tmp_path / "students_sample.csv"
    df.to_csv(csv_path, index=False)
    train_df, test_df = ingest_data(cfg, source_path=csv_path)
    assert len(train_df) + len(test_df) == 120


def test_single_student_validation_and_alias() -> None:
    cfg = load_config()
    valid_payload = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch_type": "standard",  # alias for lunch
        "test_preparation_course": "none",
        "reading_score": 72.0,
        "writing_score": 74.0,
    }
    clean = validate_single_student(valid_payload, cfg)
    assert clean["lunch"] == "standard"
    assert clean["reading_score"] == 72.0

    bad_score = {**valid_payload, "reading_score": 150.0}
    with pytest.raises(DataValidationError, match="out of range"):
        validate_single_student(bad_score, cfg)

    bad_category = {**valid_payload, "gender": "unknown"}
    with pytest.raises(DataValidationError, match="Invalid gender"):
        validate_single_student(bad_category, cfg)


def test_dataframe_validation_filters_corrupted_rows() -> None:
    cfg = load_config()
    df = generate_synthetic_students(n_rows=50, random_state=1)
    df.loc[0, "reading_score"] = -5
    df.loc[1, "math_score"] = 999
    validated = validate_student_dataframe(df, cfg, require_target=True)
    assert len(validated) == 48


def test_feature_engineering_and_pipeline_shapes() -> None:
    cfg = load_config()
    df = generate_synthetic_students(n_rows=60, random_state=7)
    X = df.drop(columns=["math_score"])

    engineer = StudentFeatureEngineer()
    eng_df = engineer.fit_transform(X)
    assert "reading_writing_avg" in eng_df.columns
    assert "reading_writing_diff" in eng_df.columns
    assert "reading_writing_ratio" in eng_df.columns

    pipeline = build_preprocessing_pipeline(cfg)
    X_mat = pipeline.fit_transform(X)
    assert X_mat.shape[0] == 60
    assert np.isfinite(X_mat).all()
