"""Deterministic offline student dataset generator and train/test ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from student_performance.core.config import AppConfig
from student_performance.core.exceptions import DataIngestionError
from student_performance.core.logger import get_logger, log_event
from student_performance.data.validation import validate_student_dataframe

logger = get_logger(__name__)

EDUCATION_EFFECT: dict[str, float] = {
    "some high school": -3.0,
    "high school": -1.5,
    "some college": 0.0,
    "associate's degree": 1.5,
    "bachelor's degree": 3.0,
    "master's degree": 4.5,
}

GROUP_EFFECT: dict[str, float] = {
    "group A": -4.0,
    "group B": -2.0,
    "group C": -0.5,
    "group D": 1.5,
    "group E": 4.5,
}


def generate_synthetic_students(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a deterministic dataset mirroring StudentsPerformance distributions."""
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["female", "male"], size=n_rows, p=[0.52, 0.48])
    group = rng.choice(
        list(GROUP_EFFECT), size=n_rows, p=[0.09, 0.19, 0.32, 0.26, 0.14]
    )
    education = rng.choice(
        list(EDUCATION_EFFECT), size=n_rows, p=[0.18, 0.20, 0.23, 0.22, 0.12, 0.05]
    )
    lunch = rng.choice(["standard", "free/reduced"], size=n_rows, p=[0.65, 0.35])
    prep = rng.choice(["none", "completed"], size=n_rows, p=[0.64, 0.36])

    # Latent academic ability drives correlation between the three scores.
    ability = rng.normal(0.0, 1.0, size=n_rows)
    base = (
        66.0
        + np.array([GROUP_EFFECT[g] for g in group])
        + np.array([EDUCATION_EFFECT[e] for e in education])
        + np.where(lunch == "standard", 5.5, -5.5) * 0.9
        + np.where(prep == "completed", 4.5, -1.5)
    )
    is_female = gender == "female"

    math = base + 11.5 * ability + np.where(is_female, -2.5, 2.5) + rng.normal(0, 5.0, n_rows)
    reading = (
        base + 3.0 + 11.0 * ability + np.where(is_female, 3.5, -3.5) + rng.normal(0, 4.5, n_rows)
    )
    writing = (
        base + 2.0 + 11.5 * ability + np.where(is_female, 4.5, -4.5) + rng.normal(0, 4.0, n_rows)
    )

    return pd.DataFrame(
        {
            "gender": gender,
            "race_ethnicity": group,
            "parental_level_of_education": education,
            "lunch": lunch,
            "test_preparation_course": prep,
            "math_score": np.clip(np.round(math), 0, 100).astype(int),
            "reading_score": np.clip(np.round(reading), 0, 100).astype(int),
            "writing_score": np.clip(np.round(writing), 0, 100).astype(int),
        }
    )


def ingest_data(
    config: AppConfig, source_path: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV (or cached/synthetic fallback), validate it, and split train/test."""
    try:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path)
            log_event(logger, "loaded_external_student_csv", path=str(source_path), rows=len(df))
        elif config.data.raw_data_path.exists():
            df = pd.read_csv(config.data.raw_data_path)
            log_event(logger, "loaded_cached_student_csv", rows=len(df))
        else:
            df = generate_synthetic_students(config.data.synthetic_rows, config.data.random_state)
            config.data.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(config.data.raw_data_path, index=False)
            log_event(logger, "generated_synthetic_students", rows=len(df))
    except Exception as exc:
        raise DataIngestionError(f"Failed to ingest student dataset: {exc}") from exc

    validated = validate_student_dataframe(df, config, require_target=True)
    prep_counts = validated["test_preparation_course"].value_counts()
    stratify = validated["test_preparation_course"] if int(prep_counts.min()) >= 2 else None

    train_df, test_df = train_test_split(
        validated,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=stratify,
    )
    config.data.train_data_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.data.train_data_path, index=False)
    test_df.to_csv(config.data.test_data_path, index=False)
    log_event(logger, "student_split_completed", train_rows=len(train_df), test_rows=len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
