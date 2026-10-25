"""Schema validation, column alias normalization, and score range checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from student_performance.core.config import AppConfig
from student_performance.core.exceptions import DataValidationError

SCORE_MIN = 0.0
SCORE_MAX = 100.0


def normalize_columns(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Rename raw dataset columns to canonical snake_case names using configured aliases."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip()
        renamed[col] = config.features.column_aliases.get(key, key.replace(" ", "_"))
    return df.rename(columns=renamed)


def validate_single_student(payload: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Validate and normalize one student record for inference."""
    record = {
        config.features.column_aliases.get(str(k), str(k)): v for k, v in payload.items()
    }
    required = list(config.features.numerical_columns) + list(config.features.categorical_columns)
    missing = [c for c in required if c not in record]
    if missing:
        raise DataValidationError(f"Missing required student fields: {missing}")

    clean: dict[str, Any] = {}
    for col in config.features.numerical_columns:
        try:
            value = float(record[col])
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"{col} must be numeric: {record[col]!r}") from exc
        if not (SCORE_MIN <= value <= SCORE_MAX):
            raise DataValidationError(f"{col} out of range [0, 100]: {value}")
        clean[col] = value

    for col in config.features.categorical_columns:
        value = str(record[col]).strip()
        allowed = config.features.categories[col]
        if value not in allowed:
            raise DataValidationError(f"Invalid {col} '{value}'. Allowed: {allowed}")
        clean[col] = value
    return clean


def validate_student_dataframe(
    df: pd.DataFrame, config: AppConfig, require_target: bool = True
) -> pd.DataFrame:
    """Validate a student DataFrame: schema, score ranges, and categorical domains."""
    if df.empty:
        raise DataValidationError("Input student DataFrame cannot be empty.")

    cleaned = normalize_columns(df, config)
    target = config.features.target_column
    required = list(config.features.numerical_columns) + list(config.features.categorical_columns)
    if require_target:
        required.append(target)

    missing = [c for c in required if c not in cleaned.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    cleaned = cleaned.dropna(subset=required).reset_index(drop=True)
    score_cols = list(config.features.numerical_columns) + ([target] if require_target else [])
    for col in score_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=score_cols).reset_index(drop=True)

    in_range = pd.Series(True, index=cleaned.index)
    for col in score_cols:
        in_range &= cleaned[col].between(SCORE_MIN, SCORE_MAX)
    cleaned = cleaned.loc[in_range].reset_index(drop=True)
    if cleaned.empty:
        raise DataValidationError("No valid rows remain after score range filtering.")

    for col in config.features.categorical_columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        invalid = set(cleaned[col].unique()) - set(config.features.categories[col])
        if invalid:
            raise DataValidationError(f"Unsupported values in '{col}': {sorted(invalid)}")
    return cleaned
