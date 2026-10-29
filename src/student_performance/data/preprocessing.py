"""Academic feature engineering and scikit-learn preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_performance.core.config import AppConfig


class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer composite literacy metrics and balance ratios between reading and writing."""

    ENGINEERED_COLUMNS: tuple[str, ...] = (
        "reading_writing_avg",
        "reading_writing_diff",
        "reading_writing_ratio",
    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> StudentFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        read = df["reading_score"].astype(float)
        write = df["writing_score"].astype(float)

        df["reading_writing_avg"] = (read + write) / 2.0
        df["reading_writing_diff"] = read - write
        df["reading_writing_ratio"] = read / np.maximum(write, 1.0)
        return df


def build_preprocessing_pipeline(config: AppConfig) -> Pipeline:
    """Build preprocessing pipeline with composite engineering, imputation, and encoding."""
    engineered_num_cols = list(config.features.numerical_columns) + list(
        StudentFeatureEngineer.ENGINEERED_COLUMNS
    )
    cat_cols = list(config.features.categorical_columns)
    cat_categories = [config.features.categories[col] for col in cat_cols]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    categories=cat_categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, engineered_num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineer", StudentFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
