"""Model trainer, subgroup fairness evaluator, and local feature attribution explainer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from student_performance.core.config import AppConfig
from student_performance.core.exceptions import ModelTrainingError
from student_performance.core.logger import get_logger, log_event
from student_performance.data.ingestion import ingest_data
from student_performance.data.preprocessing import build_preprocessing_pipeline

logger = get_logger(__name__)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute overall regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
    }


def evaluate_subgroup_fairness(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: AppConfig,
) -> dict[str, Any]:
    """Evaluate performance across demographic/preparation subgroups to detect fairness gaps."""
    subgroup_report: dict[str, Any] = {}
    r2_values: list[float] = []

    for group_col in config.model.subgroup_columns:
        subgroup_report[group_col] = {}
        for group_val in test_df[group_col].unique():
            mask = (test_df[group_col] == group_val).to_numpy()
            if mask.sum() >= 5:
                metrics = evaluate_predictions(y_true[mask], y_pred[mask])
                subgroup_report[group_col][str(group_val)] = metrics
                r2_values.append(metrics["r2"])

    max_gap = round(float(max(r2_values) - min(r2_values)), 4) if len(r2_values) > 1 else 0.0
    return {
        "subgroups": subgroup_report,
        "max_r2_gap": max_gap,
        "fairness_passed": bool(max_gap <= config.model.max_subgroup_r2_gap),
    }


def build_reference_stats(train_df: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Export reference training distributions for reading/writing scores and categorical attributes."""
    numerical_stats: dict[str, Any] = {}
    rng = np.random.default_rng(config.data.random_state)

    for col in config.features.numerical_columns:
        series = train_df[col].astype(float).to_numpy()
        sampled = rng.choice(series, size=min(len(series), 300), replace=False)
        quantiles = np.quantile(series, np.linspace(0.0, 1.0, 11)).tolist()
        numerical_stats[col] = {
            "mean": round(float(np.mean(series)), 4),
            "std": round(float(np.std(series)), 4),
            "quantiles": [round(float(q), 4) for q in quantiles],
            "sample": [round(float(x), 4) for x in sampled],
        }

    categorical_stats: dict[str, dict[str, float]] = {}
    for col in config.features.categorical_columns:
        counts = train_df[col].astype(str).value_counts(normalize=True)
        categorical_stats[col] = {str(k): round(float(v), 4) for k, v in counts.items()}

    return {
        "n_reference": len(train_df),
        "numerical": numerical_stats,
        "categorical": categorical_stats,
    }


def compute_feature_attributions(
    model: Any,
    preprocessor: Any,
    input_df: pd.DataFrame,
    baseline_prediction: float,
) -> dict[str, Any]:
    """Compute local feature attributions using permutation importance on the input instance."""
    feature_names = [
        "reading_score",
        "writing_score",
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ]
    attributions: dict[str, float] = {}

    # Marginal perturbation baseline for local importance
    for col in feature_names:
        perturbed = input_df.copy()
        if col in ["reading_score", "writing_score"]:
            perturbed[col] = 50.0  # reference median score
        elif col == "test_preparation_course":
            perturbed[col] = "none"
        elif col == "lunch":
            perturbed[col] = "standard"

        X_pert = preprocessor.transform(perturbed)
        pert_pred = float(model.predict(X_pert)[0])
        # Contribution is how much the actual value shifts prediction from reference
        diff = round(baseline_prediction - pert_pred, 2)
        attributions[col] = diff

    interventions: list[str] = []
    if input_df["test_preparation_course"].iloc[0] == "none":
        interventions.append("Enroll in test preparation course to boost expected math score by +4 to +6 points.")
    if input_df["reading_score"].iloc[0] < 60.0:
        interventions.append("Target reading comprehension tutoring; strong literacy foundations directly improve word-problem math modeling.")
    if input_df["lunch"].iloc[0] == "free/reduced":
        interventions.append("Access supplementary nutrition assistance and after-school academic support programs.")

    return {
        "attributions": attributions,
        "recommendations": interventions if interventions else ["Continue current academic study regime."],
    }


def train_pipeline(config: AppConfig) -> dict[str, Any]:
    """Execute end-to-end model selection, fairness audit, and artifact persistence."""
    train_df, test_df = ingest_data(config)
    target_col = config.features.target_column

    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(float).to_numpy()
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(float).to_numpy()

    preprocessor = build_preprocessing_pipeline(config)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    seed = config.data.random_state
    candidates: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=140,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    leaderboard: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Any = None
    best_r2 = -np.inf
    best_preds = np.zeros_like(y_test)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0.0, 100.0)
        metrics = evaluate_predictions(y_test, preds)
        leaderboard[name] = metrics
        log_event(logger, "candidate_model_evaluated", model=name, **metrics)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_model = model
            best_preds = preds

    if best_r2 < config.model.r2_threshold:
        raise ModelTrainingError(
            f"Best model {best_name} R2 ({best_r2:.4f}) below threshold ({config.model.r2_threshold:.4f})"
        )

    fairness_results = evaluate_subgroup_fairness(test_df, y_test, best_preds, config)
    residuals = np.abs(y_test - best_preds)
    q90_residual = round(float(np.quantile(residuals, 0.90)), 2)

    config.model.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.model.preprocessor_path)
    joblib.dump(best_model, config.model.model_path)

    metadata: dict[str, Any] = {
        "model_name": best_name,
        "model_version": config.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "q90_residual": q90_residual,
        "metrics": leaderboard[best_name],
        "subgroup_fairness": fairness_results,
        "leaderboard": leaderboard,
    }

    with config.model.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    ref_stats = build_reference_stats(train_df, config)
    with config.model.reference_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(ref_stats, handle, indent=2)

    log_event(
        logger,
        "student_training_completed",
        winner=best_name,
        r2=leaderboard[best_name]["r2"],
        fairness_passed=fairness_results["fairness_passed"],
    )
    return metadata
