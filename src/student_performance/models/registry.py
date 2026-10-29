"""Model registry for runtime scoring, local attributions, and intervention generation."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from student_performance.core.config import AppConfig
from student_performance.core.exceptions import InferenceError
from student_performance.core.logger import get_logger, log_event
from student_performance.data.validation import validate_single_student
from student_performance.models.trainer import (
    compute_feature_attributions,
    train_pipeline,
)

logger = get_logger(__name__)


class ModelRegistry:
    """Manages inference, local attribution generation, and model artifacts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor: Any = None
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return bool(self.preprocessor is not None and self.model is not None)

    def ensure_loaded(self, auto_train: bool = True) -> None:
        """Load trained artifacts from disk or auto-train if missing."""
        if self.is_loaded:
            return

        m_cfg = self.config.model
        exists = (
            m_cfg.preprocessor_path.exists()
            and m_cfg.model_path.exists()
            and m_cfg.metadata_path.exists()
            and m_cfg.reference_stats_path.exists()
        )
        if not exists:
            if not auto_train:
                raise InferenceError("Model artifacts missing and auto_train is disabled.")
            log_event(logger, "auto_training_triggered", reason="missing_artifacts")
            train_pipeline(self.config)

        try:
            self.preprocessor = joblib.load(m_cfg.preprocessor_path)
            self.model = joblib.load(m_cfg.model_path)
            with m_cfg.metadata_path.open("r", encoding="utf-8") as f_meta:
                self.metadata = json.load(f_meta)
            with m_cfg.reference_stats_path.open("r", encoding="utf-8") as f_ref:
                self.reference_stats = json.load(f_ref)
        except Exception as exc:
            raise InferenceError(f"Failed to load student performance artifacts: {exc}") from exc

    def predict_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate input, generate predicted score, 90% interval, attributions, and interventions."""
        self.ensure_loaded(auto_train=True)
        validated = validate_single_student(payload, self.config)
        input_df = pd.DataFrame([validated])

        try:
            X_trans = self.preprocessor.transform(input_df)
            pred_score = float(np.clip(self.model.predict(X_trans)[0], 0.0, 100.0))
        except Exception as exc:
            raise InferenceError(f"Inference execution failed: {exc}") from exc

        q90_err = float(self.metadata.get("q90_residual", 6.5))
        lower_bound = round(max(0.0, pred_score - q90_err), 1)
        upper_bound = round(min(100.0, pred_score + q90_err), 1)

        attribution_data = compute_feature_attributions(
            self.model, self.preprocessor, input_df, pred_score
        )

        return {
            "predicted_math_score": round(pred_score, 1),
            "prediction_interval": {"lower": lower_bound, "upper": upper_bound},
            "feature_attributions": attribution_data["attributions"],
            "interventions": attribution_data["recommendations"],
            "model_name": str(self.metadata.get("model_name", "Ridge")),
            "model_version": str(self.metadata.get("model_version", self.config.version)),
            "validated_payload": validated,
        }
