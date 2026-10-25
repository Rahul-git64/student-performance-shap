"""Custom exception hierarchy for student performance prediction pipeline."""

from __future__ import annotations


class StudentPerformanceError(Exception):
    """Base exception for all diamond valuation pipeline errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(StudentPerformanceError):
    """Raised when YAML configuration cannot be loaded or validated."""


class DataIngestionError(StudentPerformanceError):
    """Raised when student dataset ingestion or synthetic generation fails."""


class DataValidationError(StudentPerformanceError):
    """Raised when score ranges or demographic categories are invalid."""


class FeatureEngineeringError(StudentPerformanceError):
    """Raised when composite feature computation fails."""


class ModelTrainingError(StudentPerformanceError):
    """Raised when regression training or fairness gates fail."""


class InferenceError(StudentPerformanceError):
    """Raised when prediction or attribution inference fails."""
