"""Unit tests for model training, subgroup fairness audit, and local feature attributions."""

from __future__ import annotations

from student_performance.core.config import load_config
from student_performance.models.registry import ModelRegistry
from student_performance.models.trainer import train_pipeline


def test_student_trainer_and_fairness_audit() -> None:
    cfg = load_config()
    metadata = train_pipeline(cfg)
    assert "metrics" in metadata
    assert metadata["metrics"]["r2"] >= cfg.model.r2_threshold
    assert "subgroup_fairness" in metadata
    assert "subgroups" in metadata["subgroup_fairness"]
    assert cfg.model.model_path.exists()
    assert cfg.model.preprocessor_path.exists()
    assert cfg.model.reference_stats_path.exists()


def test_model_registry_attributions_and_interventions() -> None:
    cfg = load_config()
    registry = ModelRegistry(cfg)
    registry.ensure_loaded(auto_train=True)

    student = {
        "gender": "male",
        "race_ethnicity": "group D",
        "parental_level_of_education": "some college",
        "lunch_type": "standard",
        "test_preparation_course": "completed",
        "reading_score": 85.0,
        "writing_score": 82.0,
    }
    result = registry.predict_single(student)
    assert 0.0 <= result["predicted_math_score"] <= 100.0
    assert result["prediction_interval"]["lower"] <= result["predicted_math_score"]
    assert "reading_score" in result["feature_attributions"]
    assert len(result["interventions"]) >= 1

    # Test student with uncompleted course triggers actionable intervention
    student_unprep = {**student, "test_preparation_course": "none", "reading_score": 45.0}
    res_unprep = registry.predict_single(student_unprep)
    assert any("preparation course" in item.lower() for item in res_unprep["interventions"])
