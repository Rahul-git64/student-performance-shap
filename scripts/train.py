"""CLI script to execute student performance training pipeline."""

from __future__ import annotations

from student_performance.core.config import load_config
from student_performance.core.logger import get_logger, log_event
from student_performance.models.trainer import train_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Load config and train student performance models."""
    config = load_config()
    metadata = train_pipeline(config)
    log_event(
        logger,
        "cli_student_training_finished",
        model_name=metadata["model_name"],
        metrics=metadata["metrics"],
        fairness_passed=metadata["subgroup_fairness"]["fairness_passed"],
    )


if __name__ == "__main__":
    main()
