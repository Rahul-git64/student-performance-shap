"""FastAPI application for student performance scoring, local attributions, and drift supervision."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from student_performance.api.monitoring import DriftSupervisor
from student_performance.api.schemas import (
    DriftReportResponse,
    HealthResponse,
    StudentPredictionRequest,
    StudentPredictionResponse,
)
from student_performance.core.config import AppConfig, load_config
from student_performance.core.exceptions import StudentPerformanceError
from student_performance.core.logger import get_logger, log_event
from student_performance.models.registry import ModelRegistry

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI student performance prediction application."""
    cfg = config or load_config()
    registry = ModelRegistry(cfg)
    supervisor = DriftSupervisor(cfg)
    start_time = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        log_event(logger, "service_startup_initiated", version=cfg.version)
        registry.ensure_loaded(auto_train=True)
        log_event(
            logger,
            "service_startup_complete",
            active_model=registry.metadata.get("model_name", "unknown"),
        )
        yield

    app = FastAPI(
        title="Student Performance ML Service",
        description=(
            "Academic performance prediction service featuring per-feature local attributions, "
            "subgroup fairness monitoring, and score drift supervision."
        ),
        version=cfg.version,
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.registry = registry
    app.state.supervisor = supervisor
    app.state.start_time = start_time

    @app.get("/health", response_model=HealthResponse, tags=["Supervision"])
    def health_check() -> HealthResponse:
        """Return service health, uptime, and active model metrics with subgroup fairness audit."""
        app.state.registry.ensure_loaded(auto_train=True)
        uptime = round(time.monotonic() - app.state.start_time, 2)
        meta = app.state.registry.metadata
        metrics = {
            k: float(v)
            for k, v in meta.get("metrics", {}).items()
            if isinstance(v, (int, float))
        }
        return HealthResponse(
            status="healthy",
            service=cfg.project_name,
            version=cfg.version,
            uptime_seconds=uptime,
            active_model=str(meta.get("model_name", "unknown")),
            model_metrics=metrics,
            subgroup_fairness=meta.get("subgroup_fairness", {}),
            buffered_requests=len(app.state.supervisor.buffer),
        )

    @app.post("/predict", response_model=StudentPredictionResponse, tags=["Inference"])
    def predict_score(request: StudentPredictionRequest) -> StudentPredictionResponse:
        """Validate student profile, predict math score, generate attributions, and log drift."""
        t0 = time.perf_counter()
        payload = request.model_dump()
        try:
            res = app.state.registry.predict_single(payload)
        except StudentPerformanceError as exc:
            log_event(logger, "inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_request(res["validated_payload"])

        log_event(
            logger,
            "prediction_completed",
            predicted_score=res["predicted_math_score"],
            reading_score=request.reading_score,
            writing_score=request.writing_score,
            latency_ms=latency_ms,
        )

        return StudentPredictionResponse(
            predicted_math_score=res["predicted_math_score"],
            prediction_interval=res["prediction_interval"],
            feature_attributions=res["feature_attributions"],
            interventions=res["interventions"],
            model_name=res["model_name"],
            model_version=res["model_version"],
            latency_ms=latency_ms,
        )

    @app.get("/monitor/drift", response_model=DriftReportResponse, tags=["Supervision"])
    def monitor_drift() -> DriftReportResponse:
        """Evaluate score distribution and demographic feature drift on recent requests."""
        app.state.registry.ensure_loaded(auto_train=True)
        report = app.state.supervisor.evaluate(app.state.registry.reference_stats)
        log_event(
            logger,
            "drift_evaluation_executed",
            status=report["status"],
            sample_size=report["sample_size"],
            overall_drift_detected=report["overall_drift_detected"],
        )
        return DriftReportResponse(**report)

    return app


app = create_app()
