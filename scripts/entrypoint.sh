#!/usr/bin/env sh
set -e

if [ ! -f "artifacts/models/model.joblib" ]; then
    echo "Model artifacts not found; running initial training pipeline..."
    python scripts/train.py
fi

exec uvicorn student_performance.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
