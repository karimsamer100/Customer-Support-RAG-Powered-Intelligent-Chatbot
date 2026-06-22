# MLOps & Monitoring Plan

## Implemented Demo Features

- Lightweight evaluation script: `evaluate_rag.py` generates `evaluation_results/` with metrics and a report.
- Simple request and feedback logging: `monitoring.py` writes JSONL logs to `logs/` and summarizes them.
- Demo monitoring endpoints: `GET /metrics` and `GET /dashboard` (HTML) returning summarized metrics.
- Manual re-indexing wrapper: `refresh_index.py` to invoke `build_index.py`.
- Optional MLflow helper: `mlflow_tracking.py` (runs only if `mlflow` is installed).

## Future Production Enhancements

- Full MLflow experiment registry and tracking server.
- Azure Application Insights or Prometheus + Grafana for richer telemetry and alerting.
- Scheduled retraining and re-indexing via CI/CD or orchestration (GitHub Actions, Azure Functions).
- Human review and labeling workflow for improving generation relevance.
- Stronger authentication (Azure AD) and centralized secret management.