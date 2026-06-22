import json
from pathlib import Path

METRICS_PATH = Path("evaluation_results") / "evaluation_metrics.json"

if not METRICS_PATH.exists():
    print("Evaluation metrics not found. Run: python evaluate_rag.py --sample-size 20 --top-k 5")
    raise SystemExit(1)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

params = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_store": "FAISS",
    "top_k": 5,
}

try:
    import mlflow

    print("MLflow detected — logging metrics...")
    with mlflow.start_run(run_name="rag_evaluation_demo"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        # log numeric metrics
        for k, v in metrics.items():
            try:
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            except Exception:
                continue
    print("MLflow tracking completed successfully.")
except Exception:
    print("MLflow is not installed. To enable MLflow tracking, run: pip install mlflow")
    # save fallback JSON
    fallback = {
        "experiment_name": "rag_evaluation_demo",
        "params": params,
        "metrics": metrics,
    }
    out = Path("evaluation_results") / "mlflow_fallback_tracking.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        print(f"Saved fallback tracking to {out}")
    except Exception as e:
        print("Failed to save fallback tracking:", e)
