# Evaluation Report

Run the evaluation script to generate a human-readable evaluation report and raw metrics.

To run:

python evaluate_rag.py --sample-size 20 --top-k 5

Outputs (written to `evaluation_results/`):

- evaluation_details.csv — per-query results and metrics
- evaluation_metrics.json — aggregate metrics
- evaluation_report.md — human-readable report (also found under `evaluation_results/`)

The generated `evaluation_report.md` contains overview, setup, metrics, sample results, and notes.