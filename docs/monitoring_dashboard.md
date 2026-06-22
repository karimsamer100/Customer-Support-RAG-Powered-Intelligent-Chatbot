# Monitoring Dashboard

The project includes a lightweight monitoring dashboard available at:

GET /dashboard

The dashboard reads from request logs, feedback logs, and evaluation results and displays basic metrics:

- total requests
- success/failure count
- average latency
- average retrieved cases
- average top similarity score
- retrieval success rate
- average user satisfaction rating
- evaluation metrics (from evaluation_results)

This dashboard is intended for demo purposes. For production, consider using Azure Application Insights, Grafana, or Power BI.

Test commands:

Start the API:

```bash
uvicorn api:app --reload
```

Open the dashboard:

http://127.0.0.1:8000/dashboard

Get JSON metrics:

```bash
curl -H "x-api-key: demo-key" http://127.0.0.1:8000/metrics
```
