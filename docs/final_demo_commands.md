# Final Demo Commands

## Start API
Windows PowerShell:

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

## Health
Browser:

http://127.0.0.1:8000/health

## Dashboard
Browser:

http://127.0.0.1:8000/dashboard

## Ask Endpoint
PowerShell command using Invoke-RestMethod:

Invoke-RestMethod `-Uri "http://127.0.0.1:8000/ask"`
-Method Post `-Headers @{"x-api-key"="demo-key"}`
-ContentType "application/json" `
-Body '{"question":"Where is my order?","top_k":5}'

## Feedback Endpoint
PowerShell command using Invoke-RestMethod:

Invoke-RestMethod `-Uri "http://127.0.0.1:8000/feedback"`
-Method Post `-Headers @{"x-api-key"="demo-key"}`
-ContentType "application/json" `
-Body '{"question":"Where is my order?","answer":"Demo answer","rating":5,"comment":"Helpful"}'

## Metrics Endpoint
PowerShell:

Invoke-RestMethod `-Uri "http://127.0.0.1:8000/metrics"`
-Headers @{"x-api-key"="demo-key"}

Also include curl.exe alternative:

curl.exe -H "x-api-key: demo-key" "http://127.0.0.1:8000/metrics"

## Evaluation

python evaluate_rag.py --sample-size 5 --top-k 5

## Monitoring Summary

python monitoring.py

## Optional MLflow Tracking

python mlflow_tracking.py

## Docker

docker build -t support-rag-chatbot .
docker run -p 8000:8000 --env-file .env support-rag-chatbot

## Render

- Push to GitHub.
- Create Render Web Service using render.yaml.
- Add environment variables in Render dashboard.
- Test /health and /ask.
