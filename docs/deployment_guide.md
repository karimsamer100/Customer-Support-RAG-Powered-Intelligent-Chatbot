# Deployment Guide

## Local Run

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Windows (cmd):

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

Set the following environment variables (local `.env` for development, platform env vars for deployment):

- APP_API_KEY
- LLM_API_KEY
- LLM_BASE_URL
- LLM_MODEL
- ENVIRONMENT

Do NOT commit `.env` to version control. For platforms like Render, set these values in the app dashboard as secrets.

## Docker Run

Build and run the Docker image locally:

```bash
docker build -t support-rag-chatbot .
docker run -p 8000:8000 --env-file .env support-rag-chatbot
```

## Render Deployment

1. Push the repository to GitHub.
2. Create a new Web Service on Render and connect your GitHub repo.
3. Choose Docker as the environment or rely on `render.yaml` in the repo.
4. Add required environment variables in the Render dashboard (APP_API_KEY, LLM_API_KEY, etc.).
5. Deploy and monitor build logs.
6. Verify `/health` and test `/ask` with `x-api-key` header.

## Test Commands

Health check:

```bash
curl https://YOUR-RENDER-URL/health
```

Ask endpoint example:

```bash
curl -X POST "https://YOUR-RENDER-URL/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_APP_API_KEY" \
  -d '{"question":"Where is my order?","top_k":5}'
```

Windows PowerShell example:

```powershell
curl -X POST "https://YOUR-RENDER-URL/ask" ^
  -H "Content-Type: application/json" ^
  -H "x-api-key: YOUR_APP_API_KEY" ^
  -d "{\"question\": \"Where is my order?\", \"top_k\": 5}"
```

## Notes

- First startup may be slow due to embedding model downloads or loading FAISS index.
- Ensure `vector_store/` is included in the deployment so `faiss_index.bin` and `metadata.csv` are available.
- Keep `.env` out of source control and use platform secrets for deployments.
- The API-key check in `api.py` is simple demo security; for production consider stronger auth (OAuth2 / Azure AD).
