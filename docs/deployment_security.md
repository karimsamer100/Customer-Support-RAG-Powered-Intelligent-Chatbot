# Deployment & Security

- The app is served using FastAPI (`api.py`).
- The project can be containerized using the included `Dockerfile` and deployed to any container host.
 - The project can be containerized using the included `Dockerfile` and deployed to any container host.
 - A `render.yaml` is provided to assist with deployment to Render.com (Docker environment).
- Environment variables (from `.env`) are used for secrets such as `LLM_API_KEY` and `APP_API_KEY`.
- The `/ask` endpoint is protected with a basic API key check; this is intended for demo purposes only.
- For production, integrate a proper authentication layer (e.g., Azure AD, OAuth2) and enable TLS, rate limiting, and logging/monitoring.
- Never commit `.env` or other secret files to source control.

Notes for demo deployments:
- The included `Dockerfile` and `.dockerignore` are configured for quick demo deployments. Do not copy `.env` into images — use platform secrets.
- Render or Railway can be used for simple deployments; add secrets in the service dashboard and verify `/health` after deployment.

Recommended production steps:
- Use managed secret stores (Key Vault, Azure App Configuration)
- Use HTTPS with a proper certificate
- Configure logging to Application Insights or another centralized system
- Apply resource limits and autoscaling for the container running the API