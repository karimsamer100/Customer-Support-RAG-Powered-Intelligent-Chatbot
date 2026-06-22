# Azure Deployment Guide

This section outlines a simple approach to deploy the FastAPI app to Azure App Service using Docker.

1. Build a Docker image locally:

```bash
docker build -t support-rag-chatbot .
```

2. Push the image to a container registry (Azure Container Registry or Docker Hub).
3. Create an Azure App Service (Web App for Containers) and point it to the container image.
4. Configure environment variables in the App Service settings:

- APP_API_KEY
- LLM_API_KEY
- LLM_BASE_URL
- LLM_MODEL
- ENVIRONMENT

5. Ensure the `vector_store/` (FAISS index and metadata) is available in the container image or mounted storage.
6. Use `/health` as the health check path in Azure.

Notes:
- For production, integrate Azure AD and managed identity for secrets, and enable Application Insights for telemetry.
- The current demo uses API key auth; replace with a stronger auth method for production.
