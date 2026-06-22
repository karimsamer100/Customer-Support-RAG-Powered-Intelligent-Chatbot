# Customer Support RAG-Powered Intelligent Chatbot

## 1. Project Overview
This project implements a customer support chatbot using Retrieval-Augmented Generation (RAG). It retrieves similar historical support cases from a FAISS vector database and uses an LLM to generate a helpful, context-aware answer.

## 2. Problem Statement
Support teams receive many repeated questions. This project demonstrates a lightweight RAG system to provide faster, consistent automated responses and help human agents focus on complex cases.

## 3. Dataset
The project uses historical customer support conversations filtered for AmazonHelp support interactions. The processed corpus contains customer questions and support answers in a cleaned/structured format located in `processed_data/processed_amazon_support.csv`.

## 4. Main Features
- Data preprocessing and cleaning
- Exploratory data analysis (notebooks in the repo)
- Sentence-transformer embeddings
- FAISS vector search
- RAG-based answer generation
- FastAPI REST API
- Basic API key security
- Docker support
- Evaluation script for retrieval and generation

## 5. Project Structure
A high-level tree of the important files and folders in this repo:

- amazon_final_data.csv
- api.py
- build_index.py
- build_index.py (embeddings / index builder)
- embedder.py
- eda.ipynb
- eda_outputs/
  - amazon_support_with_categories.csv
- preprocessing.ipynb
- processed_data/
  - processed_amazon_support.csv
- vector_store/
  - metadata.csv
- rag_pipeline.py
- llm_generator.py
- evaluate_rag.py
- test_retrieval.py
- Dockerfile
- requirements.txt

(Keep notebooks and CSVs in place; they are part of the demo assets.)

## 6. Setup Instructions
Create a virtual environment and install dependencies:

Windows (PowerShell):

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

Windows (cmd):

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Linux / macOS:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Create a `.env` file based on `.env.example` (do not commit real secrets):

copy .env.example .env  # Windows
cp .env.example .env    # Linux / macOS

IMPORTANT: Real secret keys must stay in `.env` and should NOT be committed to version control.

## 7. Environment Variables
The project expects the following environment variables (do not put real values in the repo):

- LLM_API_KEY=your_llm_api_key_here
- LLM_BASE_URL=https://api.example.com/v1
- LLM_MODEL=your_model_name_here
- APP_API_KEY=demo-key
- ENVIRONMENT=development

## 8. Build Vector Index
To build the FAISS index from the processed corpus, run:

python build_index.py

(See `build_index.py` and `embedder.py` for implementation details.)

## 9. Run the API
Start the FastAPI app locally with Uvicorn:

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

## 10. API Endpoints
- GET /health — basic health check
- POST /ask — ask a question (requires API key header)

Example curl request (replace URL and API key as needed):

Windows (PowerShell / cmd):

curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -H "x-api-key: demo-key" ^
  -d "{\"question\": \"How do I return an item?\", \"top_k\": 5}"

Linux / macOS:

curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-key" \
  -d '{"question": "How do I return an item?", "top_k": 5}'

The API returns the generated answer along with the retrieved source snippets and metadata.

## 11. Evaluation
Run the evaluation script to compute retrieval and generation metrics and latency:

Basic usage:

```bash
python evaluate_rag.py --sample-size 20 --top-k 5
```

This script produces outputs under `evaluation_results/`:

- `evaluation_details.csv` — per-query results and metrics
- `evaluation_metrics.json` — aggregate metrics
- `evaluation_report.md` — human-readable report

## 12. Docker
To build and run the Docker image (Dockerfile exists):

docker build -t support-rag-chatbot .
docker run -p 8000:8000 --env-file .env support-rag-chatbot

Render deployment

This repo includes a `render.yaml` to simplify deployment on Render.com (Docker). After pushing to GitHub, connect the repo to Render and set environment variables in the Render dashboard. See `docs/deployment_guide.md` for step-by-step instructions.

## 13. Limitations
- Corpus is mainly historical support conversations; coverage may be limited.
- Integration with FAQ or product manuals is not included but can be added.
- Evaluation is simplified for a demo and focuses on retrieval and generation similarity.
- Production monitoring, robust authentication, and rate limiting are out of scope for this demo.
- LangChain is intentionally not used; the codebase implements embedding/search and simple prompt orchestration directly.

## 14. Future Work
- Deploy to Azure App Service or other cloud platforms
- Integrate Azure AD authentication for enterprise use
- Add MLflow or other model/experiment tracking
- Add monitoring/dashboard (Azure Application Insights)
- Scheduled embedding refresh when new support data arrives
- Conversation memory and richer session handling
- Integrate with a support portal or ticketing system


## Notes about existing .env
A `.env` file exists in the repository root on this machine. Do NOT commit its contents. Keep secrets only in `.env` and in your system's protected secret store.


## Quick Commands Summary
- Build index: `python build_index.py`
- Run API: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
- Run evaluation: `python evaluate_rag.py`

## Final Demo Documentation
Short-guide and speaking notes for the demo are available in the `docs/` folder and the project root:

- [docs/project_summary.md](docs/project_summary.md)
- [docs/demo_script.md](docs/demo_script.md)
- [docs/demo_checklist.md](docs/demo_checklist.md)
- [docs/future_work.md](docs/future_work.md)
- [docs/deployment_guide.md](docs/deployment_guide.md)
- [evaluation_results/evaluation_report.md](evaluation_results/evaluation_report.md)
- [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md)

## Final Demo

Quick resources for the final demo:

- Final demo commands: [docs/final_demo_commands.md](docs/final_demo_commands.md)
- Final submission checklist: [docs/final_submission_checklist.md](docs/final_submission_checklist.md)

## MLOps, Monitoring, and Re-indexing

Quick commands:

- Run evaluation:
  ```bash
  python evaluate_rag.py --sample-size 20 --top-k 5
  ```
- View monitoring summary locally:
  ```bash
  python monitoring.py
  ```
- Run optional MLflow tracking (mlflow not required):
  ```bash
  python mlflow_tracking.py
  ```
- Refresh vector index (manual):
  ```bash
  python refresh_index.py
  ```
- Dashboard (open in browser after starting API):
  ```
  http://127.0.0.1:8000/dashboard
  ```
- Metrics endpoint (requires API key):
  ```bash
  curl -H "x-api-key: demo-key" http://127.0.0.1:8000/metrics
  ```
- Feedback endpoint example:
  ```bash
  curl -X POST "http://127.0.0.1:8000/feedback" \
    -H "Content-Type: application/json" \
    -H "x-api-key: demo-key" \
    -d '{"question":"Where is my order?","answer":"Demo answer","rating":5,"comment":"Helpful"}'
  ```


Thank you — the project is organized for a quick academic/demo submission. If you'd like, I can run the evaluation or start the API locally next.