# Customer Support RAG-Powered Intelligent Chatbot

A production-oriented customer support automation project that combines **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, **SentenceTransformer embeddings**, **FastAPI**, lightweight **monitoring**, feedback logging, evaluation reports, and deployment-ready configuration.

The chatbot receives a customer support question, retrieves similar historical support cases from a vector database, and uses an LLM to generate a helpful response grounded in real support data.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Key Features](#key-features)
* [Milestone Coverage](#milestone-coverage)
* [System Architecture](#system-architecture)
* [Dataset](#dataset)
* [Preprocessing and EDA](#preprocessing-and-eda)
* [RAG Pipeline](#rag-pipeline)
* [API Endpoints](#api-endpoints)
* [Monitoring Dashboard and Feedback](#monitoring-dashboard-and-feedback)
* [Evaluation Results](#evaluation-results)
* [MLOps and Re-indexing](#mlops-and-re-indexing)
* [Deployment](#deployment)
* [Project Structure](#project-structure)
* [Setup Instructions](#setup-instructions)
* [Environment Variables](#environment-variables)
* [Run the Project](#run-the-project)
* [Demo Commands](#demo-commands)
* [Security Notes](#security-notes)
* [Limitations](#limitations)
* [Future Work](#future-work)

---

## Project Overview

This project implements an intelligent customer support chatbot powered by **Retrieval-Augmented Generation**.

Instead of relying only on the LLM’s general knowledge, the system first retrieves similar real support conversations from a vector store, then gives those retrieved cases to the LLM as context. This makes the final answer more relevant to the support domain and allows the chatbot to reuse historical support knowledge.

The project includes:

* Historical support data preprocessing
* Exploratory data analysis
* Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* FAISS vector search
* RAG-based answer generation
* FastAPI REST API
* API key security
* Monitoring dashboard
* User feedback endpoint
* Evaluation scripts and reports
* Docker and Render deployment preparation
* Azure deployment documentation
* Lightweight MLOps and re-indexing scripts

---

## Problem Statement

Customer support teams receive many repeated questions about orders, refunds, delivery, account issues, and product problems. Handling these questions manually can increase response time, support cost, and agent workload.

This project solves the problem by building a chatbot that can:

1. Understand the customer question semantically.
2. Retrieve similar historical support cases.
3. Generate a response using an LLM grounded in retrieved support examples.
4. Expose the system through a REST API that can be integrated into a support portal.
5. Track latency, retrieval quality, and user satisfaction for monitoring.

---

## Key Features

* **Data ingestion and preprocessing** for AmazonHelp support conversations
* **Text cleaning** for URLs, mentions, extra spaces, and noisy support patterns
* **EDA outputs** for issue categories and frequent question/answer terms
* **SentenceTransformer embeddings** using `all-MiniLM-L6-v2`
* **FAISS vector database** for semantic retrieval
* **RAG pipeline** with retrieval + LLM generation
* **Fallback handling** when retrieval or generation fails
* **FastAPI service** with `/ask`, `/health`, `/metrics`, `/feedback`, and `/dashboard`
* **API key authentication** using the `x-api-key` header
* **Monitoring dashboard** for latency, success/failure count, retrieval score, and feedback
* **Feedback logging** for satisfaction score tracking
* **Evaluation pipeline** with BLEU-like, ROUGE-L, latency, and retrieval metrics
* **Optional MLflow tracking helper** for evaluation metrics
* **Manual re-indexing script** to refresh embeddings and FAISS index
* **Dockerfile** and `render.yaml` for deployment preparation
* **Azure App Service deployment guide**
* **Support portal demo page** using plain HTML/JavaScript

---

## Milestone Coverage

| Milestone   | Requirement                          | Current Project Status                                                                                                   |
| ----------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Milestone 1 | Data collection and preprocessing    | Implemented using historical AmazonHelp support conversations, preprocessing notebook, processed corpus, and EDA outputs |
| Milestone 1 | Processed text corpus                | Implemented: `processed_data/processed_amazon_support.csv`                                                               |
| Milestone 1 | Preprocessing pipeline documentation | Implemented: `docs/preprocessing_pipeline.md`                                                                            |
| Milestone 1 | Support data EDA report              | Implemented through EDA notebook, EDA outputs, and documentation                                                         |
| Milestone 2 | Vector store                         | Implemented with FAISS: `vector_store/faiss_index.bin` and `vector_store/metadata.csv`                                   |
| Milestone 2 | RAG model configuration              | Implemented using SentenceTransformer embeddings + FAISS retrieval + LLM generator                                       |
| Milestone 2 | Evaluation metrics                   | Implemented: `evaluate_rag.py` and `evaluation_results/`                                                                 |
| Milestone 2 | Optimization                         | Implemented partially through similarity threshold, top-k retrieval, fallback handling, and evaluation support           |
| Milestone 3 | REST API                             | Implemented with FastAPI                                                                                                 |
| Milestone 3 | Workflow/support portal integration  | Demo implemented: `support_portal_demo/index.html`                                                                       |
| Milestone 3 | Security                             | Implemented using `x-api-key`; Azure AD is documented as future production enhancement                                   |
| Milestone 3 | Azure deployment                     | Azure App Service deployment guide provided; actual Azure deployment must be performed separately if required            |
| Milestone 4 | Experiment tracking                  | Optional helper implemented: `mlflow_tracking.py`                                                                        |
| Milestone 4 | Monitoring dashboard                 | Implemented: `/dashboard`, `/metrics`, and `monitoring.py`                                                               |
| Milestone 4 | User satisfaction tracking           | Implemented: `/feedback` endpoint and feedback logs                                                                      |
| Milestone 4 | Retraining/re-indexing               | Implemented as manual refresh script: `refresh_index.py`; scheduled automation is future work                            |
| Milestone 5 | Final documentation                  | Implemented through README and docs folder                                                                               |
| Milestone 5 | Demo presentation outline            | Implemented: `PRESENTATION_OUTLINE.md`                                                                                   |
| Milestone 5 | Business KPI impact analysis         | Implemented: `docs/business_kpi_impact.md`                                                                               |

> Note: The project is demo-ready and production-oriented. Some enterprise features such as Azure AD, fully automated scheduled retraining, and full cloud observability are documented as future production enhancements.

---

## System Architecture

```text
User Question
    ↓
FastAPI REST API
    ↓
SentenceTransformer Query Embedding
    ↓
FAISS Vector Search
    ↓
Top-K Retrieved Support Cases
    ↓
Prompt Construction
    ↓
LLM Generation
    ↓
Final Chatbot Answer + Retrieved Evidence + Monitoring Metadata
```

---

## Dataset

The project uses historical customer support conversations filtered for **AmazonHelp** interactions.

Main data files:

| File                                          | Description                                       |
| --------------------------------------------- | ------------------------------------------------- |
| `amazon_final_data.csv`                       | Filtered AmazonHelp support question/answer pairs |
| `processed_data/processed_amazon_support.csv` | Cleaned and processed support corpus              |
| `vector_store/metadata.csv`                   | Metadata used by the FAISS retrieval pipeline     |
| `vector_store/faiss_index.bin`                | FAISS vector index file                           |

Dataset summary from the current project state:

* Raw filtered AmazonHelp data: approximately **123k support pairs**
* Processed support corpus: approximately **123k cleaned rows**
* Current FAISS index metadata: **20,000 indexed rows** for demo/runtime efficiency

---

## Preprocessing and EDA

The preprocessing pipeline prepares support conversations for retrieval.

Typical preprocessing steps include:

* Loading historical support conversations
* Filtering AmazonHelp interactions
* Removing noisy text patterns
* Cleaning URLs, mentions, extra whitespace, and support-specific artifacts
* Creating cleaned question and answer fields
* Calculating basic text statistics
* Saving a processed corpus for embedding and retrieval

Important files:

* `preprocessing.ipynb`
* `eda.ipynb`
* `docs/preprocessing_pipeline.md`
* `eda_outputs/`
* `processed_data/processed_amazon_support.csv`

EDA outputs include issue category analysis and frequent words in support questions and answers.

---

## RAG Pipeline

The RAG pipeline is implemented mainly in:

* `embedder.py`
* `build_index.py`
* `rag_pipeline.py`
* `llm_generator.py`

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is lightweight and suitable for semantic similarity retrieval in a demo support chatbot.

### Vector Search

The project uses **FAISS** with normalized embeddings and inner-product similarity, which works similarly to cosine similarity.

### Retrieval

The system retrieves top-k similar support cases and filters weak matches using a similarity threshold.

Current retrieval safety features:

* Empty query handling
* Top-k clamping between 1 and 10
* Invalid FAISS index handling
* Duplicate retrieved-question removal
* Minimum similarity threshold

### Generation

The LLM receives retrieved support cases as context and generates a final answer. If the LLM call fails, the system falls back to the best retrieved support answer instead of crashing.

---

## API Endpoints

The API is implemented using **FastAPI** in `api.py`.

| Method | Endpoint     | Auth | Description                        |
| ------ | ------------ | ---- | ---------------------------------- |
| GET    | `/`          | No   | Root service status                |
| GET    | `/health`    | No   | Health check and pipeline status   |
| POST   | `/ask`       | Yes  | Ask the chatbot a support question |
| POST   | `/feedback`  | Yes  | Submit user satisfaction feedback  |
| GET    | `/metrics`   | Yes  | Return monitoring metrics as JSON  |
| GET    | `/dashboard` | No   | Demo monitoring dashboard          |

Protected endpoints use:

```text
x-api-key: <APP_API_KEY>
```

---

## Monitoring Dashboard and Feedback

The project includes a lightweight monitoring layer suitable for demo and academic evaluation.

Implemented monitoring components:

* `monitoring.py`
* `/metrics` endpoint
* `/dashboard` endpoint
* `/feedback` endpoint
* JSONL request logs
* JSONL feedback logs

The dashboard tracks:

* Total requests
* Successful requests
* Failed requests
* Average latency
* Average retrieved count
* Average top similarity score
* Retrieval success rate
* Total feedback submissions
* Average user rating
* Evaluation metrics

Open the dashboard after starting the API:

```text
http://127.0.0.1:8000/dashboard
```

Run monitoring summary locally:

```bash
python monitoring.py
```

> The dashboard is a real lightweight monitoring dashboard for the demo. For enterprise production, it can be replaced or extended with Azure Application Insights, Grafana, or Power BI.

---

## Evaluation Results

Evaluation is implemented in:

```bash
python evaluate_rag.py --sample-size 20 --top-k 5
```

The evaluation outputs are saved in:

* `evaluation_results/evaluation_details.csv`
* `evaluation_results/evaluation_metrics.json`
* `evaluation_results/evaluation_report.md`

Current evaluation metrics from the latest run:

| Metric                  |         Value |
| ----------------------- | ------------: |
| Sample Size             |             5 |
| Successful Evaluations  |             5 |
| Average Latency         | 0.653 seconds |
| Average Top Score       |         1.000 |
| Average Retrieved Count |          2.20 |
| BLEU-like Score         |         0.512 |
| ROUGE-L Score           |         0.821 |
| Retrieval Success Rate  |         1.000 |

> Evaluation is simplified for academic/demo purposes. BLEU-like and ROUGE-L scores help provide a quick sanity check but do not fully replace human review for support quality.

---

## MLOps and Re-indexing

The project includes lightweight MLOps support:

| Component              | File                  | Purpose                                        |
| ---------------------- | --------------------- | ---------------------------------------------- |
| Evaluation pipeline    | `evaluate_rag.py`     | Computes retrieval/generation metrics          |
| Evaluation outputs     | `evaluation_results/` | Stores metrics and reports                     |
| Optional MLflow helper | `mlflow_tracking.py`  | Logs evaluation metrics to MLflow if installed |
| Monitoring helpers     | `monitoring.py`       | Logs requests and feedback                     |
| Re-indexing script     | `refresh_index.py`    | Rebuilds FAISS index using `build_index.py`    |

Optional MLflow tracking:

```bash
python mlflow_tracking.py
```

Manual index refresh:

```bash
python refresh_index.py
```

In a production setting, this refresh process can be scheduled using cron, GitHub Actions, Azure Functions, or Azure ML pipelines.

---

## Deployment

The project is prepared for local, Docker, Render, and Azure-style deployment.

### Local Uvicorn

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t support-rag-chatbot .
docker run -p 8000:8000 --env-file .env support-rag-chatbot
```

### Render

The repository includes:

```text
render.yaml
```

Basic Render flow:

1. Push the project to GitHub.
2. Create a Render Web Service.
3. Use Docker deployment.
4. Add environment variables in the Render dashboard.
5. Deploy and test `/health`, `/ask`, and `/dashboard`.

### Azure

Azure deployment is documented in:

```text
docs/azure_deployment_guide.md
```

The recommended Azure path is Docker-based deployment to **Azure App Service**. Required environment variables should be configured in Azure App Service Configuration.

> The codebase is prepared for Azure App Service deployment, but an actual Azure deployment URL is not included unless you deploy it separately.

---

## Project Structure

```text
GRAD_PROJECT/
├── api.py
├── rag_pipeline.py
├── llm_generator.py
├── embedder.py
├── build_index.py
├── evaluate_rag.py
├── monitoring.py
├── refresh_index.py
├── mlflow_tracking.py
├── test_retrieval.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
├── .gitignore
├── .dockerignore
│
├── amazon_final_data.csv
├── processed_data/
│   └── processed_amazon_support.csv
│
├── vector_store/
│   ├── faiss_index.bin
│   └── metadata.csv
│
├── evaluation_results/
│   ├── evaluation_details.csv
│   ├── evaluation_metrics.json
│   └── evaluation_report.md
│
├── eda_outputs/
│   ├── issue_categories.png
│   ├── top_answer_words.png
│   └── top_question_words.png
│
├── docs/
│   ├── project_summary.md
│   ├── demo_script.md
│   ├── demo_checklist.md
│   ├── final_demo_commands.md
│   ├── final_submission_checklist.md
│   ├── preprocessing_pipeline.md
│   ├── model_pipeline.md
│   ├── evaluation_report.md
│   ├── deployment_guide.md
│   ├── deployment_security.md
│   ├── azure_deployment_guide.md
│   ├── monitoring_dashboard.md
│   ├── mlops_monitoring_plan.md
│   ├── retraining_pipeline.md
│   ├── support_portal_integration.md
│   ├── business_kpi_impact.md
│   └── future_work.md
│
├── support_portal_demo/
│   └── index.html
│
├── preprocessing.ipynb
├── eda.ipynb
├── DEPI_PROJECT.ipynb
└── PRESENTATION_OUTLINE.md
```

---

## Setup Instructions

### 1. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Then fill in the required keys inside `.env`.

---

## Environment Variables

The application expects:

| Variable       | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `APP_API_KEY`  | API key used to protect `/ask`, `/feedback`, and `/metrics` |
| `LLM_API_KEY`  | LLM provider API key                                        |
| `LLM_BASE_URL` | OpenAI-compatible LLM base URL                              |
| `LLM_MODEL`    | LLM model name                                              |
| `ENVIRONMENT`  | `development` or `production`                               |

Example `.env.example`:

```env
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=your_model_name_here
APP_API_KEY=demo-key
ENVIRONMENT=development
```

---

## Run the Project

### Build or Refresh Vector Index

```bash
python build_index.py
```

or:

```bash
python refresh_index.py
```

### Start API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check

Open:

```text
http://127.0.0.1:8000/health
```

---

## Demo Commands

### Ask Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -Headers @{"x-api-key"="demo-key"} `
  -ContentType "application/json" `
  -Body '{"question":"Where is my order?","top_k":5}'
```

### Feedback Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/feedback" `
  -Method Post `
  -Headers @{"x-api-key"="demo-key"} `
  -ContentType "application/json" `
  -Body '{"question":"Where is my order?","answer":"Demo answer","rating":5,"comment":"Helpful"}'
```

### Metrics Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/metrics" `
  -Headers @{"x-api-key"="demo-key"}
```

### Dashboard

```text
http://127.0.0.1:8000/dashboard
```

### Evaluation

```bash
python evaluate_rag.py --sample-size 5 --top-k 5
```

### Monitoring Summary

```bash
python monitoring.py
```

---

## Support Portal Demo

A simple support portal integration demo is included in:

```text
support_portal_demo/index.html
```

It demonstrates how a support portal can:

* Send a customer question to `/ask`
* Display the generated answer
* Show retrieval metadata
* Submit user satisfaction feedback to `/feedback`

---

## Security Notes

Implemented security:

* Protected endpoints use `x-api-key`
* Secrets are loaded from `.env`
* `.env` is ignored by Git
* `.env.example` is provided for safe configuration sharing
* Runtime logs are ignored using `logs/`

Important:

* Do not commit `.env`
* Do not commit API keys
* Rotate any API keys that were accidentally committed in the past
* Use stronger authentication such as Azure AD for enterprise production

---

## Limitations

* The current knowledge base mainly uses historical support conversations.
* FAQ, product manual, and full knowledge base ingestion are documented as future expansion.
* The FAISS index currently uses a subset of the processed corpus for demo efficiency.
* Evaluation is simplified and should be extended with human relevance scoring for production.
* Conversation memory is not implemented because it was not part of the core requirement.
* Azure AD is not implemented; API key authentication is used for the demo.
* Automated scheduled retraining is not implemented; manual re-indexing is available through `refresh_index.py`.

---

## Future Work

* Add FAQ, manuals, and knowledge base documents
* Add conversation memory or LangChain memory
* Deploy fully on Azure App Service
* Add Azure AD authentication
* Add CI/CD pipeline
* Add Azure Application Insights or Grafana monitoring
* Add scheduled embedding refresh
* Add human review workflow
* Add advanced human relevance evaluation
* Add real support ticket backend integration

---

## Business KPI Impact

| KPI                    | Expected Impact                                                 |
| ---------------------- | --------------------------------------------------------------- |
| First Response Time    | Faster answers for repeated customer questions                  |
| Ticket Deflection Rate | Reduces repetitive tickets handled by human agents              |
| Agent Productivity     | Agents can focus on complex cases instead of repeated questions |
| Customer Satisfaction  | Faster and more consistent support responses                    |
| Support Cost           | Lower operational load through automation                       |

---

## Final Demo Resources

| File                                      | Purpose                                            |
| ----------------------------------------- | -------------------------------------------------- |
| `docs/project_summary.md`                 | Short explanation of the project                   |
| `docs/demo_script.md`                     | Speaking script for the demo                       |
| `docs/final_demo_commands.md`             | Commands for API, metrics, feedback, and dashboard |
| `docs/final_submission_checklist.md`      | Final checklist before submission                  |
| `evaluation_results/evaluation_report.md` | Model evaluation report                            |
| `PRESENTATION_OUTLINE.md`                 | Presentation slide outline                         |

---

## Conclusion

This project demonstrates a complete RAG-powered customer support chatbot workflow: data preprocessing, vector search, LLM answer generation, REST API access, monitoring, feedback collection, evaluation, deployment preparation, and lightweight MLOps support.

It is suitable for an academic/demo submission and provides a strong foundation for future production deployment on Azure or other cloud platforms.
