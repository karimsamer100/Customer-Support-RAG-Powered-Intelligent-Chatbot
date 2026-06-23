# Customer Support RAG-Powered Intelligent Chatbot

A cloud-deployed, production-oriented **Customer Support RAG Chatbot** that combines **Retrieval-Augmented Generation**, **FAISS vector search**, **SentenceTransformer embeddings**, **FastAPI**, **monitoring dashboard**, **feedback logging**, **evaluation reports**, Docker deployment, and lightweight MLOps utilities.

The system answers customer support questions by retrieving similar historical support cases and using them as grounding context for LLM-based answer generation.

---

## Live Deployment

The chatbot API is deployed as a public cloud service on Railway.

| Resource             | URL                                                             |
| -------------------- | --------------------------------------------------------------- |
| Public API Root      | https://support-rag-chatbot-production.up.railway.app           |
| Health Check         | https://support-rag-chatbot-production.up.railway.app/health    |
| Interactive API Docs | https://support-rag-chatbot-production.up.railway.app/docs      |
| Monitoring Dashboard | https://support-rag-chatbot-production.up.railway.app/dashboard |

> Note: Protected endpoints such as `/ask`, `/feedback`, and `/metrics` require the `x-api-key` header.

---

## Project Overview

This project implements an intelligent customer support automation system using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on the LLM’s general knowledge, the chatbot first retrieves semantically similar historical support cases from a vector database, then passes those retrieved cases to the LLM as context. This makes the generated answer more grounded, relevant, and aligned with real support data.

The project includes:

* Historical customer support data preprocessing
* Exploratory data analysis
* SentenceTransformer embeddings
* FAISS semantic vector search
* RAG-based answer generation
* FastAPI REST API
* API key security
* Public cloud deployment
* Monitoring dashboard
* Feedback collection
* Evaluation metrics and reports
* Docker deployment
* Railway deployment
* Azure App Service deployment documentation
* Lightweight MLOps and re-indexing scripts

---

## Problem Statement

Customer support teams receive many repeated questions about orders, refunds, delivery delays, account issues, and product problems.

Manually answering these repeated questions can lead to:

* Slower first response time
* Higher support cost
* Increased agent workload
* Inconsistent support answers
* Lower customer satisfaction

This project addresses the problem by creating a chatbot that can retrieve relevant historical support cases and generate helpful answers automatically.

---

## Key Features

* **RAG-powered support chatbot**
* **Semantic retrieval** using FAISS
* **SentenceTransformer embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
* **FastAPI REST API**
* **Cloud deployment on Railway**
* **Dockerized application**
* **API key authentication**
* **Monitoring dashboard**
* **Request metrics endpoint**
* **User feedback endpoint**
* **Evaluation reports**
* **BLEU-like and ROUGE-L scoring**
* **Latency and retrieval tracking**
* **Optional MLflow tracking helper**
* **Manual re-indexing / retraining script**
* **Support portal integration demo**
* **Azure deployment documentation**

---

## Milestone Coverage

| Milestone   | Requirement                 | Project Status                                                |
| ----------- | --------------------------- | ------------------------------------------------------------- |
| Milestone 1 | Data ingestion              | Implemented using historical AmazonHelp support conversations |
| Milestone 1 | Text preprocessing          | Implemented with cleaning and processed corpus generation     |
| Milestone 1 | EDA                         | Implemented using notebooks and EDA outputs                   |
| Milestone 1 | Processed corpus            | Available in `processed_data/`                                |
| Milestone 1 | Preprocessing documentation | Available in `docs/preprocessing_pipeline.md`                 |
| Milestone 2 | Vector store                | Implemented using FAISS                                       |
| Milestone 2 | RAG pipeline                | Implemented using SentenceTransformer + FAISS + LLM           |
| Milestone 2 | Evaluation                  | Implemented using `evaluate_rag.py` and `evaluation_results/` |
| Milestone 2 | Optimization                | Similarity threshold, top-k retrieval, fallback handling      |
| Milestone 3 | REST API                    | Implemented using FastAPI                                     |
| Milestone 3 | Workflow integration        | Demo support portal included                                  |
| Milestone 3 | Security                    | Implemented using `x-api-key` authentication                  |
| Milestone 3 | Deployment                  | Deployed publicly on Railway; Azure deployment guide included |
| Milestone 4 | MLflow tracking             | Optional helper implemented in `mlflow_tracking.py`           |
| Milestone 4 | Monitoring dashboard        | Implemented through `/dashboard`                              |
| Milestone 4 | Accuracy/latency monitoring | Implemented through `/metrics` and evaluation reports         |
| Milestone 4 | User satisfaction tracking  | Implemented through `/feedback`                               |
| Milestone 4 | Retraining mechanism        | Manual re-indexing implemented using `refresh_index.py`       |
| Milestone 5 | Final documentation         | README and docs folder included                               |
| Milestone 5 | Demo presentation outline   | Included in `PRESENTATION_OUTLINE.md`                         |
| Milestone 5 | Business KPI impact         | Included in `docs/business_kpi_impact.md`                     |

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
Final Answer + Retrieved Evidence + Monitoring Metadata
```

---

## Technology Stack

| Layer           | Tools                                               |
| --------------- | --------------------------------------------------- |
| Backend API     | FastAPI, Uvicorn                                    |
| RAG Pipeline    | Python, SentenceTransformers, FAISS                 |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2`            |
| Vector Search   | FAISS                                               |
| LLM Integration | OpenAI-compatible chat completion API               |
| Data Processing | Pandas, NumPy                                       |
| Evaluation      | Custom BLEU-like, ROUGE-L, latency, retrieval score |
| Monitoring      | JSONL logs, `/metrics`, `/dashboard`                |
| Deployment      | Docker, Railway                                     |
| Cloud Readiness | Azure App Service documentation                     |
| MLOps           | Optional MLflow helper, re-indexing script          |

---

## Dataset

The project uses historical customer support conversations filtered for **AmazonHelp** support interactions.

Main dataset files:

| File                                          | Description                                       |
| --------------------------------------------- | ------------------------------------------------- |
| `amazon_final_data.csv`                       | Filtered AmazonHelp support question-answer pairs |
| `processed_data/processed_amazon_support.csv` | Cleaned and processed support corpus              |
| `vector_store/metadata.csv`                   | Metadata used for retrieval                       |
| `vector_store/faiss_index.bin`                | FAISS vector index                                |

Dataset summary:

* Filtered AmazonHelp dataset: approximately **123k support pairs**
* Processed corpus: approximately **123k cleaned rows**
* Current FAISS index metadata: **20,000 indexed rows** for demo/runtime efficiency

> The current implementation focuses mainly on historical customer support conversations. FAQ, product manual, and full knowledge base ingestion are documented as future enhancements.

---

## Data Preprocessing

The preprocessing pipeline prepares the support data for embedding and retrieval.

Main preprocessing steps:

* Load historical support conversations
* Filter AmazonHelp interactions
* Remove noise and irrelevant patterns
* Normalize URLs and support links
* Remove mentions and extra whitespace
* Create cleaned question and answer columns
* Generate support categories and metadata
* Save the processed corpus

Important files:

* `preprocessing.ipynb`
* `DEPI_PROJECT.ipynb`
* `depi_project.py`
* `processed_data/processed_amazon_support.csv`
* `docs/preprocessing_pipeline.md`

---

## Exploratory Data Analysis

EDA was used to understand common customer support topics and support response patterns.

EDA includes:

* Common question terms
* Common answer terms
* Issue category distribution
* Text statistics
* Support topic exploration

Important files and folders:

* `eda.ipynb`
* `eda_outputs/`
* `docs/project_summary.md`

Example issue categories:

* Delivery and order tracking
* Refunds
* Product issues
* Account issues
* General support

---

## RAG Pipeline

The RAG pipeline is implemented across:

* `embedder.py`
* `build_index.py`
* `rag_pipeline.py`
* `llm_generator.py`

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is lightweight, fast, and suitable for semantic similarity retrieval in a demo customer support system.

### Vector Store

The vector store is implemented with **FAISS**.

Files:

```text
vector_store/faiss_index.bin
vector_store/metadata.csv
```

### Retrieval

The retriever embeds the user query and searches the FAISS index for similar historical support cases.

Retrieval safety features:

* Empty query handling
* Top-k clamping
* Invalid index handling
* Duplicate retrieved-question removal
* Minimum similarity threshold

### Generation

Retrieved support cases are inserted into a prompt and sent to an LLM through an OpenAI-compatible API.

If generation fails, the system returns a fallback response based on the most relevant retrieved support answer instead of crashing.

---

## API Endpoints

The API is implemented in `api.py`.

| Method | Endpoint     | Auth | Description                           |
| ------ | ------------ | ---- | ------------------------------------- |
| GET    | `/`          | No   | Root service status                   |
| GET    | `/health`    | No   | Health check and pipeline status      |
| POST   | `/ask`       | Yes  | Ask the chatbot a support question    |
| POST   | `/feedback`  | Yes  | Submit user satisfaction feedback     |
| GET    | `/metrics`   | Yes  | Return monitoring metrics as JSON     |
| GET    | `/dashboard` | No   | Monitoring dashboard                  |
| GET    | `/docs`      | No   | FastAPI interactive API documentation |

Protected endpoints require:

```text
x-api-key: <APP_API_KEY>
```

---

## Example API Usage

### Ask Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "https://support-rag-chatbot-production.up.railway.app/ask" `
  -Method Post `
  -Headers @{"x-api-key"="YOUR_APP_API_KEY"} `
  -ContentType "application/json" `
  -Body '{"question":"Where is my order?","top_k":5}'
```

Example response includes:

* Question
* Generated answer
* Retrieved support cases
* Latency
* Top similarity score
* Retrieved count
* Status

---

### Feedback Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "https://support-rag-chatbot-production.up.railway.app/feedback" `
  -Method Post `
  -Headers @{"x-api-key"="YOUR_APP_API_KEY"} `
  -ContentType "application/json" `
  -Body '{"question":"Where is my order?","answer":"Demo answer","rating":5,"comment":"Helpful"}'
```

---

### Metrics Endpoint - PowerShell

```powershell
Invoke-RestMethod `
  -Uri "https://support-rag-chatbot-production.up.railway.app/metrics" `
  -Headers @{"x-api-key"="YOUR_APP_API_KEY"}
```

---

## Monitoring Dashboard

The project includes a lightweight monitoring dashboard available at:

```text
https://support-rag-chatbot-production.up.railway.app/dashboard
```

The dashboard tracks:

* Total requests
* Successful requests
* Failed requests
* Average latency
* Average retrieved count
* Average top similarity score
* Retrieval success rate
* Feedback count
* Average rating
* Positive feedback count
* Negative feedback count
* Evaluation metrics

Monitoring components:

* `monitoring.py`
* `/metrics`
* `/dashboard`
* `/feedback`
* JSONL request logs
* JSONL feedback logs

> This dashboard is a lightweight implemented monitoring dashboard for demo and academic purposes. In production, it can be extended using Azure Application Insights, Grafana, or Power BI.

---

## Evaluation Results

Evaluation is implemented using:

```bash
python evaluate_rag.py --sample-size 20 --top-k 5
```

Output files:

* `evaluation_results/evaluation_details.csv`
* `evaluation_results/evaluation_metrics.json`
* `evaluation_results/evaluation_report.md`

Latest evaluation metrics:

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

> Evaluation is simplified for academic/demo purposes. BLEU-like and ROUGE-L provide a quick quality sanity check, but production evaluation should include unseen questions and human relevance scoring.

---

## MLOps and Re-indexing

The project includes lightweight MLOps utilities.

| Component                | File                  | Purpose                                        |
| ------------------------ | --------------------- | ---------------------------------------------- |
| Evaluation pipeline      | `evaluate_rag.py`     | Computes retrieval and generation metrics      |
| Evaluation outputs       | `evaluation_results/` | Stores metrics and reports                     |
| Monitoring helper        | `monitoring.py`       | Logs requests and feedback                     |
| Optional MLflow tracking | `mlflow_tracking.py`  | Logs evaluation metrics to MLflow if installed |
| Re-indexing script       | `refresh_index.py`    | Rebuilds the FAISS vector index                |

### Optional MLflow Tracking

```bash
python mlflow_tracking.py
```

If MLflow is installed, metrics are logged to MLflow. If not, a fallback tracking JSON is generated.

### Manual Index Refresh

```bash
python refresh_index.py
```

In a production environment, this process can be scheduled using cron, GitHub Actions, Azure Functions, or Azure ML pipelines.

---

## Deployment

The application is deployed publicly using **Railway** and is also prepared for Docker, Render, and Azure App Service deployment.

### Railway Deployment

Live service:

```text
https://support-rag-chatbot-production.up.railway.app
```

Railway deployment includes:

* Docker-based build
* Public HTTPS endpoint
* Environment variables
* Health check endpoint
* API key protected endpoints
* Monitoring dashboard
* Feedback endpoint

---

### Docker Deployment

```bash
docker build -t support-rag-chatbot .
docker run -p 8000:8000 --env-file .env support-rag-chatbot
```

---

### Render Deployment

The repository includes:

```text
render.yaml
```

Render can deploy the same Dockerized service using environment variables and `/health` as a health check path.

---

### Azure App Service Readiness

Azure deployment is documented in:

```text
docs/azure_deployment_guide.md
```

The codebase is ready for Docker-based deployment to Azure App Service. The same containerized application can be migrated to Azure by configuring:

* Azure App Service
* Environment variables
* Health check path `/health`
* API key authentication
* Optional Azure AD / Microsoft Entra ID authentication

> Due to Azure subscription constraints, the live demo deployment was hosted on Railway. Azure deployment documentation is included to demonstrate cloud readiness and migration path.

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

### 1. Create Virtual Environment

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

---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3. Create `.env`

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Fill in the required environment variables.

---

## Environment Variables

| Variable       | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `APP_API_KEY`  | API key used to protect `/ask`, `/feedback`, and `/metrics` |
| `LLM_API_KEY`  | LLM provider API key                                        |
| `LLM_BASE_URL` | OpenAI-compatible LLM base URL                              |
| `LLM_MODEL`    | LLM model name                                              |
| `ENVIRONMENT`  | `development` or `production`                               |

Example:

```env
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=your_model_name_here
APP_API_KEY=demo-key
ENVIRONMENT=development
```

---

## Run Locally

### Build or Refresh Vector Index

```bash
python build_index.py
```

or:

```bash
python refresh_index.py
```

---

### Start API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

### Health Check

```text
http://127.0.0.1:8000/health
```

---

### Local Dashboard

```text
http://127.0.0.1:8000/dashboard
```

---

## Support Portal Demo

A simple support portal integration demo is included in:

```text
support_portal_demo/index.html
```

It demonstrates how a business support portal can:

* Send customer questions to `/ask`
* Display chatbot answers
* Show retrieval metadata
* Submit feedback to `/feedback`

---

## Security Notes

Implemented security:

* Protected endpoints use `x-api-key`
* Secrets are stored in environment variables
* `.env` is ignored by Git
* `.env.example` is provided safely
* Runtime logs are ignored
* API keys are not hardcoded

Important:

* Do not commit `.env`
* Do not commit real API keys
* Rotate any leaked keys
* Use Azure AD / Microsoft Entra ID for enterprise production

---

## Business KPI Impact

| KPI                    | Expected Impact                               |
| ---------------------- | --------------------------------------------- |
| First Response Time    | Faster answers for repeated support questions |
| Ticket Deflection Rate | Reduces repetitive tickets handled by agents  |
| Agent Productivity     | Allows agents to focus on complex issues      |
| Customer Satisfaction  | Improves speed and consistency of support     |
| Support Cost           | Reduces operational load through automation   |

---

## Limitations

* The current corpus mainly uses historical support conversations.
* FAQ, manual, and full knowledge base ingestion are future enhancements.
* The FAISS index uses a subset of the processed corpus for demo efficiency.
* Evaluation is simplified and should be extended with human scoring.
* Conversation memory is not implemented.
* Azure AD is not implemented; API key authentication is used for demo security.
* Scheduled retraining is not automated yet; manual re-indexing is implemented.
* Production observability can be improved with Azure Application Insights, Grafana, or Power BI.

---

## Future Work

* Add FAQ, product manuals, and knowledge base documents
* Add conversation memory
* Add LangChain memory if required later
* Deploy on Azure App Service
* Add Azure AD / Microsoft Entra ID authentication
* Add CI/CD pipeline
* Add scheduled embedding refresh
* Add advanced monitoring with Azure Application Insights or Grafana
* Add human review workflow
* Add advanced evaluation with unseen questions
* Integrate with a real support ticket backend

---

## Final Demo Resources

| File                                      | Purpose                    |
| ----------------------------------------- | -------------------------- |
| `docs/project_summary.md`                 | Short project explanation  |
| `docs/demo_script.md`                     | Speaking script for demo   |
| `docs/final_demo_commands.md`             | API demo commands          |
| `docs/final_submission_checklist.md`      | Final submission checklist |
| `evaluation_results/evaluation_report.md` | Model evaluation report    |
| `PRESENTATION_OUTLINE.md`                 | Slide outline              |
| `docs/business_kpi_impact.md`             | Business KPI analysis      |

---

## Conclusion

This project demonstrates a complete RAG-powered customer support chatbot workflow: data preprocessing, semantic vector search, LLM answer generation, REST API access, public cloud deployment, monitoring dashboard, feedback collection, evaluation, Docker deployment, and lightweight MLOps support.

It is suitable for an academic/demo submission and provides a strong foundation for future enterprise deployment on Azure or other cloud platforms.
