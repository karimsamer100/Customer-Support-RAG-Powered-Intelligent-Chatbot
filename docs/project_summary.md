# Project Summary: Customer Support RAG-Powered Intelligent Chatbot

## Overview
This project is a customer support chatbot using Retrieval-Augmented Generation (RAG). It retrieves similar historical support cases using vector search, and uses a Large Language Model (LLM) to generate a final, helpful answer. The chatbot is exposed through a FastAPI REST API.

## Problem
Support teams receive many repeated customer questions. Manual support can be slow and expensive. A RAG chatbot can reduce response time and help agents by surfacing relevant historical answers.

## Dataset
The project uses historical customer support conversations, filtered to focus on AmazonHelp support interactions. The final corpus contains customer questions and support answers used to compute embeddings and build the FAISS vector store. Integration with FAQ/manual/knowledge-base documents can be added later.

## Pipeline
User Question → FastAPI Endpoint → SentenceTransformer Embedding → FAISS Vector Search → Retrieved Support Cases → LLM Generation → Final Chatbot Answer

## Implemented Features
- Data preprocessing and cleaning (notebooks and scripts)
- Exploratory data analysis (EDA notebooks)
- SentenceTransformer embeddings (sentence-transformers/all-MiniLM-L6-v2)
- FAISS vector database for similarity search
- RAG answer generation using retrieved cases + LLM
- FastAPI REST API with `x-api-key` authentication
- Fallback handling if retrieval or LLM fails
- Evaluation script and saved results (`evaluation_results/`)
- Dockerfile and `render.yaml` for simple deployments
 - Monitoring dashboard (`GET /dashboard`) and JSON metrics endpoint (`GET /metrics`)
 - Feedback endpoint (`POST /feedback`) and feedback logging
 - Manual re-indexing wrapper `refresh_index.py`
 - Optional MLflow tracking script `mlflow_tracking.py`

## Evaluation Summary
The repository contains an evaluation output at `evaluation_results/evaluation_metrics.json` with aggregated metrics. Current values (if available) are:

- sample_size: 5
- avg_latency_seconds: 0.6525375366210937
- avg_top_score: 1.0000000476837159
- avg_retrieved_count: 2.2
- avg_simple_bleu_score: 0.5120737425404944
- avg_rouge_l_score: 0.8210526315789475
- retrieval_success_rate: 1.0

If you do not have these files locally, generate them with:

```bash
python evaluate_rag.py --sample-size 20 --top-k 5
```

## Limitations
- The current corpus mainly uses historical support conversations and may not cover all product details.
- FAQ/manual integration is not implemented by default.
- Evaluation is simplified for academic/demo purposes and uses approximate BLEU/ROUGE metrics.
- Conversation memory is not included yet.
- Production-grade monitoring, authentication, and logging are future work.

## Future Work
See `docs/future_work.md` for planned enhancements such as adding knowledge-base documents, conversation memory, Azure deployment, and MLflow tracking.
