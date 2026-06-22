# Demo Script

## 30-Second Summary
This project is a RAG-powered customer support chatbot that uses historical Amazon support conversations. It retrieves similar past cases using FAISS and generates answers with an LLM. The service is exposed via a FastAPI endpoint and is prepared for Docker/Render deployment.

## 2-Minute Explanation
We built a lightweight Retrieval-Augmented Generation pipeline to accelerate customer support responses. The dataset consists of historical support conversations filtered for AmazonHelp. Text is preprocessed and cleaned, then encoded with a `sentence-transformers` embedding model. Embeddings are indexed with FAISS for fast nearest-neighbor search. When a user asks a question, the API embeds the query, retrieves top-k similar cases, and includes those cases in a prompt sent to an LLM which generates the final answer. The API is protected by a simple `x-api-key` header and returns the answer plus the retrieved sources for transparency. An evaluation script provides quick quantitative checks, and Docker/Render artifacts are included for demo deployment.

## Demo Flow
1. Open the project `README.md`.
2. Show project structure and highlight `processed_data/`, `vector_store/`, `api.py`, and `rag_pipeline.py`.
3. Open `vector_store/metadata.csv` to show sample entries.
4. Show `vector_store/faiss_index.bin` exists (binary index file).
5. Open `api.py` and explain the `/health` and `/ask` endpoints and `x-api-key` check.
6. Run the API locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

7. Open `/health` in the browser or via curl.
8. Send a POST to `/ask` with `x-api-key` header and a sample question.
9. Show the generated answer and the retrieved cases returned by the API.
10. Open `evaluation_results/evaluation_report.md` to show the evaluation summary.
11. Show `Dockerfile`, `render.yaml`, and `docs/deployment_guide.md` for deployment steps.
12. Show the monitoring dashboard at `/dashboard` and the metrics endpoint `/metrics`.
13. Demonstrate sending feedback via `/feedback` and check `logs/feedback_log.jsonl`.

## Questions I Might Be Asked
Q: Why RAG?
A: RAG lets the model ground responses in real historical support content instead of relying solely on model parameters, reducing hallucination and improving usefulness.

Q: Why FAISS?
A: FAISS provides efficient vector similarity search, suitable for fast retrieval in demo and production settings.

Q: Why sentence-transformers/all-MiniLM-L6-v2?
A: It's compact, fast, and delivers good semantic similarity for demo use without heavy infrastructure.

Q: Why no LangChain memory?
A: Conversation memory was not in the core scope. The architecture focuses on retrieval + generation; memory can be added later if needed.

Q: Is this production-ready?
A: It is demo-ready: the project includes an API, Dockerfile, and basic security. Full production requires more monitoring, stronger auth, and robust CI/CD.

Q: How is it secured?
A: The `/ask` endpoint requires `x-api-key` matching `APP_API_KEY` from environment variables. Secrets should be stored in platform secret stores and never committed.

Q: How to improve?
A: Add knowledge-base docs, conversation memory, Azure deployment with AD, MLflow tracking, human feedback loop, and production monitoring.
