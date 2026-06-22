# Demo Checklist

## Before Demo

- Ensure `.env` exists locally (do NOT commit it).
- Verify `APP_API_KEY` is set and known for demo requests.
- Verify `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are set in `.env`.
- Confirm `vector_store/faiss_index.bin` exists.
- Confirm `vector_store/metadata.csv` exists and contains `clean_question` and `clean_answer`.
- Run quick syntax checks:

```bash
python -m py_compile api.py rag_pipeline.py llm_generator.py embedder.py evaluate_rag.py
```

- Run a short evaluation to sanity-check responses:

```bash
python evaluate_rag.py --sample-size 5 --top-k 5
```

- Start API locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- Test `/health` and `/ask` endpoints.

## During Demo

- Explain problem and dataset briefly.
- Show preprocessing notebooks or sample cleaned rows.
- Explain RAG architecture and how retrieval + generation works.
- Demonstrate `/ask` and show generated answer and retrieved cases.
- Open `evaluation_results/evaluation_report.md` to show metrics.
- Show `Dockerfile`, `render.yaml`, and `docs/deployment_guide.md`.

## Emergency Fallback

- If the LLM is unavailable, present the fallback guidance returned from retrieved cases.
- If model loading is slow, explain that embedding models may take time to download on first run.
- If deployment is slow, demo locally using the deployment guide notes.
