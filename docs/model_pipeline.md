# Model Pipeline

- A user query is received by the FastAPI `/ask` endpoint.
- The query is embedded using a SentenceTransformer or compatible embedder (see `embedder.py`).
- FAISS retrieves the top-k most similar support cases from the vector index.
- Retrieved cases are assembled into a prompt template along with the user query.
- The chosen LLM (via `LLM_BASE_URL` / `LLM_API_KEY`) generates the final answer.
- The API returns the generated answer plus the retrieved source snippets and metadata for transparency.

See `rag_pipeline.py` and `llm_generator.py` for implementation details.