# Retraining and Re-indexing Pipeline

This project provides a simple manual re-indexing workflow to refresh embeddings and rebuild the FAISS index.

Steps (manual demo):

1. Collect new support data and add it to the processing pipeline.
2. Run preprocessing (notebooks or scripts) to clean the text.
3. Generate embeddings and build the FAISS index by running:

```bash
python build_index.py
```

4. Optionally run the evaluation script:

```bash
python evaluate_rag.py --sample-size 20 --top-k 5
```

5. Deploy the updated `vector_store/` to your environment (container image or mounted storage).

The repository includes `refresh_index.py` as a convenience wrapper to run `build_index.py`:

```bash
python refresh_index.py
```

For production, schedule re-indexing using cron, GitHub Actions, Azure Functions, or other orchestration tools.
