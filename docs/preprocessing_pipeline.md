# Preprocessing Pipeline

- Raw support conversations are loaded from the original CSVs.
- AmazonHelp interactions are selected (filtered by source/category fields when present).
- Non-English samples are removed if language detection is applied.
- Text cleaning steps include removing URLs, @mentions, tracking/order noise, special tokens, repeated whitespace, and support agent signatures.
- The final output saved under `processed_data/` contains at minimum: `question`, `answer`, `cleaned_question`, `cleaned_answer`, `category`, plus other useful metadata fields.
- This processed corpus is used to compute embeddings and build the FAISS index for retrieval.

Keep notebooks such as `preprocessing.ipynb` and `eda.ipynb` as references for the cleaning decisions.