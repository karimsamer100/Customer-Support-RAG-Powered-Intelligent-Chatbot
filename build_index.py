import pandas as pd
import numpy as np
import faiss
from pathlib import Path

from embedder import TextEmbedder


INPUT_PATH = Path("processed_data/processed_amazon_support.csv")
OUTPUT_DIR = Path("vector_store")

INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"


def build_text_for_embedding(row):
    """
    We embed the customer question mainly.
    Adding the answer gives extra support context, but too much answer text can bias retrieval.
    So for now we use only clean_question.
    """
    return str(row["clean_question"])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Rows loaded: {len(df)}")

    # Optional: for first test, use subset to build faster.
    # Later we can use all data.
    df = df.head(20000).copy()

    df["embedding_text"] = df.apply(build_text_for_embedding, axis=1)

    texts = df["embedding_text"].tolist()

    embedder = TextEmbedder()

    print("Creating embeddings...")
    embeddings = embedder.embed_texts(texts, batch_size=64)

    embeddings = embeddings.astype("float32")

    print(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    # Since embeddings are normalized, inner product works like cosine similarity
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to FAISS index...")
    index.add(embeddings)

    print(f"Total vectors in index: {index.ntotal}")

    print("Saving FAISS index...")
    faiss.write_index(index, str(INDEX_PATH))

    metadata_cols = [
        "question",
        "answer",
        "clean_question",
        "clean_answer",
        "company",
        "question_word_count",
        "answer_word_count"
    ]

    df[metadata_cols].to_csv(METADATA_PATH, index=False)

    print("\nSaved files:")
    print(INDEX_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()