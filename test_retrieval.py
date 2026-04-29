import pandas as pd
import faiss
from pathlib import Path

from embedder import TextEmbedder


INDEX_PATH = Path("vector_store/faiss_index.bin")
METADATA_PATH = Path("vector_store/metadata.csv")


def search(query, top_k=5):
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)

    embedder = TextEmbedder()
    query_embedding = embedder.embed_query(query).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        row = metadata.iloc[idx]

        results.append({
            "score": float(score),
            "question": row["clean_question"],
            "answer": row["clean_answer"]
        })

    return results


def main():
    test_queries = [
        "My package says delivered but I did not receive it",
        "I was charged for Prime but I want a refund",
        "My account is locked and I cannot log in",
        "My Echo device is not working",
        "I received a damaged item"
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        results = search(query, top_k=3)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Matched Question: {result['question']}")
            print(f"Support Answer: {result['answer']}")


if __name__ == "__main__":
    main()