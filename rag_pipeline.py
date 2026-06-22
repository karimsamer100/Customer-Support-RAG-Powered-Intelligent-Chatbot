import pandas as pd
import faiss
from pathlib import Path

from embedder import TextEmbedder
from llm_generator import LLMGenerator



INDEX_PATH = Path("vector_store/faiss_index.bin")
METADATA_PATH = Path("vector_store/metadata.csv")


class RAGPipeline:
    def __init__(self):
        # Validate vector store files before loading
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise RuntimeError("Vector store files are missing. Please run build_index.py first.")

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.llm = LLMGenerator()

        print("Loading metadata...")
        self.metadata = pd.read_csv(METADATA_PATH)

        # Validate required metadata columns
        required_cols = {"clean_question", "clean_answer"}
        missing = required_cols.difference(set(self.metadata.columns))
        if missing:
            raise RuntimeError(f"Metadata CSV is missing required columns: {', '.join(sorted(missing))}")

        print("Loading embedder...")
        self.embedder = TextEmbedder()

        # Settings
        self.min_similarity_score = 0.60

    def retrieve(self, query, top_k=5):
        # Validate query
        if not query or str(query).strip() == "":
            return []

        # Safe top_k handling
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 5

        top_k = max(1, min(10, top_k))

        query_embedding = self.embedder.embed_query(query).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        seen_questions = set()

        for score, idx in zip(scores[0], indices[0]):
            # FAISS may return -1 for empty results
            if int(idx) < 0:
                continue

            if float(score) < float(self.min_similarity_score):
                continue

            # index bounds check
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = self.metadata.iloc[idx]

            question = str(row.get("clean_question", ""))
            answer = str(row.get("clean_answer", ""))

            # remove duplicated retrieved questions
            normalized_question = question.lower().strip()

            if normalized_question in seen_questions:
                continue

            seen_questions.add(normalized_question)

            results.append({
                "score": float(score),
                "question": question,
                "answer": answer
            })

        return results

    def generate_answer(self, query, retrieved_results):
        return self.llm.generate(query, retrieved_results)

    def ask(self, query, top_k=5):
        retrieved_results = self.retrieve(query, top_k=top_k)

        if not retrieved_results:
            fallback = (
                "I'm sorry, I couldn't find enough relevant information in the support knowledge base for this question. "
                "Please contact official customer support for more accurate help."
            )
            return fallback, []

        try:
            answer = self.generate_answer(query, retrieved_results)
        except Exception:
            # Fallback based on top retrieved answer
            top_answer = retrieved_results[0].get("answer", "")
            fallback = (
                "Based on similar support cases, here is the most relevant available guidance:\n\n" + top_answer
            )
            return fallback, retrieved_results

        return answer, retrieved_results

    def get_status(self):
        try:
            embedding_model = getattr(self.embedder, "model_name", "sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        return {
            "index_loaded": True,
            "metadata_loaded": True,
            "metadata_rows": len(self.metadata) if hasattr(self, "metadata") else 0,
            "embedding_model": embedding_model,
            "min_similarity_score": float(self.min_similarity_score)
        }


def main():
    rag = RAGPipeline()

    while True:
        query = input("\nAsk a customer support question or type 'exit': ")

        if query.lower().strip() == "exit":
            break

        answer, results = rag.ask(query, top_k=5)

        print("\n" + "=" * 80)
        print("Generated Answer")
        print("=" * 80)
        print(answer)

        print("\nRetrieved Similar Cases:")
        for i, result in enumerate(results, start=1):
            print(f"\nCase {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Similar Question: {result['question']}")
            print(f"Support Answer: {result['answer']}")


if __name__ == "__main__":
    main()