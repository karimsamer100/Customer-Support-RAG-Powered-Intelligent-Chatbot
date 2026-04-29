import pandas as pd
import faiss
from pathlib import Path

from embedder import TextEmbedder
from llm_generator import LLMGenerator



INDEX_PATH = Path("vector_store/faiss_index.bin")
METADATA_PATH = Path("vector_store/metadata.csv")


class RAGPipeline:
    def __init__(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.llm = LLMGenerator()
        print("Loading metadata...")
        self.metadata = pd.read_csv(METADATA_PATH)

        print("Loading embedder...")
        self.embedder = TextEmbedder()

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.embed_query(query).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        seen_questions = set()

        for score, idx in zip(scores[0], indices[0]):
            if float(score) < 0.60:
                continue

            row = self.metadata.iloc[idx]

            question = str(row["clean_question"])
            answer = str(row["clean_answer"])

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
        answer = self.generate_answer(query, retrieved_results)

        return answer, retrieved_results


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