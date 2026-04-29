import time
import pandas as pd
from pathlib import Path

from rag_pipeline import RAGPipeline


OUTPUT_DIR = Path("evaluation_outputs")
OUTPUT_CSV = OUTPUT_DIR / "rag_evaluation_results.csv"
SUMMARY_TXT = OUTPUT_DIR / "rag_evaluation_summary.txt"


def simple_bleu(reference, generated):
    ref_words = set(str(reference).lower().split())
    gen_words = str(generated).lower().split()

    if len(gen_words) == 0:
        return 0.0

    overlap = sum(1 for word in gen_words if word in ref_words)
    return overlap / len(gen_words)


def rouge_l(reference, generated):
    ref = str(reference).lower().split()
    gen = str(generated).lower().split()

    if len(ref) == 0 or len(gen) == 0:
        return 0.0

    dp = [[0] * (len(gen) + 1) for _ in range(len(ref) + 1)]

    for i in range(1, len(ref) + 1):
        for j in range(1, len(gen) + 1):
            if ref[i - 1] == gen[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[len(ref)][len(gen)]
    return lcs / len(ref)


def evaluate(sample_size=30):
    OUTPUT_DIR.mkdir(exist_ok=True)

    rag = RAGPipeline()
    metadata = pd.read_csv("vector_store/metadata.csv")

    sample_df = metadata.sample(
        n=min(sample_size, len(metadata)),
        random_state=42
    ).reset_index(drop=True)

    results = []

    for i, row in sample_df.iterrows():
        query = row["clean_question"]
        reference_answer = row["clean_answer"]

        print("=" * 80)
        print(f"Evaluating {i + 1}/{len(sample_df)}")
        print(f"Query: {query}")

        start_time = time.time()
        generated_answer, retrieved_results = rag.ask(query, top_k=5)
        latency = time.time() - start_time

        top_score = retrieved_results[0]["score"] if retrieved_results else 0.0
        top_match_question = retrieved_results[0]["question"] if retrieved_results else ""

        bleu = simple_bleu(reference_answer, generated_answer)
        rouge = rouge_l(reference_answer, generated_answer)

        results.append({
            "query": query,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "top_retrieval_score": top_score,
            "top_matched_question": top_match_question,
            "simple_bleu": bleu,
            "rouge_l": rouge,
            "latency_seconds": latency
        })

        print(f"Top retrieval score: {top_score:.4f}")
        print(f"BLEU-like score: {bleu:.4f}")
        print(f"ROUGE-L score: {rouge:.4f}")
        print(f"Latency: {latency:.2f}s")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "sample_size": len(results_df),
        "avg_top_retrieval_score": results_df["top_retrieval_score"].mean(),
        "avg_simple_bleu": results_df["simple_bleu"].mean(),
        "avg_rouge_l": results_df["rouge_l"].mean(),
        "avg_latency_seconds": results_df["latency_seconds"].mean()
    }

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("RAG Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nSaved evaluation results:")
    print(OUTPUT_CSV)
    print(SUMMARY_TXT)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    evaluate(sample_size=30)