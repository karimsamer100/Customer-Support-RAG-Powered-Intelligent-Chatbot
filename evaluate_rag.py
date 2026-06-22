import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rag_pipeline import RAGPipeline


OUTPUT_DIR = Path("evaluation_results")
OUTPUT_CSV = OUTPUT_DIR / "evaluation_details.csv"
OUTPUT_JSON = OUTPUT_DIR / "evaluation_metrics.json"
OUTPUT_MD = OUTPUT_DIR / "evaluation_report.md"


def normalize_text(text):
    if text is None:
        return ""
    s = str(text).lower().strip()
    # collapse repeated spaces
    s = " ".join(s.split())
    return s


def simple_bleu_like(reference, candidate):
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()

    if len(cand_tokens) == 0:
        return 0.0

    ref_set = set(ref_tokens)
    overlap = sum(1 for w in cand_tokens if w in ref_set)
    return overlap / len(cand_tokens)


def rouge_l_score(reference, candidate):
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()

    if len(ref_tokens) == 0:
        return 0.0

    # DP for LCS
    m = len(ref_tokens)
    n = len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    return lcs / m


def truncate_text(text, max_chars=500):
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def evaluate(sample_size=20, top_k=5, seed=42):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load pipeline
    try:
        rag = RAGPipeline()
    except Exception as e:
        print("Failed to initialize RAG pipeline:", e)
        return

    # Load metadata safely
    try:
        metadata = pd.read_csv("vector_store/metadata.csv")
    except Exception as e:
        print("Failed to read metadata.csv:", e)
        return

    sample_n = min(sample_size, len(metadata))
    sample_df = metadata.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    records = []
    successful = 0

    for i, row in sample_df.iterrows():
        query = row.get("clean_question", "")
        reference_answer = row.get("clean_answer", "")

        print("=" * 80)
        print(f"Evaluating {i + 1}/{len(sample_df)}")

        start_time = time.time()
        try:
            generated_answer, retrieved = rag.ask(query, top_k=top_k)
        except Exception as e:
            print("Error during RAG ask():", e)
            generated_answer = ""
            retrieved = []

        latency = time.time() - start_time

        retrieved_count = len(retrieved) if retrieved is not None else 0
        top_score = float(retrieved[0]["score"]) if retrieved_count > 0 else 0.0
        top_retrieved_question = retrieved[0].get("question", "") if retrieved_count > 0 else ""
        top_retrieved_answer = retrieved[0].get("answer", "") if retrieved_count > 0 else ""

        bleu = simple_bleu_like(reference_answer, generated_answer)
        rouge = rouge_l_score(reference_answer, generated_answer)

        records.append({
            "query": query,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "retrieved_count": retrieved_count,
            "top_score": top_score,
            "top_retrieved_question": top_retrieved_question,
            "top_retrieved_answer": top_retrieved_answer,
            "latency_seconds": latency,
            "simple_bleu_score": bleu,
            "rouge_l_score": rouge,
        })

        # count as successful even if generation used fallback
        successful += 1

        print(f"retrieved_count={retrieved_count} top_score={top_score:.4f} latency={latency:.2f}s")

    # Write details CSV
    details_df = pd.DataFrame(records)
    details_df.to_csv(OUTPUT_CSV, index=False)

    # Metrics
    successful_evaluations = successful
    avg_latency_seconds = float(details_df["latency_seconds"].mean()) if not details_df.empty else 0.0
    avg_top_score = float(details_df["top_score"].mean()) if not details_df.empty else 0.0
    avg_retrieved_count = float(details_df["retrieved_count"].mean()) if not details_df.empty else 0.0
    avg_simple_bleu_score = float(details_df["simple_bleu_score"].mean()) if not details_df.empty else 0.0
    avg_rouge_l_score = float(details_df["rouge_l_score"].mean()) if not details_df.empty else 0.0

    retrieval_success = int((details_df["retrieved_count"] > 0).sum()) if not details_df.empty else 0
    retrieval_success_rate = retrieval_success / successful_evaluations if successful_evaluations > 0 else 0.0

    metrics = {
        "sample_size": sample_n,
        "successful_evaluations": successful_evaluations,
        "avg_latency_seconds": avg_latency_seconds,
        "avg_top_score": avg_top_score,
        "avg_retrieved_count": avg_retrieved_count,
        "avg_simple_bleu_score": avg_simple_bleu_score,
        "avg_rouge_l_score": avg_rouge_l_score,
        "retrieval_success_rate": retrieval_success_rate,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Markdown report
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write("This evaluation measures retrieval and generation quality on a sampled set of support questions.\n\n")

        f.write("## Evaluation Setup\n")
        f.write(f"- sample size: {sample_n}\n")
        f.write(f"- top_k: {top_k}\n")
        f.write("- vector store: FAISS\n")
        emb_model = rag.get_status().get("embedding_model") if hasattr(rag, "get_status") else "sentence-transformers/all-MiniLM-L6-v2"
        f.write(f"- embedding model: {emb_model}\n")
        f.write(f"- generation model (LLM_MODEL env): {rag.llm.model if hasattr(rag, 'llm') and getattr(rag.llm, 'model', None) else 'unknown'}\n\n")

        f.write("## Metrics\n")
        f.write("- Retrieval success rate\n")
        f.write("- Average top similarity score\n")
        f.write("- Average latency\n")
        f.write("- BLEU-like unigram overlap\n")
        f.write("- ROUGE-L (LCS / reference length)\n\n")

        f.write("## Results Summary\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| sample_size | {metrics['sample_size']} |\n")
        f.write(f"| successful_evaluations | {metrics['successful_evaluations']} |\n")
        f.write(f"| avg_latency_seconds | {metrics['avg_latency_seconds']:.4f} |\n")
        f.write(f"| avg_top_score | {metrics['avg_top_score']:.4f} |\n")
        f.write(f"| avg_retrieved_count | {metrics['avg_retrieved_count']:.2f} |\n")
        f.write(f"| avg_simple_bleu_score | {metrics['avg_simple_bleu_score']:.4f} |\n")
        f.write(f"| avg_rouge_l_score | {metrics['avg_rouge_l_score']:.4f} |\n")
        f.write(f"| retrieval_success_rate | {metrics['retrieval_success_rate']:.4f} |\n\n")

        f.write("## Sample Results\n")
        f.write("(Up to 5 samples)\n\n")
        samples = details_df.head(5) if not details_df.empty else []
        for idx, r in samples.iterrows():
            f.write(f"### Query {idx + 1}\n")
            f.write(f"- Query: {truncate_text(r['query'], 200)}\n")
            f.write(f"- Generated answer: {truncate_text(r['generated_answer'], 400)}\n")
            f.write(f"- Top score: {r['top_score']:.4f}\n")
            f.write(f"- Latency: {r['latency_seconds']:.2f}s\n\n")

        f.write("## Notes and Limitations\n")
        f.write("- This evaluation is simplified for academic/demo purposes.\n")
        f.write("- BLEU/ROUGE are approximate and do not fully measure helpfulness.\n")
        f.write("- Human review is recommended for production-quality answers.\n")
        f.write("- Evaluation uses historical support questions from the vector store; treat this as a functionality and quality sanity check.\n")

    # Optionally also save a short text summary
    print(f"Saved details: {OUTPUT_CSV}")
    print(f"Saved metrics: {OUTPUT_JSON}")
    print(f"Saved report: {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(sample_size=args.sample_size, top_k=args.top_k, seed=args.seed)


if __name__ == "__main__":
    main()