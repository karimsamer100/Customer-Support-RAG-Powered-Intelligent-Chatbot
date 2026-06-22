import json
from pathlib import Path
from datetime import datetime, timezone


LOGS_DIR = Path("logs")
REQUESTS_LOG = LOGS_DIR / "requests_log.jsonl"
FEEDBACK_LOG = LOGS_DIR / "feedback_log.jsonl"


def current_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _ensure_logs_dir():
    try:
        LOGS_DIR.mkdir(exist_ok=True)
    except Exception:
        pass


def log_request_event(question, latency_seconds, retrieved_count, top_score, status="ok", error=None):
    _ensure_logs_dir()
    entry = {
        "timestamp": current_timestamp(),
        "question": question if question is not None else "",
        "latency_seconds": float(latency_seconds) if latency_seconds is not None else 0.0,
        "retrieved_count": int(retrieved_count) if retrieved_count is not None else 0,
        "top_score": float(top_score) if top_score is not None else 0.0,
        "status": status,
        "error": error,
    }
    try:
        with open(REQUESTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Do not crash on logging failures
        return


def log_feedback(question, answer, rating, comment=None):
    _ensure_logs_dir()
    try:
        rating_int = int(rating)
    except Exception:
        try:
            rating_int = int(float(rating))
        except Exception:
            rating_int = None

    entry = {
        "timestamp": current_timestamp(),
        "question": question if question is not None else "",
        "answer": answer if answer is not None else "",
        "rating": rating_int,
        "comment": comment,
    }

    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return


def _read_jsonl(path):
    if not path.exists():
        return []
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return items


def summarize_request_logs():
    items = _read_jsonl(REQUESTS_LOG)
    total = len(items)
    if total == 0:
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency_seconds": 0.0,
            "avg_retrieved_count": 0.0,
            "avg_top_score": 0.0,
            "retrieval_success_rate": 0.0,
        }

    successful = sum(1 for it in items if it.get("status") == "ok")
    failed = total - successful
    latencies = [float(it.get("latency_seconds", 0.0)) for it in items if isinstance(it.get("latency_seconds", None), (int, float))]
    retrieved_counts = [int(it.get("retrieved_count", 0)) for it in items if isinstance(it.get("retrieved_count", None), (int, float))]
    top_scores = [float(it.get("top_score", 0.0)) for it in items if isinstance(it.get("top_score", None), (int, float))]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_retrieved = sum(retrieved_counts) / len(retrieved_counts) if retrieved_counts else 0.0
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    retrieval_success = sum(1 for it in items if int(it.get("retrieved_count", 0)) > 0)
    retrieval_success_rate = retrieval_success / total if total > 0 else 0.0

    return {
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "avg_latency_seconds": avg_latency,
        "avg_retrieved_count": avg_retrieved,
        "avg_top_score": avg_top,
        "retrieval_success_rate": retrieval_success_rate,
    }


def summarize_feedback_logs():
    items = _read_jsonl(FEEDBACK_LOG)
    total = len(items)
    if total == 0:
        return {
            "total_feedback": 0,
            "avg_rating": 0.0,
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
        }

    ratings = []
    for it in items:
        r = it.get("rating")
        try:
            ratings.append(int(r))
        except Exception:
            continue

    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
    positive = sum(1 for r in ratings if r >= 4)
    negative = sum(1 for r in ratings if r <= 2)

    return {
        "total_feedback": total,
        "avg_rating": avg_rating,
        "positive_feedback_count": positive,
        "negative_feedback_count": negative,
    }


def load_evaluation_metrics():
    path = Path("evaluation_results") / "evaluation_metrics.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_monitoring_summary():
    return {
        "requests": summarize_request_logs(),
        "feedback": summarize_feedback_logs(),
        "evaluation": load_evaluation_metrics(),
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(get_monitoring_summary(), indent=2))
