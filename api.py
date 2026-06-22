import os
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline

# optional monitoring integration
try:
    import monitoring
except Exception:
    monitoring = None


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

APP_API_KEY = os.getenv("APP_API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if not APP_API_KEY:
    raise RuntimeError("APP_API_KEY is missing in .env")


app = FastAPI(
    title="Customer Support RAG Powered Chatbot API",
    description="Production-ready REST API for a RAG-based customer support chatbot.",
    version="1.0.0"
)

allowed_origins = ["*"] if ENVIRONMENT == "development" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

rag_pipeline = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedCase(BaseModel):
    score: float
    question: str
    answer: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_cases: List[RetrievedCase]
    latency_seconds: float
    top_score: float = 0.0
    retrieved_count: int = 0
    status: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global rag_pipeline
    logging.info("Starting Customer Support RAG API...")
    rag_pipeline = RAGPipeline()
    logging.info("RAG pipeline loaded successfully.")


def verify_api_key(x_api_key: Optional[str]):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logging.info(
        "%s %s | status=%s | latency=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response


@app.get("/")
def root():
    return {
        "service": "Customer Support RAG Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    pipeline_status = None
    rag_loaded = False

    if rag_pipeline is None:
        rag_loaded = False
    else:
        rag_loaded = True
        if hasattr(rag_pipeline, "get_status"):
            try:
                pipeline_status = rag_pipeline.get_status()
            except Exception:
                pipeline_status = None

    return {
        "status": "healthy",
        "rag_pipeline_loaded": rag_loaded,
        "environment": ENVIRONMENT,
        "pipeline_status": pipeline_status,
    }


@app.post("/ask", response_model=ChatResponse)
def ask_chatbot(
    request: ChatRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    verify_api_key(x_api_key)

    start_time = time.time()

    # Input cleanup
    question = request.question.strip() if request.question else ""
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not loaded yet")

    try:
        answer, retrieved_cases = rag_pipeline.ask(
            question,
            top_k=request.top_k
        )
    except Exception:
        latency_err = time.time() - start_time
        logging.exception("Internal error while processing the request")
        # log monitoring error event if available
        try:
            if monitoring:
                monitoring.log_request_event(question if question else "", latency_err, 0, 0.0, status="error", error="internal_server_error")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Internal server error while processing the request")

    latency = time.time() - start_time

    top_score = float(retrieved_cases[0]["score"]) if retrieved_cases else 0.0
    retrieved_count = len(retrieved_cases)

    # Log request event to monitoring
    try:
        if monitoring:
            monitoring.log_request_event(question, latency, retrieved_count, top_score, status="ok")
    except Exception:
        pass

    return {
        "question": question,
        "answer": answer,
        "retrieved_cases": retrieved_cases,
        "latency_seconds": latency,
        "top_score": top_score,
        "retrieved_count": retrieved_count,
        "status": "ok"
    }



@app.post("/feedback")
def post_feedback(
        feedback: FeedbackRequest,
        x_api_key: Optional[str] = Header(default=None)
):
        verify_api_key(x_api_key)

        # Validate rating already enforced by Pydantic
        try:
                if monitoring:
                        monitoring.log_feedback(feedback.question, feedback.answer, feedback.rating, feedback.comment)
        except Exception:
                pass

        return {"status": "received", "message": "Feedback saved successfully"}


@app.get("/metrics")
def get_metrics(x_api_key: Optional[str] = Header(default=None)):
        verify_api_key(x_api_key)
        try:
                if monitoring:
                        return monitoring.get_monitoring_summary()
        except Exception:
                pass
        # default empty metrics
        return {"requests": {}, "feedback": {}, "evaluation": {}}


from fastapi.responses import HTMLResponse


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
        # Demo-only dashboard (public) showing basic monitoring summary
        summary = {"requests": {}, "feedback": {}, "evaluation": {}}
        try:
                if monitoring:
                        summary = monitoring.get_monitoring_summary()
        except Exception:
                summary = {"requests": {}, "feedback": {}, "evaluation": {}}

        req = summary.get("requests", {})
        fb = summary.get("feedback", {})
        ev = summary.get("evaluation", {})

        html = f"""
        <html>
        <head>
            <title>RAG Demo Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .card {{ border: 1px solid #ddd; padding: 12px; margin-bottom: 12px; border-radius: 6px; }}
                h2 {{ margin-top: 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td, th {{ border: 1px solid #eee; padding: 8px; }}
            </style>
        </head>
        <body>
            <h1>RAG Demo Monitoring Dashboard</h1>
            <div class="card">
                <h2>Requests</h2>
                <table>
                    <tr><th>Total Requests</th><td>{req.get('total_requests', 0)}</td></tr>
                    <tr><th>Successful Requests</th><td>{req.get('successful_requests', 0)}</td></tr>
                    <tr><th>Failed Requests</th><td>{req.get('failed_requests', 0)}</td></tr>
                    <tr><th>Average Latency (s)</th><td>{req.get('avg_latency_seconds', 0.0):.3f}</td></tr>
                    <tr><th>Average Retrieved Count</th><td>{req.get('avg_retrieved_count', 0.0):.2f}</td></tr>
                    <tr><th>Average Top Score</th><td>{req.get('avg_top_score', 0.0):.3f}</td></tr>
                    <tr><th>Retrieval Success Rate</th><td>{req.get('retrieval_success_rate', 0.0):.3f}</td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Feedback</h2>
                <table>
                    <tr><th>Total Feedback</th><td>{fb.get('total_feedback', 0)}</td></tr>
                    <tr><th>Average Rating</th><td>{fb.get('avg_rating', 0.0):.2f}</td></tr>
                    <tr><th>Positive</th><td>{fb.get('positive_feedback_count', 0)}</td></tr>
                    <tr><th>Negative</th><td>{fb.get('negative_feedback_count', 0)}</td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Evaluation</h2>
                <table>
                    <tr><th>Sample Size</th><td>{ev.get('sample_size', 'N/A')}</td></tr>
                    <tr><th>Avg Latency (s)</th><td>{ev.get('avg_latency_seconds', 'N/A')}</td></tr>
                    <tr><th>Avg Top Score</th><td>{ev.get('avg_top_score', 'N/A')}</td></tr>
                    <tr><th>Retrieval Success Rate</th><td>{ev.get('retrieval_success_rate', 'N/A')}</td></tr>
                </table>
            </div>

            <p><em>This dashboard is for demo purposes only.</em></p>
        </body>
        </html>
        """

        return HTMLResponse(content=html, status_code=200)
