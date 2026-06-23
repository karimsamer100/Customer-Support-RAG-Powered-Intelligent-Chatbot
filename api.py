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

        # Safe formatting helpers
        def fmt_float(v, ndigits=3):
            try:
                return f"{float(v):.{ndigits}f}"
            except Exception:
                return "N/A"

        def fmt_int(v):
            try:
                return str(int(v))
            except Exception:
                return "N/A"

        def fmt_pct(v, ndigits=3):
            try:
                return f"{float(v) * 100:.{ndigits}f}%"
            except Exception:
                return "N/A"

        # Requests
        req_total = fmt_int(req.get("total_requests", 0))
        req_success = fmt_int(req.get("successful_requests", 0))
        req_failed = fmt_int(req.get("failed_requests", 0))
        req_avg_latency = fmt_float(req.get("avg_latency_seconds", 0.0), 3)
        req_avg_retrieved = fmt_float(req.get("avg_retrieved_count", 0.0), 3)
        req_avg_top_score = fmt_float(req.get("avg_top_score", 0.0), 3)
        req_retrieval_rate = fmt_pct(req.get("retrieval_success_rate", 0.0), 3)

        # Feedback
        fb_total = fmt_int(fb.get("total_feedback", 0))
        fb_avg_rating = fmt_float(fb.get("avg_rating", 0.0), 3)
        fb_positive = fmt_int(fb.get("positive_feedback_count", 0))
        fb_negative = fmt_int(fb.get("negative_feedback_count", 0))

        # Evaluation
        ev_sample = fmt_int(ev.get("sample_size", "N/A"))
        ev_avg_latency = fmt_float(ev.get("avg_latency_seconds", "N/A"), 3)
        ev_avg_top_score = fmt_float(ev.get("avg_top_score", "N/A"), 3)
        ev_retrieval_rate = fmt_pct(ev.get("retrieval_success_rate", "N/A"), 3)

        env_label = ENVIRONMENT or "unknown"

        html = f"""
        <html>
        <head>
            <meta http-equiv="refresh" content="30">
            <title>Customer Support RAG Monitoring Dashboard</title>
            <style>
                :root {{ --bg:#0b1220; --card:#0f1a2b; --muted:#9aa7bf; --accent:#1e90ff; --glass: rgba(255,255,255,0.04); }}
                html,body {{ height:100%; margin:0; font-family: Inter, Roboto, Arial, sans-serif; background: linear-gradient(180deg, #071024 0%, #081122 100%); color:#e6eef8; }}
                .container {{ max-width:1200px; margin:24px auto; padding:20px; }}
                header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }}
                .title {{ font-size:20px; font-weight:700; margin:0; color:#ffffff; }}
                .subtitle {{ margin:6px 0 0 0; color:var(--muted); font-size:13px; }}
                .badges {{ display:flex; gap:8px; align-items:center; }}
                .badge {{ background:var(--glass); border:1px solid rgba(255,255,255,0.06); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:12px; }}
                .badge.online {{ color:#bff0c6; border-color:rgba(30,200,80,0.15); background:linear-gradient(90deg, rgba(30,200,80,0.05), transparent); }}
                .grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:20px; }}
                .section {{ background:var(--card); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.03); }}
                .section h3 {{ margin:0 0 12px 0; font-size:16px; color:#dff0ff; }}
                .metrics {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; }}
                .metric {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:8px; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(255,255,255,0.02); }}
                .m-label {{ color:var(--muted); font-size:12px; }}
                .m-value {{ font-weight:700; font-size:16px; color:#fff; }}
                footer {{ margin-top:20px; color:var(--muted); font-size:13px; text-align:center; }}
                @media (max-width:900px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: 1fr 1fr; }} }}
                @media (max-width:600px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div>
                        <div class="title">Customer Support RAG Monitoring Dashboard</div>
                        <div class="subtitle">Live demo monitoring for RAG chatbot requests, feedback, and evaluation metrics</div>
                    </div>
                    <div class="badges">
                        <div class="badge online">Service: Online</div>
                        <div class="badge">Environment: {env_label}</div>
                        <div class="badge">Demo Dashboard</div>
                    </div>
                </header>

                <div class="grid">
                    <div class="section">
                        <h3>Requests</h3>
                        <div class="metrics">
                            <div class="metric"><div class="m-label">Total Requests</div><div class="m-value">{req_total}</div></div>
                            <div class="metric"><div class="m-label">Successful Requests</div><div class="m-value">{req_success}</div></div>
                            <div class="metric"><div class="m-label">Failed Requests</div><div class="m-value">{req_failed}</div></div>
                            <div class="metric"><div class="m-label">Average Latency (s)</div><div class="m-value">{req_avg_latency}</div></div>
                            <div class="metric"><div class="m-label">Average Retrieved Count</div><div class="m-value">{req_avg_retrieved}</div></div>
                            <div class="metric"><div class="m-label">Average Top Score</div><div class="m-value">{req_avg_top_score}</div></div>
                            <div class="metric" style="grid-column:span 2;"><div class="m-label">Retrieval Success Rate</div><div class="m-value">{req_retrieval_rate}</div></div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Feedback</h3>
                        <div class="metrics">
                            <div class="metric"><div class="m-label">Total Feedback</div><div class="m-value">{fb_total}</div></div>
                            <div class="metric"><div class="m-label">Average Rating</div><div class="m-value">{fb_avg_rating}</div></div>
                            <div class="metric"><div class="m-label">Positive Feedback</div><div class="m-value">{fb_positive}</div></div>
                            <div class="metric"><div class="m-label">Negative Feedback</div><div class="m-value">{fb_negative}</div></div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Evaluation</h3>
                        <div class="metrics">
                            <div class="metric"><div class="m-label">Sample Size</div><div class="m-value">{ev_sample}</div></div>
                            <div class="metric"><div class="m-label">Avg Evaluation Latency (s)</div><div class="m-value">{ev_avg_latency}</div></div>
                            <div class="metric"><div class="m-label">Avg Evaluation Top Score</div><div class="m-value">{ev_avg_top_score}</div></div>
                            <div class="metric" style="grid-column:span 2;"><div class="m-label">Evaluation Retrieval Success Rate</div><div class="m-value">{ev_retrieval_rate}</div></div>
                        </div>
                    </div>
                </div>

                <footer>
                    This lightweight dashboard is built for monitoring demo purposes. Production systems can integrate Azure Application Insights, Grafana, or Power BI.
                </footer>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html, status_code=200)
