import os
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


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
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not loaded")

    return {
        "status": "healthy",
        "rag_pipeline_loaded": True
    }


@app.post("/ask", response_model=ChatResponse)
def ask_chatbot(
    request: ChatRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    verify_api_key(x_api_key)

    start_time = time.time()

    try:
        answer, retrieved_cases = rag_pipeline.ask(
            request.question,
            top_k=request.top_k
        )
    except Exception as e:
        logging.exception("Error while processing question")
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start_time

    return {
        "question": request.question,
        "answer": answer,
        "retrieved_cases": retrieved_cases,
        "latency_seconds": latency
    }
