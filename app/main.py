from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, settings
from .logging_config import configure_logging, get_logger
from .models import ContextChunk, HealthStatus, QueryRequest, QueryResponse
from .rag_service import RAGService, RetrievalError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    This is where we configure logging and initialize long-lived
    resources such as the RAGService.
    """

    configure_logging(settings.log_level)
    app.state.settings = settings

    rag_service = RAGService(settings)
    app.state.rag_service = rag_service

    logger = get_logger(__name__)
    logger.info(
        "Application startup complete",
        extra={
            "chroma_host": settings.chroma_host,
            "chroma_port": settings.chroma_port,
            "collection": settings.chroma_collection,
        },
    )

    try:
        yield
    finally:
        logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "A simple Retrieval-Augmented Generation (RAG) API that answers "
        "questions about machine learning concepts using a ChromaDB "
        "vector database. Overlapping chunks are deduplicated before "
        "being used as context."
    ),
    lifespan=lifespan,
)

# Allow local development tools / browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
RAGDep = Annotated[RAGService, Depends(get_rag_service)]


@app.get("/health", response_model=HealthStatus, tags=["health"])
async def health(rag_service: RAGDep) -> HealthStatus:
    """Health endpoint used by Docker and observability tools.

    Returns 200 if the API itself is running, and indicates whether the
    ChromaDB backend is reachable.
    """

    chroma_ok = rag_service.healthy()
    status = "ok" if chroma_ok else "degraded"
    chroma_status = "ok" if chroma_ok else "unreachable"

    if not chroma_ok:
        # We do not raise an error here; HTTP 200 with degraded status
        # allows Kubernetes / Docker health checks to still reach this
        # endpoint while clearly signalling backend issues.
        logger.warning("Health check: ChromaDB is unreachable")

    return HealthStatus(status=status, chroma=chroma_status)


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(
    payload: QueryRequest,
    rag_service: RAGDep,
    settings: SettingsDep,
) -> QueryResponse:
    """Answer a machine learning question using RAG over ChromaDB.

    The RAG service retrieves relevant chunks, deduplicates overlapping
    context windows, and generates a concise answer.
    """

    try:
        answer, chunks = rag_service.answer_question(
            question=payload.question,
            top_k=payload.top_k or settings.max_context_chunks,
        )
    except RetrievalError as exc:
        logger.exception("Failed to answer question via RAG")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_chunks = [
        ContextChunk(
            id=chunk.id,
            text=chunk.text,
            score=chunk.distance,
            source=chunk.metadata.get("source"),
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]

    return QueryResponse(question=payload.question, answer=answer, chunks=response_chunks)


# If this file is executed directly, run a development server.
if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
