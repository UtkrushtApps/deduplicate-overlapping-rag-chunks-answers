from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for a RAG question."""

    question: str = Field(..., min_length=3, description="User question about machine learning concepts")
    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        le=20,
        description="Maximum number of unique context chunks to use",
    )


class ContextChunk(BaseModel):
    """Context snippet used to answer the question."""

    id: str
    text: str
    score: Optional[float] = Field(default=None, description="Similarity score or distance from ChromaDB")
    source: Optional[str] = Field(default=None, description="Logical source of the chunk (e.g., filename)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response schema containing the answer and supporting context."""

    question: str
    answer: str
    chunks: List[ContextChunk]


class HealthStatus(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Overall health status of the API")
    chroma: str = Field(..., description="Connectivity status for ChromaDB")
