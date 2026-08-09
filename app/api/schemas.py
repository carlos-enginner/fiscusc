"""Schemas Pydantic para a API REST do Fiscus-C."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Query ---

class QueryFilters(BaseModel):
    """Filtros opcionais para a query."""

    document_types: Optional[list[str]] = None
    agents: Optional[list[str]] = None


class QueryRequest(BaseModel):
    """Request para POST /query."""

    question: str = Field(..., min_length=3, max_length=1000)
    filters: Optional[QueryFilters] = None


class Source(BaseModel):
    """Fonte de uma resposta."""

    type: str  # "document" ou "fatura"
    document: Optional[str] = None
    document_type: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None
    snippet: Optional[str] = None


class QueryMetadata(BaseModel):
    """Metadados da query."""

    query_id: str
    latency_ms: int
    tokens_used: Optional[int] = None


class QueryResponse(BaseModel):
    """Response de POST /query."""

    answer: str
    agents_used: list[str]
    sources: list[Source]
    metadata: QueryMetadata


# --- Documents ---

class IngestResponse(BaseModel):
    """Response de POST /documents/ingest."""

    document_id: str
    filename: str
    document_type: str
    status: str
    stats: dict


class DocumentResponse(BaseModel):
    """Metadados de um documento."""

    id: str
    filename: str
    document_type: str
    version: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Response de GET /documents."""

    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


# --- Health ---

class DependencyStatus(BaseModel):
    """Status de uma dependência."""

    database: str
    ollama: str
    embeddings_model: Optional[str] = None
    llm_model: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response de GET /health."""

    status: str  # "healthy" ou "unhealthy"
    version: str
    dependencies: DependencyStatus
    timestamp: str


# --- Search ---

class SearchRequest(BaseModel):
    """Request para POST /search."""

    query: str = Field(..., min_length=2, max_length=500)
    document_type: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    """Item de resultado de busca."""

    chunk_id: str
    content: str
    page: int
    section: Optional[str] = None
    document: str
    document_type: str
    score: float


class SearchResponse(BaseModel):
    """Response de POST /search."""

    results: list[SearchResultItem]
    query_embedding_time_ms: Optional[int] = None
    search_time_ms: Optional[int] = None
