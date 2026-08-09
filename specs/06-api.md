# API REST (FastAPI)

## Visão Geral

API REST para interação com o Fiscus-C.

**Base URL**: `http://localhost:8000/api/v1`

## Endpoints

### Health Check

```
GET /health
```

Verifica status da aplicação e dependências.

**Response 200:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "ollama": "healthy",
    "embeddings_model": "loaded",
    "llm_model": "loaded"
  },
  "timestamp": "2026-08-09T15:00:00Z"
}
```

**Response 503:**
```json
{
  "status": "unhealthy",
  "dependencies": {
    "database": "healthy",
    "ollama": "unhealthy",
    "error": "Connection refused"
  }
}
```

---

### Query (Principal)

```
POST /query
```

Processa uma pergunta usando o orquestrador.

**Request:**
```json
{
  "question": "Qual o horário permitido para obras?",
  "filters": {
    "document_types": ["regimento"],
    "agents": ["docs"]
  }
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| question | string | Sim | Pergunta do usuário |
| filters.document_types | string[] | Não | Filtrar por tipo de documento |
| filters.agents | string[] | Não | Forçar uso de agentes específicos |

**Response 200:**
```json
{
  "answer": "Segundo o Regimento Interno, obras são permitidas de segunda a sábado, das 8h às 18h (Art. 15, página 8). Aos domingos e feriados não são permitidas obras que gerem ruído.",
  "agents_used": ["docs"],
  "sources": [
    {
      "type": "document",
      "document": "regimento_interno.pdf",
      "document_type": "regimento",
      "page": 8,
      "section": "Art. 15 - Obras e Reformas",
      "score": 0.92,
      "snippet": "Obras e reformas são permitidas de segunda a sábado..."
    }
  ],
  "metadata": {
    "query_id": "550e8400-e29b-41d4-a716-446655440000",
    "latency_ms": 1250,
    "tokens_used": 450
  }
}
```

**Response 400:**
```json
{
  "error": "validation_error",
  "message": "Question is required",
  "details": [
    {"field": "question", "error": "Field required"}
  ]
}
```

---

### Ingestão de Documentos

```
POST /documents/ingest
```

Faz upload e processa um documento PDF.

**Request (multipart/form-data):**
```
file: <arquivo.pdf>
document_type: regimento
version: 2024.1
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| file | file | Sim | Arquivo PDF |
| document_type | string | Sim | Tipo: regimento, convencao, manual, fatura |
| version | string | Não | Versão do documento |

**Response 201:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "regimento_interno.pdf",
  "document_type": "regimento",
  "status": "processed",
  "stats": {
    "pages": 25,
    "chunks": 87,
    "processing_time_ms": 5200
  }
}
```

**Response 409 (Documento já existe):**
```json
{
  "error": "document_exists",
  "message": "Document with same SHA256 already exists",
  "existing_document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Listar Documentos

```
GET /documents
```

Lista documentos ingeridos.

**Query Parameters:**
| Param | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| type | string | - | Filtrar por tipo |
| limit | int | 20 | Máximo de resultados |
| offset | int | 0 | Paginação |

**Response 200:**
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "regimento_interno.pdf",
      "document_type": "regimento",
      "version": "2024.1",
      "page_count": 25,
      "chunk_count": 87,
      "created_at": "2026-08-09T10:00:00Z"
    }
  ],
  "total": 3,
  "limit": 20,
  "offset": 0
}
```

---

### Detalhes do Documento

```
GET /documents/{document_id}
```

Retorna detalhes de um documento específico.

**Response 200:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "regimento_interno.pdf",
  "document_type": "regimento",
  "version": "2024.1",
  "sha256": "a1b2c3d4...",
  "page_count": 25,
  "file_size_bytes": 524288,
  "chunks": [
    {
      "id": "chunk-uuid",
      "page": 1,
      "section": "Introdução",
      "content_preview": "O presente regimento..."
    }
  ],
  "created_at": "2026-08-09T10:00:00Z",
  "updated_at": "2026-08-09T10:00:00Z"
}
```

---

### Deletar Documento

```
DELETE /documents/{document_id}
```

Remove documento e seus chunks.

**Response 204:** No content

**Response 404:**
```json
{
  "error": "not_found",
  "message": "Document not found"
}
```

---

### Busca Semântica (Direta)

```
POST /search
```

Busca direta nos chunks sem passar pelo orquestrador.

**Request:**
```json
{
  "query": "horário obras",
  "document_type": "regimento",
  "top_k": 5,
  "min_score": 0.5
}
```

**Response 200:**
```json
{
  "results": [
    {
      "chunk_id": "chunk-uuid",
      "content": "Obras são permitidas de segunda a sábado...",
      "page": 8,
      "section": "Art. 15",
      "document": "regimento_interno.pdf",
      "score": 0.92
    }
  ],
  "query_embedding_time_ms": 50,
  "search_time_ms": 25
}
```

---

### Faturas

```
GET /faturas
```

Lista faturas processadas.

**Query Parameters:**
| Param | Tipo | Descrição |
|-------|------|-----------|
| unidade | string | Filtrar por unidade (ex: "A-2002") |
| mes | string | Filtrar por mês (ex: "julho/2026") |
| limit | int | Máximo de resultados |

**Response 200:**
```json
{
  "faturas": [
    {
      "id": "fatura-uuid",
      "unidade": "A-2002",
      "mes_referencia": "julho/2026",
      "data_vencimento": "2026-08-10",
      "total_cobranca": 795.96,
      "health_check_score": 95
    }
  ],
  "total": 12
}
```

---

```
GET /faturas/{fatura_id}
```

Detalhes de uma fatura.

**Response 200:**
```json
{
  "id": "fatura-uuid",
  "condominio": {
    "nome": "Residencial Exemplo",
    "cnpj": "12.345.678/0001-90"
  },
  "unidade": {
    "bloco": "A",
    "apartamento": "2002"
  },
  "mes_referencia": "julho/2026",
  "data_vencimento": "2026-08-10",
  "minha_cobranca": {
    "itens": [
      {"descricao": "TAXA DE CONDOMÍNIO", "valor": 679.76},
      {"descricao": "FUNDO DE RESERVA", "valor": 33.99}
    ],
    "total": 795.96
  },
  "despesas_condominio": {
    "itens": [...],
    "total": 89000.00
  }
}
```

## Schemas (Pydantic)

```python
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    filters: Optional[QueryFilters] = None


class QueryFilters(BaseModel):
    document_types: Optional[list[str]] = None
    agents: Optional[list[str]] = None


class QueryResponse(BaseModel):
    answer: str
    agents_used: list[str]
    sources: list[Source]
    metadata: QueryMetadata


class Source(BaseModel):
    type: str  # "document" ou "fatura"
    document: Optional[str] = None
    document_type: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None
    snippet: Optional[str] = None


class QueryMetadata(BaseModel):
    query_id: str
    latency_ms: int
    tokens_used: Optional[int] = None


class IngestRequest(BaseModel):
    document_type: str = Field(..., pattern="^(regimento|convencao|manual|fatura)$")
    version: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    document_type: str
    version: Optional[str]
    page_count: int
    chunk_count: Optional[int] = None
    created_at: datetime
```

## Implementação

```python
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Fiscus-C API",
    description="API para gestão inteligente de condomínios",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Processa pergunta usando o orquestrador."""
    from app.orchestrator import fiscus_workflow
    
    result = fiscus_workflow.invoke({
        "query": request.question
    })
    
    return QueryResponse(
        answer=result["final_answer"],
        agents_used=[r["source"] for r in result["results"]],
        sources=extract_sources(result),
        metadata=QueryMetadata(
            query_id=str(uuid.uuid4()),
            latency_ms=calculate_latency()
        )
    )


@app.post("/api/v1/documents/ingest", status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    version: str = Form(None)
):
    """Ingere documento PDF."""
    from app.ingestion import ingest_pdf
    
    result = await ingest_pdf(
        file=file,
        document_type=document_type,
        version=version
    )
    
    return result


@app.get("/api/v1/health")
async def health_check():
    """Verifica saúde da aplicação."""
    return await check_dependencies()
```

## Testes

```python
from fastapi.testclient import TestClient


def test_query_endpoint():
    """POST /query deve retornar resposta."""
    response = client.post("/api/v1/query", json={
        "question": "Qual horário para obras?"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_ingest_document():
    """POST /documents/ingest deve processar PDF."""
    with open("fixtures/reg_interno.pdf", "rb") as f:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("test.pdf", f, "application/pdf")},
            data={"document_type": "regimento"}
        )
    
    assert response.status_code == 201
    assert response.json()["status"] == "processed"


def test_query_requires_question():
    """POST /query sem question deve retornar 400."""
    response = client.post("/api/v1/query", json={})
    
    assert response.status_code == 422
```

## Rate Limiting (Futuro)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/query")
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    ...
```
