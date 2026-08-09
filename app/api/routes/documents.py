"""Endpoints de gerenciamento de documentos."""
import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import DocumentListResponse, DocumentResponse, IngestResponse
from app.core.database import DbSession
from app.core.models import Document

router = APIRouter()


@router.post("/documents/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    db: DbSession,
    file: UploadFile = File(...),
    document_type: str = Form(...),
    version: Optional[str] = Form(None),
):
    """
    Ingere um documento PDF.

    - Extrai texto do PDF
    - Divide em chunks semânticos
    - Gera embeddings
    - Armazena no banco
    """
    # Validar tipo de documento
    valid_types = {"regimento", "convencao", "manual", "fatura"}
    if document_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"document_type deve ser um de: {valid_types}",
        )

    # Salvar arquivo temporariamente
    import tempfile
    from pathlib import Path

    content = await file.read()
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        start = time.time()

        from app.embeddings.service import EmbeddingsService
        from app.rag.ingestion import DocumentIngestionService

        embeddings = EmbeddingsService()
        svc = DocumentIngestionService(embeddings_service=embeddings, db_session=db)

        result = svc.ingest(
            path=tmp_path,
            document_type=document_type,
            version=version,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        if result.already_existed:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "document_exists",
                    "message": "Document with same SHA256 already exists",
                    "existing_document_id": result.document_id,
                },
            )

        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            document_type=result.document_type,
            status=result.status,
            stats={
                "pages": result.pages,
                "chunks": result.chunks_created,
                "processing_time_ms": elapsed_ms,
            },
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    db: DbSession,
    type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Lista documentos ingeridos com paginação."""
    query = db.query(Document)
    if type:
        query = query.filter(Document.document_type == type)

    total = query.count()
    docs = query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=str(d.id),
                filename=d.filename,
                document_type=d.document_type,
                version=d.version,
                page_count=d.page_count,
                chunk_count=len(d.chunks),
                created_at=d.created_at,
            )
            for d in docs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: DbSession):
    """Retorna detalhes de um documento específico."""
    try:
        uid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="document_id inválido")

    doc = db.query(Document).filter(Document.id == uid).first()
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Document not found"})

    return DocumentResponse(
        id=str(doc.id),
        filename=doc.filename,
        document_type=doc.document_type,
        version=doc.version,
        page_count=doc.page_count,
        chunk_count=len(doc.chunks),
        created_at=doc.created_at,
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, db: DbSession):
    """Remove documento e seus chunks."""
    try:
        uid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="document_id inválido")

    doc = db.query(Document).filter(Document.id == uid).first()
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Document not found"})

    db.delete(doc)
    db.commit()
