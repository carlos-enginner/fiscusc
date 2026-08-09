"""Serviço de ingestão de documentos PDF."""
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.embeddings.service import EmbeddingsService
from app.extraction.pdf import calculate_sha256, extract_pdf, get_file_size
from app.rag.chunker import chunk_pages


@dataclass
class IngestResult:
    """Resultado da ingestão de um documento."""

    document_id: str
    filename: str
    document_type: str
    status: str
    pages: int
    chunks_created: int
    sha256: str
    already_existed: bool = False


class DocumentIngestionService:
    """Serviço para ingestão de documentos PDF."""

    def __init__(self, embeddings_service: EmbeddingsService, db_session: Session):
        self.embeddings = embeddings_service
        self.db = db_session

    def ingest(
        self,
        path: str | Path,
        document_type: str,
        version: str | None = None,
        on_progress: Callable | None = None,
    ) -> IngestResult:
        """
        Ingere um documento PDF: extrai, chunkeia, gera embeddings e salva.

        Args:
            path: Caminho para o arquivo PDF.
            document_type: Tipo do documento (regimento, convencao, manual, fatura).
            version: Versão do documento (opcional).
            on_progress: Callback opcional chamado a cada chunk processado.
                         Assinatura: on_progress(current: int, total: int, chunk_text: str)

        Returns:
            IngestResult com status e estatísticas.
        """
        path = Path(path)
        sha256 = calculate_sha256(path)

        # Verificar se documento já existe
        existing = self.db.query(Document).filter(Document.sha256 == sha256).first()
        if existing:
            return IngestResult(
                document_id=str(existing.id),
                filename=existing.filename,
                document_type=existing.document_type,
                status="already_exists",
                pages=existing.page_count or 0,
                chunks_created=len(existing.chunks),
                sha256=sha256,
                already_existed=True,
            )

        # Extrair texto do PDF
        pages = extract_pdf(path)

        # Criar documento no banco
        document = Document(
            filename=path.name,
            document_type=document_type,
            version=version,
            sha256=sha256,
            page_count=len(pages),
            file_size_bytes=get_file_size(path),
        )
        self.db.add(document)
        self.db.flush()  # Gerar ID sem commit

        # Chunkar o texto
        chunks = chunk_pages(pages)

        # Gerar embeddings em batch e salvar
        chunk_objects = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed(chunk.content)
            if on_progress:
                on_progress(i + 1, total, chunk.content[:60])
            db_chunk = DocumentChunk(
                document_id=document.id,
                page=chunk.page,
                section=chunk.section,
                chapter=chunk.chapter,
                article=chunk.article,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_length=chunk.content_length,
                embedding=embedding,
            )
            chunk_objects.append(db_chunk)

        self.db.bulk_save_objects(chunk_objects)
        self.db.commit()

        return IngestResult(
            document_id=str(document.id),
            filename=path.name,
            document_type=document_type,
            status="success",
            pages=len(pages),
            chunks_created=len(chunk_objects),
            sha256=sha256,
        )
