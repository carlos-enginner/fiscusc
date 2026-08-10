"""Serviço de ingestão de documentos PDF."""

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.models import Document, DocumentChunk
from app.embeddings.cache import EmbeddingCache
from app.embeddings.service import EmbeddingsService
from app.extraction.pdf import calculate_sha256, extract_pdf, get_file_size
from app.rag.chunker import chunk_pages
from app.rag.metrics import IngestionMetrics


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
    metrics: IngestionMetrics | None = None


@dataclass
class ProgressCallbacks:
    """Callbacks para reportar progresso por fase da ingestão.
    
    Cada callback recebe (current, total) exceto on_phase_start que recebe (phase_name).
    """
    on_phase_start: Callable[[str], None] | None = None  # ("extraction", "chunking", "embedding", "saving")
    on_phase_end: Callable[[str], None] | None = None
    on_extraction_progress: Callable[[int, int], None] | None = None  # (current_page, total_pages)
    on_chunking_progress: Callable[[int, int], None] | None = None    # (current_chunk, total_chunks)
    on_embedding_progress: Callable[[int, int], None] | None = None   # (current, total)
    on_saving_progress: Callable[[int, int], None] | None = None      # (current, total)


class DocumentIngestionService:
    """Serviço para ingestão de documentos PDF."""

    def __init__(self, embeddings_service: EmbeddingsService, db_session: Session):
        self.embeddings = embeddings_service
        self.db = db_session
        settings = get_settings()
        self._cache = EmbeddingCache() if settings.enable_embedding_cache else None
        self._settings = settings

    def _compute_content_hash(self, content: str) -> str:
        """Calcula SHA256 hash do conteúdo do chunk."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_existing_chunks_by_hash(
        self, content_hashes: list[str]
    ) -> dict[str, list[float]]:
        """Busca chunks existentes por hash e retorna mapeamento hash -> embedding."""
        if not content_hashes:
            return {}

        existing = (
            self.db.query(DocumentChunk.content_hash, DocumentChunk.embedding)
            .filter(DocumentChunk.content_hash.in_(content_hashes))
            .all()
        )
        return {row.content_hash: row.embedding for row in existing if row.embedding}

    def ingest(
        self,
        path: str | Path,
        document_type: str,
        version: str | None = None,
        on_progress: Callable | None = None,
        progress_callbacks: ProgressCallbacks | None = None,
    ) -> IngestResult:
        """
        Ingere um documento PDF: extrai, chunkeia, gera embeddings e salva.

        Args:
            path: Caminho para o arquivo PDF.
            document_type: Tipo do documento (regimento, convencao, manual, fatura).
            version: Versão do documento (opcional).
            on_progress: Callback legado chamado durante salvamento.
                         Assinatura: on_progress(current: int, total: int, chunk_text: str)
            progress_callbacks: Callbacks por fase para progresso detalhado.

        Returns:
            IngestResult com status e estatísticas.
        """
        metrics = IngestionMetrics()
        path = Path(path)
        sha256 = calculate_sha256(path)
        cb = progress_callbacks or ProgressCallbacks()

        # Verificar se documento já existe
        existing = self.db.query(Document).filter(Document.sha256 == sha256).first()
        if existing:
            metrics.finalize()
            return IngestResult(
                document_id=str(existing.id),
                filename=existing.filename,
                document_type=existing.document_type,
                status="already_exists",
                pages=existing.page_count or 0,
                chunks_created=len(existing.chunks),
                sha256=sha256,
                already_existed=True,
                metrics=metrics,
            )

        # === FASE 1: Extrair texto do PDF ===
        if cb.on_phase_start:
            cb.on_phase_start("extraction")
        t0 = time.perf_counter()
        pages = extract_pdf(path)
        metrics.extraction_ms = (time.perf_counter() - t0) * 1000
        if cb.on_extraction_progress:
            cb.on_extraction_progress(len(pages), len(pages))
        if cb.on_phase_end:
            cb.on_phase_end("extraction")

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

        # === FASE 2: Chunkar o texto ===
        if cb.on_phase_start:
            cb.on_phase_start("chunking")
        t0 = time.perf_counter()
        chunks = chunk_pages(pages)
        metrics.chunking_ms = (time.perf_counter() - t0) * 1000
        metrics.chunks_count = len(chunks)
        if cb.on_chunking_progress:
            cb.on_chunking_progress(len(chunks), len(chunks))
        if cb.on_phase_end:
            cb.on_phase_end("chunking")

        # Calcular hashes de conteúdo para todos os chunks
        chunk_hashes = [self._compute_content_hash(c.content) for c in chunks]

        # Buscar embeddings existentes se incremental ingest está habilitado
        existing_embeddings: dict[str, list[float]] = {}
        if self._settings.enable_incremental_ingest:
            existing_embeddings = self._get_existing_chunks_by_hash(chunk_hashes)
            metrics.incremental_reused = len(existing_embeddings)

        # Preparar estrutura para embeddings
        embeddings: list[list[float] | None] = [None] * len(chunks)
        chunks_to_embed: list[tuple[int, str]] = []  # (índice original, content)

        # Verificar cache e incremental
        for i, chunk in enumerate(chunks):
            content_hash = chunk_hashes[i]

            # Tentar reusar embedding de chunk existente (incremental)
            if content_hash in existing_embeddings:
                embeddings[i] = existing_embeddings[content_hash]
                continue

            # Tentar cache
            if self._cache:
                cached = self._cache.get(chunk.content)
                if cached is not None:
                    embeddings[i] = cached
                    metrics.cache_hits += 1
                    continue
                metrics.cache_misses += 1

            # Precisa gerar embedding
            chunks_to_embed.append((i, chunk.content))

        # === FASE 3: Gerar embeddings para chunks novos em batch ===
        if cb.on_phase_start:
            cb.on_phase_start("embedding")
        t0 = time.perf_counter()
        total_to_embed = len(chunks_to_embed)
        embedded_count = 0
        
        if chunks_to_embed:
            batch_size = self._settings.embedding_batch_size
            for batch_start in range(0, len(chunks_to_embed), batch_size):
                batch = chunks_to_embed[batch_start : batch_start + batch_size]
                batch_texts = [text for _, text in batch]
                batch_embeddings = self.embeddings.embed_batch(batch_texts)

                for (orig_idx, content), emb in zip(batch, batch_embeddings):
                    embeddings[orig_idx] = emb
                    embedded_count += 1
                    # Salvar no cache
                    if self._cache:
                        self._cache.put(content, emb)
                
                # Reportar progresso de embeddings
                if cb.on_embedding_progress:
                    cb.on_embedding_progress(embedded_count, total_to_embed)

        metrics.embedding_ms = (time.perf_counter() - t0) * 1000
        if cb.on_phase_end:
            cb.on_phase_end("embedding")

        # === FASE 4: Salvar chunks no banco ===
        if cb.on_phase_start:
            cb.on_phase_start("saving")
        t0 = time.perf_counter()
        chunk_objects = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            # Callback legado
            if on_progress:
                on_progress(i + 1, total, chunk.content[:60])
            
            # Callback novo
            if cb.on_saving_progress:
                cb.on_saving_progress(i + 1, total)

            db_chunk = DocumentChunk(
                document_id=document.id,
                page=chunk.page,
                section=chunk.section,
                chapter=chunk.chapter,
                article=chunk.article,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_length=chunk.content_length,
                content_hash=chunk_hashes[i],
                embedding=embeddings[i],
            )
            chunk_objects.append(db_chunk)

        self.db.bulk_save_objects(chunk_objects)
        self.db.commit()
        metrics.db_ms = (time.perf_counter() - t0) * 1000
        if cb.on_phase_end:
            cb.on_phase_end("saving")

        # Finalizar métricas
        metrics.finalize()

        return IngestResult(
            document_id=str(document.id),
            filename=path.name,
            document_type=document_type,
            status="success",
            pages=len(pages),
            chunks_created=len(chunk_objects),
            sha256=sha256,
            metrics=metrics,
        )
