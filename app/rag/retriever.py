"""Retriever para busca semântica em documentos via pgvector."""
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.embeddings.service import EmbeddingsService


@dataclass
class SearchResult:
    """Resultado de uma busca semântica."""

    id: str
    content: str
    page: int
    section: str | None
    document_type: str
    filename: str
    score: float
    chapter: str | None = None
    article: str | None = None


class DocumentRetriever:
    """Retriever para busca semântica em documentos."""

    def __init__(self, embeddings_service: EmbeddingsService, db_session: Session):
        self.embeddings = embeddings_service
        self.db = db_session
        self._settings = get_settings()

    def search(
        self,
        query: str,
        document_type: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """
        Busca chunks similares à query.

        Args:
            query: Texto da busca.
            document_type: Filtro por tipo de documento (None = todos).
            top_k: Número máximo de resultados.
            min_score: Score mínimo de similaridade.

        Returns:
            Lista de resultados ordenados por similaridade decrescente.
        """
        settings = self._settings
        top_k = top_k or settings.top_k_results
        min_score = min_score if min_score is not None else settings.min_similarity_score

        # Gerar embedding da query
        query_embedding = self.embeddings.embed(query)

        # Formatar como string de vetor para o PostgreSQL
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Buscar usando a função search_similar_chunks
        sql = text("""
            SELECT
                c.id::text,
                c.content,
                c.page,
                c.section,
                c.chapter,
                c.article,
                d.filename,
                d.document_type,
                1 - (c.embedding <=> CAST(:embedding AS vector)) as score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE (:doc_type IS NULL OR d.document_type = :doc_type)
            AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)

        rows = self.db.execute(
            sql,
            {
                "embedding": embedding_str,
                "doc_type": document_type,
                "limit": top_k,
            },
        ).fetchall()

        results = []
        for row in rows:
            score = float(row[8])
            if score >= min_score:
                results.append(
                    SearchResult(
                        id=row[0],
                        content=row[1],
                        page=row[2],
                        section=row[3],
                        chapter=row[4],
                        article=row[5],
                        filename=row[6],
                        document_type=row[7],
                        score=score,
                    )
                )

        return results


def format_results(results: list[SearchResult]) -> str:
    """Formata resultados da busca para o agente."""
    if not results:
        return "Nenhum resultado encontrado."

    formatted = []
    for r in results:
        source = f"[{r.document_type.upper()} - página {r.page}"
        if r.section:
            source += f", seção {r.section}"
        if r.article:
            source += f", {r.article}"
        source += f"] (score: {r.score:.2f})"

        formatted.append(f"{source}\n{r.content}\n")

    return "\n---\n".join(formatted)
