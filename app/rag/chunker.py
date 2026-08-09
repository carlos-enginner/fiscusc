"""Chunking semântico de texto com preservação de metadados."""
import re
from dataclasses import dataclass, field
from typing import Any

from app.extraction.pdf import PageContent


@dataclass
class TextChunk:
    """Chunk de texto com metadados de origem."""

    content: str
    page: int
    chunk_index: int
    section: str | None = None
    chapter: str | None = None
    article: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_length: int = field(init=False)

    def __post_init__(self):
        self.content_length = len(self.content)


# Padrões de seções em documentos de condomínio
_SECTION_PATTERNS = [
    # "CAPÍTULO I", "CAPÍTULO II", etc.
    re.compile(r"(?:CAP[ÍI]TULO\s+[IVXLCDM\d]+[^\n]*)", re.IGNORECASE),
    # "TÍTULO I", etc.
    re.compile(r"(?:T[ÍI]TULO\s+[IVXLCDM\d]+[^\n]*)", re.IGNORECASE),
    # "SEÇÃO I", "SEÇÃO 1", etc.
    re.compile(r"(?:SE[ÇC][ÃA]O\s+[IVXLCDM\d]+[^\n]*)", re.IGNORECASE),
]

_ARTICLE_PATTERNS = [
    # "Art. 1.", "Artigo 1 -", "Art 15 –"
    re.compile(r"(?:Art(?:igo)?\.?\s*\d+[\.\-–]?\s*)", re.IGNORECASE),
    # "§ 1°", "§1°", "§ 1º"
    re.compile(r"(?:§\s*\d+[°º]?\s*)"),
]


def chunk_text(
    text: str,
    max_size: int = 1000,
    overlap: int = 200,
    page: int = 1,
) -> list[TextChunk]:
    """
    Divide texto em chunks com overlap.

    Args:
        text: Texto a dividir.
        max_size: Tamanho máximo de cada chunk em caracteres.
        overlap: Sobreposição entre chunks consecutivos.
        page: Número da página de origem.

    Returns:
        Lista de TextChunk.
    """
    if not text.strip():
        return []

    # Dividir em sentenças/parágrafos para não cortar no meio de frases
    splits = _split_into_segments(text)

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_size = 0
    chunk_index = 0

    for segment in splits:
        seg_len = len(segment)

        # Se o segmento sozinho excede max_size, forçar divisão por tamanho
        if seg_len > max_size:
            if current:
                _flush_chunk(chunks, current, page, chunk_index)
                chunk_index += 1
                current = []
                current_size = 0
            # Dividir o segmento longo
            for sub in _force_split(segment, max_size, overlap):
                chunks.append(TextChunk(content=sub, page=page, chunk_index=chunk_index))
                chunk_index += 1
            continue

        if current_size + seg_len > max_size and current:
            _flush_chunk(chunks, current, page, chunk_index)
            chunk_index += 1
            # Manter overlap: pegar últimas palavras do chunk atual
            overlap_text = _get_overlap_text(current, overlap)
            current = [overlap_text] if overlap_text else []
            current_size = len(overlap_text) if overlap_text else 0

        current.append(segment)
        current_size += seg_len

    if current:
        _flush_chunk(chunks, current, page, chunk_index)

    return chunks


def chunk_pages(
    pages: list[PageContent],
    max_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """
    Divide uma lista de páginas em chunks semânticos.

    Detecta automaticamente seções, capítulos e artigos.

    Args:
        pages: Lista de PageContent extraídas de um PDF.
        max_size: Tamanho máximo de cada chunk.
        overlap: Sobreposição entre chunks.

    Returns:
        Lista de TextChunk com metadados preenchidos.
    """
    all_chunks: list[TextChunk] = []
    current_section: str | None = None
    current_chapter: str | None = None
    current_article: str | None = None
    chunk_index = 0

    for page in pages:
        # Detectar seção/capítulo/artigo no texto da página
        current_chapter = _detect_chapter(page.content) or current_chapter
        current_section = _detect_section(page.content) or current_section

        page_chunks = chunk_text(page.content, max_size=max_size, overlap=overlap, page=page.page_number)

        for chunk in page_chunks:
            # Atualizar metadados com contexto atual
            article = _detect_article(chunk.content)
            chunk.chunk_index = chunk_index
            chunk.section = current_section
            chunk.chapter = current_chapter
            chunk.article = article or current_article
            if article:
                current_article = article
            all_chunks.append(chunk)
            chunk_index += 1

    return all_chunks


# --- Helpers ---

def _split_into_segments(text: str) -> list[str]:
    """Divide texto em parágrafos/segmentos naturais."""
    # Dividir por parágrafos (dupla quebra de linha)
    paragraphs = re.split(r"\n\n+", text)
    segments: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if para:
            segments.append(para + "\n\n")
    return segments


def _force_split(text: str, max_size: int, overlap: int) -> list[str]:
    """Força divisão de texto longo por tamanho."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _flush_chunk(chunks: list, current: list[str], page: int, index: int):
    content = "".join(current).strip()
    if content:
        chunks.append(TextChunk(content=content, page=page, chunk_index=index))


def _get_overlap_text(segments: list[str], overlap: int) -> str:
    """Pega os últimos `overlap` caracteres como texto de sobreposição."""
    combined = "".join(segments)
    return combined[-overlap:] if len(combined) > overlap else combined


def _detect_chapter(text: str) -> str | None:
    """Detecta capítulo no texto."""
    for pattern in _SECTION_PATTERNS[:2]:  # CAPÍTULO e TÍTULO
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _detect_section(text: str) -> str | None:
    """Detecta seção no texto."""
    match = _SECTION_PATTERNS[2].search(text)  # SEÇÃO
    return match.group(0).strip() if match else None


def _detect_article(text: str) -> str | None:
    """Detecta artigo no início do chunk."""
    # Buscar artigo nos primeiros 100 caracteres
    snippet = text[:100]
    for pattern in _ARTICLE_PATTERNS:
        match = pattern.search(snippet)
        if match:
            return match.group(0).strip()
    return None
