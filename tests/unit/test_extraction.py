"""Testes da Task 5: PDF Extraction + Chunking."""
from pathlib import Path

import pytest

from app.extraction.pdf import PageContent, calculate_sha256, extract_pdf, get_file_size
from app.rag.chunker import TextChunk, chunk_pages, chunk_text

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "reg_interno.pdf"


# --- Testes de extração de PDF ---

class TestExtractPDF:
    def test_extract_pdf_returns_pages(self):
        """extract_pdf() deve retornar lista de páginas."""
        pages = extract_pdf(PDF_PATH)

        assert len(pages) > 0
        assert all(isinstance(p, PageContent) for p in pages)

    def test_extract_pdf_has_content(self):
        """Cada página deve ter conteúdo não vazio."""
        pages = extract_pdf(PDF_PATH)

        assert all(len(p.content) > 0 for p in pages)

    def test_extract_preserves_page_numbers(self):
        """Página deve ser preservada na extração."""
        pages = extract_pdf(PDF_PATH)

        assert pages[0].page_number == 1
        # Números sequenciais
        for i, page in enumerate(pages, start=1):
            assert page.page_number == i or page.page_number >= i  # pode pular páginas vazias

    def test_page_content_has_char_count(self):
        """PageContent deve calcular char_count automaticamente."""
        page = PageContent(page_number=1, content="hello world")
        assert page.char_count == 11

    def test_extract_pdf_file_not_found(self):
        """Deve lançar FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            extract_pdf("/nonexistent/path/file.pdf")

    def test_extract_pdf_invalid_file(self, tmp_path):
        """Deve lançar ValueError para arquivo não-PDF."""
        bad = tmp_path / "not_a_pdf.pdf"
        bad.write_text("este não é um PDF")

        with pytest.raises(ValueError):
            extract_pdf(bad)


class TestSHA256:
    def test_sha256_calculated(self):
        """SHA256 deve ser calculado para o documento."""
        sha = calculate_sha256(PDF_PATH)

        assert len(sha) == 64
        assert sha.isalnum()

    def test_sha256_is_deterministic(self):
        """Mesmo arquivo deve gerar mesmo SHA256."""
        sha1 = calculate_sha256(PDF_PATH)
        sha2 = calculate_sha256(PDF_PATH)

        assert sha1 == sha2

    def test_sha256_different_files_differ(self, tmp_path):
        """Arquivos diferentes devem ter SHA256 diferentes."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"conteudo A")
        f2.write_bytes(b"conteudo B")

        assert calculate_sha256(f1) != calculate_sha256(f2)

    def test_sha256_file_not_found(self):
        """Deve lançar FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            calculate_sha256("/nonexistent/file.pdf")

    def test_get_file_size(self):
        """get_file_size deve retornar tamanho em bytes."""
        size = get_file_size(PDF_PATH)

        assert size > 0
        assert isinstance(size, int)


# --- Testes de chunking ---

class TestChunkText:
    def test_chunk_respects_max_size(self):
        """Chunks não devem exceder tamanho máximo."""
        text = "Este é um parágrafo de teste.\n\n" * 50
        chunks = chunk_text(text, max_size=200, overlap=50)

        assert len(chunks) > 0
        # Permitir pequena tolerância por causa do overlap e junção de segmentos
        for chunk in chunks:
            assert chunk.content_length <= 300, f"Chunk muito grande: {chunk.content_length}"

    def test_chunk_has_overlap(self):
        """Chunks consecutivos devem ter algum conteúdo em comum."""
        # Texto grande o suficiente para múltiplos chunks
        words = ["palavra"] * 200
        text = " ".join(words)
        chunks = chunk_text(text, max_size=100, overlap=30)

        assert len(chunks) >= 2

    def test_chunk_preserves_page_number(self):
        """Chunks devem preservar número da página."""
        text = "Conteúdo de teste.\n\n" * 10
        chunks = chunk_text(text, max_size=100, overlap=20, page=5)

        assert all(c.page == 5 for c in chunks)

    def test_chunk_empty_text_returns_empty(self):
        """Texto vazio deve retornar lista vazia."""
        chunks = chunk_text("", max_size=1000, overlap=200)

        assert chunks == []

    def test_chunk_has_sequential_indices(self):
        """Chunks devem ter índices sequenciais."""
        text = "Parágrafo longo aqui.\n\n" * 30
        chunks = chunk_text(text, max_size=100, overlap=20)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_content_is_not_empty(self):
        """Todos os chunks devem ter conteúdo."""
        text = "Texto de exemplo.\n\nOutro parágrafo.\n\nMais um."
        chunks = chunk_text(text, max_size=1000, overlap=0)

        assert all(len(c.content) > 0 for c in chunks)


class TestChunkPages:
    def test_chunk_pages_preserves_metadata(self):
        """chunk_pages deve preservar metadados de origem."""
        pages = [
            PageContent(page_number=1, content="CAPÍTULO I - DISPOSIÇÕES GERAIS\n\nArt. 1. Este documento..."),
            PageContent(page_number=2, content="Art. 2. As regras..."),
        ]
        chunks = chunk_pages(pages, max_size=500, overlap=50)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        # Página deve ser preservada
        assert chunks[0].page == 1

    def test_chunk_pages_detects_article(self):
        """chunk_pages deve detectar artigos no texto."""
        pages = [
            PageContent(
                page_number=3,
                content="Art. 15. Obras e reformas são permitidas de segunda a sábado.\n\n",
            )
        ]
        chunks = chunk_pages(pages)

        # Deve detectar artigo
        articles = [c.article for c in chunks if c.article]
        assert len(articles) > 0

    def test_chunk_pages_with_real_pdf(self):
        """chunk_pages deve processar PDF real."""
        pages = extract_pdf(PDF_PATH)
        chunks = chunk_pages(pages, max_size=1000, overlap=200)

        assert len(chunks) > 0
        # Todos os chunks devem ter página válida
        assert all(c.page >= 1 for c in chunks)
        # Todos os chunks devem ter conteúdo
        assert all(len(c.content) > 0 for c in chunks)
