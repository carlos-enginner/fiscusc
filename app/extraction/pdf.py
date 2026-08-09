"""Extração de texto de arquivos PDF usando PyMuPDF."""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Conteúdo de uma página extraída de um PDF."""

    page_number: int
    content: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)


def extract_pdf(path: str | Path) -> list[PageContent]:
    """
    Extrai texto de um PDF, retornando uma lista de páginas.

    Args:
        path: Caminho para o arquivo PDF.

    Returns:
        Lista de PageContent com conteúdo e número da página.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo não for um PDF válido.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    pages: list[PageContent] = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Não foi possível abrir o PDF: {e}") from e

    with doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            # Limpar espaços excessivos mantendo estrutura
            text = _clean_text(text)
            if text.strip():  # Ignorar páginas vazias
                pages.append(PageContent(page_number=page_num + 1, content=text))

    return pages


def calculate_sha256(path: str | Path) -> str:
    """
    Calcula o SHA256 de um arquivo.

    Args:
        path: Caminho para o arquivo.

    Returns:
        Hash SHA256 hexadecimal (64 caracteres).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_size(path: str | Path) -> int:
    """Retorna o tamanho do arquivo em bytes."""
    return Path(path).stat().st_size


def _clean_text(text: str) -> str:
    """Limpa texto extraído do PDF."""
    # Remover múltiplas linhas em branco consecutivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remover espaços no final das linhas
    text = re.sub(r" +\n", "\n", text)
    # Remover espaços iniciais em linhas (exceto indentação intencional)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    return text.strip()
