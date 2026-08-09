"""Tools do Agente de Documentos."""
from langchain_core.tools import tool

from app.rag.retriever import DocumentRetriever, format_results

# Retriever é injetado no momento da criação das tools
_retriever: DocumentRetriever | None = None


def set_retriever(retriever: DocumentRetriever):
    """Injeta o retriever nas tools."""
    global _retriever
    _retriever = retriever


def _get_retriever() -> DocumentRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever não configurado. Chame set_retriever() primeiro.")
    return _retriever


@tool
def search_regimento(query: str) -> str:
    """
    Busca informações no Regimento Interno do condomínio.

    Use para perguntas sobre:
    - Horários permitidos (obras, mudanças, festas)
    - Regras de uso de áreas comuns
    - Proibições e penalidades
    - Normas de convivência

    Args:
        query: Pergunta ou termo de busca.

    Returns:
        Trechos relevantes com página e seção.
    """
    results = _get_retriever().search(query=query, document_type="regimento", top_k=5)
    return format_results(results)


@tool
def search_convencao(query: str) -> str:
    """
    Busca informações na Convenção do Condomínio.

    Use para perguntas sobre:
    - Fração ideal
    - Direitos e deveres dos condôminos
    - Estrutura administrativa
    - Assembleias
    - Alterações estruturais

    Args:
        query: Pergunta ou termo de busca.

    Returns:
        Trechos relevantes com página e seção.
    """
    results = _get_retriever().search(query=query, document_type="convencao", top_k=5)
    return format_results(results)


@tool
def search_all_documents(query: str) -> str:
    """
    Busca informações em todos os documentos do condomínio.

    Use quando:
    - A pergunta pode estar em múltiplos documentos
    - Não está claro qual documento consultar
    - Precisa de visão geral

    Args:
        query: Pergunta ou termo de busca.

    Returns:
        Trechos relevantes de todos os documentos.
    """
    results = _get_retriever().search(query=query, document_type=None, top_k=5)
    return format_results(results)


DOCS_TOOLS = [search_regimento, search_convencao, search_all_documents]
