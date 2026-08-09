"""Agente de Documentos (RAG) usando LangChain."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.core.config import get_settings

DOCS_AGENT_PROMPT = """Você é um assistente do condomínio que ajuda os moradores a entenderem as regras do Regimento Interno e da Convenção.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nos documentos encontrados — nunca invente informações
2. Use linguagem clara, direta e respeitosa — sem juridiquês, sem excessos informais
3. Use sempre o termo do Regimento quando possível (ex: "animais domésticos", não "cachorro" ou "bichinho")
4. Prefira o plural e termos neutros para evitar ambiguidades (ex: "animais de estimação" em vez de "um cachorro")
5. Cite a fonte (artigo e página) de forma natural no texto
6. Se não encontrar a informação, informe com clareza e sugira contatar a administração

FORMATO DA RESPOSTA:
- Responda diretamente à pergunta
- Explique as regras de forma objetiva
- Liste condições quando houver mais de uma
- Indique a fonte ao final ou no corpo do texto

EXEMPLOS DE TOM:
✓ "Sim, é permitida a permanência de animais domésticos. O Regimento (Art. 53, pág. 13) estabelece que os animais devem circular nas áreas comuns com guia, coleira e focinheira."
✓ "Obras são permitidas de segunda a sábado, das 8h às 18h (Art. 15, pág. 8). Aos domingos e feriados não são permitidas obras que gerem ruído."
✓ "Não encontrei essa informação no Regimento. Recomendo entrar em contato com a administração do condomínio."

✗ "Conforme disposto no Artigo 15, § 2º, inciso III, fica vedada..." (muito formal)
✗ "Pode ter um cachorro!" (singular e informal demais)

Documentos disponíveis: Regimento Interno, Convenção do Condomínio"""


class DocsAgent:
    """
    Agente de Documentos com RAG.

    Busca nos documentos do condomínio e responde com evidências.
    """

    def __init__(self, retriever=None, llm=None):
        settings = get_settings()

        # Configurar LLM
        if llm is not None:
            self._llm = llm
        else:
            self._llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        self._retriever = retriever

        # Configurar tools se retriever disponível
        if retriever is not None:
            from app.agents.docs.tools import set_retriever, DOCS_TOOLS
            set_retriever(retriever)
            self._tools = DOCS_TOOLS
            self._agent = self._llm.bind_tools(self._tools)
        else:
            self._tools = []
            self._agent = self._llm

    def invoke(self, state: dict) -> dict:
        """
        Executa o agente de documentos.

        Args:
            state: Dict com chave "query".

        Returns:
            Dict com "results" contendo resposta e evidências.
        """
        query = state["query"]

        # Buscar contexto relevante diretamente (sem tool-calling loop completo)
        context = ""
        evidence = []

        if self._retriever:
            results = self._retriever.search(query=query, top_k=5)
            if results:
                context_parts = []
                for r in results:
                    source = f"[{r.document_type.upper()} - página {r.page}"
                    if r.section:
                        source += f", {r.section}"
                    if r.article:
                        source += f", {r.article}"
                    source += f"]"
                    context_parts.append(f"{source}\n{r.content}")
                    evidence.append({
                        "doc": r.filename,
                        "document_type": r.document_type,
                        "page": r.page,
                        "section": r.section,
                        "article": r.article,
                        "score": r.score,
                    })
                context = "\n\n---\n\n".join(context_parts)

        # Montar prompt com contexto
        if context:
            user_content = f"""CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}"""
        else:
            user_content = f"""PERGUNTA: {query}

Nenhum documento foi encontrado. Informe ao usuário que não há informação disponível."""

        messages = [
            SystemMessage(content=DOCS_AGENT_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self._llm.invoke(messages)
        answer = response.content

        return {
            "results": [
                {
                    "source": "docs",
                    "result": answer,
                    "evidence": evidence,
                }
            ]
        }


# Factory function para criação do agente com DI
def create_docs_agent(retriever=None, llm=None) -> DocsAgent:
    """Cria o agente de documentos com DI."""
    return DocsAgent(retriever=retriever, llm=llm)
