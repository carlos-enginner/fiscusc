"""Classificador de perguntas para roteamento de agentes."""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

CLASSIFIER_SYSTEM_PROMPT = """Analise a pergunta e determine quais agentes consultar.

AGENTES DISPONÍVEIS:
- docs: Regimento Interno, Convenção, regras, horários, permissões, proibições, normas de convivência, uso de áreas comuns
- finance: Faturas, despesas, cobranças, valores, pagamentos, comparativos financeiros, contas do condomínio

REGRAS:
1. Retorne APENAS os agentes relevantes para a pergunta
2. Para cada agente, crie uma sub-pergunta otimizada em português
3. Se a pergunta envolver AMBOS os domínios, retorne os dois
4. Seja preciso: não rotear para finance se a pergunta for sobre regras, nem para docs se for sobre valores

EXEMPLOS:
- "Posso fazer obra sábado?" → [{source: docs, query: "horário permitido para obras reforma"}]
- "Quanto paguei de condomínio?" → [{source: finance, query: "valor fatura cobrança condomínio"}]
- "A taxa de mudança cobrada está de acordo com o regimento?" → [
    {source: docs, query: "taxa de mudança regras valor permitido"},
    {source: finance, query: "taxa de mudança cobrada valor"}
  ]
- "Posso ter cachorro no apartamento?" → [{source: docs, query: "animais pets cachorro proibição regras"}]
- "Compare as despesas de junho e julho" → [{source: finance, query: "comparativo despesas junho julho"}]"""


class Classification(BaseModel):
    """Decisão de roteamento para um agente."""

    source: Literal["docs", "finance"] = Field(description="Agente a invocar")
    query: str = Field(description="Sub-pergunta otimizada para o agente")


class ClassificationResult(BaseModel):
    """Resultado da classificação com lista de agentes."""

    classifications: list[Classification] = Field(
        description="Lista de agentes a invocar com suas sub-perguntas"
    )


class QueryClassifier:
    """
    Classificador de perguntas para roteamento no workflow.

    Analisa a pergunta do usuário e decide quais agentes invocar.
    """

    def __init__(self, llm=None):
        self._llm = llm
        self._structured_llm = None
        if llm is not None:
            self._structured_llm = llm.with_structured_output(ClassificationResult)

    def classify(self, query: str) -> list[Classification]:
        """
        Classifica a query e retorna lista de agentes a invocar.

        Tenta structured output primeiro. Se falhar (modelos menores ou llm=None),
        usa heurística por palavras-chave como fallback.
        """
        # Sem LLM configurado → ir direto para keywords (rápido, sem latência)
        if self._llm is None and self._structured_llm is None:
            return self._classify_by_keywords(query)

        if self._structured_llm is None:
            from langchain_ollama import ChatOllama
            from app.core.config import get_settings

            settings = get_settings()
            llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
            self._llm = llm
            self._structured_llm = llm.with_structured_output(ClassificationResult)

        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        try:
            result = self._structured_llm.invoke(messages)
            if result is not None and result.classifications:
                return result.classifications
        except Exception:
            pass

        # Fallback: heurística por palavras-chave
        return self._classify_by_keywords(query)

    def _classify_by_keywords(self, query: str) -> list[Classification]:
        """Fallback simples por palavras-chave quando o LLM não gera structured output."""
        q = query.lower()

        finance_words = {
            "fatura", "valor", "paguei", "despesa", "custo", "cobrança",
            "taxa", "pagamento", "boleto", "vencimento", "conta", "gasto",
            "preço", "quanto", "dinheiro", "reais", "r$", "comparar",
        }
        docs_words = {
            "posso", "pode", "permitido", "proibido", "regra", "horário",
            "norma", "convenção", "regimento", "artigo", "área", "comum",
            "mudança", "obra", "reforma", "animal", "cachorro", "festa",
            "barulho", "ruído", "vaga", "garagem", "visitante",
        }

        has_finance = any(w in q for w in finance_words)
        has_docs = any(w in q for w in docs_words)

        # Se ambos ou nenhum → docs (mais comum)
        if has_finance and has_docs:
            return [
                Classification(source="docs", query=query),
                Classification(source="finance", query=query),
            ]
        elif has_finance:
            return [Classification(source="finance", query=query)]
        else:
            return [Classification(source="docs", query=query)]


def classify_query(state: dict, classifier: QueryClassifier | None = None) -> dict:
    """
    Node do LangGraph: classifica a query e retorna classifications.

    Args:
        state: Estado do workflow com chave "query".
        classifier: Instância do classificador (opcional, cria novo se None).

    Returns:
        Dict com "classifications" para o estado.
    """
    if classifier is None:
        classifier = QueryClassifier()

    classifications = classifier.classify(state["query"])
    return {"classifications": classifications}
