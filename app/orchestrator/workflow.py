"""Workflow LangGraph do Fiscus-C com router pattern e execução paralela."""
import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing_extensions import TypedDict

from app.orchestrator.classifier import Classification, QueryClassifier, classify_query


# --- Estado do Workflow ---

class AgentInput(TypedDict):
    """Input para cada agente."""

    query: str


class AgentOutput(TypedDict):
    """Output de cada agente."""

    source: str           # "docs" ou "finance"
    result: str           # Resposta do agente
    evidence: list[dict]  # [{doc, page, section, score}]


class FiscusState(TypedDict):
    """Estado principal do workflow."""

    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # reducer para paralelo
    final_answer: str


# --- Nodes ---

def _classify_node(state: FiscusState, classifier: QueryClassifier) -> dict:
    """Node de classificação."""
    return classify_query(state, classifier=classifier)


def _route_to_agents(state: FiscusState) -> list[Send]:
    """
    Roteador: usa Send API para execução paralela.

    Cada classificação vira um Send para o nó do agente correspondente.
    """
    return [
        Send(c.source, {"query": c.query})
        for c in state["classifications"]
    ]


def _synthesize_results(state: FiscusState, llm=None) -> dict:
    """
    Sintetiza resultados de múltiplos agentes em resposta final.

    Se apenas um agente respondeu, usa o resultado diretamente.
    Se múltiplos, pede ao LLM para sintetizar.
    """
    results = state.get("results", [])

    if not results:
        return {"final_answer": "Não foi possível processar sua pergunta."}

    # Um único agente: usar resultado direto
    if len(results) == 1:
        return {"final_answer": results[0]["result"]}

    # Múltiplos agentes: sintetizar
    if llm is None:
        # Concatenar as respostas com separador
        parts = []
        for r in results:
            parts.append(f"[{r['source'].upper()}]\n{r['result']}")
        return {"final_answer": "\n\n---\n\n".join(parts)}

    formatted = []
    for r in results:
        formatted.append(f"**{r['source'].upper()}:**\n{r['result']}")

    synthesis_response = llm.invoke([
        SystemMessage(
            content=f"""Sintetize os resultados para responder a pergunta original: "{state['query']}"

REGRAS:
1. Combine informações sem redundância
2. Se houver conflito entre fontes, mencione
3. Mantenha as citações de fontes
4. Seja conciso mas completo
5. Responda em português"""
        ),
        HumanMessage(content="\n\n---\n\n".join(formatted)),
    ])

    return {"final_answer": synthesis_response.content}


class FiscusWorkflow:
    """
    Workflow completo do Fiscus-C usando LangGraph.

    Orquestra os agentes docs e finance com roteamento inteligente.
    """

    def __init__(
        self,
        docs_agent=None,
        finance_agent=None,
        classifier: QueryClassifier | None = None,
        llm=None,
    ):
        self._docs_agent = docs_agent
        self._finance_agent = finance_agent
        self._classifier = classifier or QueryClassifier()
        self._llm = llm
        self._app = self._build_graph()

    def _build_graph(self):
        """Constrói e compila o StateGraph."""
        classifier = self._classifier
        docs_agent = self._docs_agent
        finance_agent = self._finance_agent
        llm = self._llm

        def classify_node(state: FiscusState) -> dict:
            return _classify_node(state, classifier=classifier)

        def docs_node(state: AgentInput) -> dict:
            if docs_agent is None:
                return {
                    "results": [
                        {"source": "docs", "result": "Agente de documentos não configurado.", "evidence": []}
                    ]
                }
            return docs_agent.invoke(state)

        def finance_node(state: AgentInput) -> dict:
            if finance_agent is None:
                return {
                    "results": [
                        {"source": "finance", "result": "Agente financeiro não configurado.", "evidence": []}
                    ]
                }
            return finance_agent.invoke(state)

        def synthesize_node(state: FiscusState) -> dict:
            return _synthesize_results(state, llm=llm)

        workflow = (
            StateGraph(FiscusState)
            # Nodes
            .add_node("classify", classify_node)
            .add_node("docs", docs_node)
            .add_node("finance", finance_node)
            .add_node("synthesize", synthesize_node)
            # Edges
            .add_edge(START, "classify")
            .add_conditional_edges("classify", _route_to_agents, ["docs", "finance"])
            .add_edge("docs", "synthesize")
            .add_edge("finance", "synthesize")
            .add_edge("synthesize", END)
        )

        return workflow.compile()

    def invoke(self, query: str) -> dict:
        """
        Executa o workflow para uma query.

        Args:
            query: Pergunta do usuário.

        Returns:
            Dict com query, results, classifications e final_answer.
        """
        return self._app.invoke({"query": query})


def create_fiscus_workflow(
    docs_agent=None,
    finance_agent=None,
    classifier: QueryClassifier | None = None,
    llm=None,
) -> FiscusWorkflow:
    """Factory para criação do workflow com DI."""
    return FiscusWorkflow(
        docs_agent=docs_agent,
        finance_agent=finance_agent,
        classifier=classifier,
        llm=llm,
    )
