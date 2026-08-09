# Orquestrador (LangGraph)

## Responsabilidade

Coordenar a execução dos agentes usando o **Router Pattern** do LangGraph:
1. Classificar a pergunta do usuário
2. Rotear para o(s) agente(s) correto(s)
3. Executar agentes em paralelo quando necessário
4. Sintetizar respostas em resposta final unificada

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         StateGraph                               │
│                                                                  │
│    ┌─────────┐     ┌──────────┐     ┌────────────┐              │
│    │  START  │────▶│ classify │────▶│   route    │              │
│    └─────────┘     └──────────┘     └─────┬──────┘              │
│                                           │                      │
│                          ┌────────────────┼────────────────┐     │
│                          ▼                ▼                ▼     │
│                    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│                    │   docs   │    │ finance  │    │  (both)  │ │
│                    └────┬─────┘    └────┬─────┘    └────┬─────┘ │
│                         │               │               │        │
│                         └───────────────┼───────────────┘        │
│                                         ▼                        │
│                                  ┌─────────────┐                 │
│                                  │ synthesize  │                 │
│                                  └──────┬──────┘                 │
│                                         │                        │
│                                         ▼                        │
│                                    ┌─────────┐                   │
│                                    │   END   │                   │
│                                    └─────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Estado (State)

```python
import operator
from typing import Annotated, Literal, TypedDict


class AgentInput(TypedDict):
    """Input para cada agente."""
    query: str


class AgentOutput(TypedDict):
    """Output de cada agente."""
    source: str           # "docs" ou "finance"
    result: str           # Resposta do agente
    evidence: list[dict]  # [{doc, page, section, score}]


class Classification(TypedDict):
    """Decisão de roteamento."""
    source: Literal["docs", "finance"]
    query: str  # Sub-pergunta otimizada para o agente


class FiscusState(TypedDict):
    """Estado principal do workflow."""
    query: str                                                  # Pergunta original
    classifications: list[Classification]                       # Decisões de roteamento
    results: Annotated[list[AgentOutput], operator.add]        # Resultados (reducer)
    final_answer: str                                          # Resposta sintetizada
```

O campo `results` usa um **reducer** (`operator.add`) para acumular resultados de execuções paralelas.

## Nodes

### classify (Classificador)

Analisa a pergunta e decide quais agentes usar.

```python
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class ClassificationResult(BaseModel):
    """Schema para output estruturado."""
    classifications: list[Classification] = Field(
        description="Lista de agentes a invocar com suas sub-perguntas"
    )


classifier_llm = ChatOllama(model="qwen3:8b")


def classify_query(state: FiscusState) -> dict:
    """Classifica a pergunta e decide roteamento."""
    
    structured_llm = classifier_llm.with_structured_output(ClassificationResult)
    
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """Analise a pergunta e determine quais agentes consultar.

AGENTES DISPONÍVEIS:
- docs: Regimento Interno, Convenção, regras, horários, permissões, proibições
- finance: Faturas, despesas, cobranças, valores, pagamentos, comparativos

REGRAS:
1. Retorne APENAS os agentes relevantes para a pergunta
2. Para cada agente, crie uma sub-pergunta otimizada
3. Se a pergunta envolver AMBOS os domínios, retorne os dois

EXEMPLOS:
- "Posso fazer obra sábado?" → [docs]
- "Quanto paguei de condomínio?" → [finance]
- "A taxa de mudança cobrada está de acordo com o regimento?" → [docs, finance]"""
        },
        {"role": "user", "content": state["query"]}
    ])
    
    return {"classifications": result.classifications}
```

### route (Roteador)

Envia para os agentes usando Send API para execução paralela.

```python
from langgraph.types import Send


def route_to_agents(state: FiscusState) -> list[Send]:
    """Roteia para agentes em paralelo usando Send."""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]
```

### docs (Agente de Documentos)

```python
def query_docs(state: AgentInput) -> dict:
    """Executa o agente de documentos."""
    from app.agents.docs import docs_agent
    
    result = docs_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    
    return {
        "results": [{
            "source": "docs",
            "result": result["messages"][-1].content,
            "evidence": extract_evidence(result)
        }]
    }
```

### finance (Agente Financeiro)

```python
def query_finance(state: AgentInput) -> dict:
    """Executa o agente financeiro."""
    from app.agents.finance import finance_agent
    
    result = finance_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    
    return {
        "results": [{
            "source": "finance",
            "result": result["messages"][-1].content,
            "evidence": extract_financial_sources(result)
        }]
    }
```

### synthesize (Sintetizador)

Combina resultados de múltiplos agentes.

```python
def synthesize_results(state: FiscusState) -> dict:
    """Sintetiza resultados em resposta final."""
    
    if not state["results"]:
        return {"final_answer": "Não foi possível processar sua pergunta."}
    
    # Se apenas um agente, usar resultado diretamente
    if len(state["results"]) == 1:
        return {"final_answer": state["results"][0]["result"]}
    
    # Múltiplos agentes: sintetizar
    formatted = []
    for r in state["results"]:
        formatted.append(f"**{r['source'].upper()}:**\n{r['result']}")
    
    synthesis_response = classifier_llm.invoke([
        {
            "role": "system",
            "content": f"""Sintetize os resultados para responder: "{state['query']}"

REGRAS:
1. Combine informações sem redundância
2. Se houver conflito entre fontes, mencione
3. Mantenha as citações de fontes
4. Seja conciso mas completo"""
        },
        {"role": "user", "content": "\n\n---\n\n".join(formatted)}
    ])
    
    return {"final_answer": synthesis_response.content}
```

## Montagem do Grafo

```python
from langgraph.graph import StateGraph, START, END


def create_fiscus_workflow():
    """Cria o workflow LangGraph do Fiscus-C."""
    
    workflow = (
        StateGraph(FiscusState)
        # Nodes
        .add_node("classify", classify_query)
        .add_node("docs", query_docs)
        .add_node("finance", query_finance)
        .add_node("synthesize", synthesize_results)
        # Edges
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify",
            route_to_agents,
            ["docs", "finance"]
        )
        .add_edge("docs", "synthesize")
        .add_edge("finance", "synthesize")
        .add_edge("synthesize", END)
    )
    
    return workflow.compile()


# Singleton
fiscus_workflow = create_fiscus_workflow()
```

## Uso

```python
# Query simples (um agente)
result = fiscus_workflow.invoke({
    "query": "Qual horário permitido para obras?"
})
print(result["final_answer"])
# → Roteia para "docs", responde com base no Regimento

# Query financeira (um agente)
result = fiscus_workflow.invoke({
    "query": "Quanto paguei de condomínio em julho?"
})
print(result["final_answer"])
# → Roteia para "finance", mostra dados da fatura

# Query mista (dois agentes em paralelo)
result = fiscus_workflow.invoke({
    "query": "A taxa de mudança de R$200 cobrada está de acordo com o regimento?"
})
print(result["final_answer"])
# → Roteia para AMBOS, sintetiza: "Segundo o Regimento (pág X), a taxa é de R$Y. 
#    A fatura mostra cobrança de R$200. [Conclusão]"
```

## Testes

### test_classify_routes_to_docs

```python
def test_classify_routes_to_docs():
    """Perguntas sobre regras devem ir para docs."""
    result = classify_query({"query": "Posso ter cachorro?"})
    
    sources = [c["source"] for c in result["classifications"]]
    assert "docs" in sources
    assert "finance" not in sources
```

### test_classify_routes_to_finance

```python
def test_classify_routes_to_finance():
    """Perguntas sobre valores devem ir para finance."""
    result = classify_query({"query": "Quanto paguei de energia?"})
    
    sources = [c["source"] for c in result["classifications"]]
    assert "finance" in sources
    assert "docs" not in sources
```

### test_classify_routes_to_both

```python
def test_classify_routes_to_both_when_mixed():
    """Perguntas mistas devem ir para ambos."""
    result = classify_query({
        "query": "O valor da taxa de mudança está correto segundo o regimento?"
    })
    
    sources = [c["source"] for c in result["classifications"]]
    assert "docs" in sources
    assert "finance" in sources
```

### test_parallel_execution

```python
import asyncio
import time


def test_parallel_execution_is_faster():
    """Execução paralela deve ser mais rápida que sequencial."""
    
    # Simular query que usa ambos agentes
    query = {"query": "Compare a taxa do regimento com o valor cobrado"}
    
    start = time.time()
    result = fiscus_workflow.invoke(query)
    parallel_time = time.time() - start
    
    # Paralelo deve levar ~1x tempo de um agente, não 2x
    # (assumindo que cada agente leva ~similar tempo)
    assert parallel_time < 15  # segundos (ajustar conforme hardware)
```

### test_synthesize_combines_results

```python
def test_synthesize_combines_both_sources():
    """Sintetizador deve mencionar ambas as fontes."""
    state = {
        "query": "Taxa está correta?",
        "classifications": [
            {"source": "docs", "query": "Qual taxa de mudança no regimento?"},
            {"source": "finance", "query": "Quanto foi cobrado de mudança?"}
        ],
        "results": [
            {"source": "docs", "result": "Taxa é R$ 150 (Art. 20)", "evidence": []},
            {"source": "finance", "result": "Cobrado R$ 200", "evidence": []}
        ]
    }
    
    result = synthesize_results(state)
    
    assert "regimento" in result["final_answer"].lower() or "R$ 150" in result["final_answer"]
    assert "R$ 200" in result["final_answer"]
```

## Observabilidade

```python
# Logging de cada step
import logging

logger = logging.getLogger("fiscusc.orchestrator")

def classify_query_with_logging(state: FiscusState) -> dict:
    logger.info(f"Classificando query: {state['query'][:50]}...")
    
    result = classify_query(state)
    
    agents = [c["source"] for c in result["classifications"]]
    logger.info(f"Roteando para: {agents}")
    
    return result
```

## Extensibilidade

### Adicionar novo agente

```python
# 1. Criar node
def query_new_agent(state: AgentInput) -> dict:
    """Novo agente."""
    result = new_agent.invoke(...)
    return {"results": [{"source": "new_agent", ...}]}

# 2. Adicionar no grafo
workflow = (
    StateGraph(FiscusState)
    .add_node("new_agent", query_new_agent)
    .add_conditional_edges(
        "classify",
        route_to_agents,
        ["docs", "finance", "new_agent"]  # Adicionar aqui
    )
    .add_edge("new_agent", "synthesize")
    ...
)

# 3. Atualizar prompt do classifier
"""
AGENTES DISPONÍVEIS:
- docs: ...
- finance: ...
- new_agent: [descrição do novo agente]
"""
```
