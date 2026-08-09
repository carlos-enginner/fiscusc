# Arquitetura do Fiscus-C

## Diagrama de Alto Nível

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FastAPI                                   │
│                    POST /query, /documents/ingest                   │
├─────────────────────────────────────────────────────────────────────┤
│                      LangGraph (Orquestrador)                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      StateGraph                              │    │
│  │  [input] → [classify] → [route] → [agents] → [synthesize]   │    │
│  └─────────────────────────────────────────────────────────────┘    │
├────────────────────────┬────────────────────────────────────────────┤
│    Node: DocsAgent     │           Node: FinanceAgent              │
│    (LangChain RAG)     │           (LangChain + Tools)             │
│    - Qwen3-8B          │           - Qwen3-8B                      │
│    - search_regimento  │           - get_fatura                    │
│    - search_convencao  │           - comparar_despesas             │
├────────────────────────┴────────────────────────────────────────────┤
│              Embeddings Service (Qwen3-Embedding-0.6B)              │
├─────────────────────────────────────────────────────────────────────┤
│                    PostgreSQL + pgvector                            │
│         documents | document_chunks | faturas | despesas            │
└─────────────────────────────────────────────────────────────────────┘
```

## Fluxo de Query

```
┌──────────┐     ┌───────────┐     ┌──────────────┐
│  Usuário │────▶│  FastAPI  │────▶│  Classifier  │
└──────────┘     └───────────┘     └──────┬───────┘
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   ┌──────────┐    ┌──────────┐    ┌──────────┐
                   │   Docs   │    │ Finance  │    │  Ambos   │
                   │  Agent   │    │  Agent   │    │ Paralelo │
                   └────┬─────┘    └────┬─────┘    └────┬─────┘
                        │               │               │
                        └───────────────┼───────────────┘
                                        ▼
                               ┌─────────────────┐
                               │   Synthesizer   │
                               └────────┬────────┘
                                        ▼
                               ┌─────────────────┐
                               │    Resposta     │
                               │   + Fontes      │
                               └─────────────────┘
```

1. Usuário faz pergunta via API
2. Classifier (LLM) analisa e decide quais agentes usar
3. Router envia para agentes em paralelo (Send API do LangGraph)
4. Cada agente processa e retorna resultado com evidências
5. Synthesizer combina respostas em resposta final
6. API retorna com sources/evidências

## Decisões Técnicas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Orquestração | LangGraph | Router pattern oficial, paralelo nativo |
| RAG | LangChain + PGVector | Integração madura, SQL híbrido |
| Embeddings | Qwen3-Embedding-0.6B | Leve, local, bom português |
| LLM Docs | Qwen3-8B | Bom para instrução, local |
| LLM Finance | Qwen3-8B (inicial) | Mesmo modelo, pode trocar depois |
| Vector Store | pgvector | Queries híbridas, ACID |
| API | FastAPI | Async, OpenAPI, tipagem |
| PDF | PyMuPDF | Rápido, preserva estrutura |
| Testes | pytest | TDD, fixtures, async |

## Interfaces entre Componentes

### AgentInput

```python
class AgentInput(TypedDict):
    """Input simples para cada agente."""
    query: str
```

### AgentOutput

```python
class AgentOutput(TypedDict):
    """Output de cada agente."""
    source: str      # "docs" ou "finance"
    result: str      # Resposta do agente
    evidence: list   # [{doc, page, section, score}]
```

### Classification

```python
class Classification(TypedDict):
    """Decisão de roteamento."""
    source: Literal["docs", "finance"]
    query: str  # Sub-pergunta otimizada para o agente
```

### FiscusState (Estado do Workflow)

```python
class FiscusState(TypedDict):
    """Estado principal do workflow LangGraph."""
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str
```

## Camadas da Aplicação

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                   │
│                  (FastAPI, CLI)                         │
├─────────────────────────────────────────────────────────┤
│                   Application Layer                     │
│          (Orchestrator, Agents, Services)               │
├─────────────────────────────────────────────────────────┤
│                     Domain Layer                        │
│         (Schemas, Business Logic, RAG)                  │
├─────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                   │
│     (Database, Ollama, Embeddings, PDF Extraction)      │
└─────────────────────────────────────────────────────────┘
```

## Padrões de Design

1. **Repository Pattern**: Acesso a dados isolado
2. **Service Layer**: Lógica de negócio encapsulada
3. **Dependency Injection**: Configuração via Settings
4. **Strategy Pattern**: Providers de LLM intercambiáveis
5. **Router Pattern**: Orquestração multi-agente (LangGraph)

## Configuração

Todas as configurações via variáveis de ambiente (.env):

```env
# Database
DATABASE_URL=postgresql://fiscusc:fiscusc@localhost:5432/fiscusc

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:0.6b
LLM_MODEL=qwen3:8b

# App
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

## Extensibilidade

### Adicionar novo agente

1. Criar pasta em `app/agents/novo_agente/`
2. Implementar tools e agent
3. Registrar no orchestrator
4. Atualizar classifier prompt

### Trocar modelo de LLM

1. Atualizar variável de ambiente
2. Ajustar prompts se necessário
3. Nenhuma mudança de código (strategy pattern)

### Adicionar novo tipo de documento

1. Criar parser específico se necessário
2. Adicionar document_type na tabela
3. Atualizar prompts do Agente Docs
