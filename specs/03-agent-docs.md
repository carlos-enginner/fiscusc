# Agente de Documentos (RAG)

## Responsabilidade

Responder perguntas sobre documentos do condomínio:
- Regimento Interno
- Convenção do Condomínio
- Manuais de uso
- Outros documentos normativos

## Princípio Fundamental

**Responder APENAS com base em evidências encontradas nos documentos.**

Se não houver informação suficiente, informar claramente ao usuário.

## Modelo

- **LLM**: Qwen3-8B via Ollama
- **Embeddings**: Qwen3-Embedding-0.6B via Ollama

## Tools

### search_regimento

Busca no Regimento Interno do condomínio.

```python
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
        query: Pergunta ou termo de busca
        
    Returns:
        Trechos relevantes com página e seção
    """
    results = retriever.search(
        query=query,
        document_type="regimento",
        top_k=5
    )
    return format_results(results)
```

### search_convencao

Busca na Convenção do Condomínio.

```python
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
        query: Pergunta ou termo de busca
        
    Returns:
        Trechos relevantes com página e seção
    """
    results = retriever.search(
        query=query,
        document_type="convencao",
        top_k=5
    )
    return format_results(results)
```

### search_all_documents

Busca em todos os documentos disponíveis.

```python
@tool
def search_all_documents(query: str) -> str:
    """
    Busca informações em todos os documentos do condomínio.
    
    Use quando:
    - A pergunta pode estar em múltiplos documentos
    - Não está claro qual documento consultar
    - Precisa de visão geral
    
    Args:
        query: Pergunta ou termo de busca
        
    Returns:
        Trechos relevantes de todos os documentos
    """
    results = retriever.search(
        query=query,
        document_type=None,  # Todos
        top_k=5
    )
    return format_results(results)
```

## System Prompt

```python
DOCS_AGENT_PROMPT = """Você é um especialista em documentos de condomínio.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nos documentos encontrados
2. NUNCA invente informações ou use conhecimento externo
3. SEMPRE cite a fonte (documento, página, seção/artigo)
4. Se não encontrar informação suficiente, diga claramente
5. Diferencie informação encontrada de inferência

FORMATO DA RESPOSTA:
- Comece com a resposta direta
- Cite as fontes usadas
- Se houver ambiguidade, mencione

EXEMPLO DE RESPOSTA COM EVIDÊNCIA:
"Segundo o Regimento Interno, obras são permitidas de segunda a sábado, 
das 8h às 18h (Art. 15, página 8). Aos domingos e feriados não são 
permitidas obras que gerem ruído."

EXEMPLO DE RESPOSTA SEM EVIDÊNCIA:
"Não encontrei informação específica sobre [assunto] nos documentos 
disponíveis. Sugiro consultar a administração do condomínio."

Documentos disponíveis: Regimento Interno, Convenção do Condomínio"""
```

## Implementação

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Modelo
model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434"
)

# Criar agente
docs_agent = create_agent(
    model=model,
    tools=[search_regimento, search_convencao, search_all_documents],
    system_prompt=DOCS_AGENT_PROMPT
)

# Wrapper para o LangGraph
def query_docs(state: AgentInput) -> dict:
    """Executa o agente de documentos."""
    result = docs_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    
    # Extrair evidências da resposta
    evidence = extract_evidence(result)
    
    return {
        "results": [{
            "source": "docs",
            "result": result["messages"][-1].content,
            "evidence": evidence
        }]
    }
```

## Retriever (RAG)

```python
class DocumentRetriever:
    """Retriever para busca semântica em documentos."""
    
    def __init__(self, embeddings_service, db_session):
        self.embeddings = embeddings_service
        self.db = db_session
    
    def search(
        self,
        query: str,
        document_type: str | None = None,
        top_k: int = 5,
        min_score: float = 0.5
    ) -> list[SearchResult]:
        """
        Busca chunks similares à query.
        
        Args:
            query: Texto da busca
            document_type: Filtro por tipo de documento
            top_k: Número máximo de resultados
            min_score: Score mínimo de similaridade
            
        Returns:
            Lista de resultados ordenados por similaridade
        """
        # 1. Gerar embedding da query
        query_embedding = self.embeddings.embed(query)
        
        # 2. Buscar no pgvector
        results = self.db.execute(
            """
            SELECT 
                c.id,
                c.content,
                c.page,
                c.section,
                c.chapter,
                c.article,
                d.filename,
                d.document_type,
                1 - (c.embedding <=> :embedding) as score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE (:doc_type IS NULL OR d.document_type = :doc_type)
            ORDER BY c.embedding <=> :embedding
            LIMIT :limit
            """,
            {
                "embedding": query_embedding,
                "doc_type": document_type,
                "limit": top_k
            }
        )
        
        # 3. Filtrar por score mínimo
        return [
            SearchResult(**r)
            for r in results
            if r["score"] >= min_score
        ]
```

## Formatação de Resultados

```python
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
```

## Testes

### test_docs_agent_with_evidence

```python
def test_docs_agent_returns_evidence():
    """Agente deve retornar resposta com evidência."""
    result = query_docs({"query": "Qual horário permitido para obras?"})
    
    assert result["results"][0]["source"] == "docs"
    assert "página" in result["results"][0]["result"].lower()
    assert len(result["results"][0]["evidence"]) > 0
```

### test_docs_agent_no_hallucination

```python
def test_docs_agent_admits_no_info():
    """Agente deve admitir quando não tem informação."""
    result = query_docs({"query": "Qual a cor do elevador?"})
    
    response = result["results"][0]["result"].lower()
    assert any(phrase in response for phrase in [
        "não encontrei",
        "não há informação",
        "não consta",
        "não foi possível encontrar"
    ])
```

### test_search_filters_by_document_type

```python
def test_search_filters_by_document_type():
    """Busca deve filtrar por tipo de documento."""
    retriever = DocumentRetriever(embeddings, db)
    
    results = retriever.search(
        query="horário obras",
        document_type="regimento"
    )
    
    for r in results:
        assert r.document_type == "regimento"
```

## Métricas

- Tempo de busca (retrieval)
- Score médio de similaridade
- Taxa de "não encontrei"
- Precisão das fontes citadas
