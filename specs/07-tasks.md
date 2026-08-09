# Tarefas de Implementação (TDD)

## Visão Geral

14 tarefas organizadas em 6 fases. Metodologia TDD: escrever testes primeiro, depois implementar.

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1: Fundação (Sequencial)              Tasks 1-3              │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 2: Core Components (Paralelo)         Tasks 4-6              │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 3: Agentes (Paralelo)                 Tasks 7-8              │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 4: Orquestração (Sequencial)          Tasks 9-10             │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 5: API + Integração (Sequencial)      Tasks 11-13            │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 6: Documentação                        Task 14                │
└─────────────────────────────────────────────────────────────────────┘
```

## Diagrama de Dependências

```
Task 1 (Setup)
    │
    ├──▶ Task 2 (Database)
    │        │
    │        └──▶ Task 3 (Config)
    │                 │
    │    ┌───────────┬┴───────────┐
    │    ▼           ▼            ▼
    │ Task 4     Task 5       Task 6
    │ (Embed)    (PDF)        (LLM)
    │    │           │            │
    │    └───────────┼────────────┘
    │                │
    │       ┌───────┴───────┐
    │       ▼               ▼
    │    Task 7          Task 8
    │    (Docs)         (Finance)
    │       │               │
    │       └───────┬───────┘
    │               │
    │               ▼
    │          Task 9 (Router)
    │               │
    │               ▼
    │          Task 10 (Workflow)
    │               │
    │    ┌──────────┼──────────┐
    │    ▼          ▼          ▼
    │ Task 11   Task 12    Task 13
    │ (API)     (CLI)      (E2E)
    │    │          │          │
    │    └──────────┴──────────┘
    │               │
    │               ▼
    └─────────▶ Task 14 (Docs)
```

---

## FASE 1: Fundação (Sequencial)

### Task 1: Setup do Projeto + Docker

**Objetivo**: Estrutura de pastas, dependências e infraestrutura Docker.

**Entregáveis**:
- [ ] Estrutura de diretórios conforme specs/00-overview.md
- [ ] `pyproject.toml` ou `requirements.txt` com dependências
- [ ] `docker-compose.yml` com PostgreSQL + pgvector
- [ ] `.env.example` com variáveis de ambiente
- [ ] `pytest.ini` ou `pyproject.toml` com config de testes
- [ ] `.gitignore`

**Testes TDD**:
```python
# tests/test_setup.py
def test_project_structure_exists():
    """Estrutura de pastas deve existir."""
    assert Path("app").is_dir()
    assert Path("app/agents").is_dir()
    assert Path("app/core").is_dir()
    assert Path("tests").is_dir()

def test_docker_compose_valid():
    """docker-compose.yml deve ser válido."""
    result = subprocess.run(
        ["docker-compose", "config"],
        capture_output=True
    )
    assert result.returncode == 0
```

**Dependências**: Nenhuma
**Estimativa**: 1-2 horas

---

### Task 2: Database Schema + Migrações

**Objetivo**: Criar schema do PostgreSQL conforme specs/02-database.md.

**Entregáveis**:
- [ ] Alembic configurado
- [ ] Migration inicial com todas as tabelas
- [ ] Models SQLAlchemy
- [ ] Script de inicialização `scripts/init.sql`
- [ ] Função `search_similar_chunks` no PostgreSQL

**Testes TDD**:
```python
# tests/test_database.py
def test_documents_table_exists():
    """Tabela documents deve existir."""
    result = db.execute("SELECT 1 FROM documents LIMIT 1")
    assert result is not None

def test_chunks_have_vector_column():
    """Coluna embedding deve ser do tipo vector(1024)."""
    result = db.execute("""
        SELECT data_type 
        FROM information_schema.columns 
        WHERE table_name = 'document_chunks' 
        AND column_name = 'embedding'
    """)
    assert "vector" in str(result.scalar())

def test_hnsw_index_exists():
    """Índice HNSW deve existir para busca vetorial."""
    result = db.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE indexname = 'idx_chunks_embedding'
    """)
    assert result.scalar() is not None
```

**Dependências**: Task 1
**Estimativa**: 2-3 horas

---

### Task 3: Camada de Configuração

**Objetivo**: Gerenciamento de configurações via pydantic-settings.

**Entregáveis**:
- [ ] `app/core/config.py` com Settings
- [ ] `app/core/database.py` com conexão
- [ ] Dependency injection pattern
- [ ] Carregamento de `.env`

**Testes TDD**:
```python
# tests/test_config.py
def test_settings_loads_from_env(monkeypatch):
    """Settings deve carregar de variáveis de ambiente."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    
    settings = Settings()
    assert "test" in settings.database_url

def test_settings_has_defaults():
    """Settings deve ter valores padrão sensatos."""
    settings = Settings()
    assert settings.chunk_size == 1000
    assert settings.top_k_results == 5

def test_database_connection():
    """Deve conectar ao banco de dados."""
    from app.core.database import get_db
    
    db = next(get_db())
    result = db.execute("SELECT 1")
    assert result.scalar() == 1
```

**Dependências**: Task 2
**Estimativa**: 1-2 horas

---

## FASE 2: Core Components (Paralelo)

> ⚡ Tasks 4, 5 e 6 podem ser executadas em paralelo após Task 3.

### Task 4: Embeddings Service

**Objetivo**: Serviço para gerar embeddings via Ollama.

**Entregáveis**:
- [ ] `app/embeddings/service.py`
- [ ] Interface abstrata para trocar provider
- [ ] Integração com Qwen3-Embedding-0.6B via Ollama
- [ ] Cache de embeddings (opcional)

**Testes TDD**:
```python
# tests/test_embeddings.py
def test_embed_returns_vector():
    """embed() deve retornar vetor de dimensão correta."""
    service = EmbeddingsService()
    
    vector = service.embed("texto de teste")
    
    assert len(vector) == 1024
    assert all(isinstance(v, float) for v in vector)

def test_embed_similar_texts_have_high_similarity():
    """Textos similares devem ter embeddings próximos."""
    service = EmbeddingsService()
    
    v1 = service.embed("horário de obras")
    v2 = service.embed("quando posso fazer reforma")
    v3 = service.embed("receita de bolo de chocolate")
    
    sim_12 = cosine_similarity(v1, v2)
    sim_13 = cosine_similarity(v1, v3)
    
    assert sim_12 > sim_13

def test_embed_batch():
    """embed_batch() deve processar múltiplos textos."""
    service = EmbeddingsService()
    
    vectors = service.embed_batch(["texto 1", "texto 2", "texto 3"])
    
    assert len(vectors) == 3
    assert all(len(v) == 1024 for v in vectors)
```

**Dependências**: Task 3
**Estimativa**: 2-3 horas

---

### Task 5: PDF Extraction + Chunking

**Objetivo**: Extrair texto de PDFs e dividir em chunks semânticos.

**Entregáveis**:
- [ ] `app/extraction/pdf.py` - extração de texto
- [ ] `app/rag/chunker.py` - chunking semântico
- [ ] Preservação de página, seção, artigo
- [ ] Cálculo de SHA256 para deduplicação

**Testes TDD**:
```python
# tests/test_extraction.py
def test_extract_pdf_returns_pages():
    """extract_pdf() deve retornar lista de páginas."""
    pages = extract_pdf("fixtures/reg_interno.pdf")
    
    assert len(pages) > 0
    assert all("content" in p and "page_number" in p for p in pages)

def test_extract_preserves_page_numbers():
    """Página deve ser preservada na extração."""
    pages = extract_pdf("fixtures/reg_interno.pdf")
    
    assert pages[0]["page_number"] == 1
    assert pages[-1]["page_number"] == len(pages)

def test_sha256_calculated():
    """SHA256 deve ser calculado para o documento."""
    sha = calculate_sha256("fixtures/reg_interno.pdf")
    
    assert len(sha) == 64
    assert sha.isalnum()


# tests/test_chunker.py
def test_chunk_respects_max_size():
    """Chunks não devem exceder tamanho máximo."""
    text = "A" * 5000
    chunks = chunk_text(text, max_size=1000)
    
    assert all(len(c["content"]) <= 1000 for c in chunks)

def test_chunk_has_overlap():
    """Chunks devem ter overlap."""
    text = "palavra " * 500
    chunks = chunk_text(text, max_size=100, overlap=20)
    
    # Verificar que há conteúdo comum entre chunks adjacentes
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i]["content"][-20:]
        start_of_next = chunks[i+1]["content"][:50]
        # Deve haver alguma sobreposição
        assert any(word in start_of_next for word in end_of_current.split())

def test_chunk_preserves_metadata():
    """Chunks devem preservar metadados de origem."""
    pages = [{"content": "Art. 15 - Obras...", "page_number": 8}]
    chunks = chunk_pages(pages)
    
    assert all("page" in c for c in chunks)
    assert chunks[0]["page"] == 8
```

**Dependências**: Task 3
**Estimativa**: 3-4 horas

---

### Task 6: LLM Service

**Objetivo**: Integração com Ollama para LLMs.

**Entregáveis**:
- [ ] `app/llm/service.py`
- [ ] `app/llm/ollama.py` - provider Ollama
- [ ] Interface abstrata para trocar provider
- [ ] Suporte a structured output

**Testes TDD**:
```python
# tests/test_llm.py
def test_llm_generates_response():
    """LLM deve gerar resposta."""
    service = LLMService()
    
    response = service.generate("Diga olá")
    
    assert len(response) > 0
    assert isinstance(response, str)

def test_llm_follows_system_prompt():
    """LLM deve seguir system prompt."""
    service = LLMService()
    
    response = service.generate(
        "Qual seu nome?",
        system_prompt="Você se chama Fiscus e só responde em português."
    )
    
    assert "fiscus" in response.lower() or "português" in response.lower()

def test_llm_structured_output():
    """LLM deve retornar output estruturado."""
    from pydantic import BaseModel
    
    class Response(BaseModel):
        sentiment: str
        confidence: float
    
    service = LLMService()
    result = service.generate_structured(
        "O dia está lindo!",
        output_schema=Response
    )
    
    assert isinstance(result, Response)
    assert result.sentiment in ["positivo", "negativo", "neutro"]
```

**Dependências**: Task 3
**Estimativa**: 2-3 horas

---

## FASE 3: Agentes (Paralelo)

> ⚡ Tasks 7 e 8 podem ser executadas em paralelo após Tasks 4, 5, 6.

### Task 7: Agente de Documentos (RAG)

**Objetivo**: Implementar agente RAG conforme specs/03-agent-docs.md.

**Entregáveis**:
- [ ] `app/agents/docs/agent.py`
- [ ] `app/agents/docs/tools.py`
- [ ] `app/rag/retriever.py`
- [ ] Ingestão de documentos
- [ ] Busca semântica com pgvector

**Testes TDD**:
```python
# tests/test_agent_docs.py
def test_ingest_document():
    """Deve ingerir documento e criar chunks."""
    result = ingest_document("fixtures/reg_interno.pdf", "regimento")
    
    assert result["status"] == "success"
    assert result["chunks_created"] > 0

def test_search_returns_relevant_chunks():
    """Busca deve retornar chunks relevantes."""
    # Setup: ingerir documento
    ingest_document("fixtures/reg_interno.pdf", "regimento")
    
    results = search_documents("horário obras")
    
    assert len(results) > 0
    assert results[0]["score"] > 0.5

def test_agent_cites_sources():
    """Agente deve citar fontes na resposta."""
    result = query_docs({"query": "Horário para obras?"})
    
    response = result["results"][0]["result"]
    assert "página" in response.lower() or "art." in response.lower()

def test_agent_admits_no_info():
    """Agente deve admitir quando não tem informação."""
    result = query_docs({"query": "Qual a cor do elevador?"})
    
    response = result["results"][0]["result"].lower()
    assert any(phrase in response for phrase in [
        "não encontrei", "não há informação", "não consta"
    ])
```

**Dependências**: Tasks 4, 5, 6
**Estimativa**: 4-5 horas

---

### Task 8: Agente Financeiro

**Objetivo**: Implementar agente financeiro conforme specs/04-agent-finance.md.

**Entregáveis**:
- [ ] `app/agents/finance/agent.py`
- [ ] `app/agents/finance/tools.py`
- [ ] Importação de dados do Blade (schemas)
- [ ] Consulta de faturas
- [ ] Comparativo de despesas

**Testes TDD**:
```python
# tests/test_agent_finance.py
def test_get_fatura():
    """Deve retornar fatura da unidade."""
    # Setup: inserir fatura de teste
    create_test_fatura(unidade="A-2002", total=795.96)
    
    result = get_fatura("A-2002")
    
    assert "795,96" in result or "795.96" in result

def test_comparar_despesas():
    """Deve comparar despesas entre meses."""
    # Setup
    create_test_fatura(mes="junho/2026", despesas=80000)
    create_test_fatura(mes="julho/2026", despesas=85000)
    
    result = comparar_despesas("junho/2026", "julho/2026")
    
    assert "%" in result  # Deve ter variação percentual

def test_agent_formats_currency():
    """Agente deve formatar valores em Real."""
    result = query_finance({"query": "Valor do condomínio?"})
    
    response = result["results"][0]["result"]
    assert "R$" in response

def test_listar_despesas():
    """Deve listar despesas ordenadas por valor."""
    result = listar_despesas("julho/2026")
    
    assert "energia" in result.lower() or "portaria" in result.lower()
```

**Dependências**: Tasks 4, 5, 6
**Estimativa**: 3-4 horas

---

## FASE 4: Orquestração (Sequencial)

### Task 9: Router/Classifier

**Objetivo**: Implementar classificador de perguntas.

**Entregáveis**:
- [ ] `app/orchestrator/classifier.py`
- [ ] Prompt otimizado para classificação
- [ ] Structured output para roteamento

**Testes TDD**:
```python
# tests/test_classifier.py
def test_classify_routes_to_docs():
    """Perguntas sobre regras → docs."""
    result = classify_query({"query": "Posso ter cachorro?"})
    
    sources = [c["source"] for c in result["classifications"]]
    assert "docs" in sources

def test_classify_routes_to_finance():
    """Perguntas sobre valores → finance."""
    result = classify_query({"query": "Quanto paguei de energia?"})
    
    sources = [c["source"] for c in result["classifications"]]
    assert "finance" in sources

def test_classify_routes_to_both():
    """Perguntas mistas → ambos."""
    result = classify_query({
        "query": "A taxa cobrada está de acordo com o regimento?"
    })
    
    sources = [c["source"] for c in result["classifications"]]
    assert "docs" in sources and "finance" in sources
```

**Dependências**: Tasks 7, 8
**Estimativa**: 2-3 horas

---

### Task 10: LangGraph Workflow

**Objetivo**: Montar workflow completo conforme specs/05-orchestrator.md.

**Entregáveis**:
- [ ] `app/orchestrator/workflow.py`
- [ ] StateGraph completo
- [ ] Execução paralela com Send API
- [ ] Sintetizador de respostas

**Testes TDD**:
```python
# tests/test_workflow.py
def test_workflow_single_agent():
    """Workflow com um agente deve funcionar."""
    result = fiscus_workflow.invoke({
        "query": "Horário para obras?"
    })
    
    assert "final_answer" in result
    assert len(result["final_answer"]) > 0

def test_workflow_multiple_agents():
    """Workflow com múltiplos agentes deve sintetizar."""
    result = fiscus_workflow.invoke({
        "query": "A taxa cobrada está correta segundo o regimento?"
    })
    
    assert len(result["results"]) == 2
    assert "final_answer" in result

def test_workflow_returns_sources():
    """Workflow deve retornar fontes."""
    result = fiscus_workflow.invoke({
        "query": "Posso fazer churrasco domingo?"
    })
    
    assert any(r["evidence"] for r in result["results"])
```

**Dependências**: Task 9
**Estimativa**: 3-4 horas

---

## FASE 5: API + Integração (Sequencial)

### Task 11: FastAPI Endpoints

**Objetivo**: Implementar API REST conforme specs/06-api.md.

**Entregáveis**:
- [ ] `app/api/main.py`
- [ ] `app/api/routes/query.py`
- [ ] `app/api/routes/documents.py`
- [ ] `app/api/routes/health.py`
- [ ] Schemas de request/response

**Testes TDD**:
```python
# tests/test_api.py
def test_health_endpoint():
    """GET /health deve retornar status."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    """POST /query deve processar pergunta."""
    response = client.post("/api/v1/query", json={
        "question": "Horário para obras?"
    })
    
    assert response.status_code == 200
    assert "answer" in response.json()

def test_ingest_endpoint():
    """POST /documents/ingest deve processar PDF."""
    with open("fixtures/reg_interno.pdf", "rb") as f:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("test.pdf", f)},
            data={"document_type": "regimento"}
        )
    
    assert response.status_code == 201
```

**Dependências**: Task 10
**Estimativa**: 3-4 horas

---

### Task 12: CLI Commands

**Objetivo**: Comandos de linha de comando para operações comuns.

**Entregáveis**:
- [ ] `app/cli.py` ou `scripts/`
- [ ] Comando de ingestão: `python -m app.cli ingest <pdf>`
- [ ] Comando de query: `python -m app.cli query "pergunta"`
- [ ] Comando de status: `python -m app.cli status`

**Testes TDD**:
```python
# tests/test_cli.py
def test_cli_ingest():
    """CLI ingest deve processar PDF."""
    result = runner.invoke(cli, ["ingest", "fixtures/reg_interno.pdf", "--type", "regimento"])
    
    assert result.exit_code == 0
    assert "chunks" in result.output.lower()

def test_cli_query():
    """CLI query deve retornar resposta."""
    result = runner.invoke(cli, ["query", "Horário para obras?"])
    
    assert result.exit_code == 0
    assert len(result.output) > 0

def test_cli_status():
    """CLI status deve mostrar info do sistema."""
    result = runner.invoke(cli, ["status"])
    
    assert result.exit_code == 0
    assert "database" in result.output.lower()
```

**Dependências**: Task 10
**Estimativa**: 2-3 horas

---

### Task 13: Teste E2E + Demo Final ⭐

**Objetivo**: Validar fluxo completo do sistema.

**Entregáveis**:
- [ ] `tests/e2e/test_full_flow.py`
- [ ] Teste de ingestão → query → resposta
- [ ] Teste de query mista (ambos agentes)
- [ ] Teste de "não encontrei informação"
- [ ] Script de demo

**Testes TDD**:
```python
# tests/e2e/test_full_flow.py
def test_e2e_complete_flow():
    """Fluxo completo: ingestão → query → resposta com fontes."""
    # 1. Ingerir documento
    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": open("fixtures/reg_interno.pdf", "rb")},
        data={"document_type": "regimento"}
    )
    assert response.status_code == 201
    
    # 2. Query sobre regras (Agente Docs)
    response = client.post("/api/v1/query", json={
        "question": "Qual o horário permitido para obras?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert any("regimento" in s["document"].lower() for s in data["sources"])
    
    # 3. Query sobre finanças (Agente Finance) - se houver dados
    response = client.post("/api/v1/query", json={
        "question": "Quanto foi a despesa com energia?"
    })
    assert response.status_code == 200
    
    # 4. Query mista (Ambos agentes)
    response = client.post("/api/v1/query", json={
        "question": "O valor da taxa de mudança está de acordo com o regimento?"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["agents_used"]) >= 1
    
    # 5. Query sem informação
    response = client.post("/api/v1/query", json={
        "question": "Qual a cor do teto do elevador?"
    })
    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert any(phrase in answer for phrase in [
        "não encontrei", "não há informação", "não foi possível"
    ])
```

**Dependências**: Tasks 11, 12
**Estimativa**: 3-4 horas

---

## FASE 6: Documentação

### Task 14: README + Docs

**Objetivo**: Documentação completa para uso do projeto.

**Entregáveis**:
- [ ] `README.md` completo
- [ ] Pré-requisitos e instalação
- [ ] Como rodar (Docker + local)
- [ ] Exemplos de uso (curl)
- [ ] Troubleshooting
- [ ] Arquitetura (resumo)

**Checklist do README**:
```markdown
# Fiscus-C

## Pré-requisitos
- [ ] Python 3.12+
- [ ] Docker + Docker Compose
- [ ] Ollama com modelos instalados

## Instalação
- [ ] Clone do repo
- [ ] Criar venv
- [ ] Instalar dependências
- [ ] Configurar .env

## Executando
- [ ] Subir PostgreSQL: `docker-compose up -d`
- [ ] Instalar modelos Ollama
- [ ] Rodar migrações
- [ ] Iniciar API

## Uso
- [ ] Ingerir documento
- [ ] Fazer query
- [ ] Exemplos curl

## Troubleshooting
- [ ] Ollama não conecta
- [ ] PostgreSQL erro de conexão
- [ ] Modelo não carrega
```

**Dependências**: Task 13
**Estimativa**: 2-3 horas

---

## Resumo de Estimativas

| Fase | Tasks | Horas | Tipo |
|------|-------|-------|------|
| 1 | 1, 2, 3 | 4-7h | Sequencial |
| 2 | 4, 5, 6 | 7-10h | **Paralelo** |
| 3 | 7, 8 | 7-9h | **Paralelo** |
| 4 | 9, 10 | 5-7h | Sequencial |
| 5 | 11, 12, 13 | 8-11h | Sequencial |
| 6 | 14 | 2-3h | Sequencial |

**Total Estimado**: 33-47 horas

**Caminho Crítico** (sequencial): ~26-35 horas
**Com Paralelismo**: ~20-28 horas

---

## Checklist de Conclusão

- [ ] Todos os testes passando
- [ ] Docker Compose funcionando
- [ ] API respondendo
- [ ] Ingestão de PDF funcionando
- [ ] Query RAG retornando fontes
- [ ] Query financeira retornando dados
- [ ] Query mista usando ambos agentes
- [ ] README documentado
- [ ] Demo executável
