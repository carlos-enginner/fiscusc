# Fiscus-C

Sistema inteligente de gestão de condomínios com IA. Dois agentes especializados respondem perguntas sobre documentos normativos (Regimento, Convenção) e finanças (faturas, despesas), com citação de fontes e evidências.

## Arquitetura

```
FastAPI → LangGraph Workflow → [Classifier] → [DocsAgent | FinanceAgent] → Synthesizer
                                                      ↕                ↕
                                               pgvector (RAG)     PostgreSQL
                                                      ↕
                                                    Ollama
```

- **Agente Docs**: RAG sobre PDFs (Regimento, Convenção) com citação de página e artigo
- **Agente Finance**: Consulta de faturas e despesas no banco de dados
- **Orquestrador**: LangGraph com roteamento inteligente e execução paralela
- **LLMs**: Qwen3-8B (respostas) + Qwen3-Embedding-0.6B (embeddings) via Ollama

## Pré-requisitos

- Python 3.12+
- Docker + Docker Compose
- [Ollama](https://ollama.ai) instalado localmente

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repo>
cd fiscusc
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Editar .env se necessário (defaults funcionam com Docker local)
```

### 4. Subir PostgreSQL

```bash
docker compose up -d
```

### 5. Instalar modelos no Ollama

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

> **Requisito de hardware**: mínimo 16GB RAM. Com GPU: muito mais rápido.

### 6. Rodar migrações

```bash
alembic upgrade head
```

### 7. Iniciar a API

```bash
uvicorn app.api.main:app --reload
```

API disponível em `http://localhost:8000`. Documentação em `http://localhost:8000/docs`.

---

## Uso

### Via CLI

```bash
# Ingerir documento
python -m app.cli ingest fixtures/reg_interno.pdf --type regimento

# Fazer pergunta
python -m app.cli query "Qual o horário permitido para obras?"

# Ver status do sistema
python -m app.cli status
```

### Via API (curl)

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "ollama": "healthy"
  },
  "timestamp": "2026-08-09T15:00:00Z"
}
```

#### Ingerir Documento

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@fixtures/reg_interno.pdf" \
  -F "document_type=regimento"
```

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "reg_interno.pdf",
  "document_type": "regimento",
  "status": "processed",
  "stats": {
    "pages": 25,
    "chunks": 87,
    "processing_time_ms": 5200
  }
}
```

#### Fazer Pergunta (Agente Docs)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o horário permitido para obras?"}'
```

```json
{
  "answer": "Segundo o Regimento Interno (Art. 15, página 8), obras são permitidas de segunda a sábado, das 8h às 18h. Aos domingos e feriados não são permitidas obras que gerem ruído.",
  "agents_used": ["docs"],
  "sources": [
    {
      "type": "document",
      "document": "reg_interno.pdf",
      "document_type": "regimento",
      "page": 8,
      "section": "Art. 15 - Obras e Reformas",
      "score": 0.92
    }
  ],
  "metadata": {
    "query_id": "...",
    "latency_ms": 1250
  }
}
```

#### Fazer Pergunta (Agente Finance)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quanto foi a despesa com energia em julho/2026?"}'
```

#### Pergunta Mista (Dois agentes em paralelo)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "O valor da taxa de mudança cobrada está de acordo com o regimento?"}'
```

#### Listar Documentos

```bash
curl http://localhost:8000/api/v1/documents
curl "http://localhost:8000/api/v1/documents?type=regimento"
```

---

## Executando os Testes

### Testes unitários (sem dependências externas)

```bash
pytest tests/unit/ -v -m "not integration"
```

### Testes de integração (requerem PostgreSQL)

```bash
DATABASE_URL=postgresql://fiscusc:fiscusc@localhost:5432/fiscusc \
pytest tests/integration/ -v -m integration
```

### Testes E2E com mocks

```bash
pytest tests/e2e/ -v -m "not e2e"
```

### Testes E2E completos (requerem Ollama + PostgreSQL)

```bash
DATABASE_URL=postgresql://fiscusc:fiscusc@localhost:5432/fiscusc \
pytest tests/e2e/ -v -m e2e
```

### Todos os testes unitários de uma vez

```bash
pytest tests/unit/ -v
```

---

## Estrutura do Projeto

```
fiscusc/
├── app/
│   ├── api/               # FastAPI: main.py, schemas, routes/
│   ├── agents/
│   │   ├── docs/          # Agente de documentos (RAG)
│   │   └── finance/       # Agente financeiro
│   ├── core/              # Config (pydantic-settings), Database
│   ├── embeddings/        # Serviço de embeddings (Qwen3-Embedding)
│   ├── extraction/        # Extração de texto de PDFs (PyMuPDF)
│   ├── llm/               # Integração com Ollama (ChatOllama)
│   ├── orchestrator/      # LangGraph: classifier + workflow
│   ├── rag/               # Chunker, Retriever (pgvector), Ingestion
│   └── cli.py             # CLI (Click + Rich)
├── tests/
│   ├── unit/              # Testes unitários com mocks
│   ├── integration/       # Testes com banco de dados real
│   └── e2e/               # Testes de ponta a ponta
├── alembic/               # Migrações do banco de dados
├── scripts/               # init.sql
├── fixtures/              # PDFs de exemplo
├── specs/                 # Especificações do projeto
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Modelos de LLM

| Componente | Modelo | Uso |
|---|---|---|
| Embeddings | `qwen3-embedding:0.6b` | Vetorização de texto |
| Agente Docs | `qwen3:8b` | Resposta RAG |
| Agente Finance | `qwen3:8b` | Análise financeira |
| Classifier | `qwen3:8b` | Roteamento de perguntas |

---

## Troubleshooting

### Ollama não conecta

```bash
# Verificar se Ollama está rodando
ollama list

# Testar endpoint
curl http://localhost:11434/api/tags

# Iniciar Ollama se parado
ollama serve
```

### PostgreSQL erro de conexão

```bash
# Verificar se container está rodando
docker compose ps

# Ver logs
docker compose logs postgres

# Reiniciar
docker compose restart postgres

# Testar conexão
docker compose exec postgres pg_isready -U fiscusc
```

### Modelo não carrega / VRAM insuficiente

```bash
# Verificar modelos disponíveis
ollama list

# Baixar modelos necessários
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b

# Se VRAM insuficiente, usar versão menor
# Editar .env: LLM_MODEL=qwen3:4b
```

### Migration falha

```bash
# Verificar versão atual
alembic current

# Ver histórico
alembic history

# Aplicar do zero (cuidado: apaga dados)
alembic downgrade base
alembic upgrade head
```

### Embeddings com dimensão errada

Se mudar o modelo de embedding, a dimensão do vetor muda. É necessário:

```bash
# Rodar downgrade e upgrade para recriar o schema
alembic downgrade base
alembic upgrade head
# Re-ingerir todos os documentos
```

---

## Variáveis de Ambiente

| Variável | Default | Descrição |
|---|---|---|
| `DATABASE_URL` | `postgresql://fiscusc:fiscusc@localhost:5432/fiscusc` | URL do PostgreSQL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do Ollama |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Modelo de embeddings |
| `LLM_MODEL` | `qwen3:8b` | Modelo de linguagem |
| `CHUNK_SIZE` | `1000` | Tamanho máximo dos chunks |
| `CHUNK_OVERLAP` | `200` | Sobreposição entre chunks |
| `TOP_K_RESULTS` | `5` | Resultados retornados na busca |
| `MIN_SIMILARITY_SCORE` | `0.5` | Score mínimo de similaridade |
| `LOG_LEVEL` | `INFO` | Nível de log |

---

## Futuro (Pós-MVP)

- WhatsApp integration
- MCP Server para integração com outros agentes
- Multi-tenancy (múltiplos condomínios)
- Autenticação/Autorização
- Upload de faturas com extração automática via Vision (minicpm-v)
- Dashboard web
- Motor de contraprova
- S3 para armazenamento de arquivos
