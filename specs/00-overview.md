# Fiscus-C - Visão Geral

## O que é

Sistema inteligente para gestão de condomínios com dois agentes especializados:
- **Agente Docs**: Consulta documentos (Regimento, Convenção) via RAG
- **Agente Finance**: Análise de faturas e despesas

## Stack

- Python 3.12+
- FastAPI
- LangChain + LangGraph
- PostgreSQL + pgvector
- Ollama (Qwen3-8B, Qwen3-Embedding-0.6B)
- Docker Compose

## Princípios

1. **TDD**: Testes primeiro, implementação depois
2. **Evidências**: Toda resposta cita fonte (documento/página)
3. **Modular**: Agentes independentes, fácil adicionar novos
4. **Local-first**: Modelos rodam localmente via Ollama

## Estrutura do Projeto

```
fiscusc/
├── app/
│   ├── api/            # FastAPI endpoints
│   ├── agents/         # Agentes LangChain
│   │   ├── docs/       # Agente de documentos
│   │   └── finance/    # Agente financeiro
│   ├── core/           # Configuração, database
│   ├── embeddings/     # Serviço de embeddings
│   ├── extraction/     # Extração de PDF
│   ├── llm/            # Integração com Ollama
│   ├── orchestrator/   # LangGraph workflow
│   └── rag/            # Retrieval (chunking, search)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── specs/              # Especificações (este diretório)
├── fixtures/           # PDFs de teste
├── scripts/            # Scripts auxiliares
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Funcionalidades Principais

### MVP (Fase 1)

1. **Ingestão de Documentos**
   - Upload de PDF (Regimento Interno, Convenção)
   - Extração de texto com preservação de página
   - Chunking semântico (respeita seções/artigos)
   - Geração de embeddings
   - Armazenamento em pgvector

2. **Consulta RAG (Agente Docs)**
   - Busca semântica por similaridade
   - Resposta baseada apenas no contexto
   - Citação de fontes (documento, página, seção)
   - Fallback quando não há informação suficiente

3. **Análise Financeira (Agente Finance)**
   - Consulta de faturas por unidade/período
   - Comparativo de despesas entre meses
   - Extração de dados de PDF de fatura (futuro)

4. **Orquestrador (LangGraph)**
   - Classificação automática de perguntas
   - Roteamento para agente(s) correto(s)
   - Execução paralela quando necessário
   - Síntese de respostas múltiplas

5. **API REST**
   - POST /documents/ingest
   - POST /query
   - GET /documents
   - GET /health

### Futuro (Pós-MVP)

- WhatsApp integration
- MCP Server
- Multi-tenancy (múltiplos condomínios)
- Autenticação/Autorização
- Upload de faturas (extração automática)
- Dashboard web
- Motor de contraprova
- S3 para armazenamento de arquivos

## Modelos de LLM

| Componente | Modelo | Uso |
|------------|--------|-----|
| Embeddings | Qwen3-Embedding-0.6B | Vetorização de texto |
| Agente Docs | Qwen3-8B | Resposta RAG |
| Agente Finance | Qwen3-8B | Análise financeira |
| Classifier | Qwen3-8B | Roteamento de perguntas |
| Vision (futuro) | minicpm-v | Extração de faturas |

## Requisitos de Hardware

- **Mínimo**: 16GB RAM, CPU com AVX2
- **Recomendado**: 32GB RAM, GPU com 8GB+ VRAM
- **Disco**: 20GB+ para modelos e dados
