# Plano de Evolução do RAG - Fiscus-C

## Roadmap

```
V1 (atual) → V2a (chunking estrutural) → V4 (citações) → V3 (reranker) → V2b (BM25) → V5 → V6
```

## Matriz Performance x Qualidade

| Versão | Performance | Qualidade | Impacto Principal |
|--------|:-----------:|:---------:|-------------------|
| **V2a** - Chunking estrutural | → | ⬆⬆ | Chunks mais semânticos, menos ruído |
| **V4** - Citações/Highlights | → | ⬆ | UX, confiança do usuário |
| **V3** - Reranker | ⬇ (+200-500ms) | ⬆⬆ | Precisão do top-5 |
| **V2b** - BM25 híbrido | ⬇ (leve) | ⬆⬆ | Recall, busca por termos exatos |
| **V5** - Query expansion | ⬇⬇ | ⬆⬆⬆ | Entende melhor a intenção |
| **V6** - Escala | ⬆⬆⬆ | → | Cache, async, multi-tenant |

---

## V1 - MVP ✅ (Atual)

**Status:** Implementado

**Arquitetura:**
```
PDF → chunk simples → MiniLM (384 dims) → pgvector → LLM
```

**Métricas atuais:**
- Hit Rate: 90%
- MRR: 0.72
- Tempo busca: 12ms

---

## V2a - Chunking Estrutural

**Objetivo:** Melhorar qualidade dos chunks respeitando a estrutura do documento (seções, artigos, parágrafos)

**Problema atual:** O chunker divide por tamanho fixo, podendo cortar no meio de seções ou artigos importantes.

### Implementação

1. **Detectar estrutura do PDF**
   - Extrair TOC (Table of Contents) se existir
   - Detectar headers por formatação (fonte maior, negrito)
   - Mapear hierarquia: Capítulo → Seção → Artigo → Parágrafo

2. **Chunking hierárquico**
   - Priorizar quebra em limites de seção
   - Manter artigos completos quando possível
   - Adicionar contexto pai no chunk (ex: "Capítulo III > Seção 2 > Art. 15")

3. **Metadata enriquecido**
   - `section_path`: ["Capítulo III", "Das Obras", "Art. 15"]
   - `section_level`: 1-4 (capítulo, seção, artigo, parágrafo)
   - `has_table`: bool
   - `has_list`: bool

### Arquivos a modificar/criar
- `app/extraction/pdf.py` - extrair estrutura
- `app/rag/chunker.py` - chunking hierárquico
- `app/rag/ingestion.py` - salvar metadata

### Estimativa: 2-3 dias

---

## V4 - Citações e Highlights

**Objetivo:** Melhorar UX com citações precisas e possibilidade de highlight no PDF

### Implementação

1. **Citações precisas na resposta**
   - Formato: `(Manual, Cap. III, Art. 15, p. 67)`
   - LLM instruído a citar fontes no formato padrão
   - Validar citações contra chunks usados

2. **Mapeamento chunk → PDF**
   - Salvar `bbox` (bounding box) de cada chunk no PDF
   - Endpoint `/api/v1/documents/{id}/highlight?chunk_id=xxx`
   - Retorna coordenadas para highlight no frontend

3. **API de preview**
   - Endpoint `/api/v1/documents/{id}/page/{page}/image`
   - Retorna imagem da página com highlight opcional

### Arquivos a criar
- `app/api/routes/highlights.py`
- `app/extraction/pdf.py` - extrair bbox
- Atualizar schema de chunks com bbox

### Estimativa: 2-3 dias

---

## V3 - Reranker

**Objetivo:** Melhorar precisão do ranking com cross-encoder

### Implementação

1. **Reranker service**
   - Modelo: `BAAI/bge-reranker-base` (multilíngue)
   - Buscar top-30 no pgvector
   - Rerankar para top-5

2. **Configuração**
   - `ENABLE_RERANKER=true/false`
   - `RERANKER_MODEL=BAAI/bge-reranker-base`
   - `RERANKER_TOP_K=30` (candidatos)
   - `RERANKER_FINAL_K=5` (resultado final)

3. **Integração no retriever**
   - Flag opcional no método `search()`
   - Fallback se reranker falhar

### Arquivos a criar
- `app/rag/reranker.py`
- Atualizar `app/rag/retriever.py`
- Atualizar `app/core/config.py`

### Dependência
`sentence-transformers>=2.2.0`

### Estimativa: 1-2 dias

---

## V2b - Busca Híbrida (BM25)

**Objetivo:** Combinar busca semântica (dense) com busca lexical (sparse) para melhor recall

### Implementação

1. **BM25 no PostgreSQL**
   - Usar extensão `pg_trgm` para busca full-text
   - Ou implementar BM25 manual com `tsvector`

2. **Fusion de scores**
   - Reciprocal Rank Fusion (RRF)
   - `score = 1/(k + rank_dense) + 1/(k + rank_sparse)`

3. **Configuração**
   - `ENABLE_HYBRID_SEARCH=true/false`
   - `HYBRID_ALPHA=0.7` (peso do dense vs sparse)

### Alternativa
Migrar para Qdrant (tem sparse vectors nativo)

### Arquivos a modificar
- Nova migration para índice GIN/tsvector
- `app/rag/retriever.py` - busca híbrida

### Estimativa: 2-3 dias

---

## V5 - Inteligência (Query Expansion + Histórico)

**Objetivo:** Melhorar entendimento da query e manter contexto de conversa

### Implementação

1. **Query Expansion**
   - LLM gera variações da query
   - "horário de obras" → ["horário de obras", "quando pode fazer reforma", "barulho permitido"]
   - Multi-query retrieval + merge de resultados

2. **Histórico de conversa**
   - Salvar últimas N mensagens por sessão
   - Resolver referências: "e sobre isso?" → expandir com contexto
   - Redis ou tabela `chat_sessions`

3. **Perguntas contextuais**
   - Sugerir perguntas relacionadas após resposta
   - Baseado nos chunks não usados mas relevantes

### Arquivos a criar
- `app/rag/query_expansion.py`
- `app/core/session.py`
- `app/api/routes/chat.py` (chat com histórico)

### Estimativa: 3-4 dias

---

## V6 - Escala

**Objetivo:** Preparar para produção com múltiplos manuais e usuários

### Implementação

1. **Processamento assíncrono**
   - Fila de ingestão (Celery ou similar)
   - Progress tracking por documento
   - Webhook quando terminar

2. **Cache inteligente**
   - Redis para embeddings de queries frequentes
   - Cache de respostas por hash(query + docs)
   - TTL configurável

3. **Multi-documento**
   - Filtro por documento/tipo na busca
   - Versionamento de documentos
   - Diff entre versões

4. **Observabilidade**
   - Métricas Prometheus
   - Tracing com OpenTelemetry
   - Dashboard de uso

### Arquivos a criar
- `app/workers/` - workers Celery
- `app/cache/` - camada de cache
- `app/api/routes/admin.py` - gestão

### Estimativa: 5-7 dias

---

## Resumo de Estimativas

| Versão | Foco | Tempo | Dependências |
|--------|------|-------|--------------|
| V2a | Chunking estrutural | 2-3 dias | - |
| V4 | Citações/Highlights | 2-3 dias | V2a (opcional) |
| V3 | Reranker | 1-2 dias | sentence-transformers |
| V2b | BM25 híbrido | 2-3 dias | pg_trgm ou Qdrant |
| V5 | Query expansion + histórico | 3-4 dias | Redis (opcional) |
| V6 | Escala | 5-7 dias | Celery, Redis, Prometheus |

**Total estimado:** 15-22 dias
