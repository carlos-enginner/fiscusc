# Avaliação e Otimização de Embeddings

## Problema

A ingestão de documentos via Ollama é lenta (~15-20 min para 261 páginas) porque:
- Cada embedding é uma chamada HTTP separada
- Ollama não tem batch nativo eficiente
- Overhead de rede mesmo sendo localhost

## Alternativa Proposta: FastEmbed (ONNX)

### Por que FastEmbed?

- **2-4x mais rápido** em CPU (ONNX otimizado, sem HTTP)
- **Batch nativo** real
- Modelos multilíngues de qualidade comprovada

### Modelos Candidatos

| Modelo | Dims | Tamanho | Retrieval (MTEB) | Velocidade |
|--------|------|---------|------------------|------------|
| qwen3-embedding:0.6b (atual) | 1024 | 600MB | ~68% | Baseline |
| **intfloat/multilingual-e5-small** | 384 | 471MB | **~70%** | **3-5x mais rápido** |
| intfloat/multilingual-e5-base | 768 | 1.1GB | ~73% | 2x mais rápido |
| BAAI/bge-small-en-v1.5 | 384 | 133MB | ~65% | 4x mais rápido |

**Recomendação:** `multilingual-e5-small` - melhor custo/benefício para português.

## Implementação

### 1. FastEmbedProvider

```python
# app/embeddings/fastembed_provider.py
from fastembed import TextEmbedding
from app.embeddings.service import EmbeddingsProvider

class FastEmbedProvider(EmbeddingsProvider):
    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        self._model = TextEmbedding(model)
        self._dims = 384  # e5-small
    
    def embed(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]
    
    @property
    def dimensions(self) -> int:
        return self._dims
```

### 2. Configuração

```bash
# .env
EMBEDDING_PROVIDER=fastembed  # ou "ollama"
EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

### 3. Factory

```python
# app/embeddings/factory.py
def create_embedding_provider() -> EmbeddingsProvider:
    settings = get_settings()
    if settings.embedding_provider == "fastembed":
        return FastEmbedProvider(settings.embedding_model)
    return OllamaEmbeddingsProvider(settings.embedding_model)
```

### 4. Migration de Dimensão

Se mudar de 1024 → 384 dims, precisa:
1. Recriar a coluna `embedding` com nova dimensão
2. Re-ingerir todos os documentos

```python
# Migration
op.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
op.add_column('document_chunks', sa.Column('embedding', Vector(384)))
```

## Framework de Avaliação

### Golden Set

Criar dataset de perguntas/respostas esperadas:

```python
# tests/evaluation/golden_set.py
GOLDEN_SET = [
    {
        "question": "Qual o horário permitido para obras?",
        "expected_chunks": ["Art. 15", "obras", "8h às 18h"],
        "expected_answer_contains": ["segunda", "sábado", "18h"],
    },
    {
        "question": "Pode ter cachorro no condomínio?",
        "expected_chunks": ["animal", "pet", "Art."],
        "expected_answer_contains": ["permitido", "coleira"],
    },
    # ... 15-20 perguntas
]
```

### Métricas de Retrieval

```python
@dataclass
class RetrievalMetrics:
    hit_rate: float      # % queries com chunk certo no top-k
    mrr: float           # Mean Reciprocal Rank
    precision_at_5: float
```

### Script de Benchmark

```bash
python scripts/benchmark_embeddings.py

# Output esperado:
┌─────────────────────────────┬──────┬──────────┬───────────┬──────────┬──────┐
│ Modelo                      │ Dims │ Ingestão │ Busca (ms)│ Hit Rate │  MRR │
├─────────────────────────────┼──────┼──────────┼───────────┼──────────┼──────┤
│ ollama/qwen3-embedding:0.6b │ 1024 │   135.0s │        45 │    75.0% │ 0.65 │
│ fastembed/e5-small          │  384 │    38.0s │        12 │    80.0% │ 0.72 │
└─────────────────────────────┴──────┴──────────┴───────────┴──────────┴──────┘
```

## Outras Otimizações Futuras

### Reranker (melhora qualidade)

Após retrieval, reranquear top-30 → top-5 com cross-encoder:

```python
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# ou: jinaai/jina-reranker-v2-base-multilingual

scores = reranker.predict([(query, doc) for doc in candidates])
top_5 = sorted(zip(scores, candidates), reverse=True)[:5]
```

**Custo:** ~200-500ms para 30 candidatos
**Ganho:** Melhora significativa na qualidade do retrieval

### Busca Híbrida (denso + BM25)

pgvector não tem BM25 nativo, mas pode:
1. Usar Qdrant (tem sparse vectors)
2. Implementar BM25 separado + fusão de scores

### Scalar Quantization

Reduz uso de RAM em ~4x com perda mínima:

```sql
-- pgvector 0.5+
CREATE INDEX ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Dependências

```bash
# requirements.txt
fastembed>=0.2.0
# sentence-transformers>=2.2.0  # se usar reranker
```

## Ordem de Implementação

1. **FastEmbedProvider** - ganho imediato de velocidade
2. **Benchmark framework** - medir qualidade antes/depois
3. **Reranker** - se qualidade não for suficiente
4. **Busca híbrida** - otimização avançada

## Referências

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [FastEmbed Docs](https://qdrant.github.io/fastembed/)
- [E5 Paper](https://arxiv.org/abs/2212.03533)
- [Sentence Transformers](https://www.sbert.net/)
