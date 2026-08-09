# Schema do Banco de Dados

## Visão Geral

PostgreSQL com extensão pgvector para armazenamento de embeddings.

## Extensões Necessárias

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

## Tabelas

### documents

Armazena metadados dos documentos ingeridos (PDFs).

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename        VARCHAR(255) NOT NULL,
    document_type   VARCHAR(50) NOT NULL,  -- 'regimento', 'convencao', 'manual', 'fatura'
    version         VARCHAR(50),
    sha256          VARCHAR(64) NOT NULL UNIQUE,
    page_count      INTEGER,
    file_size_bytes INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_sha256 ON documents(sha256);
CREATE INDEX idx_documents_created ON documents(created_at);
```

### document_chunks

Armazena os chunks de texto com seus embeddings.

```sql
CREATE TABLE document_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page            INTEGER NOT NULL,
    section         VARCHAR(255),
    chapter         VARCHAR(255),
    article         VARCHAR(100),
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    content_length  INTEGER NOT NULL,
    embedding       vector(1024),  -- Dimensão do Qwen3-Embedding-0.6B
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_page ON document_chunks(page);
CREATE INDEX idx_chunks_section ON document_chunks(section);

-- Índice HNSW para busca vetorial (melhor performance que IVFFlat)
CREATE INDEX idx_chunks_embedding ON document_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### faturas

Armazena faturas extraídas de PDFs.

```sql
CREATE TABLE faturas (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID REFERENCES documents(id) ON DELETE SET NULL,
    
    -- Identificação
    condominio_nome     VARCHAR(255),
    condominio_cnpj     VARCHAR(20),
    unidade_bloco       VARCHAR(10),
    unidade_apartamento VARCHAR(10),
    proprietario        VARCHAR(255),
    
    -- Período
    mes_referencia      VARCHAR(20),
    data_vencimento     DATE,
    data_emissao        DATE,
    
    -- Valores
    total_cobranca      DECIMAL(12, 2),
    codigo_barras       VARCHAR(100),
    
    -- Metadados de extração
    modelo_provider     VARCHAR(50),
    modelo_nome         VARCHAR(100),
    tempo_extracao_seg  DECIMAL(8, 2),
    health_check_score  INTEGER,
    
    -- Dados brutos
    raw_data            JSONB,
    
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_faturas_unidade ON faturas(unidade_bloco, unidade_apartamento);
CREATE INDEX idx_faturas_vencimento ON faturas(data_vencimento);
CREATE INDEX idx_faturas_referencia ON faturas(mes_referencia);
```

### fatura_itens

Itens individuais de uma fatura (cobrança, despesas, receitas).

```sql
CREATE TABLE fatura_itens (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fatura_id       UUID NOT NULL REFERENCES faturas(id) ON DELETE CASCADE,
    
    secao           VARCHAR(50) NOT NULL,  -- 'cobranca', 'despesas', 'receitas'
    grupo           VARCHAR(100),          -- Ex: 'SERVIÇOS TERCEIRIZADOS'
    descricao       VARCHAR(255) NOT NULL,
    valor           DECIMAL(12, 2) NOT NULL,
    categoria       VARCHAR(50),           -- Categoria normalizada
    
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fatura_itens_fatura ON fatura_itens(fatura_id);
CREATE INDEX idx_fatura_itens_secao ON fatura_itens(secao);
CREATE INDEX idx_fatura_itens_categoria ON fatura_itens(categoria);
```

### query_logs (Opcional - Observabilidade)

```sql
CREATE TABLE query_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query           TEXT NOT NULL,
    agents_used     VARCHAR(50)[],
    response        TEXT,
    sources         JSONB,
    latency_ms      INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_query_logs_created ON query_logs(created_at);
```

## Views

### v_chunks_with_document

View para facilitar queries de chunks com info do documento.

```sql
CREATE VIEW v_chunks_with_document AS
SELECT 
    c.id,
    c.content,
    c.page,
    c.section,
    c.chapter,
    c.article,
    c.embedding,
    d.filename,
    d.document_type,
    d.version
FROM document_chunks c
JOIN documents d ON c.document_id = d.id;
```

### v_despesas_por_categoria

View para análise de despesas por categoria.

```sql
CREATE VIEW v_despesas_por_categoria AS
SELECT 
    f.mes_referencia,
    fi.categoria,
    SUM(fi.valor) as total,
    COUNT(*) as qtd_itens
FROM fatura_itens fi
JOIN faturas f ON fi.fatura_id = f.id
WHERE fi.secao = 'despesas'
GROUP BY f.mes_referencia, fi.categoria
ORDER BY f.mes_referencia, total DESC;
```

## Funções

### search_similar_chunks

Função para busca semântica.

```sql
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(1024),
    match_count INT DEFAULT 5,
    filter_document_type VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    page INTEGER,
    section VARCHAR,
    document_type VARCHAR,
    filename VARCHAR,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.content,
        c.page,
        c.section,
        d.document_type,
        d.filename,
        1 - (c.embedding <=> query_embedding) as similarity
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE (filter_document_type IS NULL OR d.document_type = filter_document_type)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

## Migrações

Usar Alembic para gerenciar migrações:

```
alembic/
├── versions/
│   ├── 001_initial_schema.py
│   ├── 002_add_faturas.py
│   └── 003_add_query_logs.py
├── env.py
└── alembic.ini
```

## Docker Compose - PostgreSQL + pgvector

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: fiscusc-db
    environment:
      POSTGRES_USER: fiscusc
      POSTGRES_PASSWORD: fiscusc
      POSTGRES_DB: fiscusc
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fiscusc"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

## Considerações

1. **Dimensão do embedding**: 1024 (Qwen3-Embedding-0.6B)
2. **Índice HNSW**: Melhor para datasets < 1M vetores
3. **JSONB**: Flexibilidade para metadados extras
4. **UUID**: Evita colisões, bom para distribuído futuro
5. **SHA256**: Evita duplicação de documentos
6. **Cascade delete**: Chunks removidos com documento
