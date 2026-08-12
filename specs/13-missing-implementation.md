# O que falta implementar - Fiscus-C

**Data:** 2026-08-12  
**Branch atual:** experiment/llm-vision-test

---

## TL;DR - Resumo Executivo

O sistema **já tem toda a arquitetura pronta**:
- ✅ 2 agentes especializados (DocsAgent + FinanceAgent)
- ✅ Classifier inteligente (detecta qual agente usar)
- ✅ Workflow orquestrado (LangGraph com execução paralela)
- ✅ RAG funcionando (pgvector + embeddings)
- ✅ Banco estruturado (PostgreSQL com tabelas)

**O que falta:** Popular os dados (extração de campos de faturas).

---

## Status da Implementação

### ✅ O que JÁ FUNCIONA

#### 1. **DocsAgent** (RAG para documentos de texto)
- **Localização:** `app/agents/docs/`
- **Função:** Responde perguntas sobre regimentos, manuais, contratos
- **Funciona com:**
  - Regimento interno
  - Convenção do condomínio
  - Manuais de equipamentos
  - Contratos (texto puro)
- **Tecnologia:** pgvector + embeddings + LLM
- **Status:** ✅ **Funcional**

**Exemplo:**
```python
"Qual o horário permitido para obras?"
→ DocsAgent busca no RAG
→ "Segundo o Regimento Interno (Art. 15, pág. 8), 
   obras são permitidas de segunda a sábado, das 8h às 18h"
```

#### 2. **FinanceAgent** (SQL para dados estruturados)
- **Localização:** `app/agents/finance/`
- **Função:** Responde perguntas sobre finanças
- **Tools disponíveis:**
  - `listar_despesas(mes_referencia, top_n)`
  - `comparar_despesas(mes1, mes2, categoria)`
  - `get_fatura(unidade, mes_referencia)`
- **Tecnologia:** PostgreSQL + LLM + SQL queries
- **Status:** ✅ **Funcional** (aguardando dados)

**Exemplo:**
```python
"Quanto gastei em julho?"
→ FinanceAgent consulta: SELECT SUM(valor) FROM faturas WHERE mes='2026-07'
→ "Em julho você gastou R$ 795,96 (taxa de condomínio + fundo de reserva)"
```

#### 3. **Classifier** (roteamento inteligente)
- **Localização:** `app/orchestrator/classifier.py`
- **Função:** Detecta qual agente usar (ou ambos)
- **Suporta:**
  - Query simples → 1 agente
  - Query híbrida → 2 agentes em paralelo
- **Status:** ✅ **Funcional**

**Exemplos:**
```python
# Query simples
"Posso fazer obra sábado?"
→ [{"source": "docs"}]

# Query híbrida
"A taxa cobrada está de acordo com o regimento?"
→ [{"source": "docs"}, {"source": "finance"}]
```

#### 4. **Workflow** (orquestração com LangGraph)
- **Localização:** `app/orchestrator/workflow.py`
- **Função:** Orquestra execução dos agentes
- **Features:**
  - Execução paralela quando necessário
  - Synthesizer combina resultados
  - Retry e error handling
- **Status:** ✅ **Funcional**

#### 5. **Pipeline de Ingestão** (para documentos de texto)
- **Localização:** `app/rag/ingestion.py`
- **Função:** PDF → Texto → Chunks → Embeddings → pgvector
- **Tecnologia:** PyMuPDF + chunker + embeddings service
- **Status:** ✅ **Funcional**

**Fluxo atual:**
```
PDF (regimento.pdf)
  ↓ PyMuPDF
Texto puro
  ↓ Chunker
Chunks de 1000 chars
  ↓ Embeddings
Vetores (384 ou 1024 dims)
  ↓ pgvector
Armazenado para RAG
```

---

### ❌ O que FALTA IMPLEMENTAR

#### 1. **Extração de Campos Estruturados de Faturas** 🔴 CRÍTICO

**Problema:** Hoje só extrai texto puro. Não extrai campos específicos.

**Solução:** Implementar extrator com Vision LLM

**Localização sugerida:** `app/extraction/vision.py` (novo arquivo)

**O que precisa:**

```python
class VisionExtractor:
    """Extrai campos estruturados de faturas usando LLM Vision."""
    
    def extract_energy_bill(self, pdf_path: Path) -> dict:
        """
        Extrai campos de conta de energia.
        
        Returns:
            {
                "tipo": "energia",
                "fornecedor": "ENEL",
                "cliente": "João Silva",
                "instalacao": "123456789",
                "consumo_kwh": 320,
                "vencimento": "2026-08-15",
                "valor_total": 287.43,
                "historico": [
                    {"mes": "2026-01", "consumo": 310},
                    {"mes": "2026-02", "consumo": 340},
                    ...
                ]
            }
        """
        pass
    
    def extract_water_bill(self, pdf_path: Path) -> dict:
        """Extrai campos de conta de água."""
        pass
    
    def extract_invoice(self, pdf_path: Path) -> dict:
        """Extrai campos de nota fiscal."""
        pass
```

**Tecnologias possíveis:**
- **Opção A:** LLM Vision (minicpm-v, llava) - mais fácil, sem treinamento
- **Opção B:** LayoutLMv3 fine-tuned - mais preciso, requer anotação

**Esforço estimado:** 2-3 dias (com Vision), 2-3 semanas (com LayoutLMv3)

---

#### 2. **Ingestão de Dados Estruturados** 🔴 CRÍTICO

**Problema:** Pipeline atual só salva texto no pgvector. Não salva campos estruturados no PostgreSQL.

**Solução:** Estender `app/rag/ingestion.py` para lidar com documentos estruturados

**Modificações necessárias:**

```python
# app/rag/ingestion.py (modificar método ingest)

def ingest(self, path: Path, document_type: str) -> IngestResult:
    """Ingere documento - texto OU estruturado."""
    
    if document_type in ["regimento", "convencao", "manual", "contrato"]:
        # FLUXO ATUAL (já funciona)
        pages = extract_pdf(path)
        chunks = chunk_pages(pages)
        embeddings = generate_embeddings(chunks)
        save_to_pgvector(chunks, embeddings)
    
    elif document_type in ["energia", "agua", "nota_fiscal"]:
        # FLUXO NOVO (precisa implementar)
        fields = vision_extractor.extract(path)
        save_to_postgres_structured(fields)
        
        # Opcional: também salvar texto para RAG híbrido
        text = extract_pdf(path)
        chunks = chunk_pages(text)
        save_to_pgvector(chunks)  # Permite perguntas sobre o texto também
    
    else:
        raise ValueError(f"Tipo de documento desconhecido: {document_type}")
```

**Esforço estimado:** 1-2 dias

---

#### 3. **Schema do Banco para Faturas** 🟡 IMPORTANTE

**Problema:** Tabelas de faturas podem precisar de ajustes/expansão.

**Verificar schema atual:**

```bash
# Ver tabelas existentes
psql -U fiscusc -d fiscusc -c "\dt"

# Ver estrutura da tabela faturas
psql -U fiscusc -d fiscusc -c "\d faturas"
```

**Schema sugerido:**

```sql
-- Tabela genérica de faturas
CREATE TABLE IF NOT EXISTS faturas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Classificação
    tipo VARCHAR(50) NOT NULL,  -- 'energia', 'agua', 'gas', 'nota_fiscal'
    fornecedor VARCHAR(200),
    
    -- Identificação
    numero_documento VARCHAR(100),
    mes_referencia DATE NOT NULL,
    
    -- Valores
    valor_total DECIMAL(10, 2) NOT NULL,
    valor_pago DECIMAL(10, 2),
    vencimento DATE,
    data_pagamento DATE,
    
    -- Específico por tipo (JSON flexível)
    campos_especificos JSONB,
    -- Ex para energia: {"consumo_kwh": 320, "instalacao": "123456789"}
    -- Ex para água: {"consumo_m3": 32, "matricula": "987654"}
    
    -- Metadados
    documento_pdf_path VARCHAR(500),
    sha256 VARCHAR(64),
    confidence REAL,  -- Confiança da extração (0-1)
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Histórico de consumo (para gráficos)
CREATE TABLE IF NOT EXISTS consumo_historico (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fatura_id UUID REFERENCES faturas(id) ON DELETE CASCADE,
    
    mes_referencia DATE NOT NULL,
    consumo DECIMAL(10, 2),
    unidade VARCHAR(20),  -- 'kWh', 'm³', etc.
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_faturas_tipo ON faturas(tipo);
CREATE INDEX idx_faturas_mes ON faturas(mes_referencia);
CREATE INDEX idx_faturas_fornecedor ON faturas(fornecedor);
CREATE INDEX idx_consumo_historico_mes ON consumo_historico(mes_referencia);
```

**Migration:**

```bash
# Criar migration
alembic revision -m "add_structured_invoices_tables"

# Aplicar
alembic upgrade head
```

**Esforço estimado:** 4-6 horas

---

#### 4. **CLI para Ingestão de Faturas** 🟡 IMPORTANTE

**Problema:** CLI atual (`app/cli.py`) só ingere para RAG.

**Solução:** Adicionar comando para faturas

```bash
# Uso desejado
python -m app.cli ingest-invoice fixtures/conta_energia.pdf --type energia
python -m app.cli ingest-invoice fixtures/conta_agua.pdf --type agua
python -m app.cli ingest-invoice fixtures/nota_fiscal.pdf --type nota_fiscal
```

**Implementação:**

```python
# app/cli.py (adicionar comando)

@cli.command("ingest-invoice")
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", type=click.Choice(["energia", "agua", "gas", "nota_fiscal"]), required=True)
def ingest_invoice(path: str, type: str):
    """Ingere fatura/nota fiscal com extração de campos."""
    from app.extraction.vision import VisionExtractor
    from app.rag.ingestion import save_structured_data
    
    console.print(f"[blue]Extraindo campos de {Path(path).name}...")
    
    extractor = VisionExtractor()
    
    if type == "energia":
        fields = extractor.extract_energy_bill(Path(path))
    elif type == "agua":
        fields = extractor.extract_water_bill(Path(path))
    elif type == "nota_fiscal":
        fields = extractor.extract_invoice(Path(path))
    
    console.print(f"[green]✓ Campos extraídos:")
    console.print(fields)
    
    # Salvar no banco
    save_structured_data(fields)
    
    console.print(f"[green]✓ Fatura salva no banco de dados")
```

**Esforço estimado:** 2-3 horas

---

#### 5. **Testes para Extração Estruturada** 🟢 DESEJÁVEL

**Problema:** Sem testes, difícil garantir qualidade da extração.

**Solução:** Testes unitários e E2E

```python
# tests/unit/extraction/test_vision_extractor.py

def test_extract_energy_bill():
    extractor = VisionExtractor()
    result = extractor.extract_energy_bill("fixtures/conta_energia.pdf")
    
    assert result["tipo"] == "energia"
    assert result["consumo_kwh"] > 0
    assert result["valor_total"] > 0
    assert "vencimento" in result


# tests/e2e/test_invoice_ingestion.py

def test_ingest_and_query_energy_bill():
    # Ingerir fatura
    ingest_invoice("fixtures/conta_energia.pdf", type="energia")
    
    # Consultar com FinanceAgent
    response = finance_agent.invoke(
        "Quanto foi o consumo de energia em julho?"
    )
    
    assert "320 kWh" in response
    assert "287,43" in response
```

**Esforço estimado:** 1-2 dias

---

#### 6. **Documentação de Uso** 🟢 DESEJÁVEL

**Problema:** README não documenta ingestão de faturas.

**Solução:** Atualizar README.md

```markdown
## Ingestão de Documentos

### Documentos de Texto (Regimentos, Manuais)

```bash
python -m app.cli ingest fixtures/reg_interno.pdf --type regimento
```

### Faturas e Notas Fiscais

```bash
# Conta de energia
python -m app.cli ingest-invoice fixtures/conta_energia.pdf --type energia

# Conta de água
python -m app.cli ingest-invoice fixtures/conta_agua.pdf --type agua

# Nota fiscal
python -m app.cli ingest-invoice fixtures/nota_fiscal.pdf --type nota_fiscal
```

### Consultando Dados

```bash
# Via CLI
python -m app.cli query "Quanto gastei de energia em julho?"

# Via API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quanto gastei de energia em julho?"}'
```
```

**Esforço estimado:** 2-3 horas

---

## Priorização

### 🔴 Fase 1 - MVP Funcional (1 semana)

1. **Extração de campos com Vision** (2-3 dias)
   - Implementar `VisionExtractor` básico
   - Testar com 1-2 tipos de fatura
   
2. **Ingestão estruturada** (1-2 dias)
   - Modificar `ingestion.py`
   - Criar/ajustar schema do banco
   
3. **Teste E2E** (1 dia)
   - Upload de fatura → extração → consulta
   - Validar que funciona ponta a ponta

**Resultado:** Sistema completo funcionando para faturas de energia/água.

---

### 🟡 Fase 2 - Produção (2 semanas)

4. **CLI e API** (2-3 dias)
   - Comandos para ingestão de faturas
   - Endpoints REST
   
5. **Testes automatizados** (2-3 dias)
   - Cobertura >80%
   - CI/CD
   
6. **Documentação** (1 dia)
   - README atualizado
   - Guia de uso

**Resultado:** Sistema pronto para usuários finais.

---

### 🟢 Fase 3 - Otimização (1+ mês)

7. **Fine-tuning LayoutLMv3** (2-3 semanas)
   - Anotar 100+ exemplos
   - Treinar modelos específicos
   - Melhorar precisão e latência
   
8. **Active Learning** (1-2 semanas)
   - Interface de revisão
   - Retreinamento automático
   
9. **Novos tipos de documento** (contínuo)
   - Contratos (extração de cláusulas)
   - Recibos
   - Boletos

**Resultado:** Sistema evoluindo continuamente.

---

## Decisões Técnicas Pendentes

### 1. LLM Vision vs LayoutLMv3?

| Critério | LLM Vision | LayoutLMv3 |
|----------|------------|------------|
| **Setup inicial** | 1 dia | 2-3 semanas |
| **Precisão** | 85-90% | 95%+ |
| **Latência** | 3-10s | 2-3s |
| **Manutenção** | Zero | Retreinar quando layout muda |
| **Novos tipos** | 5 min (mudar prompt) | 1 semana (anotar + treinar) |

**Recomendação:** 
- **Começar com Vision** (MVP rápido)
- **Migrar para LayoutLMv3** se volume justificar (>100 faturas/mês)

### 2. Processamento Síncrono vs Assíncrono?

**Síncrono:**
- Usuário espera 3-10s
- Feedback imediato
- Simples

**Assíncrono:**
- Upload instantâneo
- Processamento em background (Celery/Dramatiq)
- Notificação quando pronto

**Recomendação:**
- **MVP: Síncrono** (mais simples)
- **Produção: Assíncrono** (melhor UX)

### 3. Salvar também no RAG?

**Opção A:** Só tabelas estruturadas
- ✅ Mais rápido
- ❌ Não responde perguntas textuais ("Por que a conta veio alta?")

**Opção B:** Estruturado + RAG
- ✅ Responde tudo (valores + explicações)
- ❌ Mais lento na ingestão

**Recomendação:** **Opção B** (híbrido) - melhor cobertura

---

## Próximos Passos Imediatos

### Para implementar agora:

1. **Testar Vision com faturas reais** ✅ (já começamos)
   ```bash
   python scripts/test_llm_vision.py fixtures/conta_energia.pdf
   ```

2. **Implementar `VisionExtractor`**
   ```bash
   # Criar arquivo
   touch app/extraction/vision.py
   
   # Implementar classe básica
   # Testar com 1-2 faturas
   ```

3. **Ajustar schema do banco**
   ```bash
   alembic revision -m "add_invoices_tables"
   # Editar migration
   alembic upgrade head
   ```

4. **Modificar ingestion.py**
   ```python
   # Adicionar branch para documentos estruturados
   ```

5. **Teste E2E**
   ```bash
   # Ingerir fatura de teste
   # Consultar com FinanceAgent
   # Validar resposta
   ```

---

## Arquivos a Criar/Modificar

### Novos Arquivos

```
app/extraction/vision.py              # Extrator com Vision LLM
tests/unit/extraction/test_vision.py  # Testes unitários
tests/e2e/test_invoice_flow.py        # Teste E2E completo
alembic/versions/XXX_invoices.py      # Migration para faturas
```

### Arquivos a Modificar

```
app/rag/ingestion.py                  # Adicionar branch estruturado
app/cli.py                            # Adicionar comando ingest-invoice
README.md                             # Documentar novo fluxo
requirements.txt                      # Adicionar deps (se necessário)
```

---

## Dependências Extras Necessárias

```txt
# Para Vision
pdf2image>=1.16.0      # Converter PDF → imagem
pillow>=10.0.0         # Processar imagens

# Já existentes (verificar)
langchain-community    # Para Ollama Vision
```

---

## Conclusão

**O que está pronto (80% do sistema):**
- ✅ Arquitetura completa (2 agentes + orchestrator)
- ✅ RAG funcionando (documentos de texto)
- ✅ Banco de dados estruturado
- ✅ API REST
- ✅ CLI básico

**O que falta (20% do sistema):**
- ❌ Extrair campos de faturas
- ❌ Salvar dados estruturados no banco

**Esforço para completar:** 1 semana para MVP, 2-3 semanas para produção.

**Decisão recomendada:** Implementar extração com Vision LLM primeiro (rápido), otimizar com LayoutLMv3 depois se necessário.
