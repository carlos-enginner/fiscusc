# Agente Financeiro

## Responsabilidade

Responder perguntas sobre finanças do condomínio:
- Faturas e cobranças
- Despesas do condomínio
- Comparativos entre períodos
- Extração de dados de PDFs de fatura (futuro)

## Princípio Fundamental

**Responder com dados precisos e sempre mostrar as fontes dos valores.**

## Modelo

- **LLM**: Qwen3-8B via Ollama (inicial)
- **Vision** (futuro): minicpm-v para extração de faturas

## Tools

### get_fatura

Retorna fatura de uma unidade.

```python
@tool
def get_fatura(
    unidade: str,
    mes_referencia: str | None = None
) -> str:
    """
    Retorna a fatura de uma unidade do condomínio.
    
    Use para perguntas sobre:
    - Valor do condomínio
    - Itens da cobrança
    - Data de vencimento
    
    Args:
        unidade: Identificação da unidade (ex: "A-2002", "B-101")
        mes_referencia: Mês de referência (ex: "julho/2026"). 
                       Se não informado, retorna a mais recente.
        
    Returns:
        Dados da fatura formatados
    """
    # Parse da unidade
    bloco, apto = parse_unidade(unidade)
    
    # Buscar no banco
    fatura = db.query(Fatura).filter(
        Fatura.unidade_bloco == bloco,
        Fatura.unidade_apartamento == apto,
        Fatura.mes_referencia == mes_referencia if mes_referencia else True
    ).order_by(Fatura.data_vencimento.desc()).first()
    
    if not fatura:
        return f"Fatura não encontrada para unidade {unidade}"
    
    return format_fatura(fatura)
```

### comparar_despesas

Compara despesas entre dois períodos.

```python
@tool
def comparar_despesas(
    mes1: str,
    mes2: str,
    categoria: str | None = None
) -> str:
    """
    Compara despesas do condomínio entre dois meses.
    
    Use para perguntas sobre:
    - Variação de despesas
    - O que aumentou/diminuiu
    - Tendências de gastos
    
    Args:
        mes1: Primeiro mês (ex: "junho/2026")
        mes2: Segundo mês (ex: "julho/2026")
        categoria: Categoria específica (opcional)
        
    Returns:
        Comparativo formatado com variações
    """
    despesas1 = get_despesas_por_mes(mes1, categoria)
    despesas2 = get_despesas_por_mes(mes2, categoria)
    
    return format_comparativo(despesas1, despesas2)
```

### listar_despesas

Lista despesas de um período.

```python
@tool
def listar_despesas(
    mes_referencia: str,
    top_n: int = 10
) -> str:
    """
    Lista as principais despesas de um mês.
    
    Use para perguntas sobre:
    - Maiores gastos do mês
    - Detalhamento de despesas
    - Categorias de despesa
    
    Args:
        mes_referencia: Mês de referência (ex: "julho/2026")
        top_n: Número de itens a retornar (default: 10)
        
    Returns:
        Lista de despesas ordenadas por valor
    """
    despesas = db.query(FaturaItem).join(Fatura).filter(
        Fatura.mes_referencia == mes_referencia,
        FaturaItem.secao == "despesas"
    ).order_by(FaturaItem.valor.desc()).limit(top_n).all()
    
    return format_despesas(despesas)
```

### extrair_fatura_pdf (Futuro)

Extrai dados de PDF de fatura usando Vision model.

```python
@tool
def extrair_fatura_pdf(pdf_path: str) -> str:
    """
    Extrai dados estruturados de um PDF de fatura.
    
    Use quando:
    - Usuário enviar um PDF de fatura
    - Precisar processar nova fatura
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Returns:
        Dados extraídos da fatura
    """
    # Usar código do Blade (extractor.py)
    from app.extraction import extract_from_pdf
    
    raw_data = extract_from_pdf(pdf_path, provider="ollama", model="minicpm-v")
    fatura = parse_extracted_data(raw_data)
    
    # Salvar no banco
    save_fatura(fatura)
    
    return format_fatura_extraida(fatura)
```

## System Prompt

```python
FINANCE_AGENT_PROMPT = """Você é um especialista em finanças de condomínio.

REGRAS OBRIGATÓRIAS:
1. Sempre mostre valores numéricos precisos
2. Cite a fonte dos dados (mês de referência, documento)
3. Use formatação de moeda brasileira (R$ X.XXX,XX)
4. Ao comparar, mostre variação absoluta e percentual
5. Agrupe despesas por categoria quando relevante

FORMATO DA RESPOSTA:
- Resposta direta com os valores
- Detalhamento quando solicitado
- Comparativos em formato de tabela quando aplicável

EXEMPLO:
"Sua fatura de julho/2026 é de R$ 795,96, com vencimento em 10/08/2026.

Composição:
- Taxa de condomínio: R$ 679,76
- Fundo de reserva: R$ 33,99
- Rateio academia: R$ 82,21

Comparado ao mês anterior, houve aumento de R$ 15,00 (+1,9%)."

CATEGORIAS DE DESPESAS:
- Pessoal: portaria, limpeza, encargos
- Consumo: água, energia, gás
- Manutenção: elevadores, bombas, reparos
- Administrativo: honorários, contabilidade
- Outros: seguros, tarifas"""
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
finance_agent = create_agent(
    model=model,
    tools=[get_fatura, comparar_despesas, listar_despesas],
    system_prompt=FINANCE_AGENT_PROMPT
)

# Wrapper para o LangGraph
def query_finance(state: AgentInput) -> dict:
    """Executa o agente financeiro."""
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

## Formatação

### format_fatura

```python
def format_fatura(fatura: Fatura) -> str:
    """Formata fatura para resposta."""
    lines = [
        f"📄 FATURA - {fatura.mes_referencia}",
        f"Unidade: Bloco {fatura.unidade_bloco}, Apto {fatura.unidade_apartamento}",
        f"Vencimento: {fatura.data_vencimento.strftime('%d/%m/%Y')}",
        f"",
        f"💰 MINHA COBRANÇA: R$ {fatura.total_cobranca:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        ""
    ]
    
    # Itens de cobrança
    itens = db.query(FaturaItem).filter(
        FaturaItem.fatura_id == fatura.id,
        FaturaItem.secao == "cobranca"
    ).all()
    
    for item in itens:
        valor_fmt = f"R$ {item.valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        lines.append(f"  • {item.descricao}: {valor_fmt}")
    
    if fatura.codigo_barras:
        lines.append(f"")
        lines.append(f"📊 Código de barras: {fatura.codigo_barras[:30]}...")
    
    return "\n".join(lines)
```

### format_comparativo

```python
def format_comparativo(despesas1: dict, despesas2: dict) -> str:
    """Formata comparativo de despesas."""
    lines = [
        f"📊 COMPARATIVO DE DESPESAS",
        f"{despesas1['mes']} vs {despesas2['mes']}",
        "",
        "| Categoria | Antes | Depois | Variação |",
        "|-----------|-------|--------|----------|"
    ]
    
    all_cats = set(despesas1["por_categoria"].keys()) | set(despesas2["por_categoria"].keys())
    
    for cat in sorted(all_cats):
        v1 = despesas1["por_categoria"].get(cat, 0)
        v2 = despesas2["por_categoria"].get(cat, 0)
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 > 0 else 0
        
        sinal = "+" if diff > 0 else ""
        lines.append(
            f"| {cat} | R$ {v1:,.2f} | R$ {v2:,.2f} | {sinal}{pct:.1f}% |"
        )
    
    # Totais
    t1 = despesas1["total"]
    t2 = despesas2["total"]
    diff_total = t2 - t1
    pct_total = (diff_total / t1 * 100) if t1 > 0 else 0
    
    lines.append(f"| **TOTAL** | **R$ {t1:,.2f}** | **R$ {t2:,.2f}** | **{'+' if diff_total > 0 else ''}{pct_total:.1f}%** |")
    
    return "\n".join(lines)
```

## Testes

### test_finance_agent_get_fatura

```python
def test_finance_agent_returns_fatura():
    """Agente deve retornar dados da fatura."""
    result = query_finance({
        "query": "Qual o valor do condomínio do apto 2002A?"
    })
    
    response = result["results"][0]["result"]
    assert "R$" in response
    assert "vencimento" in response.lower()
```

### test_finance_agent_comparativo

```python
def test_finance_agent_compara_meses():
    """Agente deve comparar despesas entre meses."""
    result = query_finance({
        "query": "Compare as despesas de junho e julho de 2026"
    })
    
    response = result["results"][0]["result"]
    assert "%" in response  # Deve ter variação percentual
```

### test_format_currency

```python
def test_format_currency_brazilian():
    """Valores devem estar em formato brasileiro."""
    fatura = create_test_fatura(total=1234.56)
    formatted = format_fatura(fatura)
    
    assert "R$ 1.234,56" in formatted
```

## Métricas

- Precisão dos valores (match com banco)
- Tempo de query
- Cobertura de faturas processadas
