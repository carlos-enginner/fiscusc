# Teste de LLM Vision

Script para testar extração estruturada de documentos usando modelos Vision (minicpm-v, llava, etc).

## Requisitos

### 1. Dependências Python

```bash
pip install pdf2image pillow langchain-community
```

### 2. Dependências do Sistema

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### 3. Modelo Vision no Ollama

```bash
# Baixar minicpm-v (recomendado - multilingual, 8B)
ollama pull minicpm-v:8b

# Ou LLaVA (alternativa)
ollama pull llava:13b
```

## Uso

```bash
# Usar modelo padrão (minicpm-v:8b)
python scripts/test_llm_vision.py fixtures/reg_interno.pdf

# Especificar modelo
python scripts/test_llm_vision.py fixtures/reg_interno.pdf llava:13b

# Salvar resultado
python scripts/test_llm_vision.py fixtures/reg_interno.pdf > resultado_vision.txt
```

## O que o script faz?

1. **Converte PDF → Imagem** (primeira página, 150 DPI)
2. **Codifica para base64** (formato aceito pelos modelos Vision)
3. **Envia para LLM Vision** com prompt específico por tipo de documento
4. **Parseia resposta JSON** (se o modelo retornar JSON válido)
5. **Exibe resultados** formatados

## Resultado Esperado

### Para Regimento Interno

```json
{
  "tipo_documento": "regimento_interno",
  "nome_condominio": "Residencial Porto Ludovico",
  "artigos": [
    {
      "numero": "15",
      "assunto": "obras e reformas",
      "conteudo": "São permitidas obras...",
      "horarios": "segunda a sábado, 8h-18h"
    },
    {
      "numero": "23",
      "assunto": "animais de estimação",
      "conteudo": "É permitida a criação...",
      "restricoes": ["áreas comuns apenas com guia"]
    }
  ],
  "capitulos": [
    "CAPÍTULO I - DAS DISPOSIÇÕES GERAIS",
    "CAPÍTULO II - DO USO DAS ÁREAS COMUNS",
    "CAPÍTULO III - DAS OBRAS E REFORMAS"
  ],
  "multas": [
    {"tipo": "obras fora de horário", "valor": "R$ 500,00"},
    {"tipo": "uso indevido de áreas comuns", "valor": "R$ 300,00"}
  ]
}
```

### Para Conta de Energia

```json
{
  "tipo_documento": "conta_energia",
  "fornecedor": "ENEL Distribuidora",
  "cliente": "João da Silva",
  "instalacao": "123456789",
  "consumo_kwh": 320,
  "vencimento": "2026-08-15",
  "valor_total": 287.43,
  "historico": [
    {"mes": "jan/26", "consumo": 310},
    {"mes": "fev/26", "consumo": 340},
    {"mes": "mar/26", "consumo": 380}
  ]
}
```

## Comparação com LayoutLMv3

| Aspecto | **LLM Vision** | **LayoutLMv3** |
|---------|----------------|----------------|
| **Setup** | Baixar modelo (10min) | Anotar + treinar (10-20h) |
| **Novos tipos** | Mudar prompt (5min) | Anotar + retreinar (10h) |
| **Precisão** | 85-90% (zero-shot) | 95%+ (após fine-tuning) |
| **Latência** | 3-10s (depende do modelo) | 2-3s (após treino) |
| **Manutenção** | Zero (ajustar prompt) | Retreinar quando layout muda |
| **Custo** | Alto (inferência de LLM grande) | Baixo (modelo pequeno) |

## Performance Esperada

### minicpm-v:8b (8B params)

```
Hardware: CPU (16 cores)
- Conversão PDF: ~0.5s
- Inferência: ~15-30s
- Total: ~16-31s

Hardware: GPU T4
- Conversão PDF: ~0.5s
- Inferência: ~3-5s
- Total: ~3.5-5.5s
```

### llava:13b (13B params)

```
Hardware: CPU (16 cores)
- Muito lento (~60s+)

Hardware: GPU T4
- Conversão PDF: ~0.5s
- Inferência: ~5-8s
- Total: ~5.5-8.5s
```

## Modelos Recomendados

### 1. minicpm-v:8b ⭐ (Recomendado)

```bash
ollama pull minicpm-v:8b
```

**Vantagens:**
- ✅ Multilingual (português nativo)
- ✅ 8B params (rápido)
- ✅ Bom desempenho em documentos
- ✅ Suporta alta resolução

**Desvantagens:**
- ⚠️ Pode alucinar em documentos muito complexos

### 2. llava:13b

```bash
ollama pull llava:13b
```

**Vantagens:**
- ✅ Muito testado
- ✅ Boa precisão

**Desvantagens:**
- ❌ Inglês principalmente
- ⚠️ Mais lento (13B)

### 3. bakllava

```bash
ollama pull bakllava
```

**Vantagens:**
- ✅ Multilingual

**Desvantagens:**
- ⚠️ Menos testado

## Troubleshooting

### Erro: "Model not found"

```bash
# Verificar modelos instalados
ollama list

# Instalar modelo
ollama pull minicpm-v:8b
```

### Erro: "Connection refused"

```bash
# Verificar se Ollama está rodando
ollama serve

# Ou verificar porta
curl http://localhost:11434/api/tags
```

### Resposta não é JSON válido

O modelo pode retornar texto natural em vez de JSON. Isso é normal em zero-shot.

**Soluções:**
1. Ajustar temperatura: `temperature=0.0` (mais determinístico)
2. Melhorar prompt: adicionar "IMPORTANTE: retorne APENAS JSON, sem texto adicional"
3. Few-shot: incluir exemplos no prompt

### Resultado impreciso

**Melhorias possíveis:**
1. Aumentar DPI da imagem (150 → 200)
2. Processar múltiplas páginas
3. Usar modelo maior (llava:34b)
4. Few-shot learning (incluir exemplos)

## Próximos Passos

### 1. Processar documento completo

```python
# Processar todas as páginas
for page in range(1, total_pages + 1):
    result = extract_page(pdf, page)
    # Combinar resultados
```

### 2. Integrar no pipeline

```python
# app/extraction/vision.py
class VisionExtractor:
    def extract(self, pdf_path: Path) -> StructuredDocument:
        # Implementação completa
        pass
```

### 3. Cache de resultados

```python
# Cachear extrações por SHA256 do documento
@cache_by_hash
def extract_with_vision(pdf_path):
    # ...
```

### 4. Fallback strategy

```python
# Usar Vision para documentos complexos, PyMuPDF para simples
if is_complex_document(pdf):
    return vision_extractor.extract(pdf)
else:
    return text_extractor.extract(pdf)
```

## Quando usar LLM Vision vs LayoutLMv3?

### Use **LLM Vision** se:

- ✅ Variedade alta de tipos de documento
- ✅ Layouts mudam frequentemente
- ✅ Poucos exemplos de cada tipo
- ✅ Prototipagem rápida
- ✅ Você já tem Ollama rodando

### Use **LayoutLMv3** se:

- ✅ Volume alto de um tipo específico
- ✅ Precisa de latência mínima
- ✅ Tem 100+ exemplos anotados
- ✅ Layout é estável
- ✅ Orçamento apertado (inferência)

### Use **Híbrido** (recomendado):

```python
if has_finetuned_model(document_type):
    return layoutlmv3_extractor.extract(pdf)
else:
    return llm_vision_extractor.extract(pdf)
```

- Começa com Vision (funciona para tudo)
- Evolui para modelos fine-tunados (melhor performance)
- Mantém Vision como fallback
