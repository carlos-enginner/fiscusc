#!/usr/bin/env python3
"""
Script de teste de LLM Vision para extração estruturada de documentos.

Testa a capacidade de um LLM com Vision de extrair dados estruturados
de documentos PDF sem necessidade de fine-tuning.

Uso:
    python scripts/test_llm_vision.py <path_to_pdf>
"""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image
    from langchain_community.llms import Ollama
except ImportError as e:
    print(f"❌ Dependência faltando: {e}")
    print("\nInstale as dependências:")
    print("  pip install pdf2image pillow langchain-community")
    print("  sudo apt-get install poppler-utils  # Ubuntu/Debian")
    sys.exit(1)


def pdf_to_base64_image(pdf_path: Path, page: int = 1, dpi: int = 150) -> str:
    """
    Converte uma página de PDF para imagem base64.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        page: Número da página (1-indexed)
        dpi: Resolução da conversão
        
    Returns:
        String base64 da imagem
    """
    print(f"📄 Convertendo página {page} para imagem (DPI: {dpi})...")
    
    images = convert_from_path(
        pdf_path,
        first_page=page,
        last_page=page,
        dpi=dpi
    )
    
    if not images:
        raise ValueError(f"Não foi possível converter a página {page}")
    
    image = images[0]
    print(f"   ✓ Imagem: {image.size[0]}x{image.size[1]} pixels")
    
    # Converter para base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return img_base64


def extract_with_vision(
    image_base64: str,
    model: str = "minicpm-v:8b",
    document_type: str = "regimento"
) -> dict:
    """
    Extrai dados estruturados de um documento usando LLM Vision.
    
    Args:
        image_base64: Imagem do documento em base64
        model: Modelo Ollama com Vision a usar
        document_type: Tipo de documento para contexto
        
    Returns:
        Dicionário com dados extraídos
    """
    print(f"🤖 Processando com {model}...")
    
    # Prompts específicos por tipo de documento
    prompts = {
        "regimento": """
Analise este regimento interno de condomínio e extraia as seguintes informações:

1. Nome do condomínio
2. Principais artigos sobre:
   - Horários permitidos para obras/reformas
   - Regras de uso de áreas comuns
   - Multas e penalidades
   - Animais de estimação
3. Estrutura hierárquica (capítulos/seções principais)

Retorne APENAS um JSON válido no formato:
{
  "tipo_documento": "regimento_interno",
  "nome_condominio": "...",
  "artigos": [
    {
      "numero": "15",
      "assunto": "obras e reformas",
      "conteudo": "texto completo do artigo",
      "horarios": "segunda a sábado, 8h-18h"
    }
  ],
  "capitulos": ["Capítulo I - ...", "Capítulo II - ..."],
  "multas": [
    {"tipo": "...", "valor": "..."}
  ]
}
""",
        "conta_energia": """
Extraia os seguintes dados desta conta de energia elétrica:

- Fornecedor/Distribuidora
- Cliente
- Instalação/UC
- Consumo (kWh)
- Vencimento
- Valor total
- Histórico de consumo (se visível)

Retorne JSON válido.
""",
        "contrato": """
Extraia as seguintes informações deste contrato:

- Partes (contratante e contratado)
- Objeto do contrato
- Vigência
- Valor mensal
- Condições de rescisão
- Multas

Retorne JSON válido.
"""
    }
    
    prompt = prompts.get(document_type, prompts["regimento"])
    
    try:
        # Usar Ollama com modelo Vision
        llm = Ollama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.1  # Baixa temperatura para respostas mais determinísticas
        )
        
        # Formato de input para modelos Vision no Ollama
        # minicpm-v aceita imagens via base64
        input_text = f"[IMAGE_BASE64]{image_base64}[/IMAGE_BASE64]\n\n{prompt}"
        
        print("   Aguardando resposta do modelo...")
        response = llm.invoke(input_text)
        
        print("   ✓ Resposta recebida")
        return {"raw_response": response}
        
    except Exception as e:
        print(f"   ❌ Erro ao processar com LLM: {e}")
        return {"error": str(e)}


def parse_llm_response(response: str) -> dict:
    """
    Tenta parsear a resposta do LLM como JSON.
    
    Args:
        response: Resposta do LLM
        
    Returns:
        Dicionário parseado ou resposta raw
    """
    # Tentar extrair JSON da resposta
    try:
        # Procurar por blocos JSON na resposta
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        else:
            return {"raw_text": response}
    except json.JSONDecodeError:
        return {"raw_text": response}


def display_results(result: dict):
    """Exibe os resultados de forma formatada."""
    print("\n" + "=" * 60)
    print("📊 RESULTADO DA EXTRAÇÃO")
    print("=" * 60)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        return
    
    raw = result.get("raw_response", "")
    parsed = parse_llm_response(raw)
    
    if "raw_text" in parsed:
        print("\n📝 Resposta do LLM (texto):")
        print("-" * 60)
        print(parsed["raw_text"])
    else:
        print("\n📝 Dados Estruturados Extraídos:")
        print("-" * 60)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/test_llm_vision.py <path_to_pdf> [model]")
        print("\nExemplos:")
        print("  python scripts/test_llm_vision.py fixtures/reg_interno.pdf")
        print("  python scripts/test_llm_vision.py fixtures/reg_interno.pdf llava:13b")
        print("\nModelos Vision disponíveis no Ollama:")
        print("  - minicpm-v:8b (recomendado, multilingual)")
        print("  - llava:13b")
        print("  - llava:34b")
        print("  - bakllava")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "minicpm-v:8b"
    
    if not pdf_path.exists():
        print(f"❌ Arquivo não encontrado: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Arquivo deve ser PDF: {pdf_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🔍 TESTE DE LLM VISION - FISCUS-C")
    print("=" * 60)
    print(f"\n📁 Arquivo: {pdf_path}")
    print(f"🤖 Modelo: {model}\n")
    
    # Detectar tipo de documento pelo nome
    filename_lower = pdf_path.name.lower()
    if "reg" in filename_lower or "regimento" in filename_lower:
        doc_type = "regimento"
    elif "energia" in filename_lower or "light" in filename_lower:
        doc_type = "conta_energia"
    elif "contrato" in filename_lower:
        doc_type = "contrato"
    else:
        doc_type = "regimento"  # default
    
    print(f"📋 Tipo detectado: {doc_type}\n")
    
    try:
        # 1. Converter PDF para imagem
        image_base64 = pdf_to_base64_image(pdf_path)
        
        # 2. Extrair com Vision
        result = extract_with_vision(image_base64, model, doc_type)
        
        # 3. Exibir resultados
        display_results(result)
        
        print("\n✅ Teste concluído!")
        print("\n" + "=" * 60)
        print("💡 PRÓXIMOS PASSOS")
        print("=" * 60)
        print("""
1. Se o modelo não está instalado:
   ollama pull minicpm-v:8b

2. Para testar outros modelos:
   python scripts/test_llm_vision.py fixtures/reg_interno.pdf llava:13b

3. Para processar todas as páginas:
   # (implementar depois) python scripts/extract_full_document.py

4. Integrar no pipeline de ingestão:
   # Adicionar VisionExtractor à app/extraction/
""")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
