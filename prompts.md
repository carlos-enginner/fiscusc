Prompt 1

Você é um engenheiro de software sênior especializado em Python, RAG, LLMs locais e arquitetura de sistemas.

Quero construir um laboratório de RAG para um futuro produto de consulta de documentos de condomínios.

OBJETIVO

Construir uma aplicação local capaz de:

1. Receber um PDF de um Regimento Interno de condomínio.
2. Extrair o texto mantendo o número da página.
3. Dividir o conteúdo em chunks semanticamente coerentes.
4. Gerar embeddings localmente usando Qwen3-Embedding-0.6B.
5. Armazenar os embeddings no PostgreSQL usando pgvector.
6. Receber uma pergunta em português.
7. Fazer busca semântica no pgvector.
8. Recuperar os chunks mais relevantes.
9. Enviar os chunks recuperados para um LLM local Qwen3-8B.
10. Gerar uma resposta baseada exclusivamente nos trechos recuperados.
11. Informar na resposta quais páginas/documentos foram utilizados como fonte.
12. Se não houver evidência suficiente, o sistema deve informar que não encontrou informação suficiente, sem inventar uma resposta.

IMPORTANTE

Não implementar ainda:
- WhatsApp
- MCP
- autenticação
- multi-tenancy
- AWS
- S3
- frontend
- produção

Este é um laboratório local para validar o RAG.

STACK

- Python 3.12+
- FastAPI
- LangChain
- PostgreSQL
- pgvector
- Docker Compose
- Qwen3-Embedding-0.6B para embeddings
- Qwen3-8B para geração
- PyMuPDF ou equivalente para extração de PDF

MODELOS LOCAIS

A aplicação deve permitir executar os modelos localmente.

Prefira Ollama como runtime local, mas isole a integração com o modelo em uma camada própria para que futuramente seja possível trocar Ollama por outra implementação.

MODELO DE EMBEDDING

Qwen3-Embedding-0.6B.

MODELO DE GERAÇÃO

Qwen3-8B.

BANCO

Utilizar PostgreSQL com extensão pgvector.

Criar uma tabela semelhante a:

documents
- id
- filename
- document_type
- version
- created_at
- sha256

document_chunks
- id
- document_id
- page
- section
- chunk_index
- content
- embedding
- metadata
- created_at

O embedding deve utilizar o tipo vector apropriado para a dimensão retornada pelo modelo.

INGESTÃO

Criar um comando/script:

python -m app.ingestion.ingest documents/regimento.pdf

O processo deve:

1. Validar se o arquivo existe.
2. Calcular SHA-256.
3. Extrair texto página por página.
4. Preservar:
   - página
   - título/seção quando possível
   - texto original
5. Fazer chunking.
6. Gerar embeddings.
7. Persistir os chunks no PostgreSQL.
8. Evitar duplicar documentos pelo SHA-256.
9. Exibir progresso no terminal.
10. Informar quantos:
   - documentos
   - páginas
   - chunks
   foram processados.

CHUNKING

Não faça chunking simplesmente por número fixo de caracteres.

Tente respeitar:
- títulos
- capítulos
- artigos
- parágrafos
- seções

Utilize overlap razoável.

Cada chunk deve manter metadata suficiente para rastrear sua origem.

Exemplo:

{
  "document_id": 1,
  "page": 18,
  "section": "5.1 - Horários",
  "chunk_index": 12
}

RETRIEVAL

Criar uma função:

search_documents(query, top_k=5)

Ela deve:

1. Gerar embedding da pergunta.
2. Executar busca por similaridade no pgvector.
3. Filtrar pelo tipo de documento quando solicitado.
4. Retornar os chunks mais relevantes.
5. Retornar score de similaridade.
6. Retornar página e seção.

IMPORTANTE:

O condomínio futuramente terá vários documentos.

Portanto, mesmo neste laboratório, desenhe o código de forma que futuramente seja possível filtrar por:

condominium_id

Não precisa implementar multi-tenancy agora, mas não acople o código a um único condomínio.

GERAÇÃO

Criar uma função:

answer_question(query)

O prompt enviado ao Qwen3-8B deve seguir o princípio:

- responder somente com base no contexto recuperado;
- não inventar informações;
- não usar conhecimento externo;
- se o contexto não responder à pergunta, informar claramente;
- citar documento e página;
- diferenciar informação encontrada de inferência.

Exemplo:

PERGUNTA:
"Posso fazer reforma no sábado?"

CONTEXTO:
[Regimento Interno — página 18]
"..."

RESPOSTA ESPERADA:

"Segundo o Regimento Interno, reformas são permitidas ..."

"Fonte: Regimento Interno, página 18."

Se não houver informação:

"Não encontrei informação suficiente no Regimento Interno disponível para responder com segurança."

API

Criar uma API FastAPI:

POST /documents/ingest

POST /query

GET /documents

GET /documents/{id}

POST /search

POST /health

Exemplo:

POST /query

{
  "question": "Qual o horário permitido para obras?"
}

Resposta:

{
  "answer": "...",
  "sources": [
    {
      "document": "regimento.pdf",
      "page": 18,
      "section": "5.1 - Horários",
      "score": 0.91
    }
  ]
}

OBSERVABILIDADE

Adicionar logs claros para:

- ingestão
- quantidade de chunks
- tempo de embedding
- tempo de retrieval
- quantidade de chunks recuperados
- tempo de geração
- modelo utilizado

Não registrar conteúdo sensível desnecessariamente.

TESTES

Criar testes automatizados para:

1. extração do PDF;
2. chunking;
3. geração/validação de metadata;
4. persistência;
5. retrieval;
6. resposta sem contexto suficiente;
7. resposta com evidências;
8. não duplicação por SHA-256.

CRIAR DOCUMENTAÇÃO

Criar README.md contendo:

1. Pré-requisitos.
2. Instalação.
3. Como iniciar PostgreSQL + pgvector.
4. Como instalar dependências.
5. Como instalar/iniciar Ollama.
6. Como baixar os modelos.
7. Como colocar o PDF em documents/.
8. Como executar a ingestão.
9. Como iniciar FastAPI.
10. Como fazer uma pergunta.
11. Exemplos de curl.
12. Troubleshooting.
13. Explicação da arquitetura.

DOCKER

Criar docker-compose.yml somente para infraestrutura necessária.

Inicialmente:

PostgreSQL + pgvector

Não containerizar o Ollama se isso dificultar o desenvolvimento local. Prefira permitir que ele rode diretamente na máquina.

ESTRUTURA

Sugestão:

app/
  api/
  ingestion/
  rag/
  embeddings/
  llm/
  database/
  models/
  services/
  config/

tests/

documents/

scripts/

docker-compose.yml

requirements.txt

.env.example

README.md

ARQUITETURA

Mantenha separadas estas responsabilidades:

PDF extraction
Chunking
Embedding
Vector storage
Retrieval
LLM generation
API

Não coloque toda a lógica em um único arquivo.

FUTURO

A arquitetura deverá permitir posteriormente adicionar:

- MCP Server
- WhatsApp
- Manual do Usuário
- Convenção do Condomínio
- Faturas
- Receitas
- Despesas estruturadas
- Motor de contraprova
- múltiplos condomínios
- S3
- autenticação

Mas NÃO implemente essas funcionalidades agora.

PRINCÍPIO MAIS IMPORTANTE

O sistema deve ser orientado por evidências.

Toda resposta do RAG deve conseguir apontar para os chunks/documentos/páginas que sustentam a resposta.

Antes de implementar, analise os requisitos, proponha uma estrutura de projeto e explique brevemente as decisões arquiteturais.

Depois implemente a solução completa.

Ao final:

1. execute os testes;
2. valide o docker-compose;
3. valide a conexão com PostgreSQL/pgvector;
4. valide a ingestão;
5. valide uma consulta RAG;
6. corrija eventuais erros;
7. apresente os comandos exatos para executar o projeto localmente.

Não faça alterações fora do diretório do projeto.
Não instale dependências globais sem necessidade.