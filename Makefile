# Fiscus-C Makefile
# Uso: make <target>

.PHONY: help install setup start stop restart logs status \
        migrate migrate-down db-reset \
        test test-unit test-integration test-e2e test-e2e-full test-all \
        ingest query api

# Detecta python/pip
PYTHON    := .venv/bin/python
PIP       := .venv/bin/pip
PYTEST    := .venv/bin/pytest
UVICORN   := .venv/bin/uvicorn
ALEMBIC   := .venv/bin/alembic
CLI       := $(PYTHON) -m app.cli

# Variáveis de ambiente
DATABASE_URL ?= postgresql://fiscusc:fiscusc@localhost:5432/fiscusc
export DATABASE_URL

# Cores
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Mostra este menu de ajuda
	@echo ""
	@echo "$(CYAN)Fiscus-C — Comandos disponíveis$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ─── Instalação ───────────────────────────────────────────────────────────────

install: ## Cria venv e instala dependências
	python3 -m venv .venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependências instaladas$(RESET)"

setup: install start migrate ollama-pull ## Setup completo (install + banco + migrations + modelos Ollama)
	@echo "$(GREEN)✓ Projeto pronto. Rode: make ingest && make api$(RESET)"

# ─── Banco de Dados ───────────────────────────────────────────────────────────

start: ## Sobe o PostgreSQL via Docker Compose
	docker compose up -d
	@echo -n "Aguardando PostgreSQL..."
	@until docker compose exec postgres pg_isready -U fiscusc -q; do sleep 1; echo -n "."; done
	@echo " $(GREEN)OK$(RESET)"

stop: ## Para o PostgreSQL
	docker compose down
	@echo "$(GREEN)✓ Banco parado$(RESET)"

restart: stop start ## Reinicia o PostgreSQL

logs: ## Exibe logs do PostgreSQL
	docker compose logs -f postgres

migrate: ## Aplica migrations pendentes
	$(ALEMBIC) upgrade head
	@echo "$(GREEN)✓ Migrations aplicadas$(RESET)"

migrate-down: ## Desfaz todas as migrations (CUIDADO: apaga dados)
	@echo "⚠ Isso apaga todos os dados. Confirma? [y/N] " && read ans && [ "$$ans" = y ]
	$(ALEMBIC) downgrade base
	@echo "$(GREEN)✓ Migrations revertidas$(RESET)"

db-reset: migrate-down migrate ## Recria o schema do zero (CUIDADO: apaga dados)

# ─── API ──────────────────────────────────────────────────────────────────────

api: ## Inicia a API FastAPI (hot-reload)
	$(UVICORN) app.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Inicia a API FastAPI (produção, sem reload)
	$(UVICORN) app.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ─── Ollama ───────────────────────────────────────────────────────────────────

ollama-pull: ## Baixa os modelos necessários (qwen3:8b + qwen3-embedding:0.6b)
	ollama pull qwen3:8b
	ollama pull qwen3-embedding:0.6b
	@echo "$(GREEN)✓ Modelos prontos$(RESET)"

ollama-list: ## Lista modelos instalados no Ollama
	ollama list

# ─── Gemini ───────────────────────────────────────────────────────────────────

gemini: ## Ativa Gemini como LLM provider (requer GOOGLE_API_KEY no .env)
	@grep -q "GOOGLE_API_KEY" .env 2>/dev/null || { echo "⚠ Adicione GOOGLE_API_KEY=sua_chave ao .env"; exit 1; }
	@sed -i 's/^LLM_PROVIDER=.*/LLM_PROVIDER=gemini/' .env
	@sed -i 's/^LLM_MODEL=.*/LLM_MODEL=gemini-2.0-flash/' .env
	@echo "$(GREEN)✓ Provider trocado para Gemini (gemini-2.0-flash)$(RESET)"
	@echo "$(GREEN)  Rode: make query Q=\"sua pergunta\"$(RESET)"

ollama-local: ## Volta para Ollama local como LLM provider
	@sed -i 's/^LLM_PROVIDER=.*/LLM_PROVIDER=ollama/' .env
	@sed -i 's/^LLM_MODEL=.*/LLM_MODEL=qwen3:1.7b/' .env
	@echo "$(GREEN)✓ Provider voltou para Ollama local (qwen3:1.7b)$(RESET)"

# ─── CLI ──────────────────────────────────────────────────────────────────────

status: ## Status do sistema (banco, Ollama, documentos)
	$(CLI) status

ingest: ## Ingere o PDF de exemplo (fixtures/reg_interno.pdf)
	$(CLI) ingest fixtures/reg_interno.pdf --type regimento

ingest-file: ## Ingere um PDF personalizado: make ingest-file FILE=path/to/file.pdf TYPE=regimento
	$(CLI) ingest $(FILE) --type $(or $(TYPE),regimento)

query: ## Faz uma pergunta: make query Q="Qual o horário para obras?"
	$(CLI) query "$(or $(Q),Qual o horário permitido para obras?)"

# ─── Testes ───────────────────────────────────────────────────────────────────

test: test-unit ## Alias para testes unitários

test-unit: ## Testes unitários (sem dependências externas)
	$(PYTEST) tests/unit/ -v -m "not integration"

test-integration: ## Testes de integração (requerem PostgreSQL rodando)
	$(PYTEST) tests/integration/ -v -m integration

test-e2e: ## Testes E2E com mocks (sem Ollama)
	$(PYTEST) tests/e2e/ -v -m "not e2e"

test-e2e-full: ## Testes E2E completos (requerem PostgreSQL + Ollama)
	$(PYTEST) tests/e2e/ -v -m e2e

test-all: ## Todos os testes (unit + integration + e2e com mocks)
	$(PYTEST) tests/unit/ tests/integration/ tests/e2e/ -v -m "not e2e"

test-coverage: ## Testes com relatório de cobertura
	$(PYTEST) tests/unit/ --cov=app --cov-report=term-missing --cov-report=html -m "not integration"
	@echo "$(GREEN)Relatório em htmlcov/index.html$(RESET)"

# ─── Desenvolvimento ──────────────────────────────────────────────────────────

lint: ## Roda o linter (ruff)
	.venv/bin/ruff check app/ tests/

fmt: ## Formata o código (ruff)
	.venv/bin/ruff format app/ tests/

# ─── Demo ─────────────────────────────────────────────────────────────────────

demo: ## Fluxo completo de demo: ingest + 3 queries de exemplo
	@echo "$(CYAN)=== DEMO Fiscus-C ===$(RESET)\n"
	@echo "$(CYAN)1. Ingerindo Regimento Interno...$(RESET)"
	$(CLI) ingest fixtures/reg_interno.pdf --type regimento
	@echo ""
	@echo "$(CYAN)2. Pergunta sobre regras (Agente Docs)...$(RESET)"
	$(CLI) query "Qual o horário permitido para obras?"
	@echo ""
	@echo "$(CYAN)3. Pergunta sobre finanças (Agente Finance)...$(RESET)"
	$(CLI) query "Quanto foi a despesa com energia em julho?"
	@echo ""
	@echo "$(CYAN)4. Pergunta mista (dois agentes)...$(RESET)"
	$(CLI) query "O valor da taxa de mudança cobrada está de acordo com o regimento?"
	@echo ""
	@echo "$(GREEN)✓ Demo concluída$(RESET)"
