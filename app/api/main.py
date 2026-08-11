"""Aplicação FastAPI principal do Fiscus-C."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query
from app.core.config import get_settings
from app.core.database import get_engine
from app.embeddings.factory import create_embedding_provider
from app.embeddings.validator import (
    EmbeddingDimensionMismatchError,
    validate_embedding_dimensions,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # Startup
    settings = get_settings()
    
    try:
        # Criar provider de embeddings
        provider = create_embedding_provider(settings)
        provider_name = settings.embedding_provider
        
        # Logar informações do provider
        logger.info(
            f"Embedding provider: {provider_name} "
            f"(model: {settings.fastembed_model if provider_name == 'fastembed' else settings.embedding_model}, "
            f"dimensions: {provider.dimensions})"
        )
        
        # Warning para Ollama sugerindo FastEmbed
        if provider_name == "ollama":
            logger.warning(
                "Usando Ollama para embeddings. "
                "Considere usar FastEmbed (EMBEDDING_PROVIDER=fastembed) "
                "para melhor performance em ingestão de documentos."
            )
        
        # Validar dimensões contra o banco
        engine = get_engine()
        validate_embedding_dimensions(provider, engine)
        logger.info("Validação de dimensões de embeddings: OK")
        
    except EmbeddingDimensionMismatchError as e:
        logger.critical(f"Falha na validação de embeddings: {e}")
        raise RuntimeError(
            f"API não pode iniciar: mismatch de dimensões de embeddings. {e}"
        ) from e
    
    yield  # Aplicação rodando
    
    # Shutdown (cleanup se necessário)


app = FastAPI(
    title="Fiscus-C API",
    description="API para gestão inteligente de condomínios com IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])


@app.get("/")
async def root():
    """Redirect info para docs."""
    return {"message": "Fiscus-C API - acesse /docs para documentação"}
