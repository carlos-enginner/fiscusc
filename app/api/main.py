"""Aplicação FastAPI principal do Fiscus-C."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query

app = FastAPI(
    title="Fiscus-C API",
    description="API para gestão inteligente de condomínios com IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
