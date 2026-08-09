"""Health check endpoint."""
from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.schemas import DependencyStatus, HealthResponse

router = APIRouter()

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saúde da aplicação e dependências."""
    deps = await _check_dependencies()
    overall = "healthy" if deps.database == "healthy" else "unhealthy"

    return HealthResponse(
        status=overall,
        version=VERSION,
        dependencies=deps,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def _check_dependencies() -> DependencyStatus:
    """Verifica cada dependência e retorna status."""
    db_status = "unhealthy"
    ollama_status = "unhealthy"
    error = None

    # Verificar banco
    try:
        from app.core.database import check_db_connection

        if check_db_connection():
            db_status = "healthy"
    except Exception as e:
        error = str(e)

    # Verificar Ollama
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            from app.core.config import get_settings

            settings = get_settings()
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_status = "healthy"
    except Exception as e:
        if error is None:
            error = f"Ollama: {e}"

    return DependencyStatus(
        database=db_status,
        ollama=ollama_status,
        error=error,
    )
