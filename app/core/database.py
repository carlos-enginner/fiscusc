"""Configuração do banco de dados e dependency injection."""
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def get_engine():
    """Cria engine SQLAlchemy a partir das configurações."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,  # Verifica conexão antes de usar
        pool_size=5,
        max_overflow=10,
    )


def get_session_factory(engine=None):
    """Cria factory de sessões."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Instâncias globais (lazy)
_engine = None
_SessionFactory = None


def _get_or_create_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_or_create_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = get_session_factory(_get_or_create_engine())
    return _SessionFactory


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection para sessão do banco.

    Uso no FastAPI:
        @app.get("/endpoint")
        def endpoint(db: DbSession):
            ...
    """
    factory = _get_or_create_factory()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Type alias para usar como anotação no FastAPI
DbSession = Annotated[Session, Depends(get_db)]


def check_db_connection() -> bool:
    """Verifica se o banco de dados está acessível."""
    try:
        engine = _get_or_create_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
