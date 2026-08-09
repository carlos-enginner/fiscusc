"""Models SQLAlchemy para o Fiscus-C."""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DECIMAL,
    TEXT,
    VARCHAR,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Metadados dos documentos ingeridos (PDFs)."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(VARCHAR(255), nullable=False)
    document_type = Column(VARCHAR(50), nullable=False)  # regimento, convencao, manual, fatura
    version = Column(VARCHAR(50), nullable=True)
    sha256 = Column(VARCHAR(64), nullable=False, unique=True)
    page_count = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_type", "document_type"),
        Index("idx_documents_sha256", "sha256"),
        Index("idx_documents_created", "created_at"),
    )


class DocumentChunk(Base):
    """Chunks de texto com embeddings."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page = Column(Integer, nullable=False)
    section = Column(VARCHAR(255), nullable=True)
    chapter = Column(VARCHAR(255), nullable=True)
    article = Column(VARCHAR(100), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(TEXT, nullable=False)
    content_length = Column(Integer, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # Qwen3-Embedding-0.6B
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_page", "page"),
        Index("idx_chunks_section", "section"),
        # Índice HNSW criado via migration (não suportado diretamente pelo SQLAlchemy)
    )


class Fatura(Base):
    """Faturas extraídas de PDFs."""

    __tablename__ = "faturas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)

    # Identificação
    condominio_nome = Column(VARCHAR(255), nullable=True)
    condominio_cnpj = Column(VARCHAR(20), nullable=True)
    unidade_bloco = Column(VARCHAR(10), nullable=True)
    unidade_apartamento = Column(VARCHAR(10), nullable=True)
    proprietario = Column(VARCHAR(255), nullable=True)

    # Período
    mes_referencia = Column(VARCHAR(20), nullable=True)
    data_vencimento = Column(Date, nullable=True)
    data_emissao = Column(Date, nullable=True)

    # Valores
    total_cobranca = Column(DECIMAL(12, 2), nullable=True)
    codigo_barras = Column(VARCHAR(100), nullable=True)

    # Metadados de extração
    modelo_provider = Column(VARCHAR(50), nullable=True)
    modelo_nome = Column(VARCHAR(100), nullable=True)
    tempo_extracao_seg = Column(DECIMAL(8, 2), nullable=True)
    health_check_score = Column(Integer, nullable=True)

    # Dados brutos
    raw_data = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    itens = relationship("FaturaItem", back_populates="fatura", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_faturas_unidade", "unidade_bloco", "unidade_apartamento"),
        Index("idx_faturas_vencimento", "data_vencimento"),
        Index("idx_faturas_referencia", "mes_referencia"),
    )


class FaturaItem(Base):
    """Itens individuais de uma fatura."""

    __tablename__ = "fatura_itens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fatura_id = Column(UUID(as_uuid=True), ForeignKey("faturas.id", ondelete="CASCADE"), nullable=False)

    secao = Column(VARCHAR(50), nullable=False)  # cobranca, despesas, receitas
    grupo = Column(VARCHAR(100), nullable=True)
    descricao = Column(VARCHAR(255), nullable=False)
    valor = Column(DECIMAL(12, 2), nullable=False)
    categoria = Column(VARCHAR(50), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    fatura = relationship("Fatura", back_populates="itens")

    __table_args__ = (
        Index("idx_fatura_itens_fatura", "fatura_id"),
        Index("idx_fatura_itens_secao", "secao"),
        Index("idx_fatura_itens_categoria", "categoria"),
    )


class QueryLog(Base):
    """Log de queries para observabilidade."""

    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(TEXT, nullable=False)
    agents_used = Column(ARRAY(VARCHAR(50)), nullable=True)
    response = Column(TEXT, nullable=True)
    sources = Column(JSONB, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_query_logs_created", "created_at"),)
