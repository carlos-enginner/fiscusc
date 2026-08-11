#!/usr/bin/env python3
"""
Benchmark script para avaliar providers de embeddings no Fiscus-C.

Mede performance de ingestão e qualidade de busca usando golden set.

Uso:
    python -m scripts.benchmark_embeddings --provider ollama
    python -m scripts.benchmark_embeddings --provider fastembed --model intfloat/multilingual-e5-small
    python -m scripts.benchmark_embeddings --skip-ingest  # Pular ingestão se já foi feita
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BenchmarkResult:
    """Resultado de um benchmark de embeddings."""

    provider: str
    model: str
    dimensions: int
    ingest_time_ms: float | None  # None se skip-ingest
    avg_search_time_ms: float
    hit_rate: float
    mrr: float
    precision_at_k: float
    k: int
    total_queries: int
    document_path: str
    chunks_count: int


# ============================================================================
# Provider Factory
# ============================================================================


def create_embedding_provider(
    provider_name: str,
    model: str | None = None,
) -> Any:  # EmbeddingsProvider
    """
    Cria provider de embeddings baseado nos parâmetros.

    Args:
        provider_name: "ollama" ou "fastembed"
        model: Nome do modelo (opcional, usa default da config)

    Returns:
        Instância de EmbeddingsProvider

    Raises:
        ImportError: Se dependências não estão instaladas
        ValueError: Se provider_name é inválido
    """
    provider_name = provider_name.lower()

    if provider_name == "ollama":
        try:
            from app.embeddings.service import OllamaEmbeddingsProvider

            return OllamaEmbeddingsProvider(model=model)
        except ImportError as e:
            raise ImportError(
                "Dependência 'ollama' não encontrada. Instale com: pip install ollama"
            ) from e

    elif provider_name == "fastembed":
        try:
            from app.embeddings.fastembed_provider import FastEmbedProvider

            return FastEmbedProvider(model=model)
        except ImportError as e:
            raise ImportError(
                "Dependência 'fastembed' não encontrada. Instale com: pip install fastembed"
            ) from e

    else:
        raise ValueError(f"Provider desconhecido: {provider_name}. Use 'ollama' ou 'fastembed'.")


def get_provider_model_name(provider: Any, provider_name: str) -> str:
    """Extrai o nome do modelo do provider."""
    if hasattr(provider, "_model_name"):
        return provider._model_name
    if hasattr(provider, "_model"):
        if isinstance(provider._model, str):
            return provider._model
        # Pode ser o objeto do modelo
        return getattr(provider._model, "model_name", str(type(provider._model).__name__))
    return f"{provider_name}-default"


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_ingest(
    document_path: Path,
    provider: Any,
    document_type: str = "regimento",
) -> tuple[float, int]:
    """
    Ingere documento medindo tempo.

    Args:
        document_path: Caminho para o PDF
        provider: EmbeddingsProvider
        document_type: Tipo do documento

    Returns:
        Tupla (tempo_ms, chunks_criados)
    """
    from app.core.database import get_engine, get_session_factory
    from app.embeddings.service import EmbeddingsService
    from app.rag.ingestion import DocumentIngestionService, ProgressCallbacks

    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()

    try:
        embeddings_service = EmbeddingsService(provider=provider)
        ingestion_service = DocumentIngestionService(
            embeddings_service=embeddings_service,
            db_session=db,
        )

        # Setup progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            tasks = {
                "extraction": progress.add_task("[cyan]Extraindo PDF...", total=None, visible=True),
                "chunking": progress.add_task("[cyan]Chunking...", total=None, visible=False),
                "embedding": progress.add_task(
                    "[cyan]Gerando embeddings...", total=None, visible=False
                ),
                "saving": progress.add_task(
                    "[cyan]Salvando no banco...", total=None, visible=False
                ),
            }

            def on_phase_start(phase: str):
                progress.update(tasks[phase], visible=True)

            def on_phase_end(phase: str):
                task = tasks[phase]
                total = progress.tasks[task].total
                if total:
                    progress.update(task, completed=total)
                else:
                    progress.update(task, total=1, completed=1)

            def on_extraction_progress(current: int, total: int):
                progress.update(
                    tasks["extraction"],
                    total=total,
                    completed=current,
                    description=f"[cyan]Extraindo PDF... ({total} páginas)",
                )

            def on_chunking_progress(current: int, total: int):
                progress.update(
                    tasks["chunking"],
                    total=total,
                    completed=current,
                    description=f"[cyan]Chunking... ({total} chunks)",
                )

            def on_embedding_progress(current: int, total: int):
                progress.update(
                    tasks["embedding"],
                    total=total,
                    completed=current,
                    description=f"[cyan]Gerando embeddings... {current}/{total}",
                )

            def on_saving_progress(current: int, total: int):
                progress.update(tasks["saving"], total=total, completed=current)

            callbacks = ProgressCallbacks(
                on_phase_start=on_phase_start,
                on_phase_end=on_phase_end,
                on_extraction_progress=on_extraction_progress,
                on_chunking_progress=on_chunking_progress,
                on_embedding_progress=on_embedding_progress,
                on_saving_progress=on_saving_progress,
            )

            start_time = time.perf_counter()
            result = ingestion_service.ingest(
                path=document_path,
                document_type=document_type,
                progress_callbacks=callbacks,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if result.already_existed:
                console.print("[yellow]⚠ Documento já existe (usando dados existentes)[/]")
                # Retornar tempo 0 pois não houve ingestão real
                return 0.0, result.chunks_created

            return elapsed_ms, result.chunks_created

    finally:
        db.close()


def run_evaluation(
    provider: Any,
    top_k: int = 5,
) -> tuple[float, Any]:  # (avg_search_time_ms, RetrievalMetrics)
    """
    Executa avaliação do golden set.

    Args:
        provider: EmbeddingsProvider
        top_k: Número de resultados por query

    Returns:
        Tupla (tempo_medio_ms, RetrievalMetrics)
    """
    from app.core.database import get_engine, get_session_factory
    from app.embeddings.service import EmbeddingsService
    from app.rag.retriever import DocumentRetriever
    from tests.evaluation.golden_set import GOLDEN_SET
    from tests.evaluation.metrics import evaluate_retrieval

    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()

    try:
        embeddings_service = EmbeddingsService(provider=provider)
        retriever = DocumentRetriever(
            embeddings_service=embeddings_service,
            db_session=db,
        )

        search_times: list[float] = []

        def timed_retriever(query: str) -> list[Any]:
            start = time.perf_counter()
            results = retriever.search(query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            search_times.append(elapsed_ms)
            return results

        # Rodar avaliação com progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Avaliando golden set..."),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Queries", total=len(GOLDEN_SET))

            def retriever_with_progress(query: str) -> list[Any]:
                result = timed_retriever(query)
                progress.advance(task)
                return result

            metrics = evaluate_retrieval(
                questions=GOLDEN_SET,
                retriever_fn=retriever_with_progress,
                k=top_k,
            )

        avg_search_time = sum(search_times) / len(search_times) if search_times else 0.0

        return avg_search_time, metrics

    finally:
        db.close()


def get_chunks_count() -> int:
    """Retorna número de chunks no banco."""
    from app.core.database import get_engine, get_session_factory
    from app.core.models import DocumentChunk

    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()

    try:
        return db.query(DocumentChunk).count()
    finally:
        db.close()


def clear_documents() -> None:
    """Remove todos os documentos e chunks do banco (para benchmark limpo)."""
    from sqlalchemy import text

    from app.core.database import get_engine, get_session_factory

    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()

    try:
        db.execute(text("DELETE FROM document_chunks"))
        db.execute(text("DELETE FROM documents"))
        db.commit()
    finally:
        db.close()


def run_benchmark(
    provider_name: str,
    model: str | None = None,
    document_path: Path | None = None,
    top_k: int = 5,
    skip_ingest: bool = False,
) -> BenchmarkResult:
    """
    Executa benchmark completo.

    Args:
        provider_name: "ollama" ou "fastembed"
        model: Nome do modelo (opcional)
        document_path: Caminho do PDF para ingerir
        top_k: Resultados por query
        skip_ingest: Pular ingestão

    Returns:
        BenchmarkResult com todas as métricas
    """
    from tests.evaluation.golden_set import GOLDEN_SET

    # Criar provider
    console.print(f"\n[bold]Criando provider [cyan]{provider_name}[/]...[/]")
    provider = create_embedding_provider(provider_name, model)
    model_name = get_provider_model_name(provider, provider_name)
    console.print(f"  Modelo: [green]{model_name}[/]")
    console.print(f"  Dimensões: [green]{provider.dimensions}[/]")

    # Ingestão
    ingest_time_ms: float | None = None
    chunks_count = 0

    if not skip_ingest:
        if document_path is None:
            document_path = Path("fixtures/reg_interno.pdf")

        if not document_path.exists():
            raise FileNotFoundError(f"Documento não encontrado: {document_path}")

        console.print(f"\n[bold]Ingerindo documento:[/] {document_path}")

        # Limpar documentos existentes para benchmark limpo
        console.print("[dim]Limpando documentos existentes...[/]")
        clear_documents()

        ingest_time_ms, chunks_count = run_ingest(document_path, provider)

        if ingest_time_ms > 0:
            console.print(
                f"[green]✓ Ingestão concluída:[/] {chunks_count} chunks em {ingest_time_ms / 1000:.1f}s"
            )
    else:
        console.print("\n[yellow]Pulando ingestão (--skip-ingest)[/]")
        chunks_count = get_chunks_count()
        if chunks_count == 0:
            console.print("[red]⚠ Nenhum chunk no banco! Execute sem --skip-ingest primeiro.[/]")
            raise ValueError("Banco vazio - execute ingestão primeiro")
        console.print(f"  Usando {chunks_count} chunks existentes")

    # Avaliação
    console.print(f"\n[bold]Avaliando retrieval ({len(GOLDEN_SET)} queries, top_k={top_k})...[/]")
    avg_search_time_ms, metrics = run_evaluation(provider, top_k)

    return BenchmarkResult(
        provider=provider_name,
        model=model_name,
        dimensions=provider.dimensions,
        ingest_time_ms=ingest_time_ms,
        avg_search_time_ms=avg_search_time_ms,
        hit_rate=metrics.hit_rate,
        mrr=metrics.mrr,
        precision_at_k=metrics.precision_at_k,
        k=metrics.k,
        total_queries=len(GOLDEN_SET),
        document_path=str(document_path) if document_path else "N/A",
        chunks_count=chunks_count,
    )


# ============================================================================
# Display Functions
# ============================================================================


def display_result(result: BenchmarkResult) -> None:
    """Exibe resultado do benchmark em tabela formatada."""

    # Cores baseadas em thresholds
    def hit_rate_color(val: float) -> str:
        if val >= 0.8:
            return "green"
        elif val >= 0.5:
            return "yellow"
        return "red"

    def mrr_color(val: float) -> str:
        if val >= 0.7:
            return "green"
        elif val >= 0.4:
            return "yellow"
        return "red"

    def time_color(val: float) -> str:
        if val < 100:
            return "green"
        elif val < 500:
            return "yellow"
        return "red"

    # Tabela de resultados
    table = Table(title="Resultado do Benchmark", show_header=True, header_style="bold")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right")

    table.add_row("Provider", result.provider)
    table.add_row("Modelo", result.model)
    table.add_row("Dimensões", str(result.dimensions))
    table.add_row("Chunks", str(result.chunks_count))
    table.add_row("", "")  # Separator

    # Tempo de ingestão
    if result.ingest_time_ms is not None and result.ingest_time_ms > 0:
        ingest_secs = result.ingest_time_ms / 1000
        table.add_row("Tempo Ingestão", f"{ingest_secs:.1f}s")
    else:
        table.add_row("Tempo Ingestão", "[dim]N/A (skip-ingest)[/]")

    # Tempo de busca
    search_color = time_color(result.avg_search_time_ms)
    table.add_row(
        "Tempo Busca (avg)",
        f"[{search_color}]{result.avg_search_time_ms:.1f}ms[/]",
    )
    table.add_row("", "")  # Separator

    # Métricas de qualidade
    hr_color = hit_rate_color(result.hit_rate)
    table.add_row(
        f"Hit Rate @{result.k}",
        f"[{hr_color}]{result.hit_rate:.1%}[/]",
    )

    mrr_c = mrr_color(result.mrr)
    table.add_row("MRR", f"[{mrr_c}]{result.mrr:.3f}[/]")

    p_color = hit_rate_color(result.precision_at_k)
    table.add_row(
        f"Precision@{result.k}",
        f"[{p_color}]{result.precision_at_k:.1%}[/]",
    )

    console.print()
    console.print(table)

    # Panel de sumário
    summary = []
    if result.hit_rate >= 0.8:
        summary.append("[green]✓ Hit rate excelente (≥80%)[/]")
    elif result.hit_rate >= 0.5:
        summary.append("[yellow]⚠ Hit rate moderado (50-80%)[/]")
    else:
        summary.append("[red]✗ Hit rate baixo (<50%)[/]")

    if result.mrr >= 0.7:
        summary.append("[green]✓ MRR excelente (≥0.7)[/]")
    elif result.mrr >= 0.4:
        summary.append("[yellow]⚠ MRR moderado (0.4-0.7)[/]")
    else:
        summary.append("[red]✗ MRR baixo (<0.4)[/]")

    if result.avg_search_time_ms < 100:
        summary.append("[green]✓ Busca rápida (<100ms)[/]")
    elif result.avg_search_time_ms < 500:
        summary.append("[yellow]⚠ Busca moderada (100-500ms)[/]")
    else:
        summary.append("[red]✗ Busca lenta (>500ms)[/]")

    console.print(Panel("\n".join(summary), title="Sumário", border_style="blue"))


def display_comparison_table(results: list[BenchmarkResult]) -> None:
    """Exibe tabela comparativa de múltiplos resultados."""
    table = Table(
        title="Comparação de Providers",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Modelo")
    table.add_column("Dims", justify="right")
    table.add_column("Tempo Ingestão", justify="right")
    table.add_column("Tempo Busca (ms)", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("MRR", justify="right")

    for r in results:
        # Cores
        hr_color = "green" if r.hit_rate >= 0.8 else "yellow" if r.hit_rate >= 0.5 else "red"
        mrr_color = "green" if r.mrr >= 0.7 else "yellow" if r.mrr >= 0.4 else "red"
        search_color = (
            "green"
            if r.avg_search_time_ms < 100
            else "yellow"
            if r.avg_search_time_ms < 500
            else "red"
        )

        ingest_str = (
            f"{r.ingest_time_ms / 1000:.1f}s"
            if r.ingest_time_ms and r.ingest_time_ms > 0
            else "[dim]N/A[/]"
        )

        table.add_row(
            r.model,
            str(r.dimensions),
            ingest_str,
            f"[{search_color}]{r.avg_search_time_ms:.1f}[/]",
            f"[{hr_color}]{r.hit_rate:.1%}[/]",
            f"[{mrr_color}]{r.mrr:.3f}[/]",
        )

    console.print()
    console.print(table)


# ============================================================================
# CLI
# ============================================================================


@click.command()
@click.option(
    "--provider",
    type=click.Choice(["ollama", "fastembed"], case_sensitive=False),
    default=None,
    help="Provider de embeddings (default: baseado em config)",
)
@click.option(
    "--model",
    default=None,
    help="Nome do modelo de embeddings (default: baseado em config)",
)
@click.option(
    "--document",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Caminho do PDF para ingerir (default: fixtures/reg_interno.pdf)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Número de resultados por query",
)
@click.option(
    "--skip-ingest",
    is_flag=True,
    default=False,
    help="Pular ingestão se já foi feita",
)
def main(
    provider: str | None,
    model: str | None,
    document: Path | None,
    top_k: int,
    skip_ingest: bool,
) -> None:
    """
    Benchmark de providers de embeddings para Fiscus-C.

    Mede tempo de ingestão, tempo de busca e qualidade de retrieval
    usando o golden set de avaliação.
    """
    console.print(
        Panel(
            "[bold blue]Fiscus-C Embedding Benchmark[/]\n"
            "Avaliação de performance e qualidade de retrieval",
            border_style="blue",
        )
    )

    try:
        # Determinar provider
        if provider is None:
            from app.core.config import get_settings

            settings = get_settings()
            provider = settings.embedding_provider
            console.print(f"[dim]Usando provider da config: {provider}[/]")

        # Determinar documento
        if document is None and not skip_ingest:
            document = Path("fixtures/reg_interno.pdf")

        # Executar benchmark
        result = run_benchmark(
            provider_name=provider,
            model=model,
            document_path=document,
            top_k=top_k,
            skip_ingest=skip_ingest,
        )

        # Exibir resultado
        display_result(result)

    except ImportError as e:
        console.print(f"[red]Erro de dependência:[/] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Arquivo não encontrado:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Erro:[/] {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
