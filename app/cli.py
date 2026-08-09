"""CLI do Fiscus-C — comandos de linha de comando."""
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Imports no topo para permitir mock nos testes
from app.agents.docs.agent import DocsAgent
from app.agents.finance.agent import FinanceAgent
from app.core.database import check_db_connection, get_engine, get_session_factory
from app.embeddings.service import EmbeddingsService
from app.orchestrator.classifier import QueryClassifier
from app.orchestrator.workflow import FiscusWorkflow
from app.rag.ingestion import DocumentIngestionService
from app.rag.retriever import DocumentRetriever

console = Console()


@click.group()
def cli():
    """Fiscus-C — Sistema inteligente de gestão de condomínios."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--type", "document_type", default="regimento",
              type=click.Choice(["regimento", "convencao", "manual", "fatura"]),
              show_default=True, help="Tipo do documento")
@click.option("--version", default=None, help="Versão do documento")
def ingest(pdf_path: Path, document_type: str, version: str | None):
    """Ingere um documento PDF no sistema."""
    console.print(f"[bold blue]Ingerindo[/] {pdf_path.name} como [yellow]{document_type}[/]...")

    try:
        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()

        embeddings = EmbeddingsService()
        svc = DocumentIngestionService(embeddings_service=embeddings, db_session=db)

        from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[chunk]}"),
            console=console,
        ) as progress:
            task = progress.add_task("Gerando embeddings...", total=None, chunk="")

            def on_progress(current: int, total: int, chunk_text: str):
                progress.update(task, total=total, completed=current, chunk=chunk_text)

            result = svc.ingest(
                path=pdf_path,
                document_type=document_type,
                version=version,
                on_progress=on_progress,
            )

        db.close()

        if result.already_existed:
            console.print(f"[yellow]⚠ Documento já existe[/] (id: {result.document_id})")
        else:
            console.print(Panel(
                f"[green]✓ Documento ingerido com sucesso[/]\n"
                f"ID: {result.document_id}\n"
                f"Páginas: {result.pages}\n"
                f"Chunks criados: {result.chunks_created}\n"
                f"SHA256: {result.sha256[:16]}...",
                title="Ingestão concluída",
                border_style="green",
            ))

    except Exception as e:
        console.print(f"[red]✗ Erro:[/] {e}")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--agent", default=None,
              type=click.Choice(["docs", "finance"]),
              help="Forçar uso de agente específico")
def query(question: str, agent: str | None):
    """Faz uma pergunta ao sistema."""
    import time
    t_inicio = time.time()
    console.print(f"[bold blue]Pergunta:[/] {question}\n")

    try:
        from app.core.config import get_settings
        from app.llm.factory import create_llm_client

        settings = get_settings()
        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()

        # Etapa 1: Embedding da query
        console.print("[dim]⏳ [1/4] Gerando embedding da pergunta...[/]", end="")
        t0 = time.time()
        embeddings = EmbeddingsService()
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)
        query_vec = embeddings.embed(question)
        console.print(f" [green]✓[/] [dim]{time.time()-t0:.1f}s[/]")

        # Etapa 2: Busca vetorial
        console.print("[dim]⏳ [2/4] Buscando chunks relevantes no banco...[/]", end="")
        t0 = time.time()
        chunks = retriever.search(question, top_k=5)
        console.print(f" [green]✓[/] [dim]{time.time()-t0:.1f}s — {len(chunks)} chunk(s) encontrado(s)[/]")
        for c in chunks[:3]:
            console.print(f"  [dim]  → pág.{c.page} score={c.score:.2f}: {c.content[:60].strip()}...[/]")

        # Etapa 3: Classificador
        provider_label = f"{settings.llm_provider}/{settings.llm_model}"
        console.print(f"[dim]⏳ [3/4] Classificando pergunta ({provider_label})...[/]", end="")
        t0 = time.time()
        llm = create_llm_client()

        # Para providers externos (Gemini), usar classificador por palavras-chave
        # para evitar latência e custo extra de uma chamada adicional à API
        if settings.llm_provider == "gemini":
            from app.orchestrator.classifier import QueryClassifier
            classifier = QueryClassifier(llm=None)  # força fallback por keywords
        else:
            classifier = QueryClassifier(llm=llm)

        workflow = FiscusWorkflow(
            docs_agent=DocsAgent(retriever=retriever, llm=llm),
            finance_agent=FinanceAgent(db_session=db, llm=llm),
            classifier=classifier,
            llm=llm,
        )

        console.print(f" [green]✓[/]")
        console.print(f"[dim]⏳ [4/4] Consultando agente ({provider_label})...[/]")

        t_start_llm = time.time()
        result = workflow.invoke(question)
        t_llm = time.time() - t_start_llm
        console.print(f"[dim]  ✓ LLM respondeu em {t_llm:.1f}s[/]\n")

        db.close()

        # Resposta
        console.print(Panel(
            result.get("final_answer", "Sem resposta."),
            title="[bold]Resposta[/]",
            border_style="blue",
        ))

        # Agentes usados
        agents_used = {r["source"] for r in result.get("results", [])}
        console.print(f"[dim]Agentes usados: {', '.join(agents_used)} | Tempo total: {time.time()-t_inicio:.1f}s[/]")

        # Fontes
        all_evidence = []
        for r in result.get("results", []):
            all_evidence.extend(r.get("evidence", []))

        if all_evidence:
            table = Table(title="Fontes", show_header=True)
            table.add_column("Documento", style="cyan")
            table.add_column("Página", justify="right")
            table.add_column("Seção")
            table.add_column("Score", justify="right")

            for ev in all_evidence[:5]:
                if "doc" in ev:
                    table.add_row(
                        ev.get("doc", ""),
                        str(ev.get("page", "")),
                        ev.get("section") or ev.get("article") or "",
                        f"{ev.get('score', 0):.2f}" if ev.get("score") else "",
                    )
            console.print(table)

    except Exception as e:
        console.print(f"[red]✗ Erro:[/] {e}")
        sys.exit(1)


@cli.command()
def status():
    """Exibe status do sistema e dependências."""
    console.print("[bold blue]Fiscus-C Status[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Componente")
    table.add_column("Status")
    table.add_column("Detalhe")

    # Database
    try:
        db_ok = check_db_connection()
        table.add_row("PostgreSQL", "[green]✓ OK[/]" if db_ok else "[red]✗ Falhou[/]", "")
    except Exception as e:
        table.add_row("PostgreSQL", "[red]✗ Erro[/]", str(e)[:50])

    # Ollama
    try:
        from app.core.config import get_settings
        settings = get_settings()
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            table.add_row("Ollama", "[green]✓ OK[/]", f"{len(models)} modelo(s)")
        else:
            table.add_row("Ollama", "[red]✗ Erro[/]", f"HTTP {resp.status_code}")
    except Exception as e:
        table.add_row("Ollama", "[red]✗ Indisponível[/]", str(e)[:50])

    # Documentos no banco
    try:
        from app.core.models import Document, DocumentChunk
        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()
        n_docs = db.query(Document).count()
        n_chunks = db.query(DocumentChunk).count()
        db.close()
        table.add_row("Documentos", "[green]✓[/]", f"{n_docs} doc(s), {n_chunks} chunk(s)")
    except Exception as e:
        table.add_row("Documentos", "[yellow]?[/]", str(e)[:50])

    console.print(table)


if __name__ == "__main__":
    cli()
