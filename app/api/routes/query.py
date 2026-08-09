"""Endpoint de query (principal)."""
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryMetadata, QueryRequest, QueryResponse, Source

router = APIRouter()


def _extract_sources(results: list[dict]) -> list[Source]:
    """Extrai e formata fontes dos resultados dos agentes."""
    sources = []
    seen = set()

    for r in results:
        for ev in r.get("evidence", []):
            if ev.get("type") == "fatura":
                key = f"fatura:{ev.get('mes')}:{ev.get('unidade')}"
                if key not in seen:
                    seen.add(key)
                    sources.append(Source(
                        type="fatura",
                        document=f"Fatura {ev.get('mes', '')}",
                        snippet=f"Total: R$ {ev.get('total', 0):,.2f}",
                    ))
            else:
                key = f"{ev.get('doc')}:{ev.get('page')}"
                if key not in seen:
                    seen.add(key)
                    sources.append(Source(
                        type="document",
                        document=ev.get("doc"),
                        document_type=ev.get("document_type"),
                        page=ev.get("page"),
                        section=ev.get("section") or ev.get("article"),
                        score=ev.get("score"),
                    ))

    return sources


def _get_workflow():
    """Retorna instância do workflow (lazy init)."""
    from app.agents.docs.agent import DocsAgent
    from app.agents.finance.agent import FinanceAgent
    from app.core.config import get_settings
    from app.core.database import get_session_factory, get_engine
    from app.embeddings.service import EmbeddingsService
    from app.orchestrator.classifier import QueryClassifier
    from app.orchestrator.workflow import FiscusWorkflow
    from app.rag.retriever import DocumentRetriever
    from langchain_ollama import ChatOllama

    settings = get_settings()
    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()

    embeddings = EmbeddingsService()
    retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)

    return FiscusWorkflow(
        docs_agent=DocsAgent(retriever=retriever, llm=llm),
        finance_agent=FinanceAgent(db_session=db, llm=llm),
        classifier=QueryClassifier(llm=llm),
        llm=llm,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Processa pergunta usando o orquestrador.

    Classifica automaticamente e roteia para os agentes adequados.
    """
    start = time.time()

    try:
        workflow = _get_workflow()
        result = workflow.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar query: {str(e)}")

    latency_ms = int((time.time() - start) * 1000)

    agents_used = list({r["source"] for r in result.get("results", [])})
    sources = _extract_sources(result.get("results", []))

    return QueryResponse(
        answer=result.get("final_answer", "Não foi possível processar sua pergunta."),
        agents_used=agents_used,
        sources=sources,
        metadata=QueryMetadata(
            query_id=str(uuid.uuid4()),
            latency_ms=latency_ms,
        ),
    )
