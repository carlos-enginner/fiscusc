"""Agente Financeiro usando LangChain."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from sqlalchemy.orm import Session

from app.core.config import get_settings

FINANCE_AGENT_PROMPT = """Você é um assistente simpático do condomínio que ajuda os moradores a entenderem suas faturas e as despesas do condomínio.

REGRAS OBRIGATÓRIAS:
1. Sempre mostre valores numéricos precisos em formato brasileiro (R$ X.XXX,XX)
2. Use linguagem simples e amigável — como explicaria para um vizinho
3. Cite o mês de referência para contextualizar os dados
4. Ao comparar, explique o que aumentou ou diminuiu de forma clara
5. Se não houver dados, diga com gentileza

FORMATO DA RESPOSTA:
- Responda diretamente com os valores
- Explique o que compõe o valor quando relevante
- Use emojis com moderação para tornar mais visual (💰 para valores, 📊 para comparativos)

EXEMPLOS DE TOM:
✓ "Sua fatura de julho/2026 é de R$ 795,96 com vencimento em 10/08. Ela é composta por taxa de condomínio (R$ 679,76) e fundo de reserva (R$ 33,99)."
✓ "As despesas de julho ficaram R$ 5.000 acima de junho, principalmente por causa do aumento nas contas de energia. Nada alarmante, dentro do esperado para o inverno!"
✓ "Não encontrei dados de fatura para essa unidade ainda. Se você acabou de se mudar, pode levar alguns dias para aparecer no sistema."

CATEGORIAS DE DESPESAS (para explicar ao morador):
- Pessoal: porteiros, zeladores, faxineiros
- Consumo: água, energia, gás
- Manutenção: elevadores, bombas, reparos
- Administrativo: honorários, contabilidade
- Outros: seguros, tarifas bancárias"""


class FinanceAgent:
    """
    Agente Financeiro.

    Consulta faturas e despesas do banco de dados e responde perguntas financeiras.
    """

    def __init__(self, db_session: Session | None = None, llm=None):
        settings = get_settings()

        # Configurar LLM
        if llm is not None:
            self._llm = llm
        else:
            self._llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        self._db = db_session

        # Configurar tools se sessão disponível
        if db_session is not None:
            from app.agents.finance.tools import set_db_session, FINANCE_TOOLS

            set_db_session(db_session)
            self._tools = FINANCE_TOOLS
        else:
            self._tools = []

    def invoke(self, state: dict) -> dict:
        """
        Executa o agente financeiro.

        Args:
            state: Dict com chave "query".

        Returns:
            Dict com "results" contendo resposta e evidências.
        """
        query = state["query"]

        # Tentar buscar dados financeiros diretamente
        context = ""
        evidence = []

        if self._db is not None:
            context, evidence = self._fetch_financial_context(query)

        # Montar prompt com contexto
        if context:
            user_content = f"""DADOS FINANCEIROS:
{context}

PERGUNTA: {query}"""
        else:
            user_content = f"""PERGUNTA: {query}

Nenhum dado financeiro foi encontrado no banco de dados.
Informe ao usuário que não há dados disponíveis para esta consulta."""

        messages = [
            SystemMessage(content=FINANCE_AGENT_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self._llm.invoke(messages)
        answer = response.content

        return {
            "results": [
                {
                    "source": "finance",
                    "result": answer,
                    "evidence": evidence,
                }
            ]
        }

    def _fetch_financial_context(self, query: str) -> tuple[str, list]:
        """Tenta extrair contexto financeiro relevante para a query."""
        from app.core.models import Fatura, FaturaItem
        from app.agents.finance.tools import parse_unidade, format_fatura, get_despesas_por_mes

        context_parts = []
        evidence = []

        query_lower = query.lower()

        # Tentar identificar unidade na query
        import re
        unidade_match = re.search(r"\b([A-Za-z])[- ]?(\d{3,4})\b|\b(\d{3,4})[- ]?([A-Za-z])\b", query)
        if unidade_match:
            try:
                if unidade_match.group(1):
                    bloco, apto = unidade_match.group(1).upper(), unidade_match.group(2)
                else:
                    bloco, apto = unidade_match.group(4).upper(), unidade_match.group(3)

                faturas = (
                    self._db.query(Fatura)
                    .filter(Fatura.unidade_bloco == bloco, Fatura.unidade_apartamento == apto)
                    .order_by(Fatura.data_vencimento.desc())
                    .limit(3)
                    .all()
                )
                for f in faturas:
                    context_parts.append(format_fatura(f, self._db))
                    evidence.append({
                        "type": "fatura",
                        "unidade": f"{f.unidade_bloco}-{f.unidade_apartamento}",
                        "mes": f.mes_referencia,
                        "total": float(f.total_cobranca or 0),
                    })
            except Exception:
                pass

        # Verificar se é consulta de despesas gerais
        if any(w in query_lower for w in ["despesa", "gasto", "custo", "comparar", "mês"]):
            try:
                faturas_recentes = (
                    self._db.query(Fatura.mes_referencia)
                    .distinct()
                    .order_by(Fatura.mes_referencia.desc())
                    .limit(2)
                    .all()
                )
                for (mes,) in faturas_recentes:
                    if mes:
                        d = get_despesas_por_mes(mes, None, self._db)
                        if d["total"] > 0:
                            context_parts.append(
                                f"Despesas de {mes}: R$ {d['total']:,.2f}"
                            )
                            evidence.append({"type": "despesas", "mes": mes, "total": d["total"]})
            except Exception:
                pass

        return "\n\n".join(context_parts), evidence


def create_finance_agent(db_session: Session | None = None, llm=None) -> FinanceAgent:
    """Cria o agente financeiro com DI."""
    return FinanceAgent(db_session=db_session, llm=llm)
