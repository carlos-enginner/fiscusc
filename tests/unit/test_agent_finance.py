"""Testes da Task 8: Agente Financeiro.

Testes unitários usam mocks do LLM e banco de dados.
Testes de integração requerem banco com dados de teste.
"""
import uuid
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from app.agents.finance.agent import FinanceAgent, create_finance_agent
from app.agents.finance.tools import (
    format_comparativo,
    format_currency,
    format_fatura,
    parse_unidade,
)


# --- Mocks ---

class FakeLLM:
    """LLM fake para testes."""

    def __init__(self, response: str = "Resposta financeira de teste"):
        self._response = response

    def invoke(self, messages):
        msg = MagicMock()
        msg.content = self._response
        return msg


def make_fatura_mock(
    bloco="A",
    apto="2002",
    mes="julho/2026",
    total=795.96,
    vencimento=date(2026, 8, 10),
):
    """Cria mock de fatura."""
    f = MagicMock()
    f.id = uuid.uuid4()
    f.unidade_bloco = bloco
    f.unidade_apartamento = apto
    f.mes_referencia = mes
    f.total_cobranca = Decimal(str(total))
    f.data_vencimento = vencimento
    f.codigo_barras = None
    return f


class FakeDB:
    """Sessão de banco fake para testes."""

    def __init__(self, faturas=None):
        self._faturas = faturas or []

    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def join(self, *args):
        return self

    def order_by(self, *args):
        return self

    def limit(self, *args):
        return self

    def first(self):
        return self._faturas[0] if self._faturas else None

    def all(self):
        return self._faturas

    def distinct(self):
        return self


# --- Testes de parse_unidade ---

class TestParseUnidade:
    def test_format_bloco_hifen_apto(self):
        """Formato A-2002 deve ser parseado corretamente."""
        bloco, apto = parse_unidade("A-2002")
        assert bloco == "A"
        assert apto == "2002"

    def test_format_bloco_espaco_apto(self):
        """Formato A 2002 deve ser parseado."""
        bloco, apto = parse_unidade("A 2002")
        assert bloco == "A"
        assert apto == "2002"

    def test_format_apto_bloco(self):
        """Formato 2002A deve ser parseado."""
        bloco, apto = parse_unidade("2002A")
        assert bloco == "A"
        assert apto == "2002"

    def test_uppercase_bloco(self):
        """Bloco deve ser normalizado para uppercase."""
        bloco, _ = parse_unidade("b-101")
        assert bloco == "B"


# --- Testes de formatação ---

class TestFormatCurrency:
    def test_format_basic_value(self):
        """Valor simples deve ser formatado em Real."""
        result = format_currency(795.96)
        assert "R$" in result
        assert "795" in result

    def test_format_thousands(self):
        """Valor com milhar deve usar ponto como separador."""
        result = format_currency(1234.56)
        assert "1.234,56" in result

    def test_format_decimal(self):
        """Decimal deve ser aceito."""
        result = format_currency(Decimal("795.96"))
        assert "R$" in result


# --- Testes do FinanceAgent ---

class TestFinanceAgent:
    @pytest.fixture
    def agent_with_data(self):
        fatura = make_fatura_mock()
        db = FakeDB(faturas=[fatura])
        llm = FakeLLM("Sua fatura de julho/2026 é de R$ 795,96, com vencimento em 10/08/2026.")
        agent = FinanceAgent(db_session=db, llm=llm)
        return agent

    @pytest.fixture
    def agent_no_data(self):
        db = FakeDB(faturas=[])
        llm = FakeLLM("Nenhum dado financeiro encontrado.")
        return FinanceAgent(db_session=db, llm=llm)

    def test_agent_returns_result_dict(self, agent_with_data):
        """invoke() deve retornar dict com 'results'."""
        result = agent_with_data.invoke({"query": "Qual o valor do condomínio?"})

        assert "results" in result
        assert len(result["results"]) == 1

    def test_agent_result_has_required_fields(self, agent_with_data):
        """Resultado deve ter source, result e evidence."""
        result = agent_with_data.invoke({"query": "Valor do condomínio A-2002?"})
        r = result["results"][0]

        assert r["source"] == "finance"
        assert isinstance(r["result"], str)
        assert isinstance(r["evidence"], list)

    def test_agent_response_contains_currency(self, agent_with_data):
        """Resposta deve conter formatação de Real."""
        result = agent_with_data.invoke({"query": "Valor do condomínio A-2002?"})
        response = result["results"][0]["result"]

        assert "R$" in response or "795" in response

    def test_agent_no_data_returns_result(self, agent_no_data):
        """Agente sem dados deve retornar resposta mesmo assim."""
        result = agent_no_data.invoke({"query": "Valor do condomínio?"})

        assert "results" in result
        assert result["results"][0]["source"] == "finance"

    def test_create_finance_agent_factory(self):
        """create_finance_agent deve retornar instância."""
        agent = create_finance_agent(db_session=FakeDB(), llm=FakeLLM())
        assert isinstance(agent, FinanceAgent)


# --- Testes de comparativo ---

class TestFormatComparativo:
    def test_comparativo_has_header(self):
        """Comparativo deve ter cabeçalho com os meses."""
        d1 = {"mes": "junho/2026", "total": 80000.0, "por_categoria": {"Pessoal": 40000, "Consumo": 40000}}
        d2 = {"mes": "julho/2026", "total": 85000.0, "por_categoria": {"Pessoal": 42000, "Consumo": 43000}}

        result = format_comparativo(d1, d2)

        assert "junho/2026" in result
        assert "julho/2026" in result

    def test_comparativo_has_percentage(self):
        """Comparativo deve ter variação percentual."""
        d1 = {"mes": "junho/2026", "total": 80000.0, "por_categoria": {"Pessoal": 80000}}
        d2 = {"mes": "julho/2026", "total": 84000.0, "por_categoria": {"Pessoal": 84000}}

        result = format_comparativo(d1, d2)

        assert "%" in result
        assert "+5" in result  # variação positiva de 5%

    def test_comparativo_has_total_row(self):
        """Comparativo deve ter linha de total."""
        d1 = {"mes": "jun", "total": 100.0, "por_categoria": {"A": 100}}
        d2 = {"mes": "jul", "total": 110.0, "por_categoria": {"A": 110}}

        result = format_comparativo(d1, d2)

        assert "TOTAL" in result


# --- Testes de integração ---

@pytest.mark.integration
class TestFinanceAgentIntegration:
    @pytest.fixture
    def db_with_fatura(self):
        """Cria fatura de teste no banco."""
        from app.core.database import get_session_factory, get_engine
        from app.core.models import Fatura, FaturaItem

        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()

        fatura = Fatura(
            condominio_nome="Residencial Teste",
            unidade_bloco="A",
            unidade_apartamento="2002",
            mes_referencia="julho/2026",
            data_vencimento=date(2026, 8, 10),
            total_cobranca=Decimal("795.96"),
        )
        db.add(fatura)
        db.flush()

        itens = [
            FaturaItem(fatura_id=fatura.id, secao="cobranca", descricao="TAXA DE CONDOMÍNIO", valor=Decimal("679.76"), categoria="Taxa"),
            FaturaItem(fatura_id=fatura.id, secao="cobranca", descricao="FUNDO DE RESERVA", valor=Decimal("33.99"), categoria="Taxa"),
            FaturaItem(fatura_id=fatura.id, secao="despesas", descricao="PORTARIA", valor=Decimal("30000"), categoria="Pessoal", grupo="SERVIÇOS"),
            FaturaItem(fatura_id=fatura.id, secao="despesas", descricao="ENERGIA ELÉTRICA", valor=Decimal("5000"), categoria="Consumo", grupo="UTILIDADES"),
        ]
        db.bulk_save_objects(itens)
        db.commit()

        yield db, fatura

        # Cleanup
        db.delete(fatura)
        db.commit()
        db.close()

    def test_get_fatura_tool(self, db_with_fatura):
        """Tool get_fatura deve retornar dados da fatura."""
        from app.agents.finance.tools import set_db_session, get_fatura

        db, _ = db_with_fatura
        set_db_session(db)

        result = get_fatura.invoke({"unidade": "A-2002", "mes_referencia": "julho/2026"})
        assert "795" in result or "julio" in result.lower() or "julho" in result.lower()

    def test_listar_despesas_tool(self, db_with_fatura):
        """Tool listar_despesas deve retornar despesas."""
        from app.agents.finance.tools import set_db_session, listar_despesas

        db, _ = db_with_fatura
        set_db_session(db)

        result = listar_despesas.invoke({"mes_referencia": "julho/2026", "top_n": 10})
        assert "PORTARIA" in result.upper() or "portaria" in result.lower()

    def test_agent_formats_currency(self, db_with_fatura):
        """Agente deve formatar valores em Real."""
        db, _ = db_with_fatura
        llm = FakeLLM("Sua fatura é de R$ 795,96.")
        agent = FinanceAgent(db_session=db, llm=llm)

        result = agent.invoke({"query": "Valor do condomínio A-2002?"})
        response = result["results"][0]["result"]

        assert "R$" in response
