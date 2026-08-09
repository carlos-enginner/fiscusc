"""Tools do Agente Financeiro."""
from decimal import Decimal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.core.models import Fatura, FaturaItem

# Sessão é injetada no momento da criação das tools
_db_session: Session | None = None


def set_db_session(session: Session):
    """Injeta sessão de banco nas tools."""
    global _db_session
    _db_session = session


def _get_db() -> Session:
    if _db_session is None:
        raise RuntimeError("DB session não configurada. Chame set_db_session() primeiro.")
    return _db_session


def parse_unidade(unidade: str) -> tuple[str, str]:
    """
    Parse de identificação de unidade.

    Formatos aceitos: "A-2002", "A 2002", "bloco A apto 2002", "2002A", etc.
    Retorna (bloco, apartamento).
    """
    import re

    unidade = unidade.strip()

    # Formato "A-2002" ou "A 2002"
    match = re.match(r"^([A-Za-z])[- ](\d+)$", unidade)
    if match:
        return match.group(1).upper(), match.group(2)

    # Formato "2002A" ou "2002-A"
    match = re.match(r"^(\d+)[- ]?([A-Za-z])$", unidade)
    if match:
        return match.group(2).upper(), match.group(1)

    # Formato "bloco A apto 2002" ou similar
    match = re.search(r"bloco\s*([A-Za-z])", unidade, re.IGNORECASE)
    bloco = match.group(1).upper() if match else None

    match = re.search(r"(?:apto?|apt|apartamento)\s*(\d+)", unidade, re.IGNORECASE)
    apto = match.group(1) if match else None

    if bloco and apto:
        return bloco, apto

    # Último recurso: assume que letras são bloco e números são apto
    letters = re.findall(r"[A-Za-z]+", unidade)
    numbers = re.findall(r"\d+", unidade)
    bloco = letters[0].upper() if letters else "A"
    apto = numbers[0] if numbers else unidade

    return bloco, apto


def format_currency(value: float | Decimal) -> str:
    """Formata valor em Real brasileiro."""
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_fatura(fatura: Fatura, db: Session) -> str:
    """Formata fatura para resposta do agente."""
    lines = [
        f"📄 FATURA - {fatura.mes_referencia or 'N/A'}",
        f"Unidade: Bloco {fatura.unidade_bloco}, Apto {fatura.unidade_apartamento}",
    ]
    if fatura.data_vencimento:
        lines.append(f"Vencimento: {fatura.data_vencimento.strftime('%d/%m/%Y')}")
    lines.append("")

    # Itens de cobrança
    itens_cobranca = (
        db.query(FaturaItem)
        .filter(FaturaItem.fatura_id == fatura.id, FaturaItem.secao == "cobranca")
        .all()
    )

    if itens_cobranca:
        lines.append(f"💰 MINHA COBRANÇA: {format_currency(fatura.total_cobranca or 0)}")
        for item in itens_cobranca:
            lines.append(f"  • {item.descricao}: {format_currency(item.valor)}")
    elif fatura.total_cobranca:
        lines.append(f"💰 TOTAL: {format_currency(fatura.total_cobranca)}")

    if fatura.codigo_barras:
        lines.append(f"")
        lines.append(f"📊 Código de barras: {fatura.codigo_barras[:40]}...")

    return "\n".join(lines)


def get_despesas_por_mes(mes: str, categoria: str | None, db: Session) -> dict:
    """Retorna despesas de um mês agrupadas por categoria."""
    query = (
        db.query(FaturaItem)
        .join(Fatura)
        .filter(
            Fatura.mes_referencia == mes,
            FaturaItem.secao == "despesas",
        )
    )
    if categoria:
        query = query.filter(FaturaItem.categoria == categoria)

    itens = query.all()

    por_categoria: dict[str, float] = {}
    for item in itens:
        cat = item.categoria or item.grupo or "Outros"
        por_categoria[cat] = por_categoria.get(cat, 0) + float(item.valor)

    return {
        "mes": mes,
        "total": sum(por_categoria.values()),
        "por_categoria": por_categoria,
        "itens": itens,
    }


def format_comparativo(despesas1: dict, despesas2: dict) -> str:
    """Formata comparativo de despesas entre dois meses."""
    lines = [
        "📊 COMPARATIVO DE DESPESAS",
        f"{despesas1['mes']} vs {despesas2['mes']}",
        "",
        "| Categoria | Antes | Depois | Variação |",
        "|-----------|-------|--------|----------|",
    ]

    all_cats = set(despesas1["por_categoria"].keys()) | set(despesas2["por_categoria"].keys())

    for cat in sorted(all_cats):
        v1 = despesas1["por_categoria"].get(cat, 0)
        v2 = despesas2["por_categoria"].get(cat, 0)
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 > 0 else 0
        sinal = "+" if diff > 0 else ""
        v1_fmt = format_currency(v1)
        v2_fmt = format_currency(v2)
        lines.append(f"| {cat} | {v1_fmt} | {v2_fmt} | {sinal}{pct:.1f}% |")

    t1 = despesas1["total"]
    t2 = despesas2["total"]
    diff_total = t2 - t1
    pct_total = (diff_total / t1 * 100) if t1 > 0 else 0
    sinal = "+" if diff_total > 0 else ""
    lines.append(
        f"| **TOTAL** | **{format_currency(t1)}** | **{format_currency(t2)}** | **{sinal}{pct_total:.1f}%** |"
    )

    return "\n".join(lines)


@tool
def get_fatura(unidade: str, mes_referencia: str = "") -> str:
    """
    Retorna a fatura de uma unidade do condomínio.

    Use para perguntas sobre:
    - Valor do condomínio
    - Itens da cobrança
    - Data de vencimento

    Args:
        unidade: Identificação da unidade (ex: "A-2002", "B-101").
        mes_referencia: Mês de referência (ex: "julho/2026"). Se vazio, retorna a mais recente.

    Returns:
        Dados da fatura formatados.
    """
    db = _get_db()
    bloco, apto = parse_unidade(unidade)

    query = db.query(Fatura).filter(
        Fatura.unidade_bloco == bloco,
        Fatura.unidade_apartamento == apto,
    )
    if mes_referencia:
        query = query.filter(Fatura.mes_referencia == mes_referencia)

    fatura = query.order_by(Fatura.data_vencimento.desc()).first()

    if not fatura:
        return f"Fatura não encontrada para unidade {unidade}"

    return format_fatura(fatura, db)


@tool
def comparar_despesas(mes1: str, mes2: str, categoria: str = "") -> str:
    """
    Compara despesas do condomínio entre dois meses.

    Use para perguntas sobre:
    - Variação de despesas
    - O que aumentou/diminuiu
    - Tendências de gastos

    Args:
        mes1: Primeiro mês (ex: "junho/2026").
        mes2: Segundo mês (ex: "julho/2026").
        categoria: Categoria específica (opcional).

    Returns:
        Comparativo formatado com variações percentuais.
    """
    db = _get_db()
    cat = categoria or None
    d1 = get_despesas_por_mes(mes1, cat, db)
    d2 = get_despesas_por_mes(mes2, cat, db)

    if d1["total"] == 0 and d2["total"] == 0:
        return f"Nenhuma despesa encontrada para {mes1} ou {mes2}."

    return format_comparativo(d1, d2)


@tool
def listar_despesas(mes_referencia: str, top_n: int = 10) -> str:
    """
    Lista as principais despesas de um mês.

    Use para perguntas sobre:
    - Maiores gastos do mês
    - Detalhamento de despesas
    - Categorias de despesa

    Args:
        mes_referencia: Mês de referência (ex: "julho/2026").
        top_n: Número de itens a retornar (default: 10).

    Returns:
        Lista de despesas ordenadas por valor.
    """
    db = _get_db()
    despesas = (
        db.query(FaturaItem)
        .join(Fatura)
        .filter(Fatura.mes_referencia == mes_referencia, FaturaItem.secao == "despesas")
        .order_by(FaturaItem.valor.desc())
        .limit(top_n)
        .all()
    )

    if not despesas:
        return f"Nenhuma despesa encontrada para {mes_referencia}."

    lines = [f"📊 PRINCIPAIS DESPESAS - {mes_referencia}", ""]
    for i, item in enumerate(despesas, 1):
        grupo = f" ({item.grupo})" if item.grupo else ""
        lines.append(f"{i}. {item.descricao}{grupo}: {format_currency(item.valor)}")

    return "\n".join(lines)


FINANCE_TOOLS = [get_fatura, comparar_despesas, listar_despesas]
