"""
Golden set of evaluation questions for Fiscus-C RAG system.

Contains curated questions about condominium regulations (regimento interno)
with expected keywords for retrieval quality evaluation.
"""

from dataclasses import dataclass, field


@dataclass
class GoldenQuestion:
    """
    A golden question for RAG evaluation.

    Attributes:
        question: The question to ask the system.
        expected_keywords: Keywords that should appear in retrieved documents.
        expected_doc_type: Expected document type ('regimento', 'convencao', or None for any).
        min_expected_matches: Minimum number of retrieved docs that should match keywords.
    """

    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_doc_type: str | None = None
    min_expected_matches: int = 1


# Golden set with 20 questions about condominium regulations
GOLDEN_SET: list[GoldenQuestion] = [
    # Horário de obras
    GoldenQuestion(
        question="Qual o horário permitido para obras no apartamento?",
        expected_keywords=["obra", "horário", "reforma", "8h", "18h", "segunda", "sábado"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Posso fazer reforma no domingo?",
        expected_keywords=["obra", "domingo", "feriado", "reforma", "proibido", "ruído"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Animais de estimação
    GoldenQuestion(
        question="Posso ter animais de estimação no apartamento?",
        expected_keywords=["animal", "estimação", "pet", "cachorro", "gato", "permitido"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Quais são as regras para circulação de animais nas áreas comuns?",
        expected_keywords=["animal", "área comum", "coleira", "elevador", "circulação"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Mudanças
    GoldenQuestion(
        question="Qual o horário permitido para mudanças?",
        expected_keywords=["mudança", "horário", "agendamento", "elevador", "carga"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Preciso agendar mudança com antecedência?",
        expected_keywords=["mudança", "agendamento", "antecedência", "portaria", "síndico"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Barulho e silêncio
    GoldenQuestion(
        question="Qual o horário de silêncio no condomínio?",
        expected_keywords=["silêncio", "horário", "22h", "8h", "barulho", "ruído"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Posso fazer festa no apartamento?",
        expected_keywords=["festa", "barulho", "silêncio", "som", "vizinho", "horário"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Piscina
    GoldenQuestion(
        question="Qual o horário de funcionamento da piscina?",
        expected_keywords=["piscina", "horário", "funcionamento", "uso", "área comum"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Quais são as regras para uso da piscina?",
        expected_keywords=["piscina", "regra", "uso", "traje", "banho", "criança"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Salão de festas
    GoldenQuestion(
        question="Como reservar o salão de festas?",
        expected_keywords=["salão", "festa", "reserva", "agendamento", "taxa", "antecedência"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Qual o valor da taxa para usar o salão de festas?",
        expected_keywords=["salão", "taxa", "valor", "reserva", "pagamento"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Garagem
    GoldenQuestion(
        question="Quantas vagas de garagem tenho direito por apartamento?",
        expected_keywords=["vaga", "garagem", "apartamento", "direito", "estacionamento"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Posso emprestar minha vaga de garagem?",
        expected_keywords=["vaga", "garagem", "emprestar", "alugar", "cessão", "terceiro"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Visitantes
    GoldenQuestion(
        question="Como funciona o cadastro de visitantes?",
        expected_keywords=["visitante", "cadastro", "portaria", "entrada", "autorização"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Visitantes podem usar as áreas comuns?",
        expected_keywords=["visitante", "área comum", "uso", "acompanhado", "morador"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Lixo e reciclagem
    GoldenQuestion(
        question="Qual o horário para colocar o lixo?",
        expected_keywords=["lixo", "horário", "coleta", "descarte", "lixeira"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Onde devo descartar móveis e objetos grandes?",
        expected_keywords=["descarte", "móvel", "objeto", "grande", "entulho", "lixo"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    # Áreas comuns gerais
    GoldenQuestion(
        question="Posso usar a churrasqueira do condomínio?",
        expected_keywords=["churrasqueira", "uso", "reserva", "área comum", "agendamento"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
    GoldenQuestion(
        question="Quais são as regras da academia do condomínio?",
        expected_keywords=["academia", "horário", "uso", "regra", "equipamento"],
        expected_doc_type="regimento",
        min_expected_matches=1,
    ),
]
