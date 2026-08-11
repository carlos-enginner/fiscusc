"""
Unit tests for evaluation metrics.

Tests cover:
- Hit rate calculation (100%, 0%, partial)
- MRR calculation (various rank positions)
- Precision@K calculation
- Complete evaluation pipeline
- Edge cases (empty results, mismatched lengths)
"""

import pytest

from tests.evaluation.golden_set import GoldenQuestion
from tests.evaluation.metrics import (
    RetrievalMetrics,
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_at_k,
    evaluate_retrieval,
)


class TestCalculateHitRate:
    """Tests for calculate_hit_rate function."""

    def test_perfect_hit_rate(self):
        """All queries have at least one relevant result."""
        results = [
            ["documento sobre obras e reformas"],
            ["regras de animais de estimação"],
            ["horário de silêncio no condomínio"],
        ]
        keywords = [
            ["obras", "reforma"],
            ["animais", "pet"],
            ["silêncio", "barulho"],
        ]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_zero_hit_rate(self):
        """No queries have any relevant results."""
        results = [
            ["documento irrelevante"],
            ["outro documento sem relação"],
            ["texto aleatório"],
        ]
        keywords = [
            ["obras", "reforma"],
            ["animais", "pet"],
            ["silêncio", "barulho"],
        ]
        assert calculate_hit_rate(results, keywords) == 0.0

    def test_partial_hit_rate(self):
        """Some queries have relevant results."""
        results = [
            ["documento sobre obras"],  # hit
            ["documento irrelevante"],  # miss
            ["regras de silêncio"],  # hit
            ["texto aleatório"],  # miss
        ]
        keywords = [
            ["obras"],
            ["animais"],
            ["silêncio"],
            ["piscina"],
        ]
        assert calculate_hit_rate(results, keywords) == 0.5

    def test_hit_in_second_result(self):
        """Hit rate should count if keyword is in any result."""
        results = [
            ["irrelevante", "documento sobre obras"],
        ]
        keywords = [["obras"]]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_empty_results(self):
        """Empty results list returns 0.0."""
        assert calculate_hit_rate([], []) == 0.0

    def test_mismatched_lengths_raises_error(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_hit_rate([["doc"]], [["kw1"], ["kw2"]])

    def test_dict_results(self):
        """Results as dictionaries with 'content' key."""
        results = [
            [{"content": "documento sobre obras"}],
        ]
        keywords = [["obras"]]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_case_insensitive_matching(self):
        """Keyword matching should be case-insensitive."""
        results = [["DOCUMENTO SOBRE OBRAS"]]
        keywords = [["obras"]]
        assert calculate_hit_rate(results, keywords) == 1.0


class TestCalculateMRR:
    """Tests for calculate_mrr function."""

    def test_perfect_mrr_all_first(self):
        """All relevant results at position 1 -> MRR = 1.0."""
        results = [
            ["documento sobre obras"],
            ["regras de animais"],
        ]
        keywords = [
            ["obras"],
            ["animais"],
        ]
        assert calculate_mrr(results, keywords) == 1.0

    def test_mrr_all_second_position(self):
        """All relevant results at position 2 -> MRR = 0.5."""
        results = [
            ["irrelevante", "documento sobre obras"],
            ["outro irrelevante", "regras de animais"],
        ]
        keywords = [
            ["obras"],
            ["animais"],
        ]
        assert calculate_mrr(results, keywords) == 0.5

    def test_mrr_mixed_positions(self):
        """Mixed positions: (1/1 + 1/2) / 2 = 0.75."""
        results = [
            ["documento sobre obras"],  # position 1
            ["irrelevante", "regras de animais"],  # position 2
        ]
        keywords = [
            ["obras"],
            ["animais"],
        ]
        assert calculate_mrr(results, keywords) == 0.75

    def test_mrr_no_hits(self):
        """No relevant results -> MRR = 0.0."""
        results = [
            ["irrelevante", "outro irrelevante"],
            ["mais irrelevante"],
        ]
        keywords = [
            ["obras"],
            ["animais"],
        ]
        assert calculate_mrr(results, keywords) == 0.0

    def test_mrr_partial_hits(self):
        """Some queries hit, some miss: (1/1 + 0) / 2 = 0.5."""
        results = [
            ["documento sobre obras"],  # hit at position 1
            ["irrelevante"],  # no hit
        ]
        keywords = [
            ["obras"],
            ["animais"],
        ]
        assert calculate_mrr(results, keywords) == 0.5

    def test_mrr_empty_results(self):
        """Empty results list returns 0.0."""
        assert calculate_mrr([], []) == 0.0

    def test_mrr_third_position(self):
        """Relevant at position 3 -> 1/3."""
        results = [
            ["a", "b", "documento sobre obras"],
        ]
        keywords = [["obras"]]
        assert abs(calculate_mrr(results, keywords) - (1 / 3)) < 0.001


class TestCalculatePrecisionAtK:
    """Tests for calculate_precision_at_k function."""

    def test_perfect_precision_at_k(self):
        """All top-k results are relevant."""
        results = [
            ["obras info", "reforma detalhes", "mais sobre obras"],
        ]
        keywords = [["obras", "reforma"]]
        assert calculate_precision_at_k(results, keywords, k=3) == 1.0

    def test_zero_precision_at_k(self):
        """No top-k results are relevant."""
        results = [
            ["irrelevante 1", "irrelevante 2", "irrelevante 3"],
        ]
        keywords = [["obras"]]
        assert calculate_precision_at_k(results, keywords, k=3) == 0.0

    def test_partial_precision_at_k(self):
        """2 out of 3 top-k results relevant -> 2/3."""
        results = [
            ["obras info", "irrelevante", "reforma detalhes"],
        ]
        keywords = [["obras", "reforma"]]
        precision = calculate_precision_at_k(results, keywords, k=3)
        assert abs(precision - (2 / 3)) < 0.001

    def test_precision_at_k_with_fewer_results(self):
        """When results < k, use actual result count."""
        results = [
            ["obras info", "reforma detalhes"],  # only 2 results
        ]
        keywords = [["obras", "reforma"]]
        # Both are relevant, so precision = 2/2 = 1.0
        assert calculate_precision_at_k(results, keywords, k=5) == 1.0

    def test_precision_at_k_truncates_results(self):
        """Only consider top-k results."""
        results = [
            ["obras info", "irrelevante", "irrelevante", "irrelevante", "reforma"],
        ]
        keywords = [["obras", "reforma"]]
        # k=2: 1 relevant out of 2 = 0.5
        assert calculate_precision_at_k(results, keywords, k=2) == 0.5

    def test_precision_at_k_empty_results(self):
        """Empty results list returns 0.0."""
        assert calculate_precision_at_k([], [], k=5) == 0.0

    def test_precision_at_k_invalid_k_raises_error(self):
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            calculate_precision_at_k([["doc"]], [["kw"]], k=0)

    def test_precision_at_k_empty_query_results(self):
        """Query with no results contributes 0.0 precision."""
        results = [
            ["obras info"],
            [],  # empty results for this query
        ]
        keywords = [["obras"], ["animais"]]
        # (1.0 + 0.0) / 2 = 0.5
        assert calculate_precision_at_k(results, keywords, k=3) == 0.5

    def test_precision_at_k_average_across_queries(self):
        """Average precision across multiple queries."""
        results = [
            ["obras", "reforma"],  # 2/2 = 1.0
            ["animais", "irrelevante", "irrelevante"],  # 1/3
        ]
        keywords = [["obras", "reforma"], ["animais"]]
        precision = calculate_precision_at_k(results, keywords, k=3)
        expected = (1.0 + 1 / 3) / 2
        assert abs(precision - expected) < 0.001


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval function."""

    def test_evaluate_retrieval_perfect_retriever(self):
        """Perfect retriever that always returns relevant results."""

        def perfect_retriever(question: str) -> list[str]:
            return [f"documento sobre {question}"]

        questions = [
            GoldenQuestion(
                question="obras",
                expected_keywords=["obras"],
            ),
            GoldenQuestion(
                question="animais",
                expected_keywords=["animais"],
            ),
        ]

        metrics = evaluate_retrieval(questions, perfect_retriever, k=5)

        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.hit_rate == 1.0
        assert metrics.mrr == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.k == 5

    def test_evaluate_retrieval_poor_retriever(self):
        """Poor retriever that never returns relevant results."""

        def poor_retriever(question: str) -> list[str]:
            return ["documento irrelevante"]

        questions = [
            GoldenQuestion(
                question="obras",
                expected_keywords=["obras"],
            ),
        ]

        metrics = evaluate_retrieval(questions, poor_retriever, k=5)

        assert metrics.hit_rate == 0.0
        assert metrics.mrr == 0.0
        assert metrics.precision_at_k == 0.0

    def test_evaluate_retrieval_mixed_quality(self):
        """Retriever with mixed quality results."""
        call_count = {"n": 0}

        def mixed_retriever(question: str) -> list[str]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["documento sobre obras", "mais sobre reforma"]
            else:
                return ["irrelevante"]

        questions = [
            GoldenQuestion(question="horário de obras?", expected_keywords=["obras", "reforma"]),
            GoldenQuestion(question="regras de animais?", expected_keywords=["animais"]),
        ]

        metrics = evaluate_retrieval(questions, mixed_retriever, k=5)

        assert metrics.hit_rate == 0.5  # 1 of 2 queries hit
        assert metrics.mrr == 0.5  # (1/1 + 0) / 2
        assert metrics.k == 5

    def test_evaluate_retrieval_respects_k(self):
        """Evaluate retrieval should truncate results to k."""

        def verbose_retriever(question: str) -> list[str]:
            return [f"result_{i}" for i in range(100)]

        questions = [
            GoldenQuestion(question="test", expected_keywords=["result_0"]),
        ]

        metrics = evaluate_retrieval(questions, verbose_retriever, k=3)
        assert metrics.k == 3
        # Only top-3 results considered for precision
        # result_0 is relevant, so precision = 1/3
        assert abs(metrics.precision_at_k - (1 / 3)) < 0.001

    def test_evaluate_retrieval_empty_questions(self):
        """Empty questions list returns zero metrics."""

        def retriever(q: str) -> list[str]:
            return ["something"]

        metrics = evaluate_retrieval([], retriever, k=5)

        assert metrics.hit_rate == 0.0
        assert metrics.mrr == 0.0
        assert metrics.precision_at_k == 0.0


class TestEdgeCases:
    """Edge case tests for metrics functions."""

    def test_dict_with_text_key(self):
        """Results with 'text' key instead of 'content'."""
        results = [[{"text": "documento sobre obras"}]]
        keywords = [["obras"]]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_multiple_keywords_any_match(self):
        """Any keyword match counts as a hit."""
        results = [["documento sobre reforma"]]
        keywords = [["obras", "reforma", "construção"]]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_keyword_partial_match(self):
        """Partial keyword match (substring) should work."""
        results = [["informações sobre reformas residenciais"]]
        keywords = [["reforma"]]  # 'reforma' is substring of 'reformas'
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_special_characters_in_keywords(self):
        """Keywords with special characters."""
        results = [["horário: 8h às 18h"]]
        keywords = [["8h", "18h"]]
        assert calculate_hit_rate(results, keywords) == 1.0

    def test_unicode_keywords(self):
        """Keywords with unicode/accents."""
        results = [["horário de silêncio após 22h"]]
        keywords = [["silêncio", "horário"]]
        assert calculate_hit_rate(results, keywords) == 1.0
