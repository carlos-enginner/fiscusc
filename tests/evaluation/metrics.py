"""
Retrieval evaluation metrics for Fiscus-C RAG system.

Provides standard IR metrics:
- Hit Rate: Percentage of queries with at least one relevant result
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- Precision@K: Precision considering only top-k results
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tests.evaluation.golden_set import GoldenQuestion


@dataclass
class RetrievalMetrics:
    """
    Container for retrieval evaluation metrics.

    Attributes:
        hit_rate: Percentage of queries where at least one result contains expected keyword.
        mrr: Mean Reciprocal Rank - average of 1/rank of first relevant result.
        precision_at_k: Precision considering only top-k results.
        k: The k value used for precision@k calculation.
    """

    hit_rate: float
    mrr: float
    precision_at_k: float
    k: int


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching (lowercase, strip)."""
    return text.lower().strip()


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    """
    Check if text contains any of the expected keywords.

    Args:
        text: Text to search in.
        keywords: List of keywords to look for.

    Returns:
        True if any keyword is found in text.
    """
    normalized_text = _normalize_text(text)
    return any(_normalize_text(kw) in normalized_text for kw in keywords)


def _get_result_text(result: Any) -> str:
    """
    Extract text content from a retrieval result.

    Handles different result formats:
    - String: returns as-is
    - Dict with 'content' or 'text' key
    - Object with 'content' or 'page_content' attribute (LangChain Document)

    Args:
        result: A retrieval result in various formats.

    Returns:
        The text content of the result.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("content", result.get("text", str(result)))
    # LangChain Document or similar object
    if hasattr(result, "page_content"):
        return result.page_content
    if hasattr(result, "content"):
        return result.content
    return str(result)


def calculate_hit_rate(results: list[list[Any]], expected_keywords: list[list[str]]) -> float:
    """
    Calculate hit rate: percentage of queries where at least one result contains expected keyword.

    Args:
        results: List of result lists, one per query. Each result can be a string,
                 dict with 'content'/'text', or object with 'page_content'/'content'.
        expected_keywords: List of keyword lists, one per query.

    Returns:
        Hit rate as float between 0.0 and 1.0.

    Raises:
        ValueError: If results and expected_keywords have different lengths.
    """
    if len(results) != len(expected_keywords):
        raise ValueError(
            f"Results ({len(results)}) and keywords ({len(expected_keywords)}) must have same length"
        )

    if not results:
        return 0.0

    hits = 0
    for query_results, keywords in zip(results, expected_keywords, strict=True):
        for result in query_results:
            text = _get_result_text(result)
            if _contains_any_keyword(text, keywords):
                hits += 1
                break  # Only count one hit per query

    return hits / len(results)


def calculate_mrr(results: list[list[Any]], expected_keywords: list[list[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average of reciprocal ranks of the first relevant result for each query.
    If no relevant result is found, the reciprocal rank for that query is 0.

    Args:
        results: List of result lists, one per query.
        expected_keywords: List of keyword lists, one per query.

    Returns:
        MRR as float between 0.0 and 1.0.

    Raises:
        ValueError: If results and expected_keywords have different lengths.
    """
    if len(results) != len(expected_keywords):
        raise ValueError(
            f"Results ({len(results)}) and keywords ({len(expected_keywords)}) must have same length"
        )

    if not results:
        return 0.0

    reciprocal_ranks = []
    for query_results, keywords in zip(results, expected_keywords, strict=True):
        rr = 0.0
        for rank, result in enumerate(query_results, start=1):
            text = _get_result_text(result)
            if _contains_any_keyword(text, keywords):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_precision_at_k(
    results: list[list[Any]], expected_keywords: list[list[str]], k: int
) -> float:
    """
    Calculate Precision@K: average proportion of relevant results in top-k.

    Args:
        results: List of result lists, one per query.
        expected_keywords: List of keyword lists, one per query.
        k: Number of top results to consider.

    Returns:
        Precision@K as float between 0.0 and 1.0.

    Raises:
        ValueError: If results and expected_keywords have different lengths, or k < 1.
    """
    if len(results) != len(expected_keywords):
        raise ValueError(
            f"Results ({len(results)}) and keywords ({len(expected_keywords)}) must have same length"
        )

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    if not results:
        return 0.0

    precisions = []
    for query_results, keywords in zip(results, expected_keywords, strict=True):
        top_k = query_results[:k]
        if not top_k:
            precisions.append(0.0)
            continue

        relevant_count = sum(
            1 for result in top_k if _contains_any_keyword(_get_result_text(result), keywords)
        )
        precisions.append(relevant_count / len(top_k))

    return sum(precisions) / len(precisions)


def evaluate_retrieval(
    questions: list[GoldenQuestion],
    retriever_fn: Callable[[str], list[Any]],
    k: int = 5,
) -> RetrievalMetrics:
    """
    Run complete retrieval evaluation on a set of golden questions.

    Args:
        questions: List of GoldenQuestion objects to evaluate.
        retriever_fn: Function that takes a question string and returns list of results.
        k: Number of top results to retrieve and use for precision@k.

    Returns:
        RetrievalMetrics with hit_rate, mrr, precision_at_k, and k.
    """
    results: list[list[Any]] = []
    expected_keywords: list[list[str]] = []

    for question in questions:
        query_results = retriever_fn(question.question)
        # Ensure we only consider top-k results
        results.append(query_results[:k])
        expected_keywords.append(question.expected_keywords)

    hit_rate = calculate_hit_rate(results, expected_keywords)
    mrr = calculate_mrr(results, expected_keywords)
    precision_at_k = calculate_precision_at_k(results, expected_keywords, k)

    return RetrievalMetrics(
        hit_rate=hit_rate,
        mrr=mrr,
        precision_at_k=precision_at_k,
        k=k,
    )
