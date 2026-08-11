"""
Evaluation framework for Fiscus-C RAG system.

This module provides:
- Golden set of test questions for evaluation
- Retrieval metrics (hit rate, MRR, precision@k)
- Tools for benchmarking retrieval quality
"""

from tests.evaluation.golden_set import GOLDEN_SET, GoldenQuestion
from tests.evaluation.metrics import (
    RetrievalMetrics,
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_at_k,
    evaluate_retrieval,
)

__all__ = [
    "GOLDEN_SET",
    "GoldenQuestion",
    "RetrievalMetrics",
    "calculate_hit_rate",
    "calculate_mrr",
    "calculate_precision_at_k",
    "evaluate_retrieval",
]
