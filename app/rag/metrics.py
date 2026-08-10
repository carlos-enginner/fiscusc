"""Metrics collection for document ingestion pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class IngestionMetrics:
    """Metrics collected during document ingestion.

    Tracks timing, counts, and throughput for each stage of the
    ingestion pipeline: extraction, chunking, embedding, and database storage.
    """

    # Timings (milliseconds)
    extraction_ms: float = 0.0
    chunking_ms: float = 0.0
    embedding_ms: float = 0.0
    db_ms: float = 0.0
    total_ms: float = 0.0

    # Counts
    chunks_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    incremental_reused: int = 0

    # Throughput
    chunks_per_sec: float = 0.0
    tokens_per_sec: float = 0.0

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def finalize(self) -> None:
        """Calculate total_ms and throughput metrics.

        Should be called after all pipeline stages complete.
        Sets end_time if not already set.
        """
        if self.end_time is None:
            self.end_time = datetime.now()

        # Calculate total time from timestamps
        delta = self.end_time - self.start_time
        self.total_ms = delta.total_seconds() * 1000

        # Calculate chunks per second
        if self.total_ms > 0:
            self.chunks_per_sec = (self.chunks_count / self.total_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Export metrics to a nested dictionary for JSON/logging.

        Returns:
            Dictionary with structure:
            {
                "timings": {...},
                "counts": {...},
                "throughput": {...},
                "timestamps": {...}
            }
        """
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        )

        return {
            "timings": {
                "extraction_ms": self.extraction_ms,
                "chunking_ms": self.chunking_ms,
                "embedding_ms": self.embedding_ms,
                "db_ms": self.db_ms,
                "total_ms": self.total_ms,
            },
            "counts": {
                "chunks_count": self.chunks_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "incremental_reused": self.incremental_reused,
            },
            "throughput": {
                "chunks_per_sec": self.chunks_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "cache_hit_rate": cache_hit_rate,
            },
            "timestamps": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
            },
        }
