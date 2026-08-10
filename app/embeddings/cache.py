"""LRU cache with TTL for embeddings."""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """A cached embedding with its expiration time."""

    embedding: list[float]
    expires_at: float


class EmbeddingCache:
    """LRU cache with TTL for embedding vectors.

    Uses SHA256 hash of content as cache key. Entries expire after TTL seconds
    and are evicted when max_size is exceeded (least recently used first).

    Attributes:
        max_size: Maximum number of entries in the cache.
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 86400.0,  # 24 hours
    ) -> None:
        """Initialize the embedding cache.

        Args:
            max_size: Maximum number of entries to store. Defaults to 10000.
            ttl_seconds: TTL for entries in seconds. Defaults to 86400 (24h).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content.

        Args:
            content: The text content to hash.

        Returns:
            Hexadecimal string of the SHA256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, content: str) -> Optional[list[float]]:
        """Retrieve an embedding from cache.

        Args:
            content: The text content to look up.

        Returns:
            The cached embedding vector, or None if not found or expired.
        """
        key = self._hash_content(content)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check if entry has expired
        if time.time() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.embedding

    def put(self, content: str, embedding: list[float]) -> None:
        """Store an embedding in cache.

        Args:
            content: The text content (used to generate cache key).
            embedding: The embedding vector to cache.
        """
        key = self._hash_content(content)
        expires_at = time.time() + self.ttl_seconds

        # If key exists, update and move to end
        if key in self._cache:
            self._cache[key] = CacheEntry(embedding=embedding, expires_at=expires_at)
            self._cache.move_to_end(key)
            return

        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(embedding=embedding, expires_at=expires_at)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0.
            Returns 0.0 if no requests have been made.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
