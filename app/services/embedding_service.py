import os
import time
from collections.abc import Callable
from openai import OpenAI
from ..exceptions import EmbeddingError


class EmbeddingService:
    """OpenAI embedding client; it deliberately has no database knowledge."""
    def __init__(self, model: str, dimension: int, max_retries: int = 3, retry_base_seconds: float = 1.0, client=None):
        self.model, self.dimension = model, dimension
        self.max_retries, self.retry_base_seconds = max_retries, retry_base_seconds
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"): raise EmbeddingError("OpenAI embedding service is not configured")
            self._client = OpenAI()
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts): raise EmbeddingError("Embedding input must contain non-empty text")
        response = self._retry(lambda: self.client.embeddings.create(model=self.model, input=texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts): raise EmbeddingError("Embedding response count does not match request")
        vectors = [item.embedding for item in ordered]
        if any(len(vector) != self.dimension for vector in vectors): raise EmbeddingError(f"Embedding dimension must be {self.dimension}")
        return vectors

    def _retry(self, operation: Callable):
        last_error = None
        for attempt in range(self.max_retries):
            try: return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries: time.sleep(self.retry_base_seconds * (2 ** attempt))
        raise EmbeddingError("Embedding request failed after retries") from last_error
