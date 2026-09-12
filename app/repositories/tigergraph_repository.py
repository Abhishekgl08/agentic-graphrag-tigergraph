import json
import time
from collections.abc import Callable
from ..embedding_models import ChunkRecord
from ..exceptions import EmbeddingError


class TigerGraphRepository:
    """Maps validated chunks to the versioned Chunk vertex contract."""
    def __init__(self, connection_factory, max_retries: int = 3, retry_base_seconds: float = 1.0):
        self._connection_factory, self._connection = connection_factory, None
        self.max_retries, self.retry_base_seconds = max_retries, retry_base_seconds

    @property
    def connection(self):
        if self._connection is None: self._connection = self._connection_factory()
        return self._connection

    def upsert_chunks(self, dataset_id: str, chunks: list[ChunkRecord], embeddings: list[list[float]], model: str, version: str, hashes: list[str]) -> None:
        if not (len(chunks) == len(embeddings) == len(hashes)): raise EmbeddingError("Chunk and embedding batch lengths differ")
        for chunk, vector, content_hash in zip(chunks, embeddings, hashes):
            attrs = {"text": chunk.text, "document_id": chunk.doc_id, "dataset_id": dataset_id, "title": chunk.title, "wikidata_qid": chunk.wikidata_qid or "", "wikipedia_pageid": int(chunk.wikipedia_pageid) if chunk.wikipedia_pageid is not None else 0, "chunk_index": chunk.chunk_index, "chunk_type": chunk.chunk_type, "section": chunk.section, "section_path": json.dumps(chunk.section_path, ensure_ascii=False), "approx_tokens": chunk.approx_tokens, "embedding": vector, "embedding_model": model, "embedding_version": version, "content_hash": content_hash}
            self._retry(lambda attrs=attrs, chunk_id=chunk.chunk_id: self.connection.upsertVertex("Chunk", chunk_id, attrs))

    def _retry(self, operation: Callable):
        last_error = None
        for attempt in range(self.max_retries):
            try: return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries: time.sleep(self.retry_base_seconds * (2 ** attempt))
        raise EmbeddingError("TigerGraph upsert failed after retries") from last_error
