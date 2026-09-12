import json
from pathlib import Path
from app.services.ingestion_service import IngestionService
from app.storage.local_storage import LocalDatasetStorage


class FakeEmbeddingService:
    model = "text-embedding-3-small"
    def __init__(self): self.calls = 0
    def embed_texts(self, texts):
        self.calls += 1
        return [[0.0] * 1536 for _ in texts]


class FakeRepository:
    def __init__(self): self.calls = []
    def upsert_chunks(self, dataset_id, chunks, embeddings, model, version, hashes): self.calls.append((dataset_id, chunks, embeddings, model, version, hashes))


def make_dataset(tmp_path: Path):
    storage = LocalDatasetStorage(tmp_path / "processed")
    paths = storage.create("dataset-1")
    chunks = [
        {"chunk_id": "chunk-1", "doc_id": "doc-1", "title": "Title", "wikidata_qid": None, "wikipedia_pageid": None, "chunk_index": 0, "chunk_type": "text", "section": "introduction", "section_path": ["introduction"], "approx_tokens": 2, "text": "first text"},
        {"chunk_id": "chunk-2", "doc_id": "doc-1", "title": "Title", "wikidata_qid": None, "wikipedia_pageid": None, "chunk_index": 1, "chunk_type": "text", "section": "introduction", "section_path": ["introduction"], "approx_tokens": 2, "text": "second text"},
    ]
    (paths.working_dir / "chunks.tmp.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    storage.commit(paths, {"dataset_id": "dataset-1", "status": "completed"}, {"passed": True})
    return storage


def test_ingestion_batches_and_skips_unchanged_restart(tmp_path):
    storage, embeddings, repository = make_dataset(tmp_path), FakeEmbeddingService(), FakeRepository()
    service = IngestionService(storage, embeddings, repository, batch_size=2, embedding_version="v1")
    first = service.queue("dataset-1")
    result = service.run("dataset-1", first.job_id)
    assert result.status == "completed" and result.stored_chunks == 2 and len(repository.calls) == 1
    second = service.queue("dataset-1")
    restarted = service.run("dataset-1", second.job_id)
    assert restarted.status == "completed" and restarted.skipped_chunks == 2 and len(repository.calls) == 1
