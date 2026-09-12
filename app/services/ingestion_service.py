import hashlib
import json
import logging
from itertools import islice
from time import monotonic
from uuid import uuid4
from ..embedding_models import ChunkRecord, IngestionJob
from ..exceptions import IngestionConflictError

logger = logging.getLogger(__name__)


class IngestionService:
    """Restartable chunks -> embeddings -> TigerGraph workflow, independent of HTTP."""
    def __init__(self, storage, embedding_service, repository, batch_size: int, embedding_version: str):
        self.storage, self.embedding_service, self.repository = storage, embedding_service, repository
        self.batch_size, self.embedding_version = batch_size, embedding_version

    def queue(self, dataset_id: str) -> IngestionJob:
        self.storage.dataset_dir(dataset_id)
        prior = self._load(dataset_id)
        if prior and prior.status in {"queued", "processing"}: raise IngestionConflictError("An ingestion job is already active for this dataset")
        job = IngestionJob(job_id=str(uuid4()), dataset_id=dataset_id, status="queued", embedding_model=self.embedding_service.model, embedding_version=self.embedding_version)
        self._save(job, completed_hashes=(prior or {}).model_extra.get("completed_hashes", {}) if prior else {})
        return job

    def run(self, dataset_id: str, job_id: str) -> IngestionJob:
        job = self._load(dataset_id)
        if job is None or job.job_id != job_id: raise ValueError("Unknown ingestion job")
        job.status, job.error = "processing", None
        completed_hashes = job.model_extra.get("completed_hashes", {})
        started = monotonic(); self._save(job, completed_hashes)
        try:
            chunk_iter = self._read_chunks(dataset_id)
            while batch := list(islice(chunk_iter, self.batch_size)):
                job.total_chunks += len(batch)
                hashes = [hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in batch]
                pending = [(chunk, content_hash) for chunk, content_hash in zip(batch, hashes) if completed_hashes.get(chunk.chunk_id) != content_hash]
                job.skipped_chunks += len(batch) - len(pending)
                if pending:
                    chunks, pending_hashes = zip(*pending)
                    vectors = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
                    job.embedded_chunks += len(chunks)
                    self.repository.upsert_chunks(dataset_id, list(chunks), vectors, self.embedding_service.model, self.embedding_version, list(pending_hashes))
                    completed_hashes.update({chunk.chunk_id: content_hash for chunk, content_hash in pending})
                    job.stored_chunks += len(chunks)
                job.processed_chunks += len(batch)
                self._save(job, completed_hashes)
                logger.info("ingestion_progress dataset_id=%s job_id=%s processed=%s stored=%s skipped=%s", dataset_id, job_id, job.processed_chunks, job.stored_chunks, job.skipped_chunks)
            job.status = "completed"
        except Exception as exc:
            job.status, job.error = "failed", "Ingestion failed; see server logs for safe diagnostic details"
            logger.exception("ingestion_failed dataset_id=%s job_id=%s", dataset_id, job_id)
        self._save(job, completed_hashes)
        logger.info("ingestion_finished dataset_id=%s job_id=%s status=%s elapsed_seconds=%.3f", dataset_id, job_id, job.status, monotonic() - started)
        return job

    def get_job(self, dataset_id: str): return self._load(dataset_id)

    def _read_chunks(self, dataset_id):
        for raw in self.storage.iter_chunks(dataset_id):
            yield ChunkRecord.model_validate(raw)

    def _job_path(self, dataset_id): return self.storage.dataset_dir(dataset_id) / "ingestion.json"
    def _load(self, dataset_id):
        path = self._job_path(dataset_id)
        return IngestionJob.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
    def _save(self, job, completed_hashes):
        payload = job.model_dump(); payload["completed_hashes"] = completed_hashes
        self._job_path(job.dataset_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
