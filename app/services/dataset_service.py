import json
import logging
import time
from pathlib import Path
from ..chunking.constants import CHUNKING_VERSION
from ..exceptions import DatasetValidationError, UploadTooLargeError
from ..models import ProcessingStats
from .chunking_service import ChunkingService
from .validation_service import ValidationService

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, storage, max_upload_size_mb: int, chunking=None, validator=None):
        self.storage = storage
        self.max_upload_bytes = max_upload_size_mb * 1024 * 1024
        self.chunking = chunking or ChunkingService()
        self.validator = validator or ValidationService()

    def ingest(self, source, source_filename: str) -> dict:
        paths = self.storage.create()
        stats, seen_ids, total_bytes, started = ProcessingStats(), set(), 0, time.monotonic()
        validation = {"passed": True, "missing_documents": [], "unknown_documents": [], "token_mismatches": []}
        try:
            with (paths.working_dir / "chunks.tmp.jsonl").open("w", encoding="utf-8") as output:
                for line_number, raw_line in enumerate(source, 1):
                    total_bytes += len(raw_line)
                    if total_bytes > self.max_upload_bytes: raise UploadTooLargeError("Uploaded file exceeds configured size limit")
                    if not raw_line.strip(): continue
                    try: record = json.loads(raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise DatasetValidationError(f"Line {line_number}: invalid JSONL") from exc
                    self.validator.validate_record(record, line_number, seen_ids)
                    chunks = self.chunking.create_chunks(record)
                    self.validator.validate_chunks(chunks, {record["doc_id"]})
                    integrity = self.validator.integrity({record["doc_id"]: record["text"]}, {record["doc_id"]: chunks})
                    if not integrity["passed"]: raise DatasetValidationError(f"Line {line_number}: generated chunks failed content integrity")
                    stats.document_count += 1
                    for chunk in chunks:
                        output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        stats.chunk_count += 1; stats.approx_tokens += chunk["approx_tokens"]
                        if chunk["chunk_type"] == "infobox": stats.infobox_chunks += 1
                        elif chunk["chunk_type"] == "table": stats.table_chunks += 1
                        else: stats.text_chunks += 1
            if not stats.document_count: raise DatasetValidationError("Dataset contains no records")
            metadata = {"dataset_id": paths.dataset_id, "source_filename": Path(source_filename).name, "status": "completed", "document_count": stats.document_count, "chunk_count": stats.chunk_count, "infobox_chunks": stats.infobox_chunks, "text_chunks": stats.text_chunks, "table_chunks": stats.table_chunks, "approx_tokens": stats.approx_tokens, "chunking_version": CHUNKING_VERSION, "processing_duration_seconds": round(time.monotonic() - started, 3)}
            self.storage.commit(paths, metadata, validation)
            logger.info("dataset_completed dataset_id=%s filename=%s documents=%s chunks=%s duration=%s", paths.dataset_id, metadata["source_filename"], stats.document_count, stats.chunk_count, metadata["processing_duration_seconds"])
            return metadata
        except Exception:
            self.storage.discard(paths)
            logger.exception("dataset_failed dataset_id=%s filename=%s", paths.dataset_id, Path(source_filename).name)
            raise

    def get_metadata(self, dataset_id): return self.storage.metadata(dataset_id)
    def list_chunks(self, dataset_id, offset=0, limit=100):
        chunks = self.storage.iter_chunks(dataset_id)
        items = [chunk for index, chunk in enumerate(chunks) if offset <= index < offset + limit]
        return {"items": items, "offset": offset, "limit": limit, "count": len(items)}
    def get_chunk(self, dataset_id, chunk_id):
        for chunk in self.storage.iter_chunks(dataset_id):
            if chunk["chunk_id"] == chunk_id: return chunk
        return None
