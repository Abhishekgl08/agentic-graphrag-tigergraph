import re
from collections import Counter
from ..chunking.constants import SECTION_NAMES
from ..exceptions import DatasetValidationError


REQUIRED_RECORD_FIELDS = ("doc_id", "title", "text")
REQUIRED_CHUNK_FIELDS = ("chunk_id", "doc_id", "title", "chunk_index", "chunk_type", "section", "section_path", "approx_tokens", "text")


class ValidationService:
    def validate_record(self, record: object, line_number: int, seen_ids: set[str]) -> None:
        if not isinstance(record, dict): raise DatasetValidationError(f"Line {line_number}: record must be an object")
        missing = [name for name in REQUIRED_RECORD_FIELDS if name not in record]
        if missing: raise DatasetValidationError(f"Line {line_number}: missing required fields: {', '.join(missing)}")
        if not all(isinstance(record[name], str) for name in REQUIRED_RECORD_FIELDS): raise DatasetValidationError(f"Line {line_number}: doc_id, title, and text must be strings")
        if not record["doc_id"].strip() or not record["title"].strip() or not record["text"].strip(): raise DatasetValidationError(f"Line {line_number}: required fields must be non-empty")
        if record["doc_id"] in seen_ids: raise DatasetValidationError(f"Line {line_number}: duplicate doc_id")
        seen_ids.add(record["doc_id"])

    def validate_chunks(self, chunks: list[dict], document_ids: set[str]) -> None:
        ids = set()
        for chunk in chunks:
            missing = [name for name in REQUIRED_CHUNK_FIELDS if name not in chunk]
            if missing or chunk["chunk_id"] in ids or chunk["doc_id"] not in document_ids or not chunk["text"].strip() or chunk["chunk_type"] not in {"text", "table", "infobox"} or chunk["approx_tokens"] < 1 or chunk["chunk_index"] < 0:
                raise DatasetValidationError("Invalid generated chunk")
            ids.add(chunk["chunk_id"])

    def integrity(self, originals: dict[str, str], chunks_by_doc: dict[str, list[dict]]) -> dict:
        missing_docs, unknown_docs, mismatches = [], [], []
        for doc_id, text in originals.items():
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks: missing_docs.append(doc_id); continue
            original_tokens = self._tokens(self._remove_headings(text))
            rebuilt_tokens = self._tokens("\n".join(c["text"] for c in sorted(doc_chunks, key=lambda c: c["chunk_index"])))
            if Counter(original_tokens) != Counter(rebuilt_tokens): mismatches.append(doc_id)
        unknown_docs = sorted(set(chunks_by_doc) - set(originals))
        return {"passed": not missing_docs and not unknown_docs and not mismatches, "missing_documents": missing_docs, "unknown_documents": unknown_docs, "token_mismatches": mismatches}

    @staticmethod
    def _tokens(text): return re.findall(r"\S+", text.replace("\r\n", "\n").replace("\r", "\n"))
    @staticmethod
    def _remove_headings(text):
        return "\n".join(line.strip() for line in text.splitlines() if line.strip() and not ("|" not in line and "!!" not in line and " ".join(line.strip().lower().split()) in SECTION_NAMES))
