from .models import RetrievedChunk


def document_aware_deduplicate(chunks: list[RetrievedChunk], max_chunks_per_document: int = 1) -> list[RetrievedChunk]:
    if max_chunks_per_document < 1: raise ValueError("max_chunks_per_document must be positive")
    result, counts = [], {}
    for chunk in chunks:
        count = counts.get(chunk.doc_id, 0)
        if count < max_chunks_per_document:
            result.append(chunk); counts[chunk.doc_id] = count + 1
    return result
