from .models import RetrievedChunk


def select_context(chunks: list[RetrievedChunk], max_chunks: int = 20, token_budget: int | None = None) -> list[RetrievedChunk]:
    selected, used = [], 0
    for chunk in chunks:
        if token_budget is not None and selected and used + chunk.approx_tokens > token_budget: continue
        selected.append(chunk); used += chunk.approx_tokens
        if len(selected) == max_chunks: break
    return selected


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Source {index} | chunk_id={chunk.chunk_id} | document_id={chunk.doc_id} | title={chunk.title} | section={chunk.section}]\n{chunk.text}" for index, chunk in enumerate(chunks, start=1))
