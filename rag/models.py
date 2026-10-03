from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    section: str = ""
    chunk_index: int = 0
    chunk_type: str = "text"
    approx_tokens: int = 0
    text: str
    score: float | None = None


class RAGAnswer(BaseModel):
    answer: str
    evidence: list[RetrievedChunk]
    candidate_counts: dict[str, int]
    context_approx_tokens: int
    model: str
