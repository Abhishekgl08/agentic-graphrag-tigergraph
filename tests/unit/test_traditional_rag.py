from rag.context_builder import select_context
from rag.deduplication import document_aware_deduplicate
from rag.hybrid_retriever import reciprocal_rank_fusion
from rag.models import RetrievedChunk
from rag.pipeline import TraditionalRAGPipeline


def chunk(identifier, document, text="evidence"):
    return RetrievedChunk(chunk_id=identifier, doc_id=document, text=text, title=document, approx_tokens=10)


def test_rrf_is_deterministic_and_dedup_keeps_best_chunk():
    semantic, lexical = [chunk("a1", "a"), chunk("b1", "b")], [chunk("b1", "b"), chunk("a2", "a")]
    fused = reciprocal_rank_fusion(semantic, lexical, limit=3)
    assert [item.chunk_id for item in fused] == ["b1", "a1", "a2"]
    assert [item.chunk_id for item in document_aware_deduplicate([chunk("a1", "a"), chunk("a2", "a"), chunk("b1", "b")])] == ["a1", "b1"]


def test_context_honors_limit_and_optional_budget():
    items = [chunk("a1", "a"), chunk("b1", "b"), chunk("c1", "c")]
    assert len(select_context(items, max_chunks=2)) == 2
    assert [item.chunk_id for item in select_context(items, max_chunks=3, token_budget=15)] == ["a1"]


class FakeVector:
    def search(self, question, limit): return [chunk("a1", "a"), chunk("a2", "a"), chunk("b1", "b")]
class FakeKeyword:
    def search(self, dataset_id, question, limit): return [chunk("b1", "b"), chunk("c1", "c")]
class FakeReranker:
    def rerank(self, question, chunks): return list(reversed(chunks))
class FakeAnswer:
    model = "test-model"
    def answer(self, instructions, user_input): return "Grounded answer [Source 1]"


def test_pipeline_keeps_300_stage_contract_and_returns_evidence():
    result = TraditionalRAGPipeline(FakeVector(), FakeKeyword(), FakeReranker(), FakeAnswer(), candidate_k=300, context_k=20).answer("dataset", "question")
    assert result.answer.startswith("Grounded")
    assert result.candidate_counts["hnsw"] == 3
    assert result.candidate_counts["context"] == 3
    assert len({item.doc_id for item in result.evidence}) == 3
