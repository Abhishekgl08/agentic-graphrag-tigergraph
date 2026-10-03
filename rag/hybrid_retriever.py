from collections import defaultdict


def reciprocal_rank_fusion(*rankings, constant: int = 60, limit: int = 300):
    scores = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] += 1.0 / (constant + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [chunks[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in ranked_ids]
