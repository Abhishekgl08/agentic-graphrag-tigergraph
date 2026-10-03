from .models import RetrievedChunk


class TigerGraphVectorRetriever:
    def __init__(self, embedding_service, connection_factory, query_name: str):
        self.embedding_service, self.connection_factory, self.query_name, self._connection = embedding_service, connection_factory, query_name, None
    @property
    def connection(self):
        if self._connection is None: self._connection = self.connection_factory()
        return self._connection
    def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        vector = self.embedding_service.embed_texts([question])[0]
        result = self.connection.runInstalledQuery(self.query_name, params={"query_vector": vector})
        vertices, distances = [], {}
        for item in result:
            vertices = item.get("results", vertices)
            distances = item.get("@@distances", distances)
        chunks = []
        for vertex in vertices:
            attrs, chunk_id = vertex.get("attributes", {}), vertex["v_id"]
            chunks.append(RetrievedChunk(chunk_id=chunk_id, doc_id=attrs.get("document_id", ""), title=attrs.get("title", ""), section=attrs.get("section", ""), chunk_index=attrs.get("chunk_index", 0), chunk_type=attrs.get("chunk_type", "text"), approx_tokens=attrs.get("approx_tokens", 0), text=attrs.get("text", ""), score=distances.get(chunk_id)))
        return sorted(chunks, key=lambda chunk: float("inf") if chunk.score is None else chunk.score)[:limit]
