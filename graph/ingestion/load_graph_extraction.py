"""Idempotent loader for one validated GraphRAG chunk extraction."""

import time
from collections.abc import Callable

from app.exceptions import TigerGraphLoadError
from app.config import settings
from app.tigergraph import create_connection
from graph.schema_contract import load_graph_contract

class GraphExtractionLoader:
    def __init__(self, connection_factory: Callable = create_connection, max_retries: int | None = None):
        self._connection_factory, self._connection, self.max_retries = connection_factory, None, max_retries if max_retries is not None else settings.graph_max_retries

    @property
    def connection(self):
        if self._connection is None:
            self._connection = self._connection_factory()
        return self._connection

    def load(self, chunk: dict, extraction: dict) -> None:
        """Upsert provenance and graph facts. Safe to call again after a crash."""
        document_id, chunk_id = extraction["document_id"], extraction["chunk_id"]
        contract = load_graph_contract()
        provenance = contract["provenance"]
        if chunk.get("doc_id") != document_id or chunk.get("chunk_id") != chunk_id:
            raise TigerGraphLoadError("Extraction does not belong to the supplied source chunk")
        self._retry(lambda: self.connection.upsertVertex(provenance["document_vertex"], document_id, {provenance["document_title_attribute"]: chunk.get("title", "")}))
        # Chunks are normally already written by the embedding pipeline. Upsert
        # the minimum portable attributes so GraphRAG also works independently.
        self._retry(lambda: self.connection.upsertVertex(provenance["chunk_vertex"], chunk_id, {provenance["chunk_text_attribute"]: chunk.get("text", ""), provenance["chunk_document_attribute"]: document_id}))
        self._retry(lambda: self.connection.upsertEdge(provenance["document_vertex"], document_id, provenance["document_chunk_edge"], provenance["chunk_vertex"], chunk_id, {}))
        for entity in extraction["entities"]:
            entity_contract = contract["entity_types"][entity["type"]]
            vertex_type = entity_contract["vertex_type"]
            attrs = {entity_contract["name_attribute"]: entity["name"]}
            self._retry(lambda vt=vertex_type, e=entity, a=attrs: self.connection.upsertVertex(vt, e["id"], a))
            mention_edge = entity_contract.get("mention_edge")
            if mention_edge:
                self._retry(lambda me=mention_edge, vt=vertex_type, e=entity: self.connection.upsertEdge("Chunk", chunk_id, me, vt, e["id"], {}))
        for relationship in extraction["relationships"]:
            attrs = {"medal": relationship["medal"]} if "medal" in relationship else {}
            self._retry(lambda r=relationship, a=attrs: self.connection.upsertEdge(
                r["source_type"], r["source"], r["relationship"], r["target_type"], r["target"], a))

    def verify_load(self, extraction: dict) -> None:
        """Confirm expected vertices and knowledge edges before journaling completion."""
        for entity in extraction["entities"]:
            contract = load_graph_contract()
            try:
                result = self.connection.getVerticesById(contract["entity_types"][entity["type"]]["vertex_type"], [entity["id"]])
            except Exception as exc:
                raise TigerGraphLoadError("TigerGraph load verification failed") from exc
            if not result:
                raise TigerGraphLoadError(f"TigerGraph did not return entity {entity['id']}")
        for relationship in extraction["relationships"]:
            try:
                result = self.connection.getEdges(
                    relationship["source_type"], relationship["source"], relationship["relationship"],
                    relationship["target_type"], relationship["target"],
                )
            except Exception as exc:
                raise TigerGraphLoadError("TigerGraph edge verification failed") from exc
            if not result:
                raise TigerGraphLoadError(
                    f"TigerGraph did not return {relationship['relationship']} for {relationship['source']}"
                )

    def _retry(self, operation: Callable) -> None:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                operation()
                return
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep((2, 5, 10)[attempt])
        raise TigerGraphLoadError("TigerGraph upsert failed after retries") from last_error
