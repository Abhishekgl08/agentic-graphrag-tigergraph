import json

import pytest

from app.exceptions import GraphExtractionError
from graph.ingestion.load_graph_extraction import GraphExtractionLoader
from graph.validation.validate_extraction import validate_extraction


def extraction():
    return {
        "document_id": "Q1", "chunk_id": "Q1_chunk_0001", "model": "test",
        "entities": [
            {"id": "event_one", "name": " Event One ", "type": "Event", "source_chunk_id": "Q1_chunk_0001"},
            {"id": "athlete_one", "name": "Athlete One", "type": "Athlete", "source_chunk_id": "Q1_chunk_0001"},
        ],
        "relationships": [{"source": "event_one", "source_type": "Event", "relationship": "WON_BY", "target": "athlete_one", "target_type": "Athlete", "source_chunk_id": "Q1_chunk_0001", "medal": "GOLD"}],
        "usage": {}, "status": "extracted",
    }


def test_validation_normalizes_safe_whitespace_and_checks_endpoint_type():
    valid = validate_extraction(extraction())
    assert valid["entities"][0]["name"] == "Event One"
    invalid = extraction()
    invalid["relationships"][0]["target_type"] = "Country"
    with pytest.raises(GraphExtractionError):
        validate_extraction(invalid)
    invalid = extraction()
    invalid["relationships"][0]["relationship"] = "SUB_EVENT_OF"
    with pytest.raises(GraphExtractionError):
        validate_extraction(invalid)


class FakeTigerGraph:
    def __init__(self):
        self.vertices, self.edges = {}, []

    def upsertVertex(self, vertex_type, vertex_id, attrs):
        self.vertices[(vertex_type, vertex_id)] = attrs

    def upsertEdge(self, *args):
        self.edges.append(args)

    def getVerticesById(self, vertex_type, ids):
        return [{"v_id": item} for item in ids if (vertex_type, item) in self.vertices]

    def getEdges(self, source_type, source_id, edge_type, target_type, target_id):
        return [edge for edge in self.edges if edge[:5] == (source_type, source_id, edge_type, target_type, target_id)]


def test_loader_writes_provenance_and_verifies_entities():
    graph = FakeTigerGraph()
    loader = GraphExtractionLoader(connection_factory=lambda: graph, max_retries=1)
    result = validate_extraction(extraction())
    loader.load({"doc_id": "Q1", "chunk_id": "Q1_chunk_0001", "title": "Doc", "text": "text"}, result)
    loader.verify_load(result)
    assert ("Document", "Q1") in graph.vertices
    assert ("Chunk", "Q1_chunk_0001") in graph.vertices
    assert any(edge[2] == "MENTIONS_EVENT" for edge in graph.edges)
    assert any(edge[2] == "WON_BY" and edge[-1] == {"medal": "GOLD"} for edge in graph.edges)
