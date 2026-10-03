"""Single source of truth for the GraphRAG-to-TigerGraph schema contract."""

import json
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_CONTRACT_PATH = Path(__file__).with_name("schema_contract.json")


@lru_cache(maxsize=4)
def load_graph_contract(path: str | Path | None = None) -> dict:
    contract_path = Path(path or os.getenv("GRAPH_SCHEMA_CONTRACT_PATH", DEFAULT_CONTRACT_PATH))
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load graph schema contract: {contract_path}") from exc
    entities, relationships = contract.get("entity_types"), contract.get("relationships")
    if not isinstance(entities, dict) or not entities or not isinstance(relationships, dict) or not relationships:
        raise RuntimeError("Graph schema contract must define entity_types and relationships")
    for name, definition in entities.items():
        if not isinstance(name, str) or not isinstance(definition, dict) or not definition.get("vertex_type"):
            raise RuntimeError(f"Invalid entity contract for {name!r}")
    for name, definition in relationships.items():
        if not isinstance(definition, dict) or definition.get("source_type") not in entities or definition.get("target_type") not in entities:
            raise RuntimeError(f"Invalid relationship contract for {name!r}")
    return contract


def extraction_instructions(contract: dict | None = None) -> str:
    contract = contract or load_graph_contract()
    entity_types = ", ".join(contract["entity_types"])
    relationship_rules = "; ".join(
        f"{name}: {rule['source_type']} -> {rule['target_type']}" for name, rule in contract["relationships"].items()
    )
    return (
        "You are a knowledge graph extraction system. Extract only facts explicitly supported by the supplied chunk; never invent facts. "
        "Return JSON only with exactly the keys entities and relationships, both arrays. "
        f"Allowed entity types: {entity_types}. "
        "Do not create date, year, number, medal, ranking, score, or count entities. "
        f"Allowed relationships and REQUIRED endpoint types: {relationship_rules}. "
        "Do not emit a relationship if its endpoint types do not exactly match this contract; omitted facts are preferable to invalid facts. "
        "Each entity must have id, name, type, source_chunk_id. Each relationship must have source, source_type, relationship, target, target_type, source_chunk_id. "
        "Every relationship endpoint must be an entity extracted in this chunk. "
        "For WON_BY include medal only when explicitly stated, using exactly GOLD, SILVER, or BRONZE."
    )
