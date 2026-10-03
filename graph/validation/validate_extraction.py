from copy import deepcopy

from app.exceptions import GraphExtractionError
from graph.schema_contract import load_graph_contract


def validate_extraction(extraction: dict) -> dict:
    """Validate and safely normalize one chunk-level Groq extraction.

    References are deliberately limited to entities in the same response.  This
    keeps a chunk independently loadable and avoids accepting unsupported,
    cross-chunk references before entity resolution is introduced.
    """
    if not isinstance(extraction, dict) or not isinstance(extraction.get("entities"), list) or not isinstance(extraction.get("relationships"), list):
        raise GraphExtractionError("Extraction must contain entities and relationships lists")
    contract = load_graph_contract()
    entity_types, relationship_types = set(contract["entity_types"]), set(contract["relationships"])
    normalized = deepcopy(extraction)
    chunk_id = normalized.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise GraphExtractionError("Extraction is missing a chunk_id")
    normalized["chunk_id"] = chunk_id.strip()
    ids: dict[str, str] = {}
    for entity in normalized["entities"]:
        required = {"id", "name", "type", "source_chunk_id"}
        if not isinstance(entity, dict) or not required <= entity.keys():
            raise GraphExtractionError("Invalid entity extraction")
        for field in ("id", "name", "type", "source_chunk_id"):
            if not isinstance(entity[field], str):
                raise GraphExtractionError("Entity fields must be strings")
            entity[field] = " ".join(entity[field].strip().split())
        if entity["type"] not in entity_types or entity["source_chunk_id"] != normalized["chunk_id"] or not entity["id"] or not entity["name"] or entity["id"] in ids:
            raise GraphExtractionError("Invalid entity extraction")
        ids[entity["id"]] = entity["type"]
    for relationship in normalized["relationships"]:
        required = {"source", "source_type", "relationship", "target", "target_type", "source_chunk_id"}
        if not isinstance(relationship, dict) or not required <= relationship.keys():
            raise GraphExtractionError("Invalid relationship extraction")
        for field in required:
            if not isinstance(relationship[field], str):
                raise GraphExtractionError("Relationship fields must be strings")
            relationship[field] = " ".join(relationship[field].strip().split())
        rule = contract["relationships"].get(relationship["relationship"])
        if relationship["relationship"] not in relationship_types or relationship["source_type"] not in entity_types or relationship["target_type"] not in entity_types or relationship["source_chunk_id"] != normalized["chunk_id"] or relationship["source"] not in ids or relationship["target"] not in ids or ids[relationship["source"]] != relationship["source_type"] or ids[relationship["target"]] != relationship["target_type"] or (relationship["source_type"], relationship["target_type"]) != (rule["source_type"], rule["target_type"]):
            raise GraphExtractionError("Invalid relationship extraction")
        if "medal" in relationship:
            allowed_medals = (rule.get("attributes", {}).get("medal", [])) if rule else []
            if relationship["relationship"] != "WON_BY" or relationship["medal"] not in allowed_medals:
                raise GraphExtractionError("Invalid relationship medal")
    return normalized
