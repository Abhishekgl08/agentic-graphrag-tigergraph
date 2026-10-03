import json
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the project root
sys.path.insert(0, str(PROJECT_ROOT))


from app.tigergraph import create_connection


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_FILE = (
    PROJECT_ROOT
    / "graph"
    / "data"
    / "q303623_graph_canonical.json"
)

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c"
    / "chunks.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

DOCUMENT_ID = "Q303623"


# Maps extraction entity types -> TigerGraph vertex types
ENTITY_TYPE_MAP = {
    "Event": "Event",
    "Athlete": "Athlete",
    "Country": "Country",
    "Venue": "Venue",
    "Sport": "Sport",
    "OlympicGames": "OlympicGames",
    "Organization": "Organization",
    "Person": "Person",
    "Other": "Other",
}


# Maps entity type -> provenance edge
MENTION_EDGE_MAP = {
    "Event": "MENTIONS_EVENT",
    "Athlete": "MENTIONS_ATHLETE",
    "Country": "MENTIONS_COUNTRY",
    "Venue": "MENTIONS_VENUE",
    "Sport": "MENTIONS_SPORT",
    "OlympicGames": "MENTIONS_GAMES",
}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD SOURCE CHUNKS
# ============================================================

def load_document_chunks():

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:

            record = json.loads(line)

            if record.get("doc_id") == DOCUMENT_ID:
                chunks.append(record)

    return chunks


# ============================================================
# GET ATTRIBUTE VALUE
# ============================================================

def get_attribute(attributes, key):

    value = attributes.get(key)

    if value is None:
        return None

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GraphRAG - Load Q303623 into TigerGraph")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    graph_data = load_json(CANONICAL_FILE)
    chunks = load_document_chunks()

    print(f"\nCanonical entities: {len(graph_data['entities'])}")
    print(
        f"Canonical relationships: "
        f"{len(graph_data['relationships'])}"
    )
    print(f"Document chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            f"No chunks found for document {DOCUMENT_ID}"
        )

    # --------------------------------------------------------
    # Connect to TigerGraph
    # --------------------------------------------------------

    print("\nConnecting to TigerGraph...")

    tg = create_connection()

    print("Connected.")

    # --------------------------------------------------------
    # 1. DOCUMENT
    # --------------------------------------------------------

    title = chunks[0].get("title", "")

    print("\n" + "-" * 70)
    print("1. Loading Document")
    print("-" * 70)

    tg.upsertVertex(
        "Document",
        DOCUMENT_ID,
        {
            "title": title
        },
    )

    print(f"Document loaded: {DOCUMENT_ID}")

    # --------------------------------------------------------
    # 2. DOCUMENT -> CHUNK
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. Loading Document -> Chunk relationships")
    print("-" * 70)

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]

        tg.upsertEdge(
            "Document",
            DOCUMENT_ID,
            "HAS_CHUNK",
            "Chunk",
            chunk_id,
            {},
        )

        print(
            f"  {DOCUMENT_ID}"
            f" --HAS_CHUNK--> "
            f"{chunk_id}"
        )

    # --------------------------------------------------------
    # 3. ENTITY VERTICES
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. Loading entity vertices")
    print("-" * 70)

    entity_lookup = {}

    entity_count = 0

    for entity in graph_data["entities"]:

        entity_type = entity["type"]

        if entity_type not in ENTITY_TYPE_MAP:
            print(
                f"Skipping unsupported entity type: "
                f"{entity_type}"
            )
            continue

        vertex_type = ENTITY_TYPE_MAP[entity_type]

        entity_id = entity["id"]
        name = entity["name"]

        attributes = entity.get("attributes", {})

        # ----------------------------------------------------
        # Build TigerGraph attributes
        # ----------------------------------------------------

        vertex_attributes = {
            "name": name
        }

        # Event-specific attributes
        if vertex_type == "Event":

            competitor_count = get_attribute(
                attributes,
                "competitor_count",
            )

            event_date = get_attribute(
                attributes,
                "date",
            )

            if competitor_count is not None:

                try:
                    vertex_attributes[
                        "competitor_count"
                    ] = int(competitor_count)

                except (ValueError, TypeError):
                    pass

            if event_date is not None:

                vertex_attributes[
                    "event_date"
                ] = str(event_date)

        # OlympicGames-specific attributes
        if vertex_type == "OlympicGames":

            year = get_attribute(
                attributes,
                "year",
            )

            if year is not None:

                try:
                    vertex_attributes["year"] = int(year)

                except (ValueError, TypeError):
                    pass

        tg.upsertVertex(
            vertex_type,
            entity_id,
            vertex_attributes,
        )

        entity_lookup[entity_id] = {
            "type": vertex_type,
            "name": name,
        }

        entity_count += 1

        print(
            f"  [{vertex_type}] "
            f"{name} "
            f"-> {entity_id}"
        )

    print(f"\nEntity vertices loaded: {entity_count}")

    # --------------------------------------------------------
    # 4. CHUNK -> ENTITY PROVENANCE EDGES
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. Loading Chunk -> Entity provenance")
    print("-" * 70)

    mention_count = 0

    for entity in graph_data["entities"]:

        entity_type = entity["type"]

        if entity_type not in MENTION_EDGE_MAP:
            continue

        vertex_type = ENTITY_TYPE_MAP[entity_type]

        edge_type = MENTION_EDGE_MAP[entity_type]

        entity_id = entity["id"]

        for chunk_id in entity.get(
            "source_chunk_ids",
            [],
        ):

            tg.upsertEdge(
                "Chunk",
                chunk_id,
                edge_type,
                vertex_type,
                entity_id,
                {},
            )

            mention_count += 1

            print(
                f"  {chunk_id}"
                f" --{edge_type}--> "
                f"{entity_id}"
            )

    print(
        f"\nProvenance edges loaded: "
        f"{mention_count}"
    )

    # --------------------------------------------------------
    # 5. KNOWLEDGE GRAPH RELATIONSHIPS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. Loading knowledge relationships")
    print("-" * 70)

    relationship_count = 0

    for relationship in graph_data[
        "relationships"
    ]:

        source_id = relationship["source_id"]
        target_id = relationship["target_id"]
        edge_type = relationship["type"]

        if source_id not in entity_lookup:
            print(
                f"WARNING: source entity not found: "
                f"{source_id}"
            )
            continue

        if target_id not in entity_lookup:
            print(
                f"WARNING: target entity not found: "
                f"{target_id}"
            )
            continue

        source_type = entity_lookup[
            source_id
        ]["type"]

        target_type = entity_lookup[
            target_id
        ]["type"]

        # Event -> Event PART_OF relationships represent
        # sub-events, so map them to SUB_EVENT_OF.
        if (
            edge_type == "PART_OF"
            and source_type == "Event"
            and target_type == "Event"
        ):
            edge_type = "SUB_EVENT_OF"

        edge_attributes = {}



        # WON_BY has the medal attribute
        if edge_type == "WON_BY":

            medal = relationship.get(
                "attributes",
                {},
            ).get("medal")

            if medal:
                edge_attributes[
                    "medal"
                ] = str(medal)

        tg.upsertEdge(
            source_type,
            source_id,
            edge_type,
            target_type,
            target_id,
            edge_attributes,
        )

        relationship_count += 1

        attribute_text = (
            f" {edge_attributes}"
            if edge_attributes
            else ""
        )

        print(
            f"  {source_id}"
            f" --{edge_type}--> "
            f"{target_id}"
            f"{attribute_text}"
        )

    print(
        f"\nKnowledge relationships loaded: "
        f"{relationship_count}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPH LOAD COMPLETE")
    print("=" * 70)

    print(f"Document: {DOCUMENT_ID}")
    print(f"Chunks: {len(chunks)}")
    print(f"Entity vertices: {entity_count}")
    print(f"Provenance edges: {mention_count}")
    print(
        f"Knowledge relationships: "
        f"{relationship_count}"
    )

    print("\nTigerGraph now contains the Q303623 graph.")


if __name__ == "__main__":
    main()