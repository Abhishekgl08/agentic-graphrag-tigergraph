import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c"
    / "chunks.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "graph"
    / "data"
    / "q303623_graph_extraction.json"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_DOCUMENT_ID = "Q303623"

load_dotenv(ENV_FILE)

client = OpenAI()


# ============================================================
# LOAD DOCUMENT CHUNKS
# ============================================================

def load_document_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("doc_id") == TARGET_DOCUMENT_ID:
                chunks.append(record)

    return chunks


# ============================================================
# LLM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an information extraction system for a GraphRAG pipeline.

Extract factual entities, entity attributes, and relationships
from the supplied source text.

IMPORTANT RULES:

1. Extract ONLY facts explicitly supported by the source text.
2. Never invent facts.
3. Do not infer facts that are not stated.
4. Keep entity names canonical and concise.
5. Do not create a separate entity for a date.
6. Do not create a separate entity for a numeric value.
7. Dates, counts, scores, and similar values should be stored
   as attributes of the appropriate entity.

Allowed entity types:

- Event
- Athlete
- Country
- Venue
- Sport
- OlympicGames
- Organization
- Person
- Other

For this Olympic corpus, prefer:
Event, Athlete, Country, Venue, Sport, OlympicGames.

Relationship rules:

Use concise relationship types such as:

- PART_OF
- AT_VENUE
- IN_SPORT
- WON_BY
- REPRESENTS
- PARTICIPATED_IN

For medal relationships:

Use:

Event --WON_BY--> Athlete

and put the medal in the relationship attribute:

"medal": "GOLD"

or:

"medal": "SILVER"

or:

"medal": "BRONZE"

Do NOT create separate GOLD_WON_BY,
SILVER_WON_BY or BRONZE_WON_BY relationship types.

For each entity mention, include the source chunk ID.

For each relationship, include the source chunk ID.

Entity attributes should only contain facts explicitly stated
in the source.

Examples of useful Event attributes:

- date
- competitor_count
- result
- medal
- gender

Do not force attributes if the source does not provide them.

Return valid JSON only.
"""


# ============================================================
# EXTRACT ONE CHUNK
# ============================================================

def extract_from_chunk(chunk):

    user_prompt = f"""
SOURCE CHUNK ID:
{chunk["chunk_id"]}

DOCUMENT ID:
{chunk["doc_id"]}

DOCUMENT TITLE:
{chunk.get("title", "")}

SOURCE TEXT:
{chunk["text"]}

Return exactly this JSON structure:

{{
  "entities": [
    {{
      "name": "canonical entity name",
      "type": "Event | Athlete | Country | Venue | Sport | OlympicGames | Organization | Person | Other",
      "attributes": {{}},
      "source_chunk_id": "{chunk["chunk_id"]}"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "type": "PART_OF | AT_VENUE | IN_SPORT | WON_BY | REPRESENTS | PARTICIPATED_IN",
      "target": "target entity name",
      "attributes": {{}},
      "source_chunk_id": "{chunk["chunk_id"]}"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-5.6",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


# ============================================================
# MERGE ENTITIES
# ============================================================

def merge_entities(all_results):

    merged = {}

    for result in all_results:

        for entity in result.get("entities", []):

            key = (
                entity["type"].strip().lower(),
                entity["name"].strip().lower(),
            )

            if key not in merged:

                merged[key] = {
                    "name": entity["name"].strip(),
                    "type": entity["type"].strip(),
                    "attributes": {},
                    "source_chunk_ids": [],
                }

            # Merge attributes
            merged[key]["attributes"].update(
                entity.get("attributes", {})
            )

            # Preserve provenance
            chunk_id = entity.get("source_chunk_id")

            if (
                chunk_id
                and chunk_id not in merged[key]["source_chunk_ids"]
            ):
                merged[key]["source_chunk_ids"].append(chunk_id)

    return list(merged.values())


# ============================================================
# MERGE RELATIONSHIPS
# ============================================================

def merge_relationships(all_results):

    merged = {}

    for result in all_results:

        for rel in result.get("relationships", []):

            source = rel["source"].strip()
            rel_type = rel["type"].strip()
            target = rel["target"].strip()

            attributes = rel.get("attributes", {})

            key = (
                source.lower(),
                rel_type.lower(),
                target.lower(),
                json.dumps(attributes, sort_keys=True),
            )

            if key not in merged:

                merged[key] = {
                    "source": source,
                    "type": rel_type,
                    "target": target,
                    "attributes": attributes,
                    "source_chunk_ids": [],
                }

            chunk_id = rel.get("source_chunk_id")

            if (
                chunk_id
                and chunk_id not in merged[key]["source_chunk_ids"]
            ):
                merged[key]["source_chunk_ids"].append(chunk_id)

    return list(merged.values())


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GraphRAG - Full Document Extraction")
    print("=" * 70)

    print(f"\nDocument: {TARGET_DOCUMENT_ID}")

    chunks = load_document_chunks()

    print(f"Chunks found: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            f"No chunks found for {TARGET_DOCUMENT_ID}"
        )

    all_results = []

    for index, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 70)
        print(
            f"Processing chunk {index}/{len(chunks)}: "
            f"{chunk['chunk_id']}"
        )
        print("-" * 70)

        result = extract_from_chunk(chunk)

        print(
            f"Entities extracted: "
            f"{len(result.get('entities', []))}"
        )

        print(
            f"Relationships extracted: "
            f"{len(result.get('relationships', []))}"
        )

        all_results.append(result)

    # --------------------------------------------------------
    # Merge duplicate entities and relationships
    # --------------------------------------------------------

    entities = merge_entities(all_results)
    relationships = merge_relationships(all_results)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = {
        "document_id": TARGET_DOCUMENT_ID,
        "chunk_count": len(chunks),
        "entities": entities,
        "relationships": relationships,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"Document: {TARGET_DOCUMENT_ID}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Unique entities: {len(entities)}")
    print(f"Unique relationships: {len(relationships)}")

    print("\nEntity types:")

    type_counts = {}

    for entity in entities:

        entity_type = entity["type"]

        type_counts[entity_type] = (
            type_counts.get(entity_type, 0) + 1
        )

    for entity_type, count in sorted(type_counts.items()):

        print(f"  {entity_type}: {count}")

    print("\nRelationships:")

    for relationship in relationships:

        attributes = relationship.get("attributes", {})

        attribute_text = (
            f" {attributes}"
            if attributes
            else ""
        )

        print(
            f"  {relationship['source']}"
            f" --{relationship['type']}-->"
            f" {relationship['target']}"
            f"{attribute_text}"
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()