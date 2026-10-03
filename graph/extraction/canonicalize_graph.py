import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

INPUT_FILE = (
    PROJECT_ROOT
    / "graph"
    / "data"
    / "q303623_graph_extraction.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "graph"
    / "data"
    / "q303623_graph_canonical.json"
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv(ENV_FILE)

client = OpenAI()


# ============================================================
# LLM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an entity resolution system for a GraphRAG knowledge graph.

You are given entities extracted independently from multiple
chunks of the SAME document.

Your job is to identify entities that refer to the same real-world
entity and assign them one canonical representation.

IMPORTANT RULES:

1. Merge only entities that clearly refer to the same entity.
2. Do not merge different people, events, venues, countries, etc.
3. Preserve the most informative canonical name.
4. Country abbreviations must be normalized when obvious:
   HUN -> Hungary
   POR -> Portugal
   GER -> Germany
5. Minor formatting differences should be merged.

Examples:

"2012 Olympics"
"2012 Summer Olympics"
"2012 Summer Olympic Games"

-> one OlympicGames entity:
"2012 Summer Olympics"

Similarly:

"Men's canoe sprint K-2 1000 metres"
"Men's canoe sprint K-2 1,000 metres"

may refer to the same event and should be merged when
the surrounding information confirms that.

For every canonical entity create a stable ID.

Use these ID formats:

Event:
event_<slug>

Athlete:
athlete_<slug>

Country:
country_<slug>

Venue:
venue_<slug>

Sport:
sport_<slug>

OlympicGames:
games_<slug>

Organization:
organization_<slug>

Person:
person_<slug>

Other:
other_<slug>

Do not create new entities that were not present in the input.

Relationships must be updated to use the canonical entity names.

Preserve all source_chunk_ids.

Return valid JSON only.
"""


# ============================================================
# LOAD EXTRACTION
# ============================================================

def load_extraction():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# CANONICALIZATION
# ============================================================

def canonicalize_graph(data):

    prompt = f"""
Here is the extracted graph for document:

DOCUMENT ID:
{data["document_id"]}

ENTITIES:
{json.dumps(data["entities"], indent=2, ensure_ascii=False)}

RELATIONSHIPS:
{json.dumps(data["relationships"], indent=2, ensure_ascii=False)}

Create a canonical graph.

Return exactly:

{{
  "entities": [
    {{
      "id": "stable_id",
      "name": "canonical name",
      "type": "entity type",
      "attributes": {{}},
      "source_chunk_ids": []
    }}
  ],
  "relationships": [
    {{
      "source_id": "canonical source ID",
      "type": "RELATIONSHIP_TYPE",
      "target_id": "canonical target ID",
      "attributes": {{}},
      "source_chunk_ids": []
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
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(
        response.choices[0].message.content
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_graph(graph):

    entity_ids = {
        entity["id"]
        for entity in graph["entities"]
    }

    errors = []

    for entity in graph["entities"]:

        required = [
            "id",
            "name",
            "type",
            "source_chunk_ids",
        ]

        for field in required:

            if field not in entity:
                errors.append(
                    f"Entity missing field '{field}': {entity}"
                )

    for relationship in graph["relationships"]:

        if relationship["source_id"] not in entity_ids:
            errors.append(
                f"Unknown source entity: "
                f"{relationship['source_id']}"
            )

        if relationship["target_id"] not in entity_ids:
            errors.append(
                f"Unknown target entity: "
                f"{relationship['target_id']}"
            )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GraphRAG - Entity Resolution")
    print("=" * 70)

    data = load_extraction()

    print(
        f"\nInput entities: "
        f"{len(data['entities'])}"
    )

    print(
        f"Input relationships: "
        f"{len(data['relationships'])}"
    )

    print("\nRunning entity resolution...")

    canonical = canonicalize_graph(data)

    errors = validate_graph(canonical)

    if errors:

        print("\nVALIDATION ERRORS:")

        for error in errors:
            print(f"  - {error}")

        raise RuntimeError(
            "Canonical graph failed validation."
        )

    output = {
        "document_id": data["document_id"],
        "chunk_count": data["chunk_count"],
        "entities": canonical["entities"],
        "relationships": canonical["relationships"],
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
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANONICAL GRAPH SUMMARY")
    print("=" * 70)

    print(
        f"Canonical entities: "
        f"{len(output['entities'])}"
    )

    print(
        f"Canonical relationships: "
        f"{len(output['relationships'])}"
    )

    print("\nEntities:")

    for entity in output["entities"]:

        print(
            f"  [{entity['type']}] "
            f"{entity['name']} "
            f"-> {entity['id']}"
        )

    print("\nRelationships:")

    for relationship in output["relationships"]:

        print(
            f"  {relationship['source_id']}"
            f" --{relationship['type']}--> "
            f"{relationship['target_id']}"
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()