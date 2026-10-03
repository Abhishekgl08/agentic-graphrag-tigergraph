import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


# ============================================================
# PATHS
# ============================================================

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
    / "batch_extraction_5docs_gpt54nano.json"
)


# ============================================================
# CONFIG
# ============================================================

MODEL = "gpt-5.4-nano"

TEST_DOCUMENT_IDS = [
    "Q1001497",
    "Q1004860",
    "Q1004861",
    "Q1004863",
    "Q1004868",
    "Q1004869",
    "Q1004870",
    "Q1004871",
    "Q1004888",
    "Q1004892",
]


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# EXTRACTION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an information extraction system for a GraphRAG pipeline.

Your task is to extract factual entities and relationships from the
provided source text.

IMPORTANT RULES:

1. Extract only information explicitly supported by the text.
2. Do not invent facts.
3. Preserve the meaning of the source.
4. Use only these entity types when applicable:
   - Event
   - Athlete
   - Country
   - Venue
   - Sport
   - OlympicGames
   - Organization
   - Person
   - Other

5. Prefer meaningful domain entities rather than extracting every word.
6. Do not create separate entities for dates, numbers, or counts.
   Store these as attributes when appropriate.
7. Preserve the source chunk ID for every extracted entity and relationship.
8. Use canonical relationship names where appropriate:
   - PART_OF
   - AT_VENUE
   - IN_SPORT
   - WON_BY
   - REPRESENTS
   - PARTICIPATED_IN
   - SUB_EVENT_OF

9. For WON_BY, use an attribute called "medal" when the medal is known.
   Example:
   Event -> WON_BY -> Athlete
   attributes: {"medal": "GOLD"}

10. If a heat, semifinal, final, qualification, or similar event is a
    sub-event of another event, use SUB_EVENT_OF.

Return valid JSON only.
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(chunk):
    return f"""
Extract the entities and relationships from this source chunk.

SOURCE CHUNK ID:
{chunk["chunk_id"]}

DOCUMENT ID:
{chunk["doc_id"]}

TITLE:
{chunk.get("title", "")}

SOURCE TEXT:
{chunk["text"]}

Return JSON in exactly this structure:

{{
  "entities": [
    {{
      "id": "stable_entity_id",
      "name": "Entity name",
      "type": "Event",
      "attributes": {{}},
      "source_chunk_ids": ["{chunk["chunk_id"]}"]
    }}
  ],
  "relationships": [
    {{
      "source_id": "source_entity_id",
      "target_id": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "attributes": {{}},
      "source_chunk_ids": ["{chunk["chunk_id"]}"]
    }}
  ]
}}
"""


# ============================================================
# LOAD TEST CHUNKS
# ============================================================

def load_test_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("doc_id") in TEST_DOCUMENT_IDS:
                chunks.append(record)

    return chunks


# ============================================================
# EXTRACT ONE CHUNK
# ============================================================

def extract_chunk(chunk):
    user_prompt = build_user_prompt(chunk)

    response = client.chat.completions.create(
        model=MODEL,
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

    content = response.choices[0].message.content

    result = json.loads(content)

    usage = response.usage

    usage_data = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

    return result, usage_data


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GraphRAG A/B Test - GPT-5.4 nano")
    print("=" * 70)

    print(f"\nModel: {MODEL}")

    print("\nTest documents:")
    for doc_id in TEST_DOCUMENT_IDS:
        print(f"  - {doc_id}")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_test_chunks()

    print(f"\nChunks found: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No test chunks found.")

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    results = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for index, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 70)
        print(
            f"Processing chunk {index}/{len(chunks)}"
        )
        print(f"Document: {chunk['doc_id']}")
        print(f"Chunk:    {chunk['chunk_id']}")

        try:

            extraction, usage = extract_chunk(chunk)

            entities = extraction.get("entities", [])
            relationships = extraction.get("relationships", [])

            print(
                f"Entities: {len(entities)} | "
                f"Relationships: {len(relationships)}"
            )

            print(
                f"Tokens: {usage['total_tokens']}"
            )

            if usage["prompt_tokens"]:
                total_prompt_tokens += usage["prompt_tokens"]

            if usage["completion_tokens"]:
                total_completion_tokens += usage["completion_tokens"]

            if usage["total_tokens"]:
                total_tokens += usage["total_tokens"]

            results.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("title", ""),
                    "extraction": extraction,
                    "usage": usage,
                }
            )

        except Exception as e:

            print(f"ERROR: {e}")

            results.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("title", ""),
                    "error": str(e),
                }
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "GraphRAG extraction A/B test",
        "model": MODEL,
        "documents": TEST_DOCUMENT_IDS,
        "document_count": len(TEST_DOCUMENT_IDS),
        "chunk_count": len(chunks),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "results": results,
    }

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

    successful = sum(
        1
        for r in results
        if "extraction" in r
    )

    failed = len(results) - successful

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Model:              {MODEL}")
    print(f"Documents:          {len(TEST_DOCUMENT_IDS)}")
    print(f"Chunks:             {len(chunks)}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {failed}")
    print(f"Prompt tokens:      {total_prompt_tokens}")
    print(f"Completion tokens:  {total_completion_tokens}")
    print(f"Total tokens:       {total_tokens}")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()