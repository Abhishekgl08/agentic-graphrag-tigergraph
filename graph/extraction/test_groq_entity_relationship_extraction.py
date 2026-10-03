import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
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
    / "groq_extraction_2docs.json"
)

MODEL = "openai/gpt-oss-120b"

# Only 2 documents for this test
MAX_DOCUMENTS = 2


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set in .env"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# EXTRACTION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a knowledge graph extraction system.

Extract ONLY information explicitly supported by the
provided text chunk.

Do NOT invent facts.

Return valid JSON with exactly:

{
  "entities": [],
  "relationships": []
}

ENTITY TYPES:

- Document
- OlympicGames
- Event
- Athlete
- Country
- Venue
- Sport

Do NOT create entities for dates, years, numbers,
medals, rankings, scores, or counts.

RELATIONSHIP TYPES:

- PART_OF
- SUB_EVENT_OF
- AT_VENUE
- IN_SPORT
- WON_BY
- REPRESENTS
- PARTICIPATED_IN

ENTITY FORMAT:

{
  "id": "stable_entity_id",
  "name": "display name",
  "type": "EntityType",
  "source_chunk_id": "chunk_id"
}

RELATIONSHIP FORMAT:

{
  "source": "source_entity_id",
  "source_type": "EntityType",
  "relationship": "RELATIONSHIP_TYPE",
  "target": "target_entity_id",
  "target_type": "EntityType",
  "source_chunk_id": "chunk_id"
}

For WON_BY, if the medal is explicitly known,
include:

"medal": "GOLD"

or

"medal": "SILVER"

or

"medal": "BRONZE"

RULES:

1. Extract only explicitly supported facts.
2. Never invent missing information.
3. Do not create date/year entities.
4. Use stable and meaningful entity IDs.
5. Preserve the actual entity names from the source.
6. Use SUB_EVENT_OF for heats, semifinals, finals, etc.
   when the parent event is identifiable.
7. Use PART_OF for Event → OlympicGames.
8. Use AT_VENUE for Event → Venue.
9. Use IN_SPORT for Event → Sport.
10. Use WON_BY for Event → Athlete.
11. Use REPRESENTS for Athlete → Country.
12. Use PARTICIPATED_IN for Athlete → Event.
13. Every relationship must refer to an entity that is
    extracted or clearly identifiable in the chunk.
14. Preserve source_chunk_id on every entity and
    relationship.

Return JSON only.
"""


# ============================================================
# LOAD ALL CHUNKS
# ============================================================

def load_chunks():

    chunks = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                chunks.append(
                    json.loads(line)
                )

    return chunks


# ============================================================
# SELECT FIRST 2 DOCUMENTS
# ============================================================

def select_documents(chunks):

    documents = {}

    for chunk in chunks:

        doc_id = (
            chunk.get("doc_id")
            or chunk.get("document_id")
        )

        if not doc_id:
            continue

        if doc_id not in documents:
            documents[doc_id] = []

        documents[doc_id].append(chunk)

    selected_ids = list(documents.keys())[:MAX_DOCUMENTS]

    selected_chunks = []

    for doc_id in selected_ids:
        selected_chunks.extend(
            documents[doc_id]
        )

    return selected_ids, selected_chunks


# ============================================================
# EXTRACT ONE CHUNK
# ============================================================

def extract_chunk(chunk):

    chunk_id = chunk["chunk_id"]

    user_prompt = f"""
SOURCE CHUNK ID:
{chunk_id}

DOCUMENT ID:
{chunk.get("doc_id")}

DOCUMENT TITLE:
{chunk.get("title", "")}

TEXT:
{chunk["text"]}

Extract all supported entities and relationships.
"""


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
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Empty response from Groq."
        )

    extraction = json.loads(content)

    return extraction, response


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GROQ GRAPHRAG EXTRACTION TEST")
    print("=" * 70)

    print()
    print(f"Model: {MODEL}")
    print(f"Document limit: {MAX_DOCUMENTS}")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    all_chunks = load_chunks()

    print(
        f"Total chunks available: {len(all_chunks)}"
    )

    # --------------------------------------------------------
    # Select documents
    # --------------------------------------------------------

    document_ids, selected_chunks = select_documents(
        all_chunks
    )

    print()
    print("Selected documents:")

    for doc_id in document_ids:
        print(f"  - {doc_id}")

    print()
    print(
        f"Documents selected: {len(document_ids)}"
    )

    print(
        f"Chunks selected:    {len(selected_chunks)}"
    )

    print()
    print("-" * 70)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    successful = 0
    failed = 0

    for index, chunk in enumerate(
        selected_chunks,
        start=1
    ):

        print()
        print(
            f"Processing chunk "
            f"{index}/{len(selected_chunks)}"
        )

        print(
            f"Document: {chunk.get('doc_id')}"
        )

        print(
            f"Chunk:    {chunk['chunk_id']}"
        )

        try:

            extraction, response = extract_chunk(
                chunk
            )

            entities = extraction.get(
                "entities",
                []
            )

            relationships = extraction.get(
                "relationships",
                []
            )

            usage = response.usage

            prompt_tokens = (
                usage.prompt_tokens
                if usage
                else 0
            )

            completion_tokens = (
                usage.completion_tokens
                if usage
                else 0
            )

            chunk_total_tokens = (
                usage.total_tokens
                if usage
                else 0
            )

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += chunk_total_tokens

            successful += 1

            results.append(
                {
                    "document_id": chunk.get(
                        "doc_id"
                    ),
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "entities": entities,
                    "relationships": relationships,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": chunk_total_tokens,
                    },
                }
            )

            print(
                f"Entities:      {len(entities)}"
            )

            print(
                f"Relationships: {len(relationships)}"
            )

            print(
                f"Tokens:        {chunk_total_tokens}"
            )

        except Exception as error:

            failed += 1

            print(
                f"ERROR: {error}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "experiment": "Groq GraphRAG extraction",
        "model": MODEL,
        "documents": document_ids,
        "statistics": {
            "documents_selected": len(
                document_ids
            ),
            "chunks_selected": len(
                selected_chunks
            ),
            "chunks_successful": successful,
            "chunks_failed": failed,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Model:              {MODEL}"
    )

    print(
        f"Documents:          {len(document_ids)}"
    )

    print(
        f"Chunks:             {len(selected_chunks)}"
    )

    print(
        f"Successful:         {successful}"
    )

    print(
        f"Failed:             {failed}"
    )

    print()
    print(
        f"Prompt tokens:      {total_prompt_tokens}"
    )

    print(
        f"Completion tokens:  {total_completion_tokens}"
    )

    print(
        f"Total tokens:       {total_tokens}"
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()