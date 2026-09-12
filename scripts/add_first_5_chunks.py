import sys
from pathlib import Path

# ============================================================
# 1. Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Imports
# ============================================================

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tigergraph import create_connection


# ============================================================
# 3. Configuration
# ============================================================

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c"
    / "chunks.jsonl"
)

EMBEDDING_MODEL = "text-embedding-3-small"

NUMBER_OF_CHUNKS = 5


# ============================================================
# 4. Load environment
# ============================================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")


# ============================================================
# 5. Create clients
# ============================================================

openai_client = OpenAI(api_key=openai_api_key)

tg = create_connection()


# ============================================================
# 6. Read first 5 chunks
# ============================================================

print("Reading chunks from:")
print(CHUNKS_FILE)

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"chunks.jsonl not found at: {CHUNKS_FILE}"
    )


chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        chunk = json.loads(line)
        chunks.append(chunk)

        if len(chunks) == NUMBER_OF_CHUNKS:
            break


if not chunks:
    raise RuntimeError("No chunks found in chunks.jsonl")


print(f"\nLoaded {len(chunks)} chunks.\n")


# ============================================================
# 7. Display chunks before embedding
# ============================================================

for index, chunk in enumerate(chunks, start=1):

    print("=" * 70)
    print(f"CHUNK {index}")
    print("=" * 70)

    print("chunk_id :", chunk.get("chunk_id"))
    print("doc_id   :", chunk.get("doc_id"))
    print("title    :", chunk.get("title"))
    print("section  :", chunk.get("section"))
    print("type     :", chunk.get("chunk_type"))
    print("tokens   :", chunk.get("approx_tokens"))

    print("\nText:")
    print(chunk.get("text", ""))

    print()


# ============================================================
# 8. Extract text for embedding
# ============================================================

texts = []

for chunk in chunks:

    text = chunk.get("text")

    if not text:
        raise RuntimeError(
            f"Chunk {chunk.get('chunk_id')} has no text."
        )

    texts.append(text)


# ============================================================
# 9. Generate embeddings
# ============================================================

print("=" * 70)
print("Generating embeddings...")
print("=" * 70)

response = openai_client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=texts,
)

embeddings = [item.embedding for item in response.data]

print(f"Generated {len(embeddings)} embeddings.")

for index, embedding in enumerate(embeddings, start=1):
    print(
        f"Chunk {index}: "
        f"{len(embedding)} dimensions"
    )


# ============================================================
# 10. Insert chunks into TigerGraph
# ============================================================

print("\n" + "=" * 70)
print("Inserting chunks into TigerGraph...")
print("=" * 70)

for chunk, embedding in zip(chunks, embeddings):

    chunk_id = chunk["chunk_id"]

    # Your JSON uses doc_id.
    # Our TigerGraph schema uses document_id.
    document_id = chunk.get("doc_id", "")

    # The current JSON structure does not contain page_number.
    # We use -1 to represent "not available".
    page_number = chunk.get("page_number", -1)

    attributes = {
        "text": chunk["text"],
        "document_id": document_id,
        "page_number": page_number,
        "embedding": embedding,
    }

    result = tg.upsertVertex(
        "Chunk",
        chunk_id,
        attributes,
    )

    print(
        f"Inserted {chunk_id} "
        f"→ TigerGraph result: {result}"
    )


# ============================================================
# 11. Verify inserted chunks
# ============================================================

print("\n" + "=" * 70)
print("Verifying inserted chunks...")
print("=" * 70)

chunk_ids = [chunk["chunk_id"] for chunk in chunks]

stored_chunks = tg.getVerticesById(
    "Chunk",
    chunk_ids,
)

print(f"\nTigerGraph returned {len(stored_chunks)} chunks.\n")

for chunk in stored_chunks:

    print("-" * 70)
    print("chunk_id    :", chunk["v_id"])
    print("attributes  :", chunk["attributes"])


# ============================================================
# 12. Finished
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)

print(
    f"First {len(chunks)} chunks were embedded "
    "and stored in TigerGraph."
)