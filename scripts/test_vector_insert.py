import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tigergraph import create_connection


# ============================================================
# 1. Load environment
# ============================================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")


# ============================================================
# 2. Create clients
# ============================================================

openai_client = OpenAI(api_key=openai_api_key)

tg = create_connection()


# ============================================================
# 3. Sample chunk
# ============================================================

chunk_id = "test_chunk_001"

chunk_text = """
TigerGraph is a graph database that can store relationships
between entities and support vector search for
retrieval-augmented generation applications.
"""

document_id = "test_document_001"

page_number = 1


# ============================================================
# 4. Generate embedding
# ============================================================

print("Generating OpenAI embedding...")

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=chunk_text,
)

embedding = response.data[0].embedding

print("Embedding generated successfully.")
print("Embedding dimensions:", len(embedding))


# ============================================================
# 5. Insert into TigerGraph
# ============================================================

print("\nInserting Chunk into TigerGraph...")

attributes = {
    "text": chunk_text,
    "document_id": document_id,
    "page_number": page_number,
    "embedding": embedding,
}

result = tg.upsertVertex(
    "Chunk",
    chunk_id,
    attributes,
)

print("TigerGraph upsert result:")
print(result)


# ============================================================
# 6. Read it back
# ============================================================

print("\nReading Chunk back from TigerGraph...")

stored_chunk = tg.getVerticesById(
    "Chunk",
    [chunk_id],
)
print("Stored Chunk:")
print(stored_chunk)


print("\n========================================")
print("SUCCESS!")
print("OpenAI embedding stored in TigerGraph.")
print("========================================")