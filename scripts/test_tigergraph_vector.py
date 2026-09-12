import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tigergraph import create_connection


# ============================================================
# 1. Load configuration
# ============================================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")


# ============================================================
# 2. Create connections
# ============================================================

openai_client = OpenAI(api_key=openai_api_key)
tg = create_connection()


# ============================================================
# 3. Sample chunk
# ============================================================

chunk_id = "test_chunk_001"

chunk_text = """
TigerGraph is a graph database that can store relationships
between entities and also support vector search for
retrieval-augmented generation applications.
"""


# ============================================================
# 4. Generate OpenAI embedding
# ============================================================

print("Generating OpenAI embedding...")

embedding_response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=chunk_text,
)

embedding = embedding_response.data[0].embedding

print("Embedding generated.")
print("Dimensions:", len(embedding))


# ============================================================
# 5. Insert chunk + embedding into TigerGraph
# ============================================================

print("\nInserting chunk into TigerGraph...")

attributes = {
    "text": chunk_text,
    "embedding": embedding,
}

result = tg.upsertVertex(
    "Chunk",
    chunk_id,
    attributes,
)

print("Insert result:")
print(result)


# ============================================================
# 6. Read the chunk back
# ============================================================

print("\nReading chunk from TigerGraph...")

result = tg.getVertex(
    "Chunk",
    chunk_id,
)

print("Stored vertex:")
print(result)


print("\nSUCCESS!")
print("Chunk and embedding were inserted into TigerGraph.")