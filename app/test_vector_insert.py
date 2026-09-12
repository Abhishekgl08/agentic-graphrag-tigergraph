import os

from dotenv import load_dotenv
from openai import OpenAI

from tigergraph import create_connection


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")


# ---------------------------------------------------------
# Create clients
# ---------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tg_conn = create_connection()


# ---------------------------------------------------------
# Sample chunk
# ---------------------------------------------------------
chunk_id = "test_chunk_001"

chunk_text = """
TigerGraph is a graph database that can store relationships
between entities and can also support vector search for
retrieval-augmented generation applications.
"""

document_id = "test_document_001"
page_number = 1


# ---------------------------------------------------------
# Step 1: Create Chunk vertex + vector attribute
# ---------------------------------------------------------
print("Checking TigerGraph schema...")

schema = tg_conn.getSchema()

vertex_types = [
    vertex["Name"]
    for vertex in schema.get("VertexTypes", [])
]

if "Chunk" not in vertex_types:

    print("Chunk vertex does not exist. Creating it...")

    gsql = """
    USE GRAPH AgenticGraphRAG

    CREATE SCHEMA_CHANGE JOB add_chunk_schema FOR GRAPH AgenticGraphRAG {

        ADD VERTEX Chunk (
            PRIMARY_ID chunk_id STRING,
            text STRING,
            document_id STRING,
            page_number INT
        );

        ALTER VERTEX Chunk
        ADD VECTOR ATTRIBUTE embedding(
            DIMENSION=1536,
            METRIC="COSINE",
            INDEXTYPE="HNSW",
            DATATYPE="FLOAT"
        );
    }

    RUN SCHEMA_CHANGE JOB add_chunk_schema
    """

    result = tg_conn.gsql(gsql)

    print("Schema creation result:")
    print(result)

else:

    print("Chunk vertex already exists.")


# ---------------------------------------------------------
# Step 2: Generate OpenAI embedding
# ---------------------------------------------------------
print("\nGenerating OpenAI embedding...")

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=chunk_text,
)

embedding = response.data[0].embedding

print("Embedding generated.")
print("Embedding dimensions:", len(embedding))


# ---------------------------------------------------------
# Step 3: Insert chunk + embedding into TigerGraph
# ---------------------------------------------------------
print("\nInserting chunk into TigerGraph...")

attributes = {
    "text": chunk_text,
    "document_id": document_id,
    "page_number": page_number,
    "embedding": embedding,
}

result = tg_conn.upsertVertex(
    "Chunk",
    chunk_id,
    attributes,
)

print("TigerGraph upsert result:", result)


# ---------------------------------------------------------
# Step 4: Read the chunk back
# ---------------------------------------------------------
print("\nReading chunk back from TigerGraph...")

stored_chunk = tg_conn.getVerticesById(
    "Chunk",
    chunk_id,
    select="text,document_id,page_number",
)

print("\nStored chunk:")
print(stored_chunk)

print("\nSUCCESS: Chunk and embedding were inserted into TigerGraph.")