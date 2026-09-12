import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.tigergraph import create_connection


GRAPH_NAME = "AgenticGraphRAG"


def main():
    print("Connecting to TigerGraph...")

    tg = create_connection()

    print("Connected successfully.\n")

    # ---------------------------------------------------------
    # 1. Create Chunk vertex
    # ---------------------------------------------------------

    print("Creating Chunk vertex...")

    create_chunk_job = f"""
    USE GRAPH {GRAPH_NAME}

    CREATE SCHEMA_CHANGE JOB create_chunk FOR GRAPH {GRAPH_NAME} {{
        ADD VERTEX Chunk (
            PRIMARY_ID chunk_id STRING,
            text STRING,
            document_id STRING,
            page_number INT
        );
    }}

    RUN SCHEMA_CHANGE JOB create_chunk
    """

    result = tg.gsql(create_chunk_job)

    print("Chunk vertex creation result:")
    print(result)

    # ---------------------------------------------------------
    # 2. Add vector attribute
    # ---------------------------------------------------------

    print("\nAdding embedding vector attribute...")

    add_vector_job = f"""
    USE GRAPH {GRAPH_NAME}

    CREATE SCHEMA_CHANGE JOB add_chunk_embedding FOR GRAPH {GRAPH_NAME} {{
        ALTER VERTEX Chunk
        ADD VECTOR ATTRIBUTE embedding(
            DIMENSION=1536,
            METRIC="COSINE",
            INDEXTYPE="HNSW",
            DATATYPE="FLOAT"
        );
    }}

    RUN SCHEMA_CHANGE JOB add_chunk_embedding -N
    """

    result = tg.gsql(add_vector_job)

    print("Vector attribute creation result:")
    print(result)

    # ---------------------------------------------------------
    # 3. Check final schema
    # ---------------------------------------------------------

    print("\nChecking final TigerGraph schema...")

    schema = tg.getSchema()

    print(schema)

    print("\nSchema setup completed.")


if __name__ == "__main__":
    main()