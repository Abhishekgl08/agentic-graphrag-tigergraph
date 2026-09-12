import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.tigergraph import create_connection


GRAPH_NAME = "AgenticGraphRAG"


def main():
    print("Connecting to TigerGraph...")

    tg = create_connection()

    print("Connected successfully.\n")

    # ---------------------------------------------------------
    # Create a temporary query that deletes all Chunk vertices
    # ---------------------------------------------------------

    delete_query = f"""
    USE GRAPH {GRAPH_NAME}

    CREATE OR REPLACE QUERY clear_chunks() FOR GRAPH {GRAPH_NAME} {{
        
        chunks = {{Chunk.*}};
        
        DELETE c FROM chunks:c;
    }}

    INSTALL QUERY clear_chunks
    """

    print("Creating deletion query...")

    result = tg.gsql(delete_query)

    print(result)

    # ---------------------------------------------------------
    # Run deletion query
    # ---------------------------------------------------------

    print("\nDeleting all Chunk vertices...")

    result = tg.runInstalledQuery("clear_chunks")

    print("Deletion result:")
    print(result)

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    print("\nChecking remaining Chunk vertices...")

    remaining = tg.getVertices(
        "Chunk",
        limit=10,
    )

    print("Remaining:")
    print(remaining)

    if len(remaining) == 0:
        print("\nSUCCESS!")
        print("Chunk data is now EMPTY.")
        print("The Chunk schema and vector configuration remain.")
    else:
        print(
            f"\nWARNING: {len(remaining)} Chunk vertices still exist."
        )


if __name__ == "__main__":
    main()