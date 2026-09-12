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

    gsql = f"""
    USE GRAPH {GRAPH_NAME}

    CREATE OR REPLACE QUERY vector_search (LIST<float> query_vector)
    FOR GRAPH {GRAPH_NAME}
    SYNTAX v3 {{
        
        MapAccum<Vertex, Float> @@distances;

        results = vectorSearch(
            {{Chunk.embedding}},
            query_vector,
            5,
            {{distance_map: @@distances}}
        );

        PRINT results;

        PRINT @@distances;
    }}

    INSTALL QUERY vector_search
    """

    print("Creating vector search query...")

    result = tg.gsql(gsql)

    print("\nTigerGraph result:")
    print(result)

    print("\nVector search query setup completed.")


if __name__ == "__main__":
    main()