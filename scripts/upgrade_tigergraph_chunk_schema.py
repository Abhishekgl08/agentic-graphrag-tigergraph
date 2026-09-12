"""One-time, idempotent schema migration for the production Chunk payload.

Review this against the deployed TigerGraph schema before executing it. It
preserves the tested VECTOR(1536), COSINE, HNSW embedding attribute.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.tigergraph import create_connection


def main() -> None:
    graph = create_connection()
    graph_name = os.environ["TG_GRAPHNAME"]
    gsql = f'''USE GRAPH {graph_name}
CREATE SCHEMA_CHANGE JOB upgrade_chunk_metadata FOR GRAPH {graph_name} {{
  ALTER VERTEX Chunk ADD ATTRIBUTE (
    dataset_id STRING, title STRING, wikidata_qid STRING, wikipedia_pageid INT,
    chunk_index INT, chunk_type STRING, section STRING, section_path STRING,
    approx_tokens INT, embedding_model STRING, embedding_version STRING,
    content_hash STRING
  );
}}
RUN SCHEMA_CHANGE JOB upgrade_chunk_metadata
'''
    print(graph.gsql(gsql))


if __name__ == "__main__":
    main()
