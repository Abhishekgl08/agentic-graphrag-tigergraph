import os
from app.tigergraph import create_connection


def test_list_tigergraph_graphs():
    conn = create_connection()

    graphs = conn.listGraphs()

    print("\n=== TigerGraph Graphs ===")
    print(graphs)

    target_graph = os.getenv("TG_GRAPHNAME")

    print(f"\nTarget graph: {target_graph}")

    assert graphs is not None

    # pyTigerGraph returns graph metadata dictionaries.
    graph_names = [graph["graphName"] for graph in graphs]

    print(f"Graph names: {graph_names}")

    if target_graph in graph_names:
        print("Target graph FOUND.")
    else:
        print("Target graph NOT FOUND.")

    assert target_graph in graph_names, (
        f"Graph '{target_graph}' does not exist in the "
        f"connected TigerGraph workspace/database. "
        f"Available graphs: {graph_names}"
    )