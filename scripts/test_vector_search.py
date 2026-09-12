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
# 3. User query
# ============================================================

query = "What database can be used for GraphRAG and vector search?"

print("Query:")
print(query)


# ============================================================
# 4. Generate query embedding
# ============================================================

print("\nGenerating query embedding...")

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)

query_embedding = response.data[0].embedding

print("Query embedding generated.")
print("Dimensions:", len(query_embedding))


# ============================================================
# 5. Run TigerGraph vector search
# ============================================================

print("\nSearching TigerGraph...")

results = tg.runInstalledQuery(
    "vector_search",
    params={
        "query_vector": query_embedding,
    },
)

print("\nTigerGraph vector search result:")
print(results)


# ============================================================
# 6. Done
# ============================================================

print("\n========================================")
print("VECTOR SEARCH TEST COMPLETED")
print("========================================")