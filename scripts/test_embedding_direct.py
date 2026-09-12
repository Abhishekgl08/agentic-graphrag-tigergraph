import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

text = """
TigerGraph is a graph database that can store relationships
between entities and support vector search for
retrieval-augmented generation applications.
"""

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
)

embedding = response.data[0].embedding

print("SUCCESS!")
print("Embedding dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])