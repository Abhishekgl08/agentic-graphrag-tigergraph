import json

INPUT_PATH = "data/raw/corpus.jsonl"

docs = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))

docs.sort(key=lambda x: x.get("approx_tokens", 0))

print("=" * 80)
print("10 SMALLEST DOCUMENTS")
print("=" * 80)

for i, doc in enumerate(docs[:10], start=1):
    print(
        f"{i}. {doc['approx_tokens']} tokens | "
        f"{doc['doc_id']} | "
        f"{doc['title']}"
    )
    