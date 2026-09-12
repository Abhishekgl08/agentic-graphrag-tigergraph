import json

INPUT_PATH = "data/raw/corpus.jsonl"

documents = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))


candidates = []

for doc in documents:
    title = doc.get("title", "")
    text = doc.get("text", "")

    if "Olympics" not in title:
        continue

    has_results_heading = (
        "\nResults" in text
        or text.startswith("Results")
    )

    table_lines = [
        line
        for line in text.splitlines()
        if line.count("|") >= 2 or "!!" in line
    ]

    if has_results_heading and len(table_lines) == 0:
        candidates.append(doc)


# Sort smallest first
candidates.sort(
    key=lambda x: x.get("approx_tokens", 0)
)


print("=" * 80)
print("OLYMPIC POTENTIAL MISSING-TABLE DOCUMENTS")
print("=" * 80)

print("Total candidates:", len(candidates))


# Inspect first 10
for i, doc in enumerate(candidates[:10], start=1):

    print("\n" + "=" * 80)
    print(f"CANDIDATE {i}")
    print("=" * 80)

    print("Doc ID:", doc["doc_id"])
    print("Title:", doc["title"])
    print("Approx tokens:", doc["approx_tokens"])
    print("Characters:", len(doc["text"]))

    print("\n--- RAW CORPUS TEXT ---")
    print(doc["text"])

    print("\n--- END DOCUMENT ---")


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)