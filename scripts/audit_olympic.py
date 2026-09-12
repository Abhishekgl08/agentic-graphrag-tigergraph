import json

INPUT_PATH = "data/raw/corpus.jsonl"

documents = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))


olympic_docs = []

for doc in documents:
    title = doc.get("title", "")
    text = doc.get("text", "")

    if "Olympics" in title:
        olympic_docs.append(doc)


print("=" * 80)
print("OLYMPIC DOCUMENT COMPLETENESS AUDIT")
print("=" * 80)

print("Total Olympic documents:", len(olympic_docs))


with_results_heading = []
with_results_table = []
with_table_like_content = []
missing_table_candidates = []


for doc in olympic_docs:

    text = doc.get("text", "")

    has_results_heading = (
        "\nResults" in text
        or text.startswith("Results")
    )

    has_results_table = "Results table" in text

    table_lines = [
        line
        for line in text.splitlines()
        if line.count("|") >= 2 or "!!" in line
    ]

    has_table_like = len(table_lines) >= 2

    if has_results_heading:
        with_results_heading.append(doc)

    if has_results_table:
        with_results_table.append(doc)

    if has_table_like:
        with_table_like_content.append(doc)

    # Potentially suspicious:
    # Olympic page with Results heading but
    # no obvious table content.
    if has_results_heading and not has_table_like:
        missing_table_candidates.append(doc)


print()
print("-" * 80)
print("SUMMARY")
print("-" * 80)

print("Olympic documents:", len(olympic_docs))
print("With Results heading:", len(with_results_heading))
print("With 'Results table' text:", len(with_results_table))
print("With table-like content:", len(with_table_like_content))
print(
    "Results heading but no table-like content:",
    len(missing_table_candidates)
)


print()
print("=" * 80)
print("POTENTIAL MISSING-TABLE OLYMPIC DOCUMENTS")
print("=" * 80)

for doc in sorted(
    missing_table_candidates,
    key=lambda x: x.get("approx_tokens", 0)
):
    print(
        f"{doc.get('approx_tokens', 0):>6} tokens | "
        f"{doc.get('doc_id')} | "
        f"{doc.get('title')}"
    )


print()
print("=" * 80)
print("Q1005331")
print("=" * 80)

target = next(
    (d for d in olympic_docs if d.get("doc_id") == "Q1005331"),
    None
)

if target:
    text = target["text"]

    print("Title:", target["title"])
    print("Tokens:", target["approx_tokens"])
    print("Results heading:", "\nResults" in text or text.startswith("Results"))
    print("Results table text:", "Results table" in text)

    table_lines = [
        line
        for line in text.splitlines()
        if line.count("|") >= 2 or "!!" in line
    ]

    print("Table-like lines:", len(table_lines))

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)