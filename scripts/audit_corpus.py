import json
from collections import Counter

INPUT_PATH = "data/raw/corpus.jsonl"

documents = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))


print("=" * 80)
print("CORPUS COMPLETENESS AUDIT")
print("=" * 80)

print(f"Total documents: {len(documents)}")


# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

token_counts = [
    d.get("approx_tokens", 0)
    for d in documents
]

print()
print("DOCUMENT SIZE")
print("-" * 80)
print(f"Minimum tokens: {min(token_counts):,}")
print(f"Maximum tokens: {max(token_counts):,}")
print(f"Average tokens: {sum(token_counts) / len(token_counts):,.2f}")


# ---------------------------------------------------------
# Detect structural/content signals
# ---------------------------------------------------------

stats = Counter()

small_documents = []
very_small_documents = []

infobox_without_results = []
results_documents = []
table_documents = []

for doc in documents:

    text = doc.get("text", "")
    tokens = doc.get("approx_tokens", 0)

    has_infobox = "[Infobox" in text
    has_results_heading = "\nResults" in text or text.startswith("Results")
    has_results_table = "Results table" in text

    # Simple table signal
    table_lines = 0

    for line in text.splitlines():
        if line.count("|") >= 2 or "!!" in line:
            table_lines += 1

    has_table_like_content = table_lines >= 2

    if has_infobox:
        stats["infobox"] += 1

    if has_results_heading:
        stats["results_heading"] += 1
        results_documents.append(doc)

    if has_results_table:
        stats["results_table_text"] += 1

    if has_table_like_content:
        stats["table_like"] += 1
        table_documents.append(doc)

    if has_infobox and not has_results_heading:
        stats["infobox_without_results"] += 1
        infobox_without_results.append(doc)

    if tokens < 200:
        very_small_documents.append(doc)

    if tokens < 300:
        small_documents.append(doc)


# ---------------------------------------------------------
# Print structural statistics
# ---------------------------------------------------------

print()
print("CONTENT SIGNALS")
print("-" * 80)

for key, value in stats.items():
    print(f"{key:30} {value:,}")


# ---------------------------------------------------------
# Very small documents
# ---------------------------------------------------------

print()
print("=" * 80)
print("DOCUMENTS UNDER 200 TOKENS")
print("=" * 80)

for doc in sorted(
    very_small_documents,
    key=lambda x: x.get("approx_tokens", 0)
):
    print(
        f"{doc['approx_tokens']:>5} tokens | "
        f"{doc['doc_id']} | "
        f"{doc['title']}"
    )


# ---------------------------------------------------------
# Infobox but no Results heading
# ---------------------------------------------------------

print()
print("=" * 80)
print("DOCUMENTS WITH INFOBOX BUT NO RESULTS HEADING")
print("=" * 80)

for doc in infobox_without_results[:100]:
    print(
        f"{doc['approx_tokens']:>5} tokens | "
        f"{doc['doc_id']} | "
        f"{doc['title']}"
    )

if len(infobox_without_results) > 100:
    print(
        f"... and {len(infobox_without_results) - 100} more"
    )


# ---------------------------------------------------------
# Documents with table-like content
# ---------------------------------------------------------

print()
print("=" * 80)
print("SAMPLE DOCUMENTS WITH TABLE-LIKE CONTENT")
print("=" * 80)

for doc in table_documents[:30]:
    print(
        f"{doc['approx_tokens']:>5} tokens | "
        f"{doc['doc_id']} | "
        f"{doc['title']}"
    )


# ---------------------------------------------------------
# Specifically inspect Q1005331
# ---------------------------------------------------------

print()
print("=" * 80)
print("Q1005331 CHECK")
print("=" * 80)

target = next(
    (d for d in documents if d["doc_id"] == "Q1005331"),
    None
)

if target:
    print("Title:", target["title"])
    print("Tokens:", target["approx_tokens"])
    print("Characters:", len(target["text"]))
    print("Has infobox:", "[Infobox" in target["text"])
    print("Has Results heading:", "\nResults" in target["text"])
    print("Has Results table:", "Results table" in target["text"])

    table_lines = [
        line
        for line in target["text"].splitlines()
        if line.count("|") >= 2 or "!!" in line
    ]

    print("Table-like lines:", len(table_lines))

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)