import json

INPUT_PATH = "data/raw/corpus.jsonl"

documents = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))


olympic_docs = [
    d for d in documents
    if "Olympics" in d.get("title", "")
]


empty_results = []
results_with_content = []
results_with_pipe_table = []


for doc in olympic_docs:

    text = doc.get("text", "")
    lines = text.splitlines()

    # Find Results heading
    results_index = None

    for i, line in enumerate(lines):
        if line.strip().lower() == "results":
            results_index = i
            break

    if results_index is None:
        continue

    # Everything after Results
    after_results = [
        line.strip()
        for line in lines[results_index + 1:]
        if line.strip()
    ]

    if not after_results:
        empty_results.append(doc)
        continue

    # Check whether there is conventional table-like formatting
    pipe_lines = [
        line
        for line in after_results
        if line.count("|") >= 2 or "!!" in line
    ]

    if pipe_lines:
        results_with_pipe_table.append(doc)
    else:
        results_with_content.append(doc)


print("=" * 80)
print("RESULTS CONTENT AUDIT")
print("=" * 80)

print("Olympic documents:", len(olympic_docs))
print()
print("Results section completely empty:", len(empty_results))
print("Results section has content:", len(results_with_content))
print("Results has pipe/table formatting:", len(results_with_pipe_table))


print()
print("=" * 80)
print("EMPTY RESULTS SECTIONS")
print("=" * 80)

for doc in sorted(
    empty_results,
    key=lambda x: x.get("approx_tokens", 0)
):
    print(
        f"{doc.get('approx_tokens', 0):>6} tokens | "
        f"{doc.get('doc_id')} | "
        f"{doc.get('title')}"
    )


print()
print("=" * 80)
print("RESULTS WITH CONTENT BUT NO PIPE TABLE")
print("=" * 80)

for doc in sorted(
    results_with_content,
    key=lambda x: x.get("approx_tokens", 0)
)[:100]:

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
    (
        d for d in olympic_docs
        if d.get("doc_id") == "Q1005331"
    ),
    None
)

if target:
    text = target["text"]
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.strip().lower() == "results":
            print("Results line:", i + 1)
            print("Content after Results:")

            content = [
                x.strip()
                for x in lines[i + 1:]
                if x.strip()
            ]

            if content:
                for x in content:
                    print("  ", x)
            else:
                print("  <EMPTY>")

            break


print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)