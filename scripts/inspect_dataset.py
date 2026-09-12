import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"


def load_records():
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            yield json.loads(line)


def print_document(record, index):
    print()
    print("=" * 100)
    print(f"DOCUMENT #{index}")
    print("=" * 100)

    print(f"doc_id:            {record.get('doc_id')}")
    print(f"title:             {record.get('title')}")
    print(f"wikidata_qid:      {record.get('wikidata_qid')}")
    print(f"wikipedia_pageid:  {record.get('wikipedia_pageid')}")
    print(f"approx_tokens:     {record.get('approx_tokens')}")
    print(f"url:               {record.get('url')}")

    text = record.get("text", "")

    print()
    print("-" * 100)
    print("TEXT")
    print("-" * 100)

    print(text[:8000])


def main():
    records = list(load_records())

    print(f"Total documents: {len(records)}")

    # ---------------------------------------------------------
    # Inspect selected documents
    # ---------------------------------------------------------

    selected_titles = [
        "Jurassic Park",
        "Mitt Romney",
        "Bill Clinton",
        "2006 Lebanon War",
        "Bob Dylan",
        "Vladimir Putin",
    ]

    print()
    print("SELECTED DOCUMENTS")
    print("-" * 100)

    found = 0

    for record in records:
        if record.get("title") in selected_titles:
            found += 1
            print_document(record, found)

    # ---------------------------------------------------------
    # Inspect largest documents
    # ---------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("LARGEST DOCUMENTS")
    print("=" * 100)

    largest = sorted(
        records,
        key=lambda r: r.get("approx_tokens", 0),
        reverse=True,
    )[:10]

    for i, record in enumerate(largest, start=1):
        print(
            f"{i:2}. "
            f"{record.get('approx_tokens'):>6} tokens | "
            f"{record.get('title')}"
        )

    # ---------------------------------------------------------
    # Inspect table-like documents
    # ---------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("TABLE-LIKE DOCUMENTS")
    print("=" * 100)

    table_docs = [
        record
        for record in records
        if "!!" in record.get("text", "")
        or "\n|" in record.get("text", "")
    ]

    for i, record in enumerate(table_docs[:10], start=1):
        text = record.get("text", "")

        print()
        print(
            f"{i}. {record.get('title')} "
            f"({record.get('approx_tokens')} tokens)"
        )

        print(text[:3000])


if __name__ == "__main__":
    main()