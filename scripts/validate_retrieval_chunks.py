import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_chunks.jsonl"
)


REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "title",
    "chunk_index",
    "chunk_type",
    "section",
    "section_path",
    "approx_tokens",
    "text",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc


def main():
    print("=" * 70)
    print("RETRIEVAL CHUNK VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load original documents
    # ---------------------------------------------------------
    documents = {}

    for line_number, record in load_jsonl(RAW_PATH):
        doc_id = record.get("doc_id")

        if not doc_id:
            raise ValueError(
                f"Missing doc_id in raw corpus line {line_number}"
            )

        documents[doc_id] = record

    print()
    print("RAW CORPUS")
    print(f"Documents: {len(documents):,}")

    # ---------------------------------------------------------
    # 2. Load retrieval chunks
    # ---------------------------------------------------------
    chunks = []

    for line_number, chunk in load_jsonl(CHUNKS_PATH):
        missing = REQUIRED_FIELDS - set(chunk.keys())

        if missing:
            raise ValueError(
                f"Chunk line {line_number} is missing fields: "
                f"{sorted(missing)}"
            )

        chunks.append(chunk)

    print()
    print("RETRIEVAL CHUNKS")
    print(f"Chunks:    {len(chunks):,}")

    # ---------------------------------------------------------
    # 3. Basic validation
    # ---------------------------------------------------------
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    duplicate_chunk_ids = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    ]

    empty_chunks = [
        chunk
        for chunk in chunks
        if not str(chunk["text"]).strip()
    ]

    missing_doc_ids = [
        chunk
        for chunk in chunks
        if chunk["doc_id"] not in documents
    ]

    invalid_chunk_types = [
        chunk
        for chunk in chunks
        if chunk["chunk_type"] not in {
            "text",
            "table",
            "infobox",
        }
    ]

    invalid_token_counts = [
        chunk
        for chunk in chunks
        if not isinstance(chunk["approx_tokens"], int)
        or chunk["approx_tokens"] <= 0
    ]

    print()
    print("BASIC CHECKS")
    print(f"Duplicate chunk IDs:    {len(duplicate_chunk_ids)}")
    print(f"Empty chunks:            {len(empty_chunks)}")
    print(f"Unknown doc IDs:         {len(missing_doc_ids)}")
    print(f"Invalid chunk types:     {len(invalid_chunk_types)}")
    print(f"Invalid token counts:    {len(invalid_token_counts)}")

    # ---------------------------------------------------------
    # 4. Document coverage
    # ---------------------------------------------------------
    docs_with_chunks = {
        chunk["doc_id"]
        for chunk in chunks
    }

    missing_documents = (
        set(documents.keys()) - docs_with_chunks
    )

    extra_documents = (
        docs_with_chunks - set(documents.keys())
    )

    print()
    print("DOCUMENT COVERAGE")
    print(f"Documents with chunks:   {len(docs_with_chunks):,}")
    print(f"Documents missing:       {len(missing_documents)}")
    print(f"Unknown documents:       {len(extra_documents)}")

    if missing_documents:
        print()
        print("First missing documents:")

        for doc_id in sorted(missing_documents)[:20]:
            record = documents[doc_id]

            print(
                f"  {doc_id} | {record.get('title')}"
            )

    # ---------------------------------------------------------
    # 5. Chunk type distribution
    # ---------------------------------------------------------
    type_counts = Counter(
        chunk["chunk_type"]
        for chunk in chunks
    )

    print()
    print("CHUNK TYPES")

    for chunk_type, count in type_counts.most_common():
        print(
            f"{chunk_type:12s}: {count:>7,}"
        )

    # ---------------------------------------------------------
    # 6. Token distribution
    # ---------------------------------------------------------
    token_counts = [
        chunk["approx_tokens"]
        for chunk in chunks
    ]

    token_counts_sorted = sorted(token_counts)

    def percentile(values, p):
        index = int((len(values) - 1) * p)

        return values[index]

    print()
    print("TOKEN DISTRIBUTION")

    print(
        f"Minimum:     {min(token_counts_sorted):,}"
    )
    print(
        f"P25:         {percentile(token_counts_sorted, 0.25):,}"
    )
    print(
        f"Median:      {percentile(token_counts_sorted, 0.50):,}"
    )
    print(
        f"P75:         {percentile(token_counts_sorted, 0.75):,}"
    )
    print(
        f"P90:         {percentile(token_counts_sorted, 0.90):,}"
    )
    print(
        f"P95:         {percentile(token_counts_sorted, 0.95):,}"
    )
    print(
        f"P99:         {percentile(token_counts_sorted, 0.99):,}"
    )
    print(
        f"Maximum:     {max(token_counts_sorted):,}"
    )

    # ---------------------------------------------------------
    # 7. Very small chunks
    # ---------------------------------------------------------
    small_chunks = [
        chunk
        for chunk in chunks
        if chunk["approx_tokens"] < 50
    ]

    tiny_chunks = [
        chunk
        for chunk in chunks
        if chunk["approx_tokens"] < 10
    ]

    print()
    print("SMALL CHUNKS")

    print(
        f"< 50 tokens: {len(small_chunks):,}"
    )
    print(
        f"< 10 tokens: {len(tiny_chunks):,}"
    )

    if tiny_chunks:
        print()
        print("Examples of <10-token chunks:")

        for chunk in tiny_chunks[:20]:
            print(
                f"  {chunk['chunk_id']} | "
                f"{chunk['approx_tokens']} tokens | "
                f"{chunk['section']} | "
                f"{chunk['text'][:100]!r}"
            )

    # ---------------------------------------------------------
    # 8. Very large chunks
    # ---------------------------------------------------------
    large_chunks = [
        chunk
        for chunk in chunks
        if chunk["approx_tokens"] > 1000
    ]

    huge_chunks = [
        chunk
        for chunk in chunks
        if chunk["approx_tokens"] > 2000
    ]

    print()
    print("LARGE CHUNKS")

    print(
        f"> 1,000 tokens: {len(large_chunks):,}"
    )
    print(
        f"> 2,000 tokens: {len(huge_chunks):,}"
    )

    if huge_chunks:
        print()
        print("Largest chunks:")

        for chunk in sorted(
            huge_chunks,
            key=lambda x: x["approx_tokens"],
            reverse=True,
        )[:20]:
            print(
                f"  {chunk['approx_tokens']:>6,} tokens | "
                f"{chunk['chunk_type']:8s} | "
                f"{chunk['doc_id']} | "
                f"{chunk['section']}"
            )

    # ---------------------------------------------------------
    # 9. Chunks per document
    # ---------------------------------------------------------
    chunks_per_doc = Counter(
        chunk["doc_id"]
        for chunk in chunks
    )

    print()
    print("CHUNKS PER DOCUMENT")

    values = list(chunks_per_doc.values())

    print(
        f"Minimum:     {min(values):,}"
    )
    print(
        f"Median:      {percentile(sorted(values), 0.50):,}"
    )
    print(
        f"Maximum:     {max(values):,}"
    )

    # ---------------------------------------------------------
    # 10. Verify important known documents
    # ---------------------------------------------------------
    important_doc_ids = {
        "Q1005331",
        "Q1222259",
        "Q1005328",
        "Q1222255",
        "Q1222251",
    }

    print()
    print("KNOWN SMALL OLYMPIC DOCUMENTS")

    for doc_id in important_doc_ids:
        matching = [
            chunk
            for chunk in chunks
            if chunk["doc_id"] == doc_id
        ]

        if not matching:
            print(
                f"  {doc_id}: MISSING"
            )
            continue

        total_tokens = sum(
            chunk["approx_tokens"]
            for chunk in matching
        )

        print(
            f"  {doc_id}: "
            f"{len(matching)} chunks, "
            f"{total_tokens:,} approx tokens"
        )

        for chunk in matching:
            print(
                f"      {chunk['chunk_id']} | "
                f"{chunk['chunk_type']} | "
                f"{chunk['section']} | "
                f"{chunk['approx_tokens']} tokens"
            )

    # ---------------------------------------------------------
    # 11. Section distribution
    # ---------------------------------------------------------
    sections = Counter(
        chunk["section"]
        for chunk in chunks
    )

    print()
    print("TOP SECTIONS")

    for section, count in sections.most_common(20):
        print(
            f"{count:>7,} | {section}"
        )

    # ---------------------------------------------------------
    # 12. Final status
    # ---------------------------------------------------------
    errors = (
        len(duplicate_chunk_ids)
        + len(empty_chunks)
        + len(missing_doc_ids)
        + len(invalid_chunk_types)
        + len(invalid_token_counts)
        + len(missing_documents)
        + len(extra_documents)
    )

    print()
    print("=" * 70)

    if errors == 0:
        print("VALIDATION STATUS: PASS")
    else:
        print(
            f"VALIDATION STATUS: CHECK REQUIRED "
            f"({errors} issues)"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()