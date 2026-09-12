import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_chunks.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_for_comparison(text):
    """
    Normalize formatting while preserving actual textual tokens.

    We intentionally:
    - ignore whitespace
    - ignore section-heading-only lines
    - preserve words, numbers, punctuation, table markers, etc.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    tokens = re.findall(r"\S+", text)

    return tokens


def remove_section_headings(text):
    """
    Remove standalone section-heading lines.

    These are represented in the processed dataset as metadata
    rather than duplicated inside the chunk text.
    """
    lines = text.splitlines()
    kept = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # A standalone heading has no table syntax and is short.
        # We only remove lines that are clearly headings based on
        # the same known section vocabulary used by the chunker.
        normalized = " ".join(stripped.lower().split())

        if (
            "|" not in stripped
            and "!!" not in stripped
            and normalized in SECTION_NAMES
        ):
            continue

        kept.append(stripped)

    return "\n".join(kept)


SECTION_NAMES = {
    "early life",
    "early life and career",
    "early life and education",
    "career",
    "personal life",
    "education",
    "history",
    "background",
    "legacy",
    "reception",
    "critical reception",
    "critical response",
    "accolades",
    "awards",
    "filmography",
    "discography",
    "bibliography",
    "references",
    "see also",
    "external links",
    "plot",
    "cast",
    "production",
    "development",
    "writing",
    "new writers",
    "casting",
    "filming",
    "post-production",
    "soundtrack",
    "release",
    "box office",
    "home media",
    "television premiere",
    "marketing",
    "sequel",
    "sequels",
    "music",
    "impact",
    "franchise",
    "theatrical",
    "re-releases",
    "effects",
    "design",
    "summary",
    "qualification",
    "qualifications",
    "competition format",
    "schedule",
    "records",
    "results",
    "results table",
    "heats",
    "semifinals",
    "quarterfinals",
    "final",
    "finals",
    "qualifying",
    "qualifying round",
    "repechage",
    "reserves",
    "political career",
    "political positions",
    "tenure",
    "committee assignments",
    "operations",
    "products and services",
    "management",
    "financial information",
    "corporate affairs",
    "controversies",
}


def main():
    raw_docs = load_jsonl(RAW_PATH)
    chunks = load_jsonl(CHUNKS_PATH)

    chunks_by_doc = {}

    for chunk in chunks:
        chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    print("=" * 70)
    print("FINAL CONTENT INTEGRITY CHECK")
    print("=" * 70)

    print(f"Raw documents:    {len(raw_docs):,}")
    print(f"Processed chunks: {len(chunks):,}")

    missing_docs = []
    extra_docs = []
    exact_matches = 0
    token_mismatches = []

    total_original_tokens = 0
    total_chunk_tokens = 0

    for record in raw_docs:
        doc_id = record["doc_id"]

        original = remove_section_headings(record["text"])
        original_tokens = normalize_for_comparison(original)

        doc_chunks = chunks_by_doc.get(doc_id, [])

        reconstructed = "\n".join(
            chunk["text"]
            for chunk in sorted(
                doc_chunks,
                key=lambda x: x["chunk_index"],
            )
        )

        reconstructed_tokens = normalize_for_comparison(reconstructed)

        total_original_tokens += len(original_tokens)
        total_chunk_tokens += len(reconstructed_tokens)

        if original_tokens == reconstructed_tokens:
            exact_matches += 1
            continue

        original_counter = Counter(original_tokens)
        reconstructed_counter = Counter(reconstructed_tokens)

        missing = original_counter - reconstructed_counter
        extra = reconstructed_counter - original_counter

        if missing or extra:
            token_mismatches.append(
                {
                    "doc_id": doc_id,
                    "title": record.get("title", ""),
                    "missing": missing,
                    "extra": extra,
                }
            )

    raw_doc_ids = {doc["doc_id"] for doc in raw_docs}
    chunk_doc_ids = set(chunks_by_doc)

    missing_docs = raw_doc_ids - chunk_doc_ids
    extra_docs = chunk_doc_ids - raw_doc_ids

    print()
    print("DOCUMENT COVERAGE")
    print("-" * 70)
    print(f"Documents missing chunks: {len(missing_docs)}")
    print(f"Unknown chunk documents:  {len(extra_docs)}")

    print()
    print("TOKEN COVERAGE")
    print("-" * 70)
    print(f"Original tokens:      {total_original_tokens:,}")
    print(f"Reconstructed tokens: {total_chunk_tokens:,}")

    if total_original_tokens:
        ratio = total_chunk_tokens / total_original_tokens
        print(f"Token ratio:          {ratio:.6f}")

    print()
    print("DOCUMENT COMPARISON")
    print("-" * 70)
    print(f"Exact token matches:  {exact_matches:,}")
    print(f"Token mismatches:     {len(token_mismatches):,}")

    if token_mismatches:
        print()
        print("FIRST 20 TRUE TOKEN DIFFERENCES")
        print("-" * 70)

        for item in token_mismatches[:20]:
            print()
            print(f"Doc:   {item['doc_id']}")
            print(f"Title: {item['title']}")

            missing = list(item["missing"].items())[:15]
            extra = list(item["extra"].items())[:15]

            if missing:
                print("Missing tokens:")
                print("  ", missing)

            if extra:
                print("Extra tokens:")
                print("  ", extra)

    print()
    print("=" * 70)

    if missing_docs or extra_docs or token_mismatches:
        print("STATUS: REVIEW REQUIRED")
        print("=" * 70)
        raise SystemExit(1)

    print("STATUS: PASS — NO TRUE TOKEN LOSS DETECTED")
    print("=" * 70)


if __name__ == "__main__":
    main()