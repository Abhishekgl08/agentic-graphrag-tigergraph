import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "retrieval_chunks.jsonl"


SECTION_NAMES = {
    "Summary",
    "Background",
    "Early life",
    "Early life and education",
    "Education",
    "Career",
    "Political career",
    "Personal life",
    "Development",
    "Production",
    "Filming",
    "Casting",
    "Cast",
    "Plot",
    "Reception",
    "Critical response",
    "Critical reception",
    "Box office",
    "Release",
    "Home media",
    "Soundtrack",
    "Accolades",
    "Legacy",
    "Qualification",
    "Competition format",
    "Schedule",
    "Records",
    "Results",
    "Results table",
    "Heats",
    "Semifinals",
    "Quarterfinals",
    "Finals",
    "Final",
    "Repechage",
    "Music",
}


def normalize_text(text):
    """
    Normalize only whitespace.

    This intentionally preserves actual words, punctuation,
    numbers, table symbols, etc.
    """
    return re.sub(r"\s+", " ", text).strip()


def is_section_heading(line):
    return line.strip() in SECTION_NAMES


def split_infobox(lines):
    """
    Match the same basic infobox treatment used by the chunker.
    """
    if not lines:
        return [], []

    infobox = []
    body = []

    inside_infobox = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[Infobox"):
            inside_infobox = True
            infobox.append(line)
            continue

        if inside_infobox:
            if line.startswith(" ") or line.startswith("\t") or not stripped:
                infobox.append(line)
                continue

            inside_infobox = False

        body.append(line)

    return infobox, body


def expected_content(record):
    """
    Reconstruct exactly what should remain in retrieval chunks.

    Section-heading lines are excluded because the chunker stores
    section names as metadata rather than retrieval text.
    Everything else is retained.
    """
    text = record.get("text", "")

    lines = text.splitlines()

    infobox_lines, body_lines = split_infobox(lines)

    kept = []

    # Infobox content
    for line in infobox_lines:
        if line.strip():
            kept.append(line)

    # Main document content
    for line in body_lines:
        stripped = line.strip()

        if not stripped:
            continue

        if is_section_heading(stripped):
            continue

        kept.append(line)

    return normalize_text(" ".join(kept))


def load_chunks():
    chunks_by_doc = defaultdict(list)

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunks_by_doc[chunk["doc_id"]].append(chunk)

    return chunks_by_doc


def main():
    print("=" * 70)
    print("LOSSLESS CONTENT VALIDATION")
    print("=" * 70)

    chunks_by_doc = load_chunks()

    total_docs = 0
    exact_matches = 0
    mismatches = []

    total_expected_chars = 0
    total_actual_chars = 0

    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            doc_id = record["doc_id"]

            total_docs += 1

            expected = expected_content(record)

            chunks = sorted(
                chunks_by_doc.get(doc_id, []),
                key=lambda x: x["chunk_index"],
            )

            actual = normalize_text(
                " ".join(chunk["text"] for chunk in chunks)
            )

            total_expected_chars += len(expected)
            total_actual_chars += len(actual)

            if expected == actual:
                exact_matches += 1
            else:
                mismatches.append(
                    {
                        "doc_id": doc_id,
                        "title": record.get("title", ""),
                        "expected_chars": len(expected),
                        "actual_chars": len(actual),
                    }
                )

    print()
    print("DOCUMENTS")
    print(f"Original documents: {total_docs:,}")
    print(f"Chunked documents:  {len(chunks_by_doc):,}")

    print()
    print("CONTENT COMPARISON")
    print(f"Exact matches:      {exact_matches:,}")
    print(f"Mismatches:         {len(mismatches):,}")

    print()
    print("CHARACTER COUNTS")
    print(f"Expected characters: {total_expected_chars:,}")
    print(f"Actual characters:   {total_actual_chars:,}")

    if total_expected_chars:
        ratio = total_actual_chars / total_expected_chars
        print(f"Character ratio:     {ratio:.6f}")

    print()

    if not mismatches:
        print("=" * 70)
        print("STATUS: PASS")
        print("=" * 70)
        print()
        print(
            "All non-structural document content is preserved exactly "
            "after whitespace normalization."
        )
        return

    print("=" * 70)
    print("STATUS: REVIEW REQUIRED")
    print("=" * 70)

    print()
    print("FIRST MISMATCHES")

    for item in mismatches[:25]:
        print(
            f"{item['doc_id']} | "
            f"{item['title']} | "
            f"expected={item['expected_chars']} | "
            f"actual={item['actual_chars']} | "
            f"delta={item['actual_chars'] - item['expected_chars']}"
        )

    print()
    print("A mismatch means the actual retrieval text differs from the")
    print("expected text after removing only known section headings.")
    print("This should be investigated before embedding the corpus.")


if __name__ == "__main__":
    main()