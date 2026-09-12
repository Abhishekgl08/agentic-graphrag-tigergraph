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


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def is_heading(line):
    return line.strip() in SECTION_NAMES


def split_infobox(lines):
    infobox = []
    body = []

    inside = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[Infobox"):
            inside = True
            infobox.append(line)
            continue

        if inside:
            if line.startswith(" ") or line.startswith("\t") or not stripped:
                infobox.append(line)
                continue

            inside = False

        body.append(line)

    return infobox, body


def expected_content(record):
    lines = record["text"].splitlines()

    infobox, body = split_infobox(lines)

    kept = []

    for line in infobox:
        if line.strip():
            kept.append(line)

    for line in body:
        stripped = line.strip()

        if not stripped:
            continue

        if is_heading(stripped):
            continue

        kept.append(line)

    return normalize(" ".join(kept))


def load_chunks():
    chunks = defaultdict(list)

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                chunks[item["doc_id"]].append(item)

    return chunks


def find_difference(expected, actual):
    """
    Find the first position where the two normalized strings differ.
    """

    limit = min(len(expected), len(actual))

    pos = 0

    while pos < limit and expected[pos] == actual[pos]:
        pos += 1

    if pos == len(expected) and pos == len(actual):
        return None

    start = max(0, pos - 100)
    end_expected = min(len(expected), pos + 200)
    end_actual = min(len(actual), pos + 200)

    return {
        "position": pos,
        "expected_context": expected[start:end_expected],
        "actual_context": actual[start:end_actual],
    }


def main():

    print("=" * 80)
    print("CONTENT MISMATCH INVESTIGATION")
    print("=" * 80)

    chunks_by_doc = load_chunks()

    mismatch_count = 0
    total_missing_chars = 0

    with RAW_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            doc_id = record["doc_id"]
            title = record.get("title", "")

            expected = expected_content(record)

            chunks = sorted(
                chunks_by_doc.get(doc_id, []),
                key=lambda x: x["chunk_index"],
            )

            actual = normalize(
                " ".join(chunk["text"] for chunk in chunks)
            )

            if expected == actual:
                continue

            mismatch_count += 1

            difference = find_difference(expected, actual)

            delta = len(actual) - len(expected)

            total_missing_chars += max(0, -delta)

            print()
            print("-" * 80)
            print(f"DOCUMENT: {doc_id}")
            print(f"TITLE:    {title}")
            print(f"EXPECTED: {len(expected):,} chars")
            print(f"ACTUAL:   {len(actual):,} chars")
            print(f"DELTA:    {delta:+,}")

            if difference:

                print()
                print(f"FIRST DIFFERENCE POSITION: {difference['position']}")

                print()
                print("EXPECTED:")
                print(
                    "..." +
                    difference["expected_context"] +
                    "..."
                )

                print()
                print("ACTUAL:")
                print(
                    "..." +
                    difference["actual_context"] +
                    "..."
                )

            if mismatch_count >= 30:
                print()
                print("Stopping after first 30 mismatches.")
                break

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Mismatches inspected: {mismatch_count}")
    print(f"Missing characters from inspected docs: {total_missing_chars:,}")


if __name__ == "__main__":
    main()