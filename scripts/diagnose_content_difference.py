import json
import re
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


def normalize_line(line: str) -> str:
    """
    Normalize one line for comparison.

    We ignore whitespace differences only.
    """
    line = line.replace("\r", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


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


def is_infobox_line(line: str) -> bool:
    """
    Identify infobox lines.

    The infobox is preserved as its own chunk, so we don't need
    to treat its formatting as missing content.
    """
    stripped = line.strip()

    return (
        stripped.startswith("[Infobox")
        or line.startswith("  ")
    )


def looks_like_heading(line: str) -> bool:
    """
    Conservative heading detector for diagnostic purposes.
    """
    text = normalize_line(line)

    if not text:
        return False

    if "|" in text or "!!" in text:
        return False

    if len(text) > 100:
        return False

    words = text.split()

    if len(words) > 12:
        return False

    known_headings = {
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

    return text.lower() in known_headings


def build_line_counter(text: str):
    """
    Build a normalized line counter.

    Counter is used instead of sequence matching, making this
    extremely fast even for large documents.
    """
    counter = Counter()

    for line in text.splitlines():
        normalized = normalize_line(line)

        if normalized:
            counter[normalized] += 1

    return counter


def main():

    print("=" * 70)
    print("FAST CONTENT DIFFERENCE DIAGNOSTIC")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load original corpus
    # ---------------------------------------------------------
    originals = {}

    for line_number, record in load_jsonl(RAW_PATH):
        originals[record["doc_id"]] = record

    print()
    print(
        f"Original documents: "
        f"{len(originals):,}"
    )

    # ---------------------------------------------------------
    # Load retrieval chunks
    # ---------------------------------------------------------
    chunks_by_doc = defaultdict(list)

    for line_number, chunk in load_jsonl(CHUNKS_PATH):
        chunks_by_doc[chunk["doc_id"]].append(chunk)

    print(
        f"Chunked documents:  "
        f"{len(chunks_by_doc):,}"
    )

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------
    documents_exact = 0
    documents_with_difference = 0

    total_missing_chars = 0

    heading_missing_chars = 0
    structural_missing_chars = 0
    content_missing_chars = 0

    missing_line_counter = Counter()

    suspicious_documents = []

    # ---------------------------------------------------------
    # Document-by-document comparison
    # ---------------------------------------------------------
    for doc_id, record in originals.items():

        original_text = record["text"]

        document_chunks = sorted(
            chunks_by_doc.get(doc_id, []),
            key=lambda x: x["chunk_index"],
        )

        if not document_chunks:
            continue

        # -----------------------------------------------------
        # Create normalized line counters.
        # -----------------------------------------------------
        original_counter = build_line_counter(
            original_text
        )

        reconstructed_text = "\n".join(
            chunk["text"]
            for chunk in document_chunks
        )

        reconstructed_counter = build_line_counter(
            reconstructed_text
        )

        # -----------------------------------------------------
        # Find lines from original that aren't represented
        # in reconstructed content.
        # -----------------------------------------------------
        missing_lines = []

        for line, original_count in original_counter.items():

            reconstructed_count = (
                reconstructed_counter.get(line, 0)
            )

            difference = (
                original_count
                - reconstructed_count
            )

            if difference <= 0:
                continue

            for _ in range(difference):
                missing_lines.append(line)

        if not missing_lines:
            documents_exact += 1
            continue

        documents_with_difference += 1

        document_heading_chars = 0
        document_structural_chars = 0
        document_content_chars = 0

        for line in missing_lines:

            line_length = len(line)

            total_missing_chars += line_length

            missing_line_counter[line] += 1

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------
            if looks_like_heading(line):

                heading_missing_chars += line_length
                document_heading_chars += line_length

            elif (
                line.startswith("[")
                and "]" in line[:100]
            ):

                structural_missing_chars += line_length
                document_structural_chars += line_length

            else:

                content_missing_chars += line_length
                document_content_chars += line_length

        if document_content_chars > 0:

            suspicious_documents.append(
                {
                    "doc_id": doc_id,
                    "title": record["title"],
                    "missing_chars": (
                        document_heading_chars
                        + document_structural_chars
                        + document_content_chars
                    ),
                    "heading_chars": (
                        document_heading_chars
                    ),
                    "structural_chars": (
                        document_structural_chars
                    ),
                    "content_chars": (
                        document_content_chars
                    ),
                    "missing_lines": missing_lines,
                }
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print()
    print("DOCUMENT COMPARISON")

    print(
        f"Documents without detected differences: "
        f"{documents_exact:,}"
    )

    print(
        f"Documents with differences: "
        f"{documents_with_difference:,}"
    )

    # ---------------------------------------------------------
    # Missing-character classification
    # ---------------------------------------------------------
    print()
    print("MISSING CONTENT CLASSIFICATION")

    print(
        f"Total detected missing chars: "
        f"{total_missing_chars:,}"
    )

    print(
        f"Heading characters: "
        f"{heading_missing_chars:,}"
    )

    print(
        f"Structural characters: "
        f"{structural_missing_chars:,}"
    )

    print(
        f"Potential content characters: "
        f"{content_missing_chars:,}"
    )

    if total_missing_chars:

        print(
            f"Heading percentage: "
            f"{heading_missing_chars / total_missing_chars * 100:.2f}%"
        )

        print(
            f"Structural percentage: "
            f"{structural_missing_chars / total_missing_chars * 100:.2f}%"
        )

        print(
            f"Potential content percentage: "
            f"{content_missing_chars / total_missing_chars * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Most common missing lines
    # ---------------------------------------------------------
    print()
    print("MOST COMMON MISSING LINES")

    for line, count in missing_line_counter.most_common(30):

        print(
            f"{count:>5}x | {line!r}"
        )

    # ---------------------------------------------------------
    # Suspicious documents
    # ---------------------------------------------------------
    suspicious_documents.sort(
        key=lambda x: x["content_chars"],
        reverse=True,
    )

    print()
    print(
        "DOCUMENTS WITH POTENTIAL CONTENT LOSS"
    )

    print(
        f"Count: {len(suspicious_documents):,}"
    )

    for result in suspicious_documents[:20]:

        print()
        print(
            f"{result['doc_id']} | "
            f"{result['title']}"
        )

        print(
            f"Missing chars: "
            f"{result['missing_chars']}"
        )

        print(
            f"Heading chars: "
            f"{result['heading_chars']}"
        )

        print(
            f"Structural chars: "
            f"{result['structural_chars']}"
        )

        print(
            f"Potential content chars: "
            f"{result['content_chars']}"
        )

        print("Examples:")

        for line in result["missing_lines"][:10]:

            print(
                f"    {line!r}"
            )

    # ---------------------------------------------------------
    # Known documents
    # ---------------------------------------------------------
    known_ids = [
        "Q303623",
        "Q1005331",
        "Q1222259",
        "Q1005328",
        "Q1222255",
        "Q1222251",
    ]

    print()
    print("KNOWN DOCUMENTS")

    for doc_id in known_ids:

        result = next(
            (
                item
                for item in suspicious_documents
                if item["doc_id"] == doc_id
            ),
            None,
        )

        if result is None:

            print(
                f"  {doc_id}: "
                f"no potential content loss detected"
            )

            continue

        print()
        print(
            f"  {doc_id} | "
            f"{result['title']}"
        )

        print(
            f"      potential content chars: "
            f"{result['content_chars']}"
        )

        for line in result["missing_lines"][:10]:

            print(
                f"      {line!r}"
            )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()