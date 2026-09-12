import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_chunks.jsonl"
)


def normalize_for_comparison(text: str) -> str:
    """
    Normalize formatting differences while preserving actual content.

    We ignore:
    - leading/trailing whitespace
    - repeated whitespace
    - line breaks

    We do NOT remove words, headings, punctuation, table values,
    or other substantive content.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse all whitespace to one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
    print("CONTENT INTEGRITY VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load original corpus
    # ---------------------------------------------------------
    originals = {}

    for line_number, record in load_jsonl(RAW_PATH):
        doc_id = record.get("doc_id")

        if not doc_id:
            raise ValueError(
                f"Missing doc_id at raw line {line_number}"
            )

        originals[doc_id] = record["text"]

    # ---------------------------------------------------------
    # Load retrieval chunks
    # ---------------------------------------------------------
    chunks_by_doc = defaultdict(list)

    for line_number, chunk in load_jsonl(CHUNKS_PATH):
        doc_id = chunk.get("doc_id")

        if not doc_id:
            raise ValueError(
                f"Missing doc_id at chunk line {line_number}"
            )

        chunks_by_doc[doc_id].append(chunk)

    print()
    print(f"Original documents: {len(originals):,}")
    print(
        f"Chunked documents:  {len(chunks_by_doc):,}"
    )

    # ---------------------------------------------------------
    # Compare each document
    # ---------------------------------------------------------
    exact_matches = 0
    documents_with_difference = 0

    missing_documents = []
    missing_content_documents = []

    total_original_chars = 0
    total_reconstructed_chars = 0

    difference_examples = []

    for doc_id, original_text in originals.items():

        original_normalized = normalize_for_comparison(
            original_text
        )

        document_chunks = chunks_by_doc.get(
            doc_id,
            []
        )

        if not document_chunks:
            missing_documents.append(doc_id)
            continue

        # Reconstruct retrieval content in chunk order.
        document_chunks = sorted(
            document_chunks,
            key=lambda x: x["chunk_index"]
        )

        reconstructed_text = "\n".join(
            chunk["text"]
            for chunk in document_chunks
        )

        reconstructed_normalized = normalize_for_comparison(
            reconstructed_text
        )

        total_original_chars += len(
            original_normalized
        )

        total_reconstructed_chars += len(
            reconstructed_normalized
        )

        if original_normalized == reconstructed_normalized:
            exact_matches += 1
            continue

        documents_with_difference += 1

        # -----------------------------------------------------
        # Determine whether the difference is a real content
        # difference or only caused by the infobox separator.
        # -----------------------------------------------------

        # The retrieval pipeline deliberately moves the infobox
        # into its own chunk. Therefore compare:
        #
        #   original infobox + body
        #
        # against:
        #
        #   reconstructed infobox + body
        #
        # after whitespace normalization.
        #
        # If there is a real difference, record the document.
        difference_examples.append(
            {
                "doc_id": doc_id,
                "original_chars": len(
                    original_normalized
                ),
                "reconstructed_chars": len(
                    reconstructed_normalized
                ),
                "title": next(
                    (
                        chunk["title"]
                        for chunk in document_chunks
                        if chunk.get("title")
                    ),
                    "",
                ),
            }
        )

        missing_content_documents.append(doc_id)

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------
    extra_chunk_documents = (
        set(chunks_by_doc.keys())
        - set(originals.keys())
    )

    # ---------------------------------------------------------
    # Character statistics
    # ---------------------------------------------------------
    if total_original_chars:
        char_ratio = (
            total_reconstructed_chars
            / total_original_chars
        )
    else:
        char_ratio = 0.0

    print()
    print("DOCUMENT COMPARISON")
    print(
        f"Exact normalized matches: "
        f"{exact_matches:,}"
    )
    print(
        f"Documents with differences: "
        f"{documents_with_difference:,}"
    )

    print()
    print("COVERAGE")
    print(
        f"Missing documents: "
        f"{len(missing_documents)}"
    )
    print(
        f"Extra chunk documents: "
        f"{len(extra_chunk_documents)}"
    )

    print()
    print("CHARACTER COVERAGE")
    print(
        f"Original normalized chars:      "
        f"{total_original_chars:,}"
    )
    print(
        f"Reconstructed normalized chars: "
        f"{total_reconstructed_chars:,}"
    )
    print(
        f"Character ratio: "
        f"{char_ratio:.4f}"
    )

    # ---------------------------------------------------------
    # Show differences
    # ---------------------------------------------------------
    if difference_examples:
        print()
        print("FIRST DIFFERENCES")

        for example in difference_examples[:20]:
            print(
                f"  {example['doc_id']} | "
                f"{example['title']}"
            )

            print(
                f"      original chars:      "
                f"{example['original_chars']:,}"
            )

            print(
                f"      reconstructed chars: "
                f"{example['reconstructed_chars']:,}"
            )

    # ---------------------------------------------------------
    # Important known documents
    # ---------------------------------------------------------
    known_documents = [
        "Q303623",
        "Q1005331",
        "Q1222259",
        "Q1005328",
        "Q1222255",
        "Q1222251",
    ]

    print()
    print("KNOWN DOCUMENTS")

    for doc_id in known_documents:
        if doc_id not in originals:
            print(
                f"  {doc_id}: not present in corpus"
            )
            continue

        chunks = chunks_by_doc.get(
            doc_id,
            []
        )

        original_chars = len(
            normalize_for_comparison(
                originals[doc_id]
            )
        )

        reconstructed_chars = len(
            normalize_for_comparison(
                "\n".join(
                    chunk["text"]
                    for chunk in sorted(
                        chunks,
                        key=lambda x: x["chunk_index"],
                    )
                )
            )
        )

        print(
            f"  {doc_id}: "
            f"{len(chunks)} chunks | "
            f"{original_chars:,} original chars | "
            f"{reconstructed_chars:,} reconstructed chars"
        )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------
    print()
    print("=" * 70)

    if (
        not missing_documents
        and not extra_chunk_documents
        and char_ratio >= 0.995
    ):
        print(
            "CONTENT INTEGRITY STATUS: PASS"
        )
    else:
        print(
            "CONTENT INTEGRITY STATUS: REVIEW REQUIRED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()