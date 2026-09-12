import json
from pathlib import Path

from process_dataset import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def get_original_content_lines(text: str):
    """
    Return all non-empty original lines.

    We intentionally ignore section-heading structure here.
    The purpose is to detect actual content loss.
    """
    return [
        normalize_line(line)
        for line in text.splitlines()
        if line.strip()
    ]


def get_chunk_content_lines(chunks):
    """
    Return all lines contained in generated chunks.

    Section names are metadata and therefore excluded.
    """

    lines = []

    for chunk in sorted(
        chunks,
        key=lambda x: x["chunk_index"],
    ):
        for line in chunk["text"].splitlines():
            if line.strip():
                lines.append(normalize_line(line))

    return lines


def audit_document(record):
    chunks = create_chunks(record)

    original_lines = get_original_content_lines(
        record["text"]
    )

    chunk_lines = get_chunk_content_lines(chunks)

    # Multiset comparison rather than set comparison.
    #
    # This matters because the same line can legitimately
    # occur multiple times in a document.
    from collections import Counter

    original_counter = Counter(original_lines)
    chunk_counter = Counter(chunk_lines)

    missing = original_counter - chunk_counter
    extra = chunk_counter - original_counter

    return missing, extra


def audit():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    total_documents = 0
    documents_without_loss = 0
    documents_with_loss = 0

    total_missing_lines = 0
    total_extra_lines = 0

    examples = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as source:

        for line_number, line in enumerate(
            source,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            total_documents += 1

            missing, extra = audit_document(record)

            missing_count = sum(missing.values())
            extra_count = sum(extra.values())

            total_missing_lines += missing_count
            total_extra_lines += extra_count

            if missing_count == 0:
                documents_without_loss += 1
            else:
                documents_with_loss += 1

                if len(examples) < 20:
                    examples.append(
                        {
                            "line_number": line_number,
                            "doc_id": record["doc_id"],
                            "title": record["title"],
                            "missing": missing,
                            "extra": extra,
                        }
                    )

    print("=" * 80)
    print("PARSER CONTENT LOSS AUDIT")
    print("=" * 80)

    print(
        f"Documents checked:       {total_documents}"
    )

    print(
        f"Documents without loss:  {documents_without_loss}"
    )

    print(
        f"Documents with loss:     {documents_with_loss}"
    )

    print(
        f"Missing content lines:   {total_missing_lines}"
    )

    print(
        f"Extra content lines:     {total_extra_lines}"
    )

    if total_documents:
        percentage = (
            documents_without_loss
            / total_documents
        ) * 100

        print(
            f"Documents lossless:      {percentage:.2f}%"
        )

    print("=" * 80)

    if examples:

        print("\nFIRST DOCUMENTS WITH MISSING CONTENT")
        print("-" * 80)

        for example in examples:

            print(
                f"\nDocument: {example['doc_id']}"
            )

            print(
                f"Title:    {example['title']}"
            )

            print(
                f"Corpus line: {example['line_number']}"
            )

            print(
                "\nMISSING:"
            )

            for line, count in example["missing"].items():
                print(
                    f"  [{count}x] {line}"
                )

            if example["extra"]:

                print(
                    "\nEXTRA:"
                )

                for line, count in example["extra"].items():
                    print(
                        f"  [{count}x] {line}"
                    )

    else:

        print(
            "\nSUCCESS: No original content lines were lost."
        )


if __name__ == "__main__":
    audit()