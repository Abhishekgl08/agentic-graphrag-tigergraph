import json
from pathlib import Path

from process_dataset import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"


def normalize_text(text: str) -> str:
    """
    Normalize formatting differences while preserving
    all meaningful text.
    """
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def reconstruct_document(chunks):
    """
    Reconstruct the original document from parsed chunks.

    Section headings are restored because the parser stores
    them separately as the 'section' metadata.
    """

    parts = []

    sorted_chunks = sorted(
        chunks,
        key=lambda x: x["chunk_index"]
    )

    for chunk in sorted_chunks:

        chunk_type = chunk["chunk_type"]
        section = chunk["section"]
        text = chunk["text"]

        # Infobox already contains its own heading:
        # [Infobox Olympic event]
        if chunk_type == "infobox":
            parts.append(text)
            continue

        # Restore section heading
        if section and section != "introduction":
            parts.append(section)

        # Add actual section content
        if text:
            parts.append(text)

    return "\n".join(parts)


def compare_content(original_text, reconstructed_text):
    """
    Compare normalized original and reconstructed text.
    """

    original = normalize_text(original_text)
    reconstructed = normalize_text(reconstructed_text)

    if original == reconstructed:
        return True, None

    original_lines = original.splitlines()
    reconstructed_lines = reconstructed.splitlines()

    max_len = max(
        len(original_lines),
        len(reconstructed_lines),
    )

    for i in range(max_len):

        original_line = (
            original_lines[i]
            if i < len(original_lines)
            else "<MISSING>"
        )

        reconstructed_line = (
            reconstructed_lines[i]
            if i < len(reconstructed_lines)
            else "<MISSING>"
        )

        if original_line != reconstructed_line:

            return False, {
                "line": i + 1,
                "original": original_line,
                "reconstructed": reconstructed_line,
            }

    return False, {
        "line": None,
        "original": "<UNKNOWN>",
        "reconstructed": "<UNKNOWN>",
    }


def audit():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    total_documents = 0
    exact_matches = 0
    differences = []

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

            chunks = create_chunks(record)

            reconstructed = reconstruct_document(chunks)

            matches, difference = compare_content(
                record["text"],
                reconstructed,
            )

            if matches:
                exact_matches += 1

            else:
                differences.append(
                    {
                        "line_number": line_number,
                        "doc_id": record["doc_id"],
                        "title": record["title"],
                        "difference": difference,
                    }
                )

    print("=" * 80)
    print("PARSER LOSSLESSNESS AUDIT")
    print("=" * 80)

    print(
        f"Documents checked:       {total_documents}"
    )

    print(
        f"Exact matches:           {exact_matches}"
    )

    print(
        f"Documents with changes:  {len(differences)}"
    )

    if total_documents:

        percentage = (
            exact_matches / total_documents
        ) * 100

        print(
            f"Lossless percentage:     {percentage:.2f}%"
        )

    print("=" * 80)

    if differences:

        print("\nFIRST DIFFERENCES")
        print("-" * 80)

        for item in differences[:20]:

            print(
                f"\nDocument: {item['doc_id']}"
            )

            print(
                f"Title:    {item['title']}"
            )

            print(
                f"Corpus line: {item['line_number']}"
            )

            diff = item["difference"]

            print(
                f"Difference line: {diff['line']}"
            )

            print(
                f"ORIGINAL:       {diff['original']}"
            )

            print(
                f"RECONSTRUCTED:  {diff['reconstructed']}"
            )

    else:

        print(
            "\nSUCCESS: Parser reconstruction is lossless "
            "under normalization."
        )


if __name__ == "__main__":
    audit()