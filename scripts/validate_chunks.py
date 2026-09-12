import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


def load_chunks():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_PATH}"
        )

    chunks = list(load_chunks())

    print()
    print("=" * 70)
    print("CHUNK VALIDATION")
    print("=" * 70)

    print(f"Total chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Chunk type distribution
    # --------------------------------------------------------

    type_counts = Counter(
        chunk["chunk_type"]
        for chunk in chunks
    )

    print()
    print("CHUNK TYPES")
    print("-" * 70)

    for chunk_type, count in type_counts.most_common():
        print(
            f"{chunk_type:15} {count:>8}"
        )

    # --------------------------------------------------------
    # Section distribution
    # --------------------------------------------------------

    section_counts = Counter(
        chunk["section"]
        for chunk in chunks
    )

    print()
    print("TOP 50 SECTIONS")
    print("-" * 70)

    for section, count in section_counts.most_common(50):
        print(
            f"{count:>6} | {section}"
        )

    # --------------------------------------------------------
    # Chunk size statistics
    # --------------------------------------------------------

    character_lengths = [
        len(chunk.get("text", ""))
        for chunk in chunks
    ]

    token_estimates = [
        len(chunk.get("text", "").split())
        for chunk in chunks
    ]

    print()
    print("CHUNK SIZE")
    print("-" * 70)

    print(
        f"Characters - min: {min(character_lengths):,}"
    )
    print(
        f"Characters - max: {max(character_lengths):,}"
    )
    print(
        f"Characters - avg: {sum(character_lengths) / len(character_lengths):,.2f}"
    )

    print(
        f"Words - min:      {min(token_estimates):,}"
    )
    print(
        f"Words - max:      {max(token_estimates):,}"
    )
    print(
        f"Words - avg:      {sum(token_estimates) / len(token_estimates):,.2f}"
    )

    # --------------------------------------------------------
    # Empty chunks
    # --------------------------------------------------------

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk.get("text", "").strip()
    ]

    print()
    print("EMPTY CHUNKS")
    print("-" * 70)

    print(
        f"Empty chunks: {len(empty_chunks)}"
    )

    # --------------------------------------------------------
    # Suspiciously short chunks
    # --------------------------------------------------------

    short_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.get("text", "").split()) < 10
    ]

    print()
    print("VERY SHORT CHUNKS")
    print("-" * 70)

    print(
        f"Chunks with <10 words: {len(short_chunks)}"
    )

    for chunk in short_chunks[:20]:
        print()
        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['title']} | "
            f"{chunk['section']}"
        )
        print(
            repr(chunk["text"][:300])
        )

    # --------------------------------------------------------
    # Very large sections
    # --------------------------------------------------------

    large_chunks = sorted(
        chunks,
        key=lambda x: len(x.get("text", "")),
        reverse=True,
    )[:20]

    print()
    print("20 LARGEST CHUNKS")
    print("-" * 70)

    for chunk in large_chunks:
        word_count = len(
            chunk.get("text", "").split()
        )

        print(
            f"{word_count:>7} words | "
            f"{chunk['chunk_type']:10} | "
            f"{chunk['title']} | "
            f"{chunk['section']}"
        )

    # --------------------------------------------------------
    # Inspect selected documents
    # --------------------------------------------------------

    selected_titles = [
        "Jurassic Park",
        "Mitt Romney",
        "Bob Dylan",
        "2006 Lebanon War",
    ]

    print()
    print("=" * 70)
    print("SELECTED DOCUMENT STRUCTURE")
    print("=" * 70)

    for title in selected_titles:

        document_chunks = [
            chunk
            for chunk in chunks
            if chunk["title"] == title
        ]

        if not document_chunks:
            print()
            print(f"{title}: NOT FOUND")
            continue

        print()
        print(f"{title}")
        print("-" * 70)

        for chunk in document_chunks:
            words = len(
                chunk.get("text", "").split()
            )

            print(
                f"{chunk['chunk_index']:>3} | "
                f"{chunk['chunk_type']:10} | "
                f"{words:>6} words | "
                f"{chunk['section']}"
            )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()