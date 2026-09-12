import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_profile.json"


REQUIRED_FIELDS = [
    "doc_id",
    "title",
    "url",
    "wikidata_qid",
    "wikipedia_pageid",
    "approx_tokens",
    "text",
]


def profile_dataset():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    malformed_lines = 0

    token_values = []
    text_char_lengths = []

    missing_fields = Counter()

    doc_ids = Counter()
    titles = Counter()
    wikidata_qids = Counter()

    empty_text_docs = []
    longest_docs = []

    infobox_count = 0
    heading_count = 0
    table_like_count = 0

    samples = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                print(
                    f"WARNING: malformed JSON at line {line_number}"
                )
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                print(
                    f"WARNING: line {line_number} is not a JSON object"
                )
                continue

            total_records += 1

            # -------------------------------------------------
            # Missing fields
            # -------------------------------------------------

            for field in REQUIRED_FIELDS:
                value = record.get(field)

                if value is None or value == "":
                    missing_fields[field] += 1

            # -------------------------------------------------
            # IDs / titles
            # -------------------------------------------------

            doc_id = record.get("doc_id")
            title = record.get("title")
            qid = record.get("wikidata_qid")

            if doc_id:
                doc_ids[str(doc_id)] += 1

            if title:
                titles[str(title)] += 1

            if qid:
                wikidata_qids[str(qid)] += 1

            # -------------------------------------------------
            # Token statistics
            # -------------------------------------------------

            approx_tokens = record.get("approx_tokens")

            if isinstance(approx_tokens, (int, float)):
                token_values.append(approx_tokens)

            # -------------------------------------------------
            # Text statistics
            # -------------------------------------------------

            text = record.get("text") or ""

            text_length = len(text)
            text_char_lengths.append(text_length)

            if not text.strip():
                empty_text_docs.append(
                    {
                        "line": line_number,
                        "doc_id": doc_id,
                        "title": title,
                    }
                )

            # -------------------------------------------------
            # Corpus structure detection
            # -------------------------------------------------

            if "[Infobox" in text:
                infobox_count += 1

            if "\n==" in text or text.startswith("=="):
                heading_count += 1

            if "!!" in text or "\n|" in text:
                table_like_count += 1

            # -------------------------------------------------
            # Longest documents
            # -------------------------------------------------

            longest_docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "approx_tokens": approx_tokens,
                    "characters": text_length,
                }
            )

            # Keep a few examples
            if len(samples) < 5:
                samples.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "approx_tokens": approx_tokens,
                        "text_preview": text[:500],
                    }
                )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    if token_values:
        token_stats = {
            "count": len(token_values),
            "total": sum(token_values),
            "min": min(token_values),
            "max": max(token_values),
            "mean": statistics.mean(token_values),
            "median": statistics.median(token_values),
        }
    else:
        token_stats = {
            "count": 0,
            "total": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    if text_char_lengths:
        character_stats = {
            "count": len(text_char_lengths),
            "min": min(text_char_lengths),
            "max": max(text_char_lengths),
            "mean": statistics.mean(text_char_lengths),
            "median": statistics.median(text_char_lengths),
        }
    else:
        character_stats = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    # ---------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------

    duplicate_doc_ids = {
        key: count
        for key, count in doc_ids.items()
        if count > 1
    }

    duplicate_titles = {
        key: count
        for key, count in titles.items()
        if count > 1
    }

    duplicate_qids = {
        key: count
        for key, count in wikidata_qids.items()
        if count > 1
    }

    # Top 10 longest documents
    longest_docs = sorted(
        longest_docs,
        key=lambda x: x["approx_tokens"] or 0,
        reverse=True,
    )[:10]

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    report = {
        "dataset": {
            "path": str(CORPUS_PATH),
            "total_records": total_records,
            "malformed_lines": malformed_lines,
        },
        "approx_tokens": token_stats,
        "text_characters": character_stats,
        "missing_fields": dict(missing_fields),
        "duplicates": {
            "doc_ids": duplicate_doc_ids,
            "titles": duplicate_titles,
            "wikidata_qids": duplicate_qids,
        },
        "empty_text_documents": empty_text_docs,
        "structure": {
            "documents_with_infobox": infobox_count,
            "documents_with_headings": heading_count,
            "documents_with_table_like_content": table_like_count,
        },
        "longest_documents": longest_docs,
        "samples": samples,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)

    print(f"Corpus:              {CORPUS_PATH}")
    print(f"Total records:       {total_records}")
    print(f"Malformed lines:     {malformed_lines}")

    print()
    print("APPROX TOKEN STATISTICS")
    print("-" * 60)
    print(f"Documents counted:   {token_stats['count']}")
    print(f"Total tokens:        {token_stats['total']:,}")
    print(f"Minimum:             {token_stats['min']}")
    print(f"Maximum:             {token_stats['max']}")
    print(f"Mean:                {token_stats['mean']:.2f}")
    print(f"Median:              {token_stats['median']}")

    print()
    print("TEXT CHARACTER STATISTICS")
    print("-" * 60)
    print(f"Minimum:             {character_stats['min']}")
    print(f"Maximum:             {character_stats['max']}")
    print(f"Mean:                {character_stats['mean']:.2f}")
    print(f"Median:              {character_stats['median']}")

    print()
    print("MISSING FIELDS")
    print("-" * 60)

    for field in REQUIRED_FIELDS:
        print(
            f"{field:20} {missing_fields.get(field, 0)}"
        )

    print()
    print("DUPLICATES")
    print("-" * 60)
    print(
        f"Duplicate doc_ids:   {len(duplicate_doc_ids)}"
    )
    print(
        f"Duplicate titles:    {len(duplicate_titles)}"
    )
    print(
        f"Duplicate QIDs:      {len(duplicate_qids)}"
    )

    print()
    print("STRUCTURE")
    print("-" * 60)
    print(
        f"Infobox documents:   {infobox_count}"
    )
    print(
        f"Heading documents:   {heading_count}"
    )
    print(
        f"Table-like docs:     {table_like_count}"
    )

    print()
    print("LONGEST DOCUMENTS")
    print("-" * 60)

    for doc in longest_docs:
        print(
            f"{doc['approx_tokens']:>6} tokens | "
            f"{doc['title']}"
        )

    print()
    print(f"Report saved to:")
    print(OUTPUT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    profile_dataset()