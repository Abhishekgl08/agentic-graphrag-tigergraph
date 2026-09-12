import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_chunks.jsonl"
)


TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 800

# Chunks below this size are candidates for merging.
# Very small source documents are protected separately.
SMALL_CHUNK_THRESHOLD = 50


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


def normalize_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def approximate_tokens(text: str) -> int:
    if not text:
        return 0

    words = re.findall(r"\S+", text)

    return max(1, int(len(words) * 1.3))


def is_section_heading(line: str) -> bool:
    text = line.strip()

    if not text:
        return False

    if "|" in text or "!!" in text:
        return False

    normalized = " ".join(text.lower().split())

    return normalized in SECTION_NAMES


def split_infobox(text: str):
    lines = text.splitlines()

    infobox_lines = []
    body_lines = []

    inside_infobox = False
    infobox_finished = False

    for line in lines:
        stripped = line.strip()

        if not infobox_finished and stripped.startswith("[Infobox"):
            inside_infobox = True
            infobox_lines.append(line)
            continue

        if inside_infobox:
            if stripped == "":
                continue

            if line.startswith("  "):
                infobox_lines.append(line)
                continue

            inside_infobox = False
            infobox_finished = True
            body_lines.append(line)
            continue

        body_lines.append(line)

    return (
        "\n".join(infobox_lines).strip(),
        "\n".join(body_lines).strip(),
    )


def split_sections(body_text: str):
    lines = body_text.splitlines()

    sections = []

    current_section = "introduction"
    current_path = ["introduction"]
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if is_section_heading(stripped):
            if current_lines:
                sections.append(
                    {
                        "section": current_section,
                        "section_path": list(current_path),
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            heading = stripped.strip(":").strip()

            current_section = heading
            current_path = [heading]
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_section,
                "section_path": list(current_path),
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections


def is_table_like(text: str) -> bool:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    table_rows = 0

    for line in lines:
        pipe_count = line.count("|")
        has_double_bang = "!!" in line

        if pipe_count >= 2 or has_double_bang:
            table_rows += 1

    if "Results table" in text and table_rows >= 1:
        return True

    if table_rows >= 2:
        return True

    return False


def split_into_paragraphs(text: str):
    raw_paragraphs = re.split(
        r"\n\s*\n+",
        text,
    )

    paragraphs = []

    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def split_sentences(text: str):
    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\[])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_large_paragraph(paragraph: str):
    paragraph_tokens = approximate_tokens(paragraph)

    if paragraph_tokens <= TARGET_MAX_TOKENS:
        return [paragraph]

    sentences = split_sentences(paragraph)

    if len(sentences) <= 1:
        words = paragraph.split()

        chunks = []
        current_words = []

        for word in words:
            current_words.append(word)

            current_text = " ".join(current_words)

            if (
                approximate_tokens(current_text)
                >= TARGET_MIN_TOKENS
            ):
                chunks.append(current_text)
                current_words = []

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = approximate_tokens(sentence)

        if (
            current_sentences
            and current_tokens + sentence_tokens
            > TARGET_MAX_TOKENS
        ):
            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


def chunk_prose(text: str):
    paragraphs = split_into_paragraphs(text)

    expanded_paragraphs = []

    for paragraph in paragraphs:
        expanded_paragraphs.extend(
            split_large_paragraph(paragraph)
        )

    chunks = []

    current_parts = []
    current_tokens = 0

    for paragraph in expanded_paragraphs:
        paragraph_tokens = approximate_tokens(paragraph)

        if not current_parts:
            current_parts.append(paragraph)
            current_tokens = paragraph_tokens
            continue

        if (
            current_tokens + paragraph_tokens
            <= TARGET_MAX_TOKENS
        ):
            current_parts.append(paragraph)
            current_tokens += paragraph_tokens
            continue

        chunks.append(
            "\n\n".join(current_parts)
        )

        current_parts = [paragraph]
        current_tokens = paragraph_tokens

    if current_parts:
        chunks.append(
            "\n\n".join(current_parts)
        )

    return chunks


def chunk_table(text: str):
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = approximate_tokens(line)

        if (
            current_lines
            and current_tokens + line_tokens
            > TARGET_MAX_TOKENS
        ):
            chunks.append(
                "\n".join(current_lines)
            )

            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append(
            "\n".join(current_lines)
        )

    return chunks


def build_chunk(
    record,
    chunk_id,
    chunk_index,
    chunk_type,
    section,
    section_path,
    text,
):
    return {
        "chunk_id": chunk_id,
        "doc_id": record["doc_id"],
        "title": record["title"],
        "wikidata_qid": record.get("wikidata_qid"),
        "wikipedia_pageid": record.get(
            "wikipedia_pageid"
        ),
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "section": section,
        "section_path": section_path,
        "approx_tokens": approximate_tokens(text),
        "text": text,
    }


def merge_small_chunks(chunks):
    """
    Safely merge tiny chunks with adjacent chunks.

    Important:
    - Do NOT merge infoboxes.
    - Do NOT merge across different documents.
    - Do NOT merge table chunks with prose.
    - Do NOT modify genuinely tiny source documents.
    - Prefer merging a small chunk forward.
    - If that would make the chunk too large, merge backward.
    """

    if len(chunks) <= 1:
        return chunks

    # If the entire document is tiny, preserve it exactly.
    document_tokens = sum(
        chunk["approx_tokens"]
        for chunk in chunks
    )

    if document_tokens <= TARGET_MIN_TOKENS:
        return chunks

    result = []
    i = 0

    while i < len(chunks):
        current = chunks[i]

        # Never merge infobox chunks.
        if current["chunk_type"] == "infobox":
            result.append(current)
            i += 1
            continue

        current_tokens = current["approx_tokens"]

        if current_tokens >= SMALL_CHUNK_THRESHOLD:
            result.append(current)
            i += 1
            continue

        # -----------------------------------------------------
        # Try merging with the next chunk.
        # -----------------------------------------------------
        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]

            same_doc = (
                current["doc_id"]
                == next_chunk["doc_id"]
            )

            compatible_type = (
                current["chunk_type"]
                == next_chunk["chunk_type"]
            )

            combined_tokens = (
                current_tokens
                + next_chunk["approx_tokens"]
            )

            if (
                same_doc
                and compatible_type
                and combined_tokens
                <= TARGET_MAX_TOKENS
            ):
                merged_text = (
                    current["text"]
                    + "\n\n"
                    + next_chunk["text"]
                )

                merged = dict(next_chunk)

                merged["text"] = merged_text
                merged["approx_tokens"] = (
                    approximate_tokens(merged_text)
                )

                # Preserve the more informative section.
                if (
                    current["section"]
                    != next_chunk["section"]
                ):
                    merged["section"] = (
                        f"{current['section']} + "
                        f"{next_chunk['section']}"
                    )

                result.append(merged)

                i += 2
                continue

        # -----------------------------------------------------
        # Otherwise try merging backward.
        # -----------------------------------------------------
        if result:
            previous = result[-1]

            same_doc = (
                previous["doc_id"]
                == current["doc_id"]
            )

            compatible_type = (
                previous["chunk_type"]
                == current["chunk_type"]
            )

            combined_tokens = (
                previous["approx_tokens"]
                + current_tokens
            )

            if (
                same_doc
                and compatible_type
                and combined_tokens
                <= TARGET_MAX_TOKENS
            ):
                merged_text = (
                    previous["text"]
                    + "\n\n"
                    + current["text"]
                )

                previous["text"] = merged_text

                previous["approx_tokens"] = (
                    approximate_tokens(
                        merged_text
                    )
                )

                if (
                    previous["section"]
                    != current["section"]
                ):
                    previous["section"] = (
                        f"{previous['section']} + "
                        f"{current['section']}"
                    )

                i += 1
                continue

        # -----------------------------------------------------
        # No safe merge.
        # Preserve the chunk.
        # -----------------------------------------------------
        result.append(current)
        i += 1

    # Re-index chunks after merging.
    for index, chunk in enumerate(result):
        chunk["chunk_index"] = index
        chunk["chunk_id"] = (
            f"{chunk['doc_id']}_chunk_{index:04d}"
            if chunk["chunk_type"] != "infobox"
            else f"{chunk['doc_id']}_infobox"
        )

    return result


def create_chunks(record):
    doc_id = record["doc_id"]

    infobox, body = split_infobox(
        record["text"]
    )

    chunks = []

    # ---------------------------------------------------------
    # Infobox
    # ---------------------------------------------------------
    if infobox:
        chunks.append(
            build_chunk(
                record=record,
                chunk_id=f"{doc_id}_infobox",
                chunk_index=0,
                chunk_type="infobox",
                section="infobox",
                section_path=["infobox"],
                text=normalize_text(infobox),
            )
        )

    # ---------------------------------------------------------
    # Body
    # ---------------------------------------------------------
    sections = split_sections(body)

    for section in sections:
        section_name = section["section"]
        section_path = section["section_path"]
        section_text = section["text"].strip()

        if not section_text:
            continue

        if is_table_like(section_text):
            pieces = chunk_table(section_text)
            chunk_type = "table"
        else:
            pieces = chunk_prose(section_text)
            chunk_type = "text"

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            chunks.append(
                build_chunk(
                    record=record,
                    chunk_id=(
                        f"{doc_id}_chunk_"
                        f"{len(chunks):04d}"
                    ),
                    chunk_index=len(chunks),
                    chunk_type=chunk_type,
                    section=section_name,
                    section_path=section_path,
                    text=piece,
                )
            )

    # Final cleanup pass.
    return merge_small_chunks(chunks)


def process_dataset():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input corpus not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_count = 0
    chunk_count = 0

    infobox_chunks = 0
    text_chunks = 0
    table_chunks = 0

    total_tokens = 0

    with (
        INPUT_PATH.open(
            "r",
            encoding="utf-8",
        ) as source,
        OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as output,
    ):
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            document_count += 1

            chunks = create_chunks(record)

            for chunk in chunks:
                output.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunk_count += 1
                total_tokens += chunk["approx_tokens"]

                if chunk["chunk_type"] == "infobox":
                    infobox_chunks += 1

                elif chunk["chunk_type"] == "table":
                    table_chunks += 1

                else:
                    text_chunks += 1

    print("=" * 70)
    print("RETRIEVAL CHUNKING COMPLETE")
    print("=" * 70)

    print(f"Input documents:       {document_count}")
    print(f"Output chunks:         {chunk_count}")
    print(f"Infobox chunks:        {infobox_chunks}")
    print(f"Text chunks:           {text_chunks}")
    print(f"Table chunks:          {table_chunks}")
    print(f"Approx chunk tokens:   {total_tokens:,}")

    print()
    print("Input:")
    print(f"  {INPUT_PATH}")

    print()
    print("Output:")
    print(f"  {OUTPUT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    process_dataset()