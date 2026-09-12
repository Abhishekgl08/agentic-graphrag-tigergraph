import json

DOC_ID = "Q1005331"
INPUT_PATH = "data/raw/corpus.jsonl"


def is_section_heading(line: str) -> bool:
    text = line.strip()

    if not text:
        return False

    if "|" in text or "!!" in text:
        return False

    normalized = " ".join(text.lower().split())

    known_sections = {
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
    }

    return normalized in known_sections


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
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            current_section = stripped.strip(":").strip()
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_section,
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


def create_chunks(record):
    doc_id = record["doc_id"]
    title = record["title"]
    text = record["text"]

    infobox, body = split_infobox(text)

    chunks = []

    if infobox:
        chunks.append(
            {
                "chunk_index": 0,
                "chunk_type": "infobox",
                "section": "infobox",
                "text": infobox,
            }
        )

    sections = split_sections(body)

    next_index = len(chunks)

    for section in sections:
        section_name = section["section"]
        section_text = section["text"]

        if not section_text:
            continue

        chunk_type = (
            "table"
            if is_table_like(section_text)
            else "text"
        )

        chunks.append(
            {
                "chunk_index": next_index,
                "chunk_type": chunk_type,
                "section": section_name,
                "text": section_text,
            }
        )

        next_index += 1

    return chunks


# ---------------------------------------------------------
# Load the requested document
# ---------------------------------------------------------

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    doc = None

    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record["doc_id"] == DOC_ID:
            doc = record
            break


if doc is None:
    raise ValueError(f"Document {DOC_ID} not found")


# ---------------------------------------------------------
# Parse it
# ---------------------------------------------------------

chunks = create_chunks(doc)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 80)
print("DOCUMENT")
print("=" * 80)

print("Doc ID:", doc["doc_id"])
print("Title:", doc["title"])
print("Original approx tokens:", doc["approx_tokens"])
print("Original characters:", len(doc["text"]))
print("Structural chunks:", len(chunks))

for chunk in chunks:
    print("\n" + "-" * 80)

    print(
        f"CHUNK {chunk['chunk_index']} | "
        f"TYPE: {chunk['chunk_type']} | "
        f"SECTION: {chunk['section']}"
    )

    print("-" * 80)
    print(chunk["text"])