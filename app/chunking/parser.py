from .constants import SECTION_NAMES


def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def is_section_heading(line: str) -> bool:
    text = line.strip()
    return bool(text) and "|" not in text and "!!" not in text and " ".join(text.lower().split()) in SECTION_NAMES


def split_infobox(text: str) -> tuple[str, str]:
    infobox_lines, body_lines = [], []
    inside_infobox = infobox_finished = False
    for line in text.splitlines():
        stripped = line.strip()
        if not infobox_finished and stripped.startswith("[Infobox"):
            inside_infobox = True; infobox_lines.append(line); continue
        if inside_infobox:
            if not stripped: continue
            if line.startswith("  "):
                infobox_lines.append(line); continue
            inside_infobox = False; infobox_finished = True; body_lines.append(line); continue
        body_lines.append(line)
    return "\n".join(infobox_lines).strip(), "\n".join(body_lines).strip()


def split_sections(body_text: str) -> list[dict]:
    sections, current_lines = [], []
    current_section, current_path = "introduction", ["introduction"]
    for line in body_text.splitlines():
        stripped = line.strip()
        if not stripped: continue
        if is_section_heading(stripped):
            if current_lines:
                sections.append({"section": current_section, "section_path": list(current_path), "text": "\n".join(current_lines).strip()})
            heading = stripped.strip(":").strip()
            current_section, current_path, current_lines = heading, [heading], []
        else: current_lines.append(line)
    if current_lines:
        sections.append({"section": current_section, "section_path": list(current_path), "text": "\n".join(current_lines).strip()})
    return sections
