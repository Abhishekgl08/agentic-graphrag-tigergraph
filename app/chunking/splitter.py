import re
from .constants import TARGET_MIN_TOKENS, TARGET_MAX_TOKENS


def approximate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, int(len(re.findall(r"\S+", text)) * 1.3))


def is_table_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = sum(line.count("|") >= 2 or "!!" in line for line in lines)
    return bool(lines) and (("Results table" in text and rows >= 1) or rows >= 2)


def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip()) if s.strip()]


def split_large_paragraph(paragraph: str) -> list[str]:
    if approximate_tokens(paragraph) <= TARGET_MAX_TOKENS: return [paragraph]
    sentences = split_sentences(paragraph)
    if len(sentences) <= 1:
        chunks, current = [], []
        for word in paragraph.split():
            current.append(word)
            if approximate_tokens(" ".join(current)) >= TARGET_MIN_TOKENS:
                chunks.append(" ".join(current)); current = []
        return chunks + ([" ".join(current)] if current else [])
    chunks, current, tokens = [], [], 0
    for sentence in sentences:
        size = approximate_tokens(sentence)
        if current and tokens + size > TARGET_MAX_TOKENS:
            chunks.append(" ".join(current)); current, tokens = [], 0
        current.append(sentence); tokens += size
    return chunks + ([" ".join(current)] if current else [])


def chunk_prose(text: str) -> list[str]:
    pieces = [p for paragraph in split_into_paragraphs(text) for p in split_large_paragraph(paragraph)]
    chunks, current, tokens = [], [], 0
    for paragraph in pieces:
        size = approximate_tokens(paragraph)
        if not current: current, tokens = [paragraph], size
        elif tokens + size <= TARGET_MAX_TOKENS: current.append(paragraph); tokens += size
        else: chunks.append("\n\n".join(current)); current, tokens = [paragraph], size
    return chunks + (["\n\n".join(current)] if current else [])


def chunk_table(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    chunks, current, tokens = [], [], 0
    for line in lines:
        size = approximate_tokens(line)
        if current and tokens + size > TARGET_MAX_TOKENS:
            chunks.append("\n".join(current)); current, tokens = [], 0
        current.append(line); tokens += size
    return chunks + (["\n".join(current)] if current else [])
