from .constants import SMALL_CHUNK_THRESHOLD, TARGET_MAX_TOKENS, TARGET_MIN_TOKENS
from .parser import normalize_text, split_infobox, split_sections
from .splitter import approximate_tokens, chunk_prose, chunk_table, is_table_like


def build_chunk(record, chunk_id, chunk_index, chunk_type, section, section_path, text):
    return {"chunk_id": chunk_id, "doc_id": record["doc_id"], "title": record["title"], "wikidata_qid": record.get("wikidata_qid"), "wikipedia_pageid": record.get("wikipedia_pageid"), "chunk_index": chunk_index, "chunk_type": chunk_type, "section": section, "section_path": section_path, "approx_tokens": approximate_tokens(text), "text": text}


def merge_small_chunks(chunks):
    if len(chunks) <= 1 or sum(c["approx_tokens"] for c in chunks) <= TARGET_MIN_TOKENS: return chunks
    result, i = [], 0
    while i < len(chunks):
        current = chunks[i]
        if current["chunk_type"] == "infobox" or current["approx_tokens"] >= SMALL_CHUNK_THRESHOLD:
            result.append(current); i += 1; continue
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if current["doc_id"] == nxt["doc_id"] and current["chunk_type"] == nxt["chunk_type"] and current["approx_tokens"] + nxt["approx_tokens"] <= TARGET_MAX_TOKENS:
                merged = dict(nxt); merged["text"] = current["text"] + "\n\n" + nxt["text"]; merged["approx_tokens"] = approximate_tokens(merged["text"])
                if current["section"] != nxt["section"]: merged["section"] = f"{current['section']} + {nxt['section']}"
                result.append(merged); i += 2; continue
        if result and result[-1]["doc_id"] == current["doc_id"] and result[-1]["chunk_type"] == current["chunk_type"] and result[-1]["approx_tokens"] + current["approx_tokens"] <= TARGET_MAX_TOKENS:
            previous = result[-1]; previous["text"] += "\n\n" + current["text"]; previous["approx_tokens"] = approximate_tokens(previous["text"])
            if previous["section"] != current["section"]: previous["section"] += f" + {current['section']}"
        else: result.append(current)
        i += 1
    for index, chunk in enumerate(result):
        chunk["chunk_index"] = index; chunk["chunk_id"] = f"{chunk['doc_id']}_infobox" if chunk["chunk_type"] == "infobox" else f"{chunk['doc_id']}_chunk_{index:04d}"
    return result


def create_chunks(record):
    doc_id = record["doc_id"]; infobox, body = split_infobox(record["text"]); chunks = []
    if infobox: chunks.append(build_chunk(record, f"{doc_id}_infobox", 0, "infobox", "infobox", ["infobox"], normalize_text(infobox)))
    for section in split_sections(body):
        kind = "table" if is_table_like(section["text"]) else "text"
        pieces = chunk_table(section["text"]) if kind == "table" else chunk_prose(section["text"])
        for piece in pieces:
            if piece.strip(): chunks.append(build_chunk(record, f"{doc_id}_chunk_{len(chunks):04d}", len(chunks), kind, section["section"], section["section_path"], piece.strip()))
    return merge_small_chunks(chunks)
