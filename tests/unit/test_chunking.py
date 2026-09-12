from app.chunking.chunker import create_chunks
from app.chunking.parser import is_section_heading
from app.chunking.splitter import approximate_tokens, chunk_table, split_large_paragraph


def test_token_approximation_and_sections():
    assert approximate_tokens("one two") == 2
    assert is_section_heading("Career")
    assert not is_section_heading("a | b | c")


def test_infobox_table_and_indices_are_preserved():
    record = {"doc_id": "d1", "title": "Document", "text": "[Infobox Person]\n  name: Example\nCareer\nA short paragraph.\n\nResults\nA | B | C\n1 | 2 | 3"}
    chunks = create_chunks(record)
    assert chunks[0]["chunk_type"] == "infobox"
    assert any(chunk["chunk_type"] == "table" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_large_prose_and_tables_split():
    assert len(split_large_paragraph(" ".join(["word"] * 1000))) > 1
    assert len(chunk_table("\n".join("a | b | c " * 25 for _ in range(40)))) > 1
