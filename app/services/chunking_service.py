from ..chunking.chunker import create_chunks


class ChunkingService:
    def create_chunks(self, record: dict) -> list[dict]:
        return create_chunks(record)
