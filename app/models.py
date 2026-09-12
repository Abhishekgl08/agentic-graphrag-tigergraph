from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessingStats:
    document_count: int = 0
    chunk_count: int = 0
    infobox_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    approx_tokens: int = 0
    document_texts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetPaths:
    dataset_id: str
    working_dir: Path
    final_dir: Path
