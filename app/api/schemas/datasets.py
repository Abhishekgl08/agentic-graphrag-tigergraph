from typing import Literal
from pydantic import BaseModel, Field


class DatasetResponse(BaseModel):
    dataset_id: str
    source_filename: str
    status: Literal["completed"]
    document_count: int
    chunk_count: int
    infobox_chunks: int
    text_chunks: int
    table_chunks: int
    approx_tokens: int
    chunking_version: str
    processing_duration_seconds: float


class ChunkResponse(BaseModel):
    chunk_id: str; doc_id: str; title: str; wikidata_qid: str | None = None; wikipedia_pageid: str | None = None
    chunk_index: int; chunk_type: Literal["text", "table", "infobox"]; section: str; section_path: list[str]; approx_tokens: int; text: str


class ChunkPage(BaseModel):
    items: list[ChunkResponse]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    count: int


class IngestionJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    total_chunks: int
    processed_chunks: int
    embedded_chunks: int
    stored_chunks: int
    skipped_chunks: int
    failed_chunks: int
    embedding_model: str
    embedding_version: str
    error: str | None = None
