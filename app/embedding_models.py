from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    wikidata_qid: str | None = None
    wikipedia_pageid: int | str | None = None
    chunk_index: int = Field(ge=0)
    chunk_type: str
    section: str
    section_path: list[str]
    approx_tokens: int = Field(ge=1)
    text: str

    @field_validator("chunk_id", "doc_id", "title", "text")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip(): raise ValueError("must not be empty")
        return value


class IngestionJob(BaseModel):
    model_config = ConfigDict(extra="allow")
    job_id: str
    dataset_id: str
    status: str
    total_chunks: int = 0
    processed_chunks: int = 0
    embedded_chunks: int = 0
    stored_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0
    embedding_model: str
    embedding_version: str
    error: str | None = None
