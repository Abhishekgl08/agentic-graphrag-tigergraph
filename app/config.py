from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic GraphRAG Dataset API"
    app_version: str = "1.0.0"
    environment: str = "development"
    data_dir: Path = Path("data")
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    max_upload_size_mb: int = 100
    chunk_target_min_tokens: int = 400
    chunk_target_max_tokens: int = 800
    small_chunk_threshold: int = 50
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 100
    embedding_version: str = "v1"
    ingestion_max_retries: int = 3
    ingestion_retry_base_seconds: float = 1.0
    rag_hnsw_k: int = 300
    rag_tfidf_k: int = 300
    rag_rrf_constant: int = 60
    rag_context_k: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-4o-mini"
    rag_vector_query_name: str = "vector_search_k300"
    groq_model: str = "openai/gpt-oss-120b"
    graph_corpus_path: Path = Path("data/processed/59aa1eeb-0c13-4f2c-9a11-b14f989cd14c/chunks.jsonl")
    graph_data_dir: Path = Path("graph/data/extraction")
    graph_state_dir: Path = Path("graph/state")
    graph_max_documents: int = 2
    graph_max_retries: int = 3
    graph_validation_retries: int = 1
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_post_init(self, __context):
        project_root = Path(__file__).resolve().parents[1]
        if not self.data_dir.is_absolute(): self.data_dir = project_root / self.data_dir
        if self.raw_data_dir is None: self.raw_data_dir = self.data_dir / "raw"
        elif not self.raw_data_dir.is_absolute(): self.raw_data_dir = project_root / self.raw_data_dir
        if self.processed_data_dir is None: self.processed_data_dir = self.data_dir / "processed"
        elif not self.processed_data_dir.is_absolute(): self.processed_data_dir = project_root / self.processed_data_dir
        if not self.graph_data_dir.is_absolute(): self.graph_data_dir = project_root / self.graph_data_dir
        if not self.graph_state_dir.is_absolute(): self.graph_state_dir = project_root / self.graph_state_dir
        if not self.graph_corpus_path.is_absolute(): self.graph_corpus_path = project_root / self.graph_corpus_path


settings = Settings()
