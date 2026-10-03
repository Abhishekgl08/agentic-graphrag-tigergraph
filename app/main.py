from fastapi import FastAPI
from .api.routes.datasets import router as datasets_router
from .config import settings
from .logging_config import configure_logging
from .services.dataset_service import DatasetService
from .services.embedding_service import EmbeddingService
from .services.ingestion_service import IngestionService
from .storage.local_storage import LocalDatasetStorage
from .repositories.tigergraph_repository import TigerGraphRepository
from .tigergraph import create_connection
from rag.keyword_retriever import TfidfRetriever
from rag.reranker import CrossEncoderReranker
from rag.vector_retriever import TigerGraphVectorRetriever
from rag.pipeline import TraditionalRAGPipeline
from llm.client import AnswerGenerationClient
from .api.routes.rag import router as rag_router


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title=settings.app_name, version=settings.app_version)
    storage = LocalDatasetStorage(settings.processed_data_dir)
    application.state.dataset_service = DatasetService(storage, settings.max_upload_size_mb)
    embeddings = EmbeddingService(settings.openai_embedding_model, settings.embedding_dimension, settings.ingestion_max_retries, settings.ingestion_retry_base_seconds)
    repository = TigerGraphRepository(create_connection, settings.ingestion_max_retries, settings.ingestion_retry_base_seconds)
    application.state.ingestion_service = IngestionService(storage, embeddings, repository, settings.embedding_batch_size, settings.embedding_version)
    keyword = TfidfRetriever(storage.iter_chunks)
    vector = TigerGraphVectorRetriever(embeddings, create_connection, settings.rag_vector_query_name)
    application.state.rag_pipeline = TraditionalRAGPipeline(vector, keyword, CrossEncoderReranker(settings.reranker_model), AnswerGenerationClient(settings.llm_model), settings.rag_hnsw_k, settings.rag_rrf_constant, settings.rag_context_k, storage.dataset_dir)

    @application.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    application.include_router(datasets_router)
    application.include_router(rag_router)
    return application


app = create_app()
