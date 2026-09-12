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


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title=settings.app_name, version=settings.app_version)
    storage = LocalDatasetStorage(settings.processed_data_dir)
    application.state.dataset_service = DatasetService(storage, settings.max_upload_size_mb)
    embeddings = EmbeddingService(settings.openai_embedding_model, settings.embedding_dimension, settings.ingestion_max_retries, settings.ingestion_retry_base_seconds)
    repository = TigerGraphRepository(create_connection, settings.ingestion_max_retries, settings.ingestion_retry_base_seconds)
    application.state.ingestion_service = IngestionService(storage, embeddings, repository, settings.embedding_batch_size, settings.embedding_version)

    @application.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    application.include_router(datasets_router)
    return application


app = create_app()
