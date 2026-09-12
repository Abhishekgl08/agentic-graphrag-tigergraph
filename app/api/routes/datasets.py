from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from ...exceptions import DatasetNotFoundError, DatasetValidationError, UploadTooLargeError, IngestionConflictError
from ..schemas.datasets import ChunkPage, ChunkResponse, DatasetResponse, IngestionJobResponse

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


def service(request: Request): return request.app.state.dataset_service
def ingestion_service(request: Request): return request.app.state.ingestion_service


@router.post("", response_model=DatasetResponse, status_code=201)
def upload_dataset(file: UploadFile = File(...), dataset_service=Depends(service)):
    if not file.filename or not file.filename.lower().endswith(".jsonl"): raise HTTPException(400, "A .jsonl file is required")
    if file.content_type and file.content_type not in {"application/json", "application/x-ndjson", "application/jsonl", "text/plain", "application/octet-stream"}: raise HTTPException(400, "Unsupported content type")
    try: return dataset_service.ingest(file.file, file.filename)
    except UploadTooLargeError as exc: raise HTTPException(413, str(exc))
    except DatasetValidationError as exc: raise HTTPException(422, str(exc))


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, dataset_service=Depends(service)):
    try: return dataset_service.get_metadata(dataset_id)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")


@router.get("/{dataset_id}/chunks", response_model=ChunkPage)
def list_chunks(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), dataset_service=Depends(service)):
    try: return dataset_service.list_chunks(dataset_id, offset, limit)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")


@router.get("/{dataset_id}/chunks/{chunk_id}", response_model=ChunkResponse)
def get_chunk(dataset_id: str, chunk_id: str, dataset_service=Depends(service)):
    try: chunk = dataset_service.get_chunk(dataset_id, chunk_id)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")
    if chunk is None: raise HTTPException(404, "Chunk not found")
    return chunk


@router.post("/{dataset_id}/ingest", response_model=IngestionJobResponse, status_code=202)
def ingest_dataset(dataset_id: str, background_tasks: BackgroundTasks, ingestion=Depends(ingestion_service)):
    try: job = ingestion.queue(dataset_id)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")
    except IngestionConflictError as exc: raise HTTPException(409, str(exc))
    background_tasks.add_task(ingestion.run, dataset_id, job.job_id)
    return job


@router.get("/{dataset_id}/ingest", response_model=IngestionJobResponse)
def get_ingestion_status(dataset_id: str, ingestion=Depends(ingestion_service)):
    try: job = ingestion.get_job(dataset_id)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")
    if job is None: raise HTTPException(404, "No ingestion job exists for this dataset")
    return job
