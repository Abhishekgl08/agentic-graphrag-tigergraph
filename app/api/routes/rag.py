from fastapi import APIRouter, Depends, HTTPException, Request
from ..schemas.rag import RAGQueryRequest, RAGQueryResponse
from ...exceptions import DatasetNotFoundError

router = APIRouter(prefix="/v1/rag", tags=["traditional-rag"])


def pipeline(request: Request): return request.app.state.rag_pipeline


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest, rag_pipeline=Depends(pipeline)):
    try: return rag_pipeline.answer(request.dataset_id, request.question)
    except DatasetNotFoundError: raise HTTPException(404, "Dataset not found")
    except ValueError as exc: raise HTTPException(422, str(exc))
