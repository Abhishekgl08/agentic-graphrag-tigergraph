from pydantic import BaseModel, Field
from rag.models import RAGAnswer


class RAGQueryRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=10000)


class RAGQueryResponse(RAGAnswer):
    pass
