import json
import os
import time
from openai import OpenAI
from app.config import settings
from app.exceptions import GraphExtractionError, GroqQuotaError
from graph.schema_contract import extraction_instructions

class GroqExtractor:
    def __init__(self, model: str, max_retries: int | None = None, client=None): self.model, self.max_retries, self._client = model, max_retries if max_retries is not None else settings.graph_max_retries, client
    @property
    def client(self):
        if self._client is None:
            key = os.getenv("GROQ_API_KEY")
            if not key: raise GraphExtractionError("Groq extraction service is not configured")
            self._client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        return self._client
    def extract(self, chunk: dict) -> dict:
        prompt = f"SOURCE CHUNK ID: {chunk['chunk_id']}\nDOCUMENT ID: {chunk['doc_id']}\nDOCUMENT TITLE: {chunk.get('title', '')}\nTEXT:\n{chunk['text']}"
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": extraction_instructions()}, {"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = response.choices[0].message.content
                if not content: raise GraphExtractionError("Groq returned an empty extraction")
                data = json.loads(content)
                usage = response.usage
                return {"document_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"], "model": self.model, "entities": data.get("entities"), "relationships": data.get("relationships"), "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0, "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "total_tokens": getattr(usage, "total_tokens", 0) or 0}, "status": "extracted"}
            except Exception as exc:
                if self._is_quota_error(exc): raise GroqQuotaError("Groq quota or credit limit reached") from exc
                if attempt + 1 == self.max_retries:
                    # Keep the provider's non-secret diagnostic in the journal;
                    # otherwise operations cannot distinguish an invalid model,
                    # auth failure, or temporary service failure.
                    raise GraphExtractionError(f"Groq extraction failed after retries: {exc}") from exc
                time.sleep((2, 5, 10)[min(attempt, 2)])
    @staticmethod
    def _is_quota_error(exc):
        text = str(exc).lower(); status = getattr(exc, "status_code", None)
        return status == 429 or any(term in text for term in ("quota", "credit", "billing", "rate limit", "daily token", "daily request", "insufficient"))
