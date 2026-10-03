from threading import Lock
from .models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str): self.model_name, self._model, self._lock = model_name, None, Lock()

    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks: return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(question, chunk.text) for chunk in chunks], show_progress_bar=False)
        return [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda pair: float(pair[0]), reverse=True)]
