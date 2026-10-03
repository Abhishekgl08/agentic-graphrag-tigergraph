import re
from threading import Lock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import RetrievedChunk


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


class TfidfRetriever:
    """Lazy, dataset-scoped TF-IDF unigram/bigram index."""
    def __init__(self, chunk_loader):
        self.chunk_loader, self._indexes, self._lock = chunk_loader, {}, Lock()

    def search(self, dataset_id: str, question: str, limit: int) -> list[RetrievedChunk]:
        vectorizer, matrix, chunks = self._index(dataset_id)
        similarities = cosine_similarity(vectorizer.transform([normalize(question)]), matrix).ravel()
        indices = similarities.argsort()[::-1][:limit]
        return [chunks[index].model_copy(update={"score": float(similarities[index])}) for index in indices]

    def _index(self, dataset_id):
        with self._lock:
            if dataset_id not in self._indexes:
                chunks = [RetrievedChunk.model_validate(raw) for raw in self.chunk_loader(dataset_id)]
                if not chunks: raise ValueError("Dataset has no chunks")
                vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=1)
                matrix = vectorizer.fit_transform([normalize(chunk.text) for chunk in chunks])
                self._indexes[dataset_id] = (vectorizer, matrix, chunks)
            return self._indexes[dataset_id]
