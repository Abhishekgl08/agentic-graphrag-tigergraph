class DatasetError(Exception):
    """Base ingestion error."""


class DatasetValidationError(DatasetError):
    pass


class DatasetNotFoundError(DatasetError):
    pass


class DatasetConflictError(DatasetError):
    pass


class UploadTooLargeError(DatasetError):
    pass


class EmbeddingError(DatasetError):
    pass


class IngestionConflictError(DatasetError):
    pass


class GraphExtractionError(DatasetError):
    pass


class GroqQuotaError(GraphExtractionError):
    pass


class TigerGraphLoadError(GraphExtractionError):
    """A validated extraction could not be durably loaded into TigerGraph."""
