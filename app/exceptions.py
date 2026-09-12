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
