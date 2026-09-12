import pytest
from app.exceptions import DatasetValidationError
from app.services.validation_service import ValidationService


def test_record_validation_rejects_missing_and_duplicates():
    validator, seen = ValidationService(), set()
    with pytest.raises(DatasetValidationError): validator.validate_record({"doc_id": "a"}, 1, seen)
    validator.validate_record({"doc_id": "a", "title": "A", "text": "text"}, 1, seen)
    with pytest.raises(DatasetValidationError): validator.validate_record({"doc_id": "a", "title": "A", "text": "text"}, 2, seen)
