import io
from fastapi.testclient import TestClient
from app.main import create_app
from app.services.dataset_service import DatasetService
from app.storage.local_storage import LocalDatasetStorage


def client(tmp_path):
    application = create_app()
    application.state.dataset_service = DatasetService(LocalDatasetStorage(tmp_path / "processed"), 1)
    return TestClient(application)


def test_dataset_upload_and_retrieval(tmp_path):
    api = client(tmp_path)
    response = api.post("/v1/datasets", files={"file": ("corpus.jsonl", io.BytesIO(b'{"doc_id":"one","title":"One","text":"Hello world."}\n'), "application/x-ndjson")})
    assert response.status_code == 201
    dataset_id = response.json()["dataset_id"]
    assert api.get("/health").json() == {"status": "ok"}
    page = api.get(f"/v1/datasets/{dataset_id}/chunks").json()
    assert page["count"] == 1
    assert api.get(f"/v1/datasets/{dataset_id}/chunks/{page['items'][0]['chunk_id']}").status_code == 200


def test_invalid_jsonl_is_controlled(tmp_path):
    response = client(tmp_path).post("/v1/datasets", files={"file": ("bad.jsonl", io.BytesIO(b"not-json\n"), "application/x-ndjson")})
    assert response.status_code == 422
