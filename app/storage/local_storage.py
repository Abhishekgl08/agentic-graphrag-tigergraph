import json
import shutil
from pathlib import Path
from uuid import uuid4
from ..exceptions import DatasetNotFoundError
from ..models import DatasetPaths


class LocalDatasetStorage:
    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processing_dir = self.processed_dir.parent / "processing"

    def create(self, dataset_id: str | None = None) -> DatasetPaths:
        dataset_id = dataset_id or str(uuid4())
        working = self.processing_dir / dataset_id
        final = self.processed_dir / dataset_id
        working.mkdir(parents=True, exist_ok=False)
        return DatasetPaths(dataset_id, working, final)

    def commit(self, paths: DatasetPaths, metadata: dict, validation: dict) -> None:
        (paths.working_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (paths.working_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        (paths.working_dir / "chunks.tmp.jsonl").replace(paths.working_dir / "chunks.jsonl")
        paths.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if paths.final_dir.exists(): raise FileExistsError(paths.dataset_id)
        paths.working_dir.replace(paths.final_dir)

    def discard(self, paths: DatasetPaths) -> None:
        if paths.working_dir.exists(): shutil.rmtree(paths.working_dir)

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.processed_dir / dataset_id
        if not path.is_dir(): raise DatasetNotFoundError("Dataset not found")
        return path

    def metadata(self, dataset_id: str) -> dict:
        return json.loads((self.dataset_dir(dataset_id) / "metadata.json").read_text(encoding="utf-8"))

    def iter_chunks(self, dataset_id: str):
        with (self.dataset_dir(dataset_id) / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)
