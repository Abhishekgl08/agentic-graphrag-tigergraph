import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from process_dataset import create_chunks


TARGET_DOC_ID = "Q303623"


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as source:

        for line in source:
            record = json.loads(line)

            if record["doc_id"] != TARGET_DOC_ID:
                continue

            print("=" * 80)
            print("RAW DOCUMENT")
            print("=" * 80)

            print(record["text"])

            print("\n")
            print("=" * 80)
            print("PARSED CHUNKS")
            print("=" * 80)

            chunks = create_chunks(record)

            for chunk in chunks:
                print("\n" + "-" * 80)

                print(
                    f"CHUNK INDEX : {chunk['chunk_index']}"
                )

                print(
                    f"TYPE        : {chunk['chunk_type']}"
                )

                print(
                    f"SECTION     : {chunk['section']}"
                )

                print(
                    f"CHARS       : {len(chunk['text'])}"
                )

                print("\n" + chunk["text"])

            return

    print(
        f"Document {TARGET_DOC_ID} not found."
    )


if __name__ == "__main__":
    main()